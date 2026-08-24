"""Constrained residual flow refiner on top of frozen v11+governor (v14).

Motivation. The semantic carrier (v11 sentence-budget JEPA + governor) is the
current best pure-generative SLP checkpoint here:
test BLEU-4 = 12.417, WER = 82.135, DTW = 0.03515. Its remaining failure is
hand_speed ratio = 0.69 (gate target ≥ 0.85). v12 proved post-hoc amplification
can reach hand_speed ≈ 0.97 but at jerk ≈ 1.41 and -0.43 BLEU on test. v13
proved a learned hand-only carrier finetune collapses recognition. The honest
next move is a small bounded residual on the FROZEN v11+governor source pose,
trained against full Phoenix train sentences with explicit gate / preservation
constraints.

Architecture
------------
Wrist-coupled residual. The model emits three bounded per-frame residuals:

  delta_body        [B, T, 8,  3]   tanh * max_body
  delta_hand_local  [B, T, 42, 3]   tanh * max_hand   (LOCAL to attached wrist)
  delta_face        [B, T, 128,3]   tanh * max_face

Output pose construction (in raw SLRTP-178 coordinates):

  body_new[t]    = body_src[t] + delta_body[t]
  rh_local[t]    = rh_src[t] - rwrist_src[t]
  out[t, 8:29]   = body_new[t, 2:3] + rh_local[t] + delta_hand_local[t, 0:21]
  out[t,29:50]   = body_new[t, 5:6] + lh_local[t] + delta_hand_local[t,21:42]
  out[t,50:178]  = face_src[t]   + delta_face[t]

So the hand-internal articulation is corrected in the wrist frame (the delta
lives where it needs to live for amplification to make physical sense), and the
hand simultaneously follows whatever small wrist correction body_new received.

The network is a 3-layer Transformer encoder over per-frame tokens. Each token
sees: pose-frame projection, frame velocity projection, gloss-sentence context
(mean of sentence gloss embeddings), per-frame progress embedding, and
sinusoidal positional encoding. The pose decoder head splits into the three
bounded residual heads with zero-init final linear so the model starts at the
identity transformation.

Losses (computed in raw pose units, mean over batch)
----------------------------------------------------
  L_speed   : two-sided band on per-clip hand-speed-ratio vs GT
              relu(target_lo - ratio) + relu(ratio - target_hi)
              plus a log-ratio smooth-L1 toward 0.95
  L_jerk    : one-sided cap on per-clip hand-jerk-ratio vs GT
              relu(ratio - jerk_cap_ratio)
  L_dev     : sum |out - src| over hands (the desired residual support)
  L_body    : body preservation, L2 on body residual
  L_face    : face preservation, L2 on face residual
  L_bone    : per-frame hand-bone length match between out and src (24 bones)
  L_smooth  : temporal smoothness on residual (residual acceleration L2)
  L_anchor  : keep refined wrist trajectory close to source wrist trajectory
              (the body delta is allowed to nudge the wrist by at most a few
              centimetres; we explicitly penalise larger drift)

Inference constraints satisfied
-------------------------------
  - frozen v11 generator: never modified at any stage of v14
  - frozen v10 governor: never modified at any stage of v14
  - no GT length at inference (input is already v11+governor sample, sampled
    with the no-leak duration policy; refiner does not change clip length)
  - no retrieval / no real-clip splicing / no train-corpus stats at inference
    (only learned model parameters; refiner sees source pose + sentence
    glosses + lengths)
  - no SLRTP back-translation evaluator distillation (supervision uses Phoenix
    GT poses only — and only their aggregate hand-speed / hand-jerk scalars,
    plus bone-length geometry priors)

Train data
----------
Source pose: v11 + governor on Phoenix train (sampled with the same no-leak
duration policy used at inference). GT pose: Phoenix train.pt provided by the
SLRTP-Sign-Production-Evaluation organizer pack (`poses_3d` field). Sentence
glosses / lengths: data/phoenix/phoenix_train.json.

Subcommands
-----------
  train : fit the refiner
  apply : refine an arbitrary v11+governor source .pt into a refined .pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_sign_jepa_slrtp178 import (  # noqa: E402
    LENGTH_BUCKETS,
    length_bucket_id,
    sinusoidal_positions,
)
from scripts.sign_jepa_motion_amplify import amplify_clip  # noqa: E402


BODY = slice(0, 8)
RH = slice(8, 29)            # right hand attached to body wrist idx 2
LH = slice(29, 50)           # left hand  attached to body wrist idx 5
FACE = slice(50, 178)
HAND = slice(8, 50)
RWRIST, LWRIST = 2, 5

# Hand-finger bone pairs in MediaPipe-style hand layout (indices 0..20 per hand,
# offset by 0 for RH and 21 for LH inside HAND[0..41]). Five fingers, four
# bones each = 20 bones per hand, plus four palm anchors -> 24 bones per hand.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17), (2, 5),    # palm web
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def as_pose(x) -> torch.Tensor:
    if isinstance(x, dict):
        x = x.get("poses_3d", x.get("pose"))
    x = torch.as_tensor(x).float()
    if x.ndim == 2:
        x = x.reshape(x.shape[0], 178, 3)
    if x.ndim != 3 or x.shape[1:] != (178, 3):
        raise ValueError(f"bad pose shape {tuple(x.shape)}")
    return x


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def hand_speed_scalar(pose: torch.Tensor) -> float:
    if pose.shape[0] < 2:
        return 0.0
    v = pose[1:, HAND] - pose[:-1, HAND]
    return float(v.norm(dim=-1).mean())


def hand_jerk_scalar(pose: torch.Tensor) -> float:
    if pose.shape[0] < 3:
        return 0.0
    v = pose[1:, HAND] - pose[:-1, HAND]
    a = v[1:] - v[:-1]
    return float(a.norm(dim=-1).mean())


class ResidualFlowDataset(Dataset):
    """Yields (src_pose, gloss_ids, gt_speed, gt_jerk) per Phoenix clip.

    Clips are padded to a max length per batch. If src and GT exist and the
    src is longer than max_T, we crop a random window during training.
    """

    def __init__(
        self,
        ids: list[str],
        src_pose_map: dict[str, torch.Tensor],
        gt_pose_map: dict[str, torch.Tensor],
        manifest_rows: dict[str, dict],
        gloss_to_id: dict[str, int],
        max_t: int,
        max_sent_len: int,
        is_train: bool,
        crop_p: float,
        source_amp_aug: bool,
        source_amp_min: float,
        source_amp_max: float,
        seed: int = 0,
    ):
        self.ids = list(ids)
        self.src_pose_map = src_pose_map
        self.gt_pose_map = gt_pose_map
        self.manifest_rows = manifest_rows
        self.gloss_to_id = gloss_to_id
        self.max_t = int(max_t)
        self.max_sent_len = int(max_sent_len)
        self.is_train = bool(is_train)
        self.crop_p = float(crop_p)
        self.source_amp_aug = bool(source_amp_aug)
        self.source_amp_min = float(source_amp_min)
        self.source_amp_max = float(source_amp_max)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        src = self.src_pose_map[sid].float()                      # [Ts, 178, 3]
        gt = self.gt_pose_map[sid].float()                        # [Tg, 178, 3]
        row = self.manifest_rows.get(sid, {"text": "", "gloss": ""})
        sent_glosses = [g for g in str(row.get("gloss", "")).split() if g]
        sent_ids = torch.zeros(self.max_sent_len, dtype=torch.long)
        sent_mask = torch.zeros(self.max_sent_len, dtype=torch.bool)
        for j, g in enumerate(sent_glosses[: self.max_sent_len]):
            sent_ids[j] = int(self.gloss_to_id.get(g, 1))
            sent_mask[j] = True
        if sent_mask.sum() == 0:
            sent_mask[0] = True
        T = src.shape[0]
        if self.is_train and T > self.max_t and self._rng.random() < self.crop_p:
            start = self._rng.randint(0, T - self.max_t)
            src_clip = src[start : start + self.max_t]
        elif T > self.max_t:
            src_clip = src[: self.max_t]
        else:
            src_clip = src
        if self.is_train and self.source_amp_aug:
            lo = min(self.source_amp_min, self.source_amp_max)
            hi = max(self.source_amp_min, self.source_amp_max)
            gain = self._rng.uniform(lo, hi)
            src_clip = amplify_clip(
                src_clip,
                g_body=1.0,
                g_hand=gain,
                g_face=1.0,
                lp_kernel=1,
                smooth_kernel=1,
            )
        gt_speed = hand_speed_scalar(gt)
        gt_jerk = hand_jerk_scalar(gt)
        return {
            "sid": sid,
            "src": src_clip.contiguous(),
            "T": int(src_clip.shape[0]),
            "sent_ids": sent_ids,
            "sent_mask": sent_mask,
            "gt_speed": torch.tensor(gt_speed, dtype=torch.float32),
            "gt_jerk": torch.tensor(gt_jerk, dtype=torch.float32),
        }


def collate_residual(batch: list[dict]) -> dict:
    B = len(batch)
    T_max = max(b["T"] for b in batch)
    src = torch.zeros(B, T_max, 178, 3)
    src_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        Ti = b["T"]
        src[i, :Ti] = b["src"]
        src_mask[i, :Ti] = True
    return {
        "src": src,
        "src_mask": src_mask,
        "sent_ids": torch.stack([b["sent_ids"] for b in batch], dim=0),
        "sent_mask": torch.stack([b["sent_mask"] for b in batch], dim=0),
        "gt_speed": torch.stack([b["gt_speed"] for b in batch], dim=0),
        "gt_jerk": torch.stack([b["gt_jerk"] for b in batch], dim=0),
        "sids": [b["sid"] for b in batch],
        "T_each": torch.tensor([b["T"] for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ResidualFlowRefiner(nn.Module):
    """Wrist-coupled bounded residual model."""

    def __init__(
        self,
        gloss_vocab: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        max_t: int,
        sent_max_len: int,
        max_res_body: float,
        max_res_hand: float,
        max_res_face: float,
    ):
        super().__init__()
        self.hidden = hidden
        self.max_t = max_t
        self.sent_max_len = sent_max_len
        self.max_res_body = max_res_body
        self.max_res_hand = max_res_hand
        self.max_res_face = max_res_face

        self.pose_in = nn.Linear(178 * 3, hidden)
        self.vel_in = nn.Linear(178 * 3, hidden)
        self.gloss_emb = nn.Embedding(gloss_vocab, hidden, padding_idx=0)
        self.sent_pos_emb = nn.Embedding(sent_max_len, hidden)
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
        self.progress_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.body_head = nn.Linear(hidden, 8 * 3)
        self.hand_head = nn.Linear(hidden, 42 * 3)
        self.face_head = nn.Linear(hidden, 128 * 3)
        for h in (self.body_head, self.hand_head, self.face_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor,
                sent_ids: torch.Tensor, sent_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T = src.shape[:2]
        device = src.device
        x = src.reshape(B, T, -1)
        v = torch.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        h = self.pose_in(x) + self.vel_in(v)

        # progress
        p = torch.linspace(0.0, 1.0, T, device=device)
        prog = torch.stack(
            [p, 1.0 - p, torch.sin(math.pi * p), torch.cos(math.pi * p)], dim=-1
        )
        h = h + self.progress_mlp(prog)[None]
        h = h + sinusoidal_positions(T, self.hidden, device)[None]

        # sentence-context: masked mean of gloss embeddings + positional emb
        S = sent_ids.shape[1]
        sent_pos = torch.arange(S, device=device)
        sent_h = self.gloss_emb(sent_ids) + self.sent_pos_emb(sent_pos)[None]
        denom = sent_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        sent_ctx = (sent_h * sent_mask.float().unsqueeze(-1)).sum(dim=1) / denom
        h = h + sent_ctx[:, None]

        # length-bucket token (using src length)
        T_each = src_mask.sum(dim=-1)
        len_id = torch.tensor(
            [length_bucket_id(int(t.item())) for t in T_each.cpu()],
            dtype=torch.long, device=device,
        )
        h = h + self.len_emb(len_id)[:, None]

        z = self.norm(self.blocks(h, src_key_padding_mask=~src_mask))

        db = torch.tanh(self.body_head(z)).reshape(B, T, 8, 3) * self.max_res_body
        dh = torch.tanh(self.hand_head(z)).reshape(B, T, 42, 3) * self.max_res_hand
        df = torch.tanh(self.face_head(z)).reshape(B, T, 128, 3) * self.max_res_face

        body_src = src[:, :, BODY]
        body_new = body_src + db

        rh_local = src[:, :, RH] - src[:, :, RWRIST : RWRIST + 1]
        rh_new = body_new[:, :, RWRIST : RWRIST + 1] + rh_local + dh[:, :, 0:21]
        lh_local = src[:, :, LH] - src[:, :, LWRIST : LWRIST + 1]
        lh_new = body_new[:, :, LWRIST : LWRIST + 1] + lh_local + dh[:, :, 21:42]
        face_new = src[:, :, FACE] + df

        out = torch.cat([body_new, rh_new, lh_new, face_new], dim=2)
        return out, {"db": db, "dh": dh, "df": df}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def masked_hand_speed(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-clip mean hand velocity magnitude, respecting the time mask."""
    B = pred.shape[0]
    v = pred[:, 1:, HAND] - pred[:, :-1, HAND]                       # [B, T-1, 42, 3]
    vm = v.norm(dim=-1).mean(dim=-1)                                 # [B, T-1]
    m = (mask[:, 1:] & mask[:, :-1]).float()                         # [B, T-1]
    num = (vm * m).sum(dim=1)
    den = m.sum(dim=1).clamp_min(1.0)
    return num / den                                                 # [B]


def masked_hand_jerk(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    v = pred[:, 1:, HAND] - pred[:, :-1, HAND]                       # [B, T-1, 42, 3]
    a = v[:, 1:] - v[:, :-1]                                         # [B, T-2, 42, 3]
    am = a.norm(dim=-1).mean(dim=-1)                                 # [B, T-2]
    m = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
    num = (am * m).sum(dim=1)
    den = m.sum(dim=1).clamp_min(1.0)
    return num / den                                                 # [B]


def hand_bones_lengths(pose: torch.Tensor) -> torch.Tensor:
    """Per-frame per-bone Euclidean lengths for both hands.

    Returns [B, T, 2*len(HAND_BONES)] tensor (RH bones then LH bones).
    """
    rh = pose[:, :, RH]                                              # [B, T, 21, 3]
    lh = pose[:, :, LH]
    parts = []
    for (i, j) in HAND_BONES:
        parts.append((rh[:, :, j] - rh[:, :, i]).norm(dim=-1))
        parts.append((lh[:, :, j] - lh[:, :, i]).norm(dim=-1))
    return torch.stack(parts, dim=-1)


def compute_losses(
    pred: torch.Tensor,
    src: torch.Tensor,
    mask: torch.Tensor,
    gt_speed: torch.Tensor,
    gt_jerk: torch.Tensor,
    deltas: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    B = pred.shape[0]
    eps = 1e-6
    mt = mask.float().unsqueeze(-1).unsqueeze(-1)                    # [B, T, 1, 1]
    n_frames = mt.sum().clamp_min(1.0)

    # --- hand speed band loss ------------------------------------------------
    pred_speed = masked_hand_speed(pred, mask)
    src_speed = masked_hand_speed(src, mask)
    # Ratio relative to GT (per-clip, no aggregation over batch yet).
    ratio = pred_speed / gt_speed.clamp_min(eps)
    log_ratio = torch.log((pred_speed + eps) / (gt_speed + eps))
    band_lo = float(args.speed_band_lo)
    band_hi = float(args.speed_band_hi)
    band_lo_t = pred.new_tensor(band_lo)
    band_hi_t = pred.new_tensor(band_hi)
    log_target = pred.new_tensor(math.log(0.5 * (band_lo + band_hi)))
    L_band = F.relu(band_lo_t - ratio).mean() + F.relu(ratio - band_hi_t).mean()
    L_logspeed = F.smooth_l1_loss(log_ratio, log_target.expand_as(log_ratio))
    L_speed = L_band + 0.1 * L_logspeed

    # --- jerk cap ------------------------------------------------------------
    pred_jerk = masked_hand_jerk(pred, mask)
    src_jerk = masked_hand_jerk(src, mask)
    jerk_ratio = pred_jerk / gt_jerk.clamp_min(eps)
    L_jerk = F.relu(jerk_ratio - float(args.jerk_cap)).mean()

    # --- source deviation on hands (encourages truly residual corrections) --
    diff_hand = (pred[:, :, HAND] - src[:, :, HAND]).abs()
    L_dev = (diff_hand * mt).sum() / (n_frames * 42 * 3)

    # --- body preservation ---------------------------------------------------
    diff_body = (pred[:, :, BODY] - src[:, :, BODY]).pow(2)
    L_body = (diff_body * mt).sum() / (n_frames * 8 * 3)

    # --- face preservation ---------------------------------------------------
    diff_face = (pred[:, :, FACE] - src[:, :, FACE]).pow(2)
    L_face = (diff_face * mt).sum() / (n_frames * 128 * 3)

    # --- wrist anchor (drift of wrist trajectory from source) ---------------
    diff_wr = (pred[:, :, [RWRIST, LWRIST]] - src[:, :, [RWRIST, LWRIST]]).pow(2)
    L_anchor = (diff_wr * mt).sum() / (n_frames * 2 * 3)

    # --- hand bone-length preservation ---------------------------------------
    bl_pred = hand_bones_lengths(pred)
    bl_src = hand_bones_lengths(src)
    bl_mask = mask.float().unsqueeze(-1)
    bone_diff = (bl_pred - bl_src).pow(2)
    L_bone = (bone_diff * bl_mask).sum() / (mask.float().sum() * bl_pred.shape[-1] + eps)

    # --- temporal smoothness on residual (residual acceleration L2) ---------
    residual = pred - src
    res_a = residual[:, 2:] - 2 * residual[:, 1:-1] + residual[:, :-2]  # [B, T-2, 178, 3]
    smask = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float().unsqueeze(-1).unsqueeze(-1)
    L_smooth = (res_a.pow(2) * smask).sum() / (smask.sum().clamp_min(1.0) * 178 * 3 + eps)

    loss = (
        args.lambda_speed * L_speed
        + args.lambda_jerk * L_jerk
        + args.lambda_dev * L_dev
        + args.lambda_body * L_body
        + args.lambda_face * L_face
        + args.lambda_anchor * L_anchor
        + args.lambda_bone * L_bone
        + args.lambda_smooth * L_smooth
    )

    logs = {
        "loss": float(loss.detach().cpu()),
        "L_speed": float(L_speed.detach().cpu()),
        "L_band": float(L_band.detach().cpu()),
        "L_jerk": float(L_jerk.detach().cpu()),
        "L_dev": float(L_dev.detach().cpu()),
        "L_body": float(L_body.detach().cpu()),
        "L_face": float(L_face.detach().cpu()),
        "L_anchor": float(L_anchor.detach().cpu()),
        "L_bone": float(L_bone.detach().cpu()),
        "L_smooth": float(L_smooth.detach().cpu()),
        "speed_ratio_mean": float(ratio.mean().detach().cpu()),
        "speed_ratio_p10": float(torch.quantile(ratio, 0.1).detach().cpu()),
        "speed_ratio_p90": float(torch.quantile(ratio, 0.9).detach().cpu()),
        "jerk_ratio_mean": float(jerk_ratio.mean().detach().cpu()),
        "src_speed_mean": float((src_speed / gt_speed.clamp_min(eps)).mean().detach().cpu()),
    }
    return loss, logs


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def build_dataset(args, src_pose_map, gt_pose_map, manifest_rows, gloss_to_id, is_train, ids):
    return ResidualFlowDataset(
        ids=ids,
        src_pose_map=src_pose_map,
        gt_pose_map=gt_pose_map,
        manifest_rows=manifest_rows,
        gloss_to_id=gloss_to_id,
        max_t=args.max_t,
        max_sent_len=args.sent_max_len,
        is_train=is_train,
        crop_p=args.crop_p,
        source_amp_aug=args.source_amp_aug,
        source_amp_min=args.source_amp_min,
        source_amp_max=args.source_amp_max,
        seed=args.seed + (0 if is_train else 1),
    )


def load_v11_artifacts(args):
    ck = torch.load(ROOT / args.v11_ckpt, map_location="cpu", weights_only=False)
    gloss_to_id = ck["gloss_to_id"]
    return gloss_to_id


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v14] loading train source poses from {args.train_src_pt}", flush=True)
    src_pose_map = load_pose_map(ROOT / args.train_src_pt)
    print(f"[v14] loaded {len(src_pose_map)} source clips", flush=True)
    print(f"[v14] loading GT train poses from {args.train_gt_pt}", flush=True)
    gt_pose_map = load_pose_map(ROOT / args.train_gt_pt)
    print(f"[v14] loaded {len(gt_pose_map)} GT clips", flush=True)
    manifest_rows = {str(r["id"]): r for r in json.load(open(ROOT / args.train_manifest_json))}
    gloss_to_id = load_v11_artifacts(args)
    print(f"[v14] gloss vocab = {len(gloss_to_id)}", flush=True)

    common_ids = [sid for sid in src_pose_map.keys() if sid in gt_pose_map and sid in manifest_rows]
    rnd = random.Random(args.seed)
    rnd.shuffle(common_ids)
    n_val = max(64, int(len(common_ids) * args.val_frac))
    val_ids = common_ids[:n_val]
    train_ids = common_ids[n_val:]
    if args.smoke:
        train_ids = train_ids[: args.smoke_train]
        val_ids = val_ids[: args.smoke_val]
    print(f"[v14] train={len(train_ids)} val={len(val_ids)}", flush=True)

    train_ds = build_dataset(args, src_pose_map, gt_pose_map, manifest_rows, gloss_to_id, True, train_ids)
    val_ds = build_dataset(args, src_pose_map, gt_pose_map, manifest_rows, gloss_to_id, False, val_ids)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), collate_fn=collate_residual, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), collate_fn=collate_residual,
    )

    model = ResidualFlowRefiner(
        gloss_vocab=len(gloss_to_id),
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        max_t=args.max_t,
        sent_max_len=args.sent_max_len,
        max_res_body=args.max_res_body,
        max_res_hand=args.max_res_hand,
        max_res_face=args.max_res_face,
    ).to(device)
    print(
        f"[v14] device={device} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log = []
    best = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_accum = {}
        n_train = 0
        for step, batch in enumerate(train_loader, start=1):
            src = batch["src"].to(device)
            src_mask = batch["src_mask"].to(device)
            sent_ids = batch["sent_ids"].to(device)
            sent_mask = batch["sent_mask"].to(device)
            gt_speed = batch["gt_speed"].to(device)
            gt_jerk = batch["gt_jerk"].to(device)
            pred, deltas = model(src, src_mask, sent_ids, sent_mask)
            loss, logs = compute_losses(pred, src, src_mask, gt_speed, gt_jerk, deltas, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            for k, v in logs.items():
                train_accum[k] = train_accum.get(k, 0.0) + v
            n_train += 1
            if args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({"step": step, "grad_norm": float(gnorm), **logs}), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        train_log = {f"train_{k}": v / max(n_train, 1) for k, v in train_accum.items()}

        model.eval()
        val_accum = {}
        n_val_b = 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if args.val_batches and bi >= args.val_batches:
                    break
                src = batch["src"].to(device)
                src_mask = batch["src_mask"].to(device)
                sent_ids = batch["sent_ids"].to(device)
                sent_mask = batch["sent_mask"].to(device)
                gt_speed = batch["gt_speed"].to(device)
                gt_jerk = batch["gt_jerk"].to(device)
                pred, deltas = model(src, src_mask, sent_ids, sent_mask)
                _, logs = compute_losses(pred, src, src_mask, gt_speed, gt_jerk, deltas, args)
                for k, v in logs.items():
                    val_accum[k] = val_accum.get(k, 0.0) + v
                n_val_b += 1
        val_log = {f"val_{k}": v / max(n_val_b, 1) for k, v in val_accum.items()}
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1), **train_log, **val_log}
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)

        score = (
            abs(rec["val_speed_ratio_mean"] - args.speed_band_target)
            + max(0.0, rec["val_jerk_ratio_mean"] - args.jerk_cap) * 2.0
            + 0.05 * rec["val_L_body"] / max(args.lambda_body, 1e-6)
        )
        if best is None or score < best["score"]:
            best = {**rec, "score": float(score)}
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best": best,
                    "gloss_to_id": gloss_to_id,
                },
                out_dir / "best.pt",
            )
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[v14] saved best -> {out_dir / 'best.pt'} score={score:.4f}", flush=True)

    print(f"[v14] done best={best}", flush=True)


# ---------------------------------------------------------------------------
# Apply (inference) driver
# ---------------------------------------------------------------------------


@torch.no_grad()
def apply(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    refiner_args = argparse.Namespace(**ck["args"])
    gloss_to_id = ck["gloss_to_id"]
    model = ResidualFlowRefiner(
        gloss_vocab=len(gloss_to_id),
        hidden=refiner_args.hidden,
        layers=refiner_args.layers,
        heads=refiner_args.heads,
        dropout=0.0,
        max_t=refiner_args.max_t,
        sent_max_len=refiner_args.sent_max_len,
        max_res_body=refiner_args.max_res_body,
        max_res_hand=refiner_args.max_res_hand,
        max_res_face=refiner_args.max_res_face,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    src_pose_map = load_pose_map(ROOT / args.in_pt)
    manifest_rows = {str(r["id"]): r for r in json.load(open(ROOT / args.manifest_json))}

    out: dict[str, torch.Tensor] = {}
    trace: dict[str, dict] = {}
    n = len(src_pose_map)
    sent_max_len = int(refiner_args.sent_max_len)
    for i, (sid, src_pose) in enumerate(src_pose_map.items(), start=1):
        row = manifest_rows.get(sid, {"text": "", "gloss": ""})
        sent_glosses = [g for g in str(row.get("gloss", "")).split() if g]
        sent_ids = torch.zeros(1, sent_max_len, dtype=torch.long, device=device)
        sent_mask = torch.zeros(1, sent_max_len, dtype=torch.bool, device=device)
        for j, g in enumerate(sent_glosses[:sent_max_len]):
            sent_ids[0, j] = int(gloss_to_id.get(g, 1))
            sent_mask[0, j] = True
        if sent_mask.sum() == 0:
            sent_mask[0, 0] = True

        # Process the full clip (we trained on padded clips; inference takes one).
        T = src_pose.shape[0]
        src_b = src_pose.float().unsqueeze(0).to(device)
        m_b = torch.ones(1, T, dtype=torch.bool, device=device)
        pred, _ = model(src_b, m_b, sent_ids, sent_mask)
        out_pose = pred[0].detach().cpu()
        out[sid] = out_pose.contiguous()
        if args.trace_json:
            trace[sid] = {
                "T": T,
                "src_hand_speed": hand_speed_scalar(src_pose),
                "out_hand_speed": hand_speed_scalar(out_pose),
                "src_hand_jerk": hand_jerk_scalar(src_pose),
                "out_hand_jerk": hand_jerk_scalar(out_pose),
            }
        if args.log_every and (i == 1 or i % args.log_every == 0):
            print(f"[v14-apply] {i}/{n}", flush=True)

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}", flush=True)
    if args.trace_json:
        tp = ROOT / args.trace_json
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(trace, indent=2))
        print(f"trace -> {tp}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--v11_ckpt", default="outputs/sota_chase/sign_jepa_v11_sentence_budget/best.pt")
    tr.add_argument("--train_src_pt", required=True,
                    help="v11+governor (or v11) pose .pt on Phoenix train (sid -> [T,178,3]).")
    tr.add_argument("--train_gt_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    tr.add_argument("--train_manifest_json", default="data/phoenix/phoenix_train.json")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_v14_residual_flow")

    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--layers", type=int, default=3)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--max_t", type=int, default=192)
    tr.add_argument("--sent_max_len", type=int, default=32)
    tr.add_argument("--crop_p", type=float, default=0.5)
    tr.add_argument("--source_amp_aug", action="store_true",
                    help="randomly hand-dampen training source clips before refinement")
    tr.add_argument("--source_amp_min", type=float, default=0.5)
    tr.add_argument("--source_amp_max", type=float, default=1.0)

    tr.add_argument("--max_res_body", type=float, default=0.008)
    tr.add_argument("--max_res_hand", type=float, default=0.04)
    tr.add_argument("--max_res_face", type=float, default=0.003)

    tr.add_argument("--speed_band_lo", type=float, default=0.85)
    tr.add_argument("--speed_band_hi", type=float, default=1.05)
    tr.add_argument("--speed_band_target", type=float, default=0.95)
    tr.add_argument("--jerk_cap", type=float, default=1.30)

    tr.add_argument("--lambda_speed", type=float, default=2.0)
    tr.add_argument("--lambda_jerk", type=float, default=1.0)
    tr.add_argument("--lambda_dev", type=float, default=0.5)
    tr.add_argument("--lambda_body", type=float, default=10.0)
    tr.add_argument("--lambda_face", type=float, default=20.0)
    tr.add_argument("--lambda_anchor", type=float, default=20.0)
    tr.add_argument("--lambda_bone", type=float, default=50.0)
    tr.add_argument("--lambda_smooth", type=float, default=200.0)

    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--eval_batch_size", type=int, default=8)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)

    tr.add_argument("--val_frac", type=float, default=0.05)
    tr.add_argument("--val_batches", type=int, default=20)
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=50)
    tr.add_argument("--max_steps", type=int, default=0)
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--smoke_train", type=int, default=64)
    tr.add_argument("--smoke_val", type=int, default=16)

    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--ckpt", required=True)
    ap_apply.add_argument("--in_pt", required=True,
                          help="source pose .pt to refine (e.g. v11+governor dev/test sample)")
    ap_apply.add_argument("--manifest_json", required=True,
                          help="Phoenix manifest with gloss/text for the input ids")
    ap_apply.add_argument("--out_pt", required=True)
    ap_apply.add_argument("--trace_json", default="")
    ap_apply.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_apply.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        apply(args)


if __name__ == "__main__":
    main()
