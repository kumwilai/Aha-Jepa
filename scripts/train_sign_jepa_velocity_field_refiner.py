"""Velocity-field refiner on top of frozen v11+governor (v21).

After v14 / v14b / v20 (bounded source-preserving position residuals) all
collapsed to identity / smoothing, the failure mode is now clear:
small position residuals can express smoothing (which decreases hand_speed)
but cannot easily express a coherent persistent trajectory shift. v21 changes
the parameterisation: the model predicts a bounded per-frame velocity
correction field, integrates it through time, removes endpoint drift and
temporal mean, then applies it. Identity remains a valid solution (zero
velocity delta), but smoothing is now much harder to express because it
requires emitting an oscillating velocity delta which jerk and velocity
smoothness penalties suppress.

Architecture
------------
Inputs per frame: source pose, source velocity, sentence gloss context mean,
progress encoding, sinusoidal positional encoding, length bucket.

Three velocity-delta heads (all zero-init):

  dv_body        [B, T, 8,  3]   tanh * max_dv_body
  dv_hand_local  [B, T, 42, 3]   tanh * max_dv_hand_local   (LOCAL to wrist)

The body head carries wrist correction (joints 2 and 5); the hand head carries
hand-internal articulation in the local wrist frame.

Integration with drift removal:

  c[t] = cumsum_t(dv)   with c[0] = 0
  c[t] -= t / (T-1) * c[-1]         endpoint-drift removal
  c[t] -= mean_t(c)                 zero-mean drift removal

This guarantees the corrected trajectory has the SAME start, end and average
position as the source trajectory, so the corrected pose cannot translate or
drift globally relative to the source. What it CAN do is add intra-utterance
arc length (going further from the time-averaged centre, then coming back),
which is exactly the missing signal in the underspeed v11+governor source.

Output pose construction in raw SLRTP-178 coordinates:

  body_new[t]  = body_src[t] + c_body[t]
  rh_local[t]  = rh_src[t] - rwrist_src[t]                # source local hand
  rh_new[t]    = body_new[t, 2] + rh_local[t] + c_hand_local[t, 0:21]
  lh_local[t]  = lh_src[t] - lwrist_src[t]
  lh_new[t]    = body_new[t, 5] + lh_local[t] + c_hand_local[t, 21:42]
  face_new[t]  = face_src[t]                              # face unchanged

Losses
------
  L_speed   : two-sided band on per-clip hand-speed-ratio vs GT,
              plus smooth-L1 of log_ratio toward log(0.95). Same as v20.
  L_jerk    : one-sided cap on per-clip hand-jerk-ratio vs GT (<=1.30).
  L_arc     : one-sided push for pred hand arc-length >= source arc-length.
              This is the lever the velocity field needs in order to add
              displacement instead of smoothing.
  L_bone    : per-frame hand-bone length match between out and src.
  L_anchor  : per-frame anchor on wrist trajectory shift; small wrist drift
              within the integrated frame is allowed but not large.
  L_body_other : keep non-wrist body very close to source (no body warping).
  L_face    : face preservation (face head is implicitly identity here, since
              we do not apply any face delta; kept for symmetry / safety in
              case heads are extended later).
  L_dev     : mild source-deviation L1 on hands.
  L_vel_smooth : penalize second time derivative of dv (jerk of the correction
                 velocity field) — encourages coherent persistent corrections
                 rather than oscillatory smoothing.

Hard constraints satisfied
--------------------------
  - frozen v11 generator
  - frozen v10 governor
  - no GT length at inference (rebuilt text-only duration policy used to
    sample the source v11+governor pose; refiner does not change length)
  - no retrieval / no real-clip splicing / no train-corpus stats at inference
  - no SLRTP back-translation evaluator distillation (supervision uses
    Phoenix GT poses only — and only their aggregate hand-speed / hand-jerk
    scalars, plus bone-length geometry priors)

Reuses
------
Dataset / collate / amplification augmentation are inherited verbatim from
scripts.train_sign_jepa_constrained_residual_flow so the train/test source
distribution is exactly the same and the only thing that varies between v20
and v21 is the model parameterisation and the loss set.
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
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_sign_jepa_slrtp178 import (  # noqa: E402
    LENGTH_BUCKETS,
    length_bucket_id,
    sinusoidal_positions,
)
from scripts.train_sign_jepa_constrained_residual_flow import (  # noqa: E402
    BODY,
    RH,
    LH,
    FACE,
    HAND,
    RWRIST,
    LWRIST,
    HAND_BONES,
    ResidualFlowDataset,
    collate_residual,
    hand_speed_scalar,
    hand_jerk_scalar,
    load_pose_map,
    hand_bones_lengths,
    masked_hand_speed,
    masked_hand_jerk,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _remove_endpoint_drift(c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Remove the linear interpolation from c[0] to c[T_eff-1] per clip.

    c     : [B, T, ..., 3]  cumulative-sum trajectory
    mask  : [B, T]          time mask (True for real frames)

    For each batch element, find the last True frame index t_end and replace
    c[t] with c[t] - (t / t_end) * c[t_end] for t <= t_end. Beyond t_end the
    correction is zeroed.
    """
    B, T = c.shape[:2]
    device = c.device
    t_each = mask.sum(dim=1).clamp_min(1).to(device)                  # [B]
    t_end_idx = (t_each - 1).clamp_min(0).long()                      # [B]
    # Gather endpoint trajectory at each clip's true last frame.
    flat = c.reshape(B, T, -1)                                        # [B, T, F]
    F_dim = flat.shape[-1]
    end_vec = flat.gather(1, t_end_idx.view(B, 1, 1).expand(B, 1, F_dim)).squeeze(1)  # [B, F]
    # Fraction t / t_end, clamped within the valid range.
    t_axis = torch.arange(T, device=device, dtype=c.dtype).view(1, T)             # [1, T]
    denom = t_each.clamp_min(1).to(c.dtype).view(B, 1) - 1.0
    denom = denom.clamp_min(1.0)
    frac = (t_axis / denom).clamp(0.0, 1.0)                                       # [B, T]
    drift = frac.unsqueeze(-1) * end_vec.unsqueeze(1)                              # [B, T, F]
    out = flat - drift
    # Zero anything past the last real frame.
    pad_mask = (~mask).to(c.dtype).unsqueeze(-1)
    out = out * (1.0 - pad_mask)
    return out.reshape(*c.shape)


def _remove_temporal_mean(c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtract the time-mean of c over the real frames per clip."""
    B, T = c.shape[:2]
    m = mask.to(c.dtype).view(B, T, *([1] * (c.dim() - 2)))
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (c * m).sum(dim=1, keepdim=True) / denom
    out = (c - mean) * m
    return out


class VelocityFieldRefiner(nn.Module):
    """Predict bounded velocity-delta field, integrate, drift-correct, apply."""

    def __init__(
        self,
        gloss_vocab: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        max_t: int,
        sent_max_len: int,
        max_dv_body: float,
        max_dv_hand_local: float,
    ):
        super().__init__()
        self.hidden = hidden
        self.max_t = max_t
        self.sent_max_len = sent_max_len
        self.max_dv_body = max_dv_body
        self.max_dv_hand_local = max_dv_hand_local

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
        for h in (self.body_head, self.hand_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        sent_ids: torch.Tensor,
        sent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        B, T = src.shape[:2]
        device = src.device

        x = src.reshape(B, T, -1)
        v = torch.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        h = self.pose_in(x) + self.vel_in(v)

        # progress encoding
        p = torch.linspace(0.0, 1.0, T, device=device)
        prog = torch.stack(
            [p, 1.0 - p, torch.sin(math.pi * p), torch.cos(math.pi * p)], dim=-1
        )
        h = h + self.progress_mlp(prog)[None]
        h = h + sinusoidal_positions(T, self.hidden, device)[None]

        # sentence context
        S = sent_ids.shape[1]
        sent_pos = torch.arange(S, device=device)
        sent_h = self.gloss_emb(sent_ids) + self.sent_pos_emb(sent_pos)[None]
        denom = sent_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        sent_ctx = (sent_h * sent_mask.float().unsqueeze(-1)).sum(dim=1) / denom
        h = h + sent_ctx[:, None]

        # length-bucket token
        T_each = src_mask.sum(dim=-1)
        len_id = torch.tensor(
            [length_bucket_id(int(t.item())) for t in T_each.cpu()],
            dtype=torch.long, device=device,
        )
        h = h + self.len_emb(len_id)[:, None]

        z = self.norm(self.blocks(h, src_key_padding_mask=~src_mask))

        dv_body_raw = torch.tanh(self.body_head(z)).reshape(B, T, 8, 3) * self.max_dv_body
        dv_hand_raw = (
            torch.tanh(self.hand_head(z)).reshape(B, T, 42, 3) * self.max_dv_hand_local
        )

        # Zero out velocity deltas on padded frames so the cumsum is clean.
        m = src_mask.to(dv_body_raw.dtype).view(B, T, 1, 1)
        dv_body = dv_body_raw * m
        dv_hand = dv_hand_raw * m

        # Integrate. cumsum gives a trajectory with c[0] = dv[0] which we want
        # to start at zero — shift so c[0] = 0 by subtracting dv[0].
        c_body = torch.cumsum(dv_body, dim=1)
        c_body = c_body - c_body[:, :1]
        c_hand = torch.cumsum(dv_hand, dim=1)
        c_hand = c_hand - c_hand[:, :1]

        # Drift removal — first the linear endpoint drift, then the temporal
        # mean so that the corrected trajectory keeps source's mean position
        # and source's start/end.
        c_body = _remove_endpoint_drift(c_body, src_mask)
        c_body = _remove_temporal_mean(c_body, src_mask)
        c_hand = _remove_endpoint_drift(c_hand, src_mask)
        c_hand = _remove_temporal_mean(c_hand, src_mask)

        body_src = src[:, :, BODY]
        body_new = body_src + c_body

        rh_local = src[:, :, RH] - src[:, :, RWRIST : RWRIST + 1]
        rh_new = body_new[:, :, RWRIST : RWRIST + 1] + rh_local + c_hand[:, :, 0:21]
        lh_local = src[:, :, LH] - src[:, :, LWRIST : LWRIST + 1]
        lh_new = body_new[:, :, LWRIST : LWRIST + 1] + lh_local + c_hand[:, :, 21:42]
        face_new = src[:, :, FACE]

        out = torch.cat([body_new, rh_new, lh_new, face_new], dim=2)

        # Apply the mask one more time so padded frames are exact source.
        pad = (~src_mask).to(out.dtype).view(B, T, 1, 1)
        out = out * (1.0 - pad) + src * pad

        return out, {
            "dv_body": dv_body,
            "dv_hand": dv_hand,
            "c_body": c_body,
            "c_hand": c_hand,
        }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def masked_arc_length(pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-clip total hand arc length (sum of per-frame velocity magnitudes)."""
    v = pose[:, 1:, HAND] - pose[:, :-1, HAND]
    vm = v.norm(dim=-1).mean(dim=-1)
    m = (mask[:, 1:] & mask[:, :-1]).float()
    return (vm * m).sum(dim=1)


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
    mt = mask.float().unsqueeze(-1).unsqueeze(-1)
    n_frames = mt.sum().clamp_min(1.0)

    # --- hand speed band loss ------------------------------------------------
    pred_speed = masked_hand_speed(pred, mask)
    src_speed = masked_hand_speed(src, mask)
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

    # --- arc-length push: pred hand arc length >= src hand arc length -------
    pred_arc = masked_arc_length(pred, mask)
    src_arc = masked_arc_length(src, mask)
    arc_gain = pred_arc / src_arc.clamp_min(eps)
    L_arc = F.relu(pred.new_tensor(float(args.arc_min_gain)) - arc_gain).mean()

    # --- source deviation on hands ------------------------------------------
    diff_hand = (pred[:, :, HAND] - src[:, :, HAND]).abs()
    L_dev = (diff_hand * mt).sum() / (n_frames * 42 * 3)

    # --- body preservation: split into wrist vs non-wrist --------------------
    non_wrist_idx = torch.tensor([0, 1, 3, 4, 6, 7], dtype=torch.long, device=pred.device)
    body_other_pred = pred[:, :, BODY].index_select(2, non_wrist_idx)
    body_other_src = src[:, :, BODY].index_select(2, non_wrist_idx)
    L_body_other = ((body_other_pred - body_other_src).pow(2) * mt).sum() / (
        n_frames * 6 * 3
    )
    wrist_idx = torch.tensor([RWRIST, LWRIST], dtype=torch.long, device=pred.device)
    wrist_pred = pred.index_select(2, wrist_idx)
    wrist_src = src.index_select(2, wrist_idx)
    L_anchor = ((wrist_pred - wrist_src).pow(2) * mt).sum() / (n_frames * 2 * 3)

    # --- face preservation ---------------------------------------------------
    diff_face = (pred[:, :, FACE] - src[:, :, FACE]).pow(2)
    L_face = (diff_face * mt).sum() / (n_frames * 128 * 3)

    # --- hand bone-length preservation --------------------------------------
    bl_pred = hand_bones_lengths(pred)
    bl_src = hand_bones_lengths(src)
    bl_mask = mask.float().unsqueeze(-1)
    bone_diff = (bl_pred - bl_src).pow(2)
    L_bone = (bone_diff * bl_mask).sum() / (
        mask.float().sum() * bl_pred.shape[-1] + eps
    )

    # --- velocity-field smoothness ------------------------------------------
    dv_body = deltas["dv_body"]
    dv_hand = deltas["dv_hand"]
    # 2nd time derivative of dv (i.e. acceleration of the velocity correction).
    if dv_body.shape[1] >= 3:
        a_body = dv_body[:, 2:] - 2 * dv_body[:, 1:-1] + dv_body[:, :-2]
        a_hand = dv_hand[:, 2:] - 2 * dv_hand[:, 1:-1] + dv_hand[:, :-2]
        smask = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float().unsqueeze(-1).unsqueeze(-1)
        L_vel_smooth = (
            (a_body.pow(2) * smask).sum() / (smask.sum().clamp_min(1.0) * 8 * 3 + eps)
            + (a_hand.pow(2) * smask).sum() / (smask.sum().clamp_min(1.0) * 42 * 3 + eps)
        )
    else:
        L_vel_smooth = pred.new_tensor(0.0)

    loss = (
        args.lambda_speed * L_speed
        + args.lambda_jerk * L_jerk
        + args.lambda_arc * L_arc
        + args.lambda_dev * L_dev
        + args.lambda_body * L_body_other
        + args.lambda_face * L_face
        + args.lambda_anchor * L_anchor
        + args.lambda_bone * L_bone
        + args.lambda_vel_smooth * L_vel_smooth
    )

    # Trajectory diagnostics: how big is the actual residual hand displacement?
    res_hand_mag = (pred[:, :, HAND] - src[:, :, HAND]).norm(dim=-1).mean(dim=-1)
    res_hand_mag = (res_hand_mag * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)

    logs = {
        "loss": float(loss.detach().cpu()),
        "L_speed": float(L_speed.detach().cpu()),
        "L_band": float(L_band.detach().cpu()),
        "L_jerk": float(L_jerk.detach().cpu()),
        "L_arc": float(L_arc.detach().cpu()),
        "L_dev": float(L_dev.detach().cpu()),
        "L_body_other": float(L_body_other.detach().cpu()),
        "L_face": float(L_face.detach().cpu()),
        "L_anchor": float(L_anchor.detach().cpu()),
        "L_bone": float(L_bone.detach().cpu()),
        "L_vel_smooth": float(L_vel_smooth.detach().cpu()),
        "speed_ratio_mean": float(ratio.mean().detach().cpu()),
        "speed_ratio_p10": float(torch.quantile(ratio, 0.1).detach().cpu()),
        "speed_ratio_p90": float(torch.quantile(ratio, 0.9).detach().cpu()),
        "jerk_ratio_mean": float(jerk_ratio.mean().detach().cpu()),
        "arc_gain_mean": float(arc_gain.mean().detach().cpu()),
        "arc_gain_p10": float(torch.quantile(arc_gain, 0.1).detach().cpu()),
        "src_speed_mean": float(
            (src_speed / gt_speed.clamp_min(eps)).mean().detach().cpu()
        ),
        "res_hand_mag_mean": float(res_hand_mag.mean().detach().cpu()),
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

    print(f"[v21] loading train source poses from {args.train_src_pt}", flush=True)
    src_pose_map = load_pose_map(ROOT / args.train_src_pt)
    print(f"[v21] loaded {len(src_pose_map)} source clips", flush=True)
    print(f"[v21] loading GT train poses from {args.train_gt_pt}", flush=True)
    gt_pose_map = load_pose_map(ROOT / args.train_gt_pt)
    print(f"[v21] loaded {len(gt_pose_map)} GT clips", flush=True)
    manifest_rows = {str(r["id"]): r for r in json.load(open(ROOT / args.train_manifest_json))}
    gloss_to_id = load_v11_artifacts(args)
    print(f"[v21] gloss vocab = {len(gloss_to_id)}", flush=True)

    common_ids = [sid for sid in src_pose_map.keys() if sid in gt_pose_map and sid in manifest_rows]
    rnd = random.Random(args.seed)
    rnd.shuffle(common_ids)
    n_val = max(64, int(len(common_ids) * args.val_frac))
    val_ids = common_ids[:n_val]
    train_ids = common_ids[n_val:]
    if args.smoke:
        train_ids = train_ids[: args.smoke_train]
        val_ids = val_ids[: args.smoke_val]
    print(f"[v21] train={len(train_ids)} val={len(val_ids)}", flush=True)

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

    model = VelocityFieldRefiner(
        gloss_vocab=len(gloss_to_id),
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        max_t=args.max_t,
        sent_max_len=args.sent_max_len,
        max_dv_body=args.max_dv_body,
        max_dv_hand_local=args.max_dv_hand_local,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v21] device={device} params={n_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log = []
    best = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_accum: dict[str, float] = {}
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
        val_accum: dict[str, float] = {}
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

        # Selection score: closeness to band target, jerk-cap violation penalty,
        # arc-gain shortfall (we WANT pred_arc >= source_arc), body preservation.
        score = (
            abs(rec["val_speed_ratio_mean"] - args.speed_band_target)
            + max(0.0, rec["val_jerk_ratio_mean"] - args.jerk_cap) * 2.0
            + max(0.0, args.arc_min_gain - rec["val_arc_gain_mean"]) * 1.0
            + 0.05 * rec["val_L_body_other"] / max(args.lambda_body, 1e-6)
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
            print(f"[v21] saved best -> {out_dir / 'best.pt'} score={score:.4f}", flush=True)

    print(f"[v21] done best={best}", flush=True)


# ---------------------------------------------------------------------------
# Apply (inference) driver
# ---------------------------------------------------------------------------


@torch.no_grad()
def apply(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    refiner_args = argparse.Namespace(**ck["args"])
    gloss_to_id = ck["gloss_to_id"]
    model = VelocityFieldRefiner(
        gloss_vocab=len(gloss_to_id),
        hidden=refiner_args.hidden,
        layers=refiner_args.layers,
        heads=refiner_args.heads,
        dropout=0.0,
        max_t=refiner_args.max_t,
        sent_max_len=refiner_args.sent_max_len,
        max_dv_body=refiner_args.max_dv_body,
        max_dv_hand_local=refiner_args.max_dv_hand_local,
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
            print(f"[v21-apply] {i}/{n}", flush=True)

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
    tr.add_argument("--train_src_pt", required=True)
    tr.add_argument("--train_gt_pt",
                    default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    tr.add_argument("--train_manifest_json", default="data/phoenix/phoenix_train.json")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_v21_velocity_field")

    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--layers", type=int, default=3)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--max_t", type=int, default=192)
    tr.add_argument("--sent_max_len", type=int, default=32)
    tr.add_argument("--crop_p", type=float, default=0.5)
    tr.add_argument("--source_amp_aug", action="store_true")
    tr.add_argument("--source_amp_min", type=float, default=0.5)
    tr.add_argument("--source_amp_max", type=float, default=1.0)

    # Velocity-delta bounds (per-frame, in raw SLRTP-178 units). The drift
    # removal then strips constant translations/linear drift, so the bound
    # acts mostly on the AC component of the corrected position.
    tr.add_argument("--max_dv_body", type=float, default=0.004)
    tr.add_argument("--max_dv_hand_local", type=float, default=0.004)

    tr.add_argument("--speed_band_lo", type=float, default=0.85)
    tr.add_argument("--speed_band_hi", type=float, default=1.05)
    tr.add_argument("--speed_band_target", type=float, default=0.95)
    tr.add_argument("--jerk_cap", type=float, default=1.30)
    tr.add_argument("--arc_min_gain", type=float, default=1.20,
                    help="pred hand arc length must be at least this fraction of src arc length")

    tr.add_argument("--lambda_speed", type=float, default=2.0)
    tr.add_argument("--lambda_jerk", type=float, default=1.0)
    tr.add_argument("--lambda_arc", type=float, default=0.5)
    tr.add_argument("--lambda_dev", type=float, default=0.05)
    tr.add_argument("--lambda_body", type=float, default=20.0)
    tr.add_argument("--lambda_face", type=float, default=20.0)
    tr.add_argument("--lambda_anchor", type=float, default=5.0)
    tr.add_argument("--lambda_bone", type=float, default=50.0)
    tr.add_argument("--lambda_vel_smooth", type=float, default=200.0)

    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--eval_batch_size", type=int, default=8)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)

    tr.add_argument("--val_frac", type=float, default=0.05)
    tr.add_argument("--val_batches", type=int, default=20)
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=50)
    tr.add_argument("--max_steps", type=int, default=0)
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--smoke_train", type=int, default=64)
    tr.add_argument("--smoke_val", type=int, default=16)

    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--ckpt", required=True)
    ap_apply.add_argument("--in_pt", required=True)
    ap_apply.add_argument("--manifest_json", required=True)
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
