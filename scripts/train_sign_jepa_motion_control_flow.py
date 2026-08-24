"""JEPA-conditioned motion-control flow.

This script tests a different algorithm from pose-space refinement:

    JEPA carrier pose -> band-limited wrist-local hand velocity controls -> pose

The learned variable is a compact DCT coefficient vector for hand velocity
corrections.  At sampling, body/face and wrists remain locked to the JEPA
carrier; generated controls are integrated into wrist-local hands and
reattached to the JEPA wrists.
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

from scripts.train_sign_jepa_ae_fullclip import WholeClipPoseAE, hand_jerk, load_pose_rows, region_speed  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import LENGTH_BUCKETS, length_bucket_id, resize_seq, sequence_features, sinusoidal_positions  # noqa: E402

PAD = 0
UNK = 1
BODY_SLICE = slice(0, 8)
RH = slice(8, 29)
LH = slice(29, 50)
HAND = slice(8, 50)
FACE_SLICE = slice(50, 178)
RWRIST = 2
LWRIST = 5


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_gloss_vocab(rows: list[dict]) -> dict[str, int]:
    glosses = sorted({g for r in rows for g in str(r.get("gloss", "")).split() if g})
    return {"<pad>": PAD, "<unk>": UNK, **{g: i + 2 for i, g in enumerate(glosses)}}


def as_pose_tensor(x) -> torch.Tensor:
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
    out = {}
    for sid, item in data.items():
        try:
            pose = as_pose_tensor(item)
        except Exception:
            continue
        if torch.isfinite(pose).all():
            out[str(sid)] = pose.contiguous()
    print(f"[control-flow] loaded source {path} clips={len(out)}", flush=True)
    return out


def dct_basis(n: int, k: int, device=None, dtype=torch.float32) -> torch.Tensor:
    t = torch.arange(n, device=device, dtype=dtype)[:, None]
    f = torch.arange(k, device=device, dtype=dtype)[None]
    b = torch.cos(math.pi / n * (t + 0.5) * f)
    b[:, 0] *= math.sqrt(1.0 / n)
    if k > 1:
        b[:, 1:] *= math.sqrt(2.0 / n)
    return b


def hand_local(pose: torch.Tensor) -> torch.Tensor:
    rh = pose[:, RH] - pose[:, RWRIST:RWRIST + 1]
    lh = pose[:, LH] - pose[:, LWRIST:LWRIST + 1]
    return torch.cat([rh, lh], dim=1)


def endpoint_locked_residual(src_local: torch.Tensor, gt_local: torch.Tensor) -> torch.Tensor:
    resid = gt_local - src_local
    a = torch.linspace(0.0, 1.0, resid.shape[0], dtype=resid.dtype, device=resid.device)[:, None, None]
    ramp = (1.0 - a) * resid[:1] + a * resid[-1:]
    return resid - ramp


def coeff_from_velocity(vel: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nk,bnjc->bkjc", basis, vel)


def velocity_from_coeff(coeff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nk,bkjc->bnjc", basis, coeff)


def hand_speed_batch(pose: torch.Tensor) -> torch.Tensor:
    v = pose[:, 1:, HAND] - pose[:, :-1, HAND]
    return v.norm(dim=-1).mean(dim=(1, 2))


def hand_jerk_batch(pose: torch.Tensor) -> torch.Tensor:
    v = pose[:, 1:, HAND] - pose[:, :-1, HAND]
    a = v[:, 1:] - v[:, :-1]
    j = a[:, 1:] - a[:, :-1]
    return j.norm(dim=-1).mean(dim=(1, 2))


def apply_controls(
    source: torch.Tensor,
    coeff: torch.Tensor,
    basis: torch.Tensor,
    endpoint_lock: bool = True,
    control_target: str = "velocity",
) -> torch.Tensor:
    B, T = source.shape[:2]
    if control_target == "velocity":
        vel = velocity_from_coeff(coeff, basis)
        resid = torch.cat([torch.zeros(B, 1, 42, 3, device=source.device, dtype=source.dtype), vel.cumsum(dim=1)], dim=1)
        if endpoint_lock:
            a = torch.linspace(0.0, 1.0, T, dtype=source.dtype, device=source.device)[None, :, None, None]
            resid = resid - a * resid[:, -1:]
    elif control_target == "displacement":
        resid = velocity_from_coeff(coeff, basis)
        resid = resid - resid[:, :1]
    elif control_target == "two_branch":
        disp_coeff, vel_coeff = coeff.chunk(2, dim=1)
        gross = velocity_from_coeff(disp_coeff, basis)
        gross = gross - gross[:, :1]
        vel_basis = dct_basis(T - 1, vel_coeff.shape[1], device=source.device, dtype=source.dtype)
        vel = velocity_from_coeff(vel_coeff, vel_basis)
        detail = torch.cat([torch.zeros(B, 1, 42, 3, device=source.device, dtype=source.dtype), vel.cumsum(dim=1)], dim=1)
        if endpoint_lock:
            a = torch.linspace(0.0, 1.0, T, dtype=source.dtype, device=source.device)[None, :, None, None]
            detail = detail - a * detail[:, -1:]
        resid = gross + detail
    else:
        raise ValueError(f"unknown control_target {control_target!r}")
    src_local = torch.cat([
        source[:, :, RH] - source[:, :, RWRIST:RWRIST + 1],
        source[:, :, LH] - source[:, :, LWRIST:LWRIST + 1],
    ], dim=2)
    local = src_local + resid
    out = source.clone()
    out[:, :, RH] = source[:, :, RWRIST:RWRIST + 1] + local[:, :, :21]
    out[:, :, LH] = source[:, :, LWRIST:LWRIST + 1] + local[:, :, 21:]
    return out


def compute_control_stats(ds: Dataset, max_items: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(ds) if not max_items else min(len(ds), max_items)
    vals = []
    for i in range(n):
        vals.append(ds[i]["coeff"].reshape(-1))
    x = torch.stack(vals)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    return mean, std


class ControlDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        source_map: dict[str, torch.Tensor],
        gloss_to_id: dict[str, int],
        basis: torch.Tensor,
        T_pose: int,
        max_gloss_len: int,
        control_target: str = "velocity",
        max_items: int = 0,
        seed: int = 0,
    ):
        rows = [r for r in rows if str(r["id"]) in source_map]
        if max_items and len(rows) > max_items:
            rows = random.Random(seed).sample(rows, max_items)
        self.rows = rows
        self.source_map = source_map
        self.gloss_to_id = gloss_to_id
        self.basis = basis.cpu()
        self.T_pose = int(T_pose)
        self.max_gloss_len = int(max_gloss_len)
        self.control_target = str(control_target)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        sid = str(row["id"])
        src_raw = resize_seq(self.source_map[sid].reshape(self.source_map[sid].shape[0], -1), self.T_pose).reshape(self.T_pose, 178, 3)
        gt_len = int(row["pose"].shape[0])
        gt_raw = resize_seq(row["pose"].reshape(gt_len, -1), self.T_pose).reshape(self.T_pose, 178, 3)
        resid = endpoint_locked_residual(hand_local(src_raw), hand_local(gt_raw))
        if self.control_target == "velocity":
            target = resid[1:] - resid[:-1]
        elif self.control_target == "displacement":
            target = hand_local(gt_raw) - hand_local(src_raw)
            target = target - target[:1]
        elif self.control_target == "two_branch":
            resid_full = hand_local(gt_raw) - hand_local(src_raw)
            resid_full = resid_full - resid_full[:1]
            gross = coeff_from_velocity(resid_full[None], self.basis)[0]
            recon_gross = velocity_from_coeff(gross[None], self.basis)[0]
            detail = resid_full - recon_gross
            detail = detail - detail[:1]
            detail_vel = detail[1:] - detail[:-1]
            vel_basis = dct_basis(self.T_pose - 1, self.basis.shape[1])
            detail_coeff = coeff_from_velocity(detail_vel[None], vel_basis)[0]
            coeff = torch.cat([gross, detail_coeff], dim=0)
            target = None
        else:
            raise ValueError(f"unknown control_target {self.control_target!r}")
        if target is not None:
            coeff = coeff_from_velocity(target[None], self.basis)[0]
        glosses = [g for g in str(row.get("gloss", "")).split() if g]
        ids = [self.gloss_to_id.get(g, UNK) for g in glosses][: self.max_gloss_len] or [UNK]
        return {
            "id": sid,
            "source": src_raw.float(),
            "gt": gt_raw.float(),
            "coeff": coeff.float(),
            "gloss_ids": torch.tensor(ids, dtype=torch.long),
            "raw_len": torch.tensor(int(self.source_map[sid].shape[0]), dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    G = max(len(x["gloss_ids"]) for x in batch)
    gloss_ids = torch.full((len(batch), G), PAD, dtype=torch.long)
    gloss_mask = torch.zeros(len(batch), G, dtype=torch.bool)
    for i, item in enumerate(batch):
        ids = item["gloss_ids"]
        gloss_ids[i, : len(ids)] = ids
        gloss_mask[i, : len(ids)] = True
    return {
        "ids": [x["id"] for x in batch],
        "source": torch.stack([x["source"] for x in batch]),
        "gt": torch.stack([x["gt"] for x in batch]),
        "coeff": torch.stack([x["coeff"] for x in batch]),
        "gloss_ids": gloss_ids,
        "gloss_mask": gloss_mask,
        "raw_len": torch.stack([x["raw_len"] for x in batch]),
    }


class ControlFlow(nn.Module):
    def __init__(self, num_gloss: int, control_dim: int, hidden: int = 384, layers: int = 4, heads: int = 4, dropout: float = 0.05):
        super().__init__()
        self.hidden = hidden
        self.control_dim = control_dim
        self.src_proj = nn.Linear(534, hidden)
        self.gloss_emb = nn.Embedding(num_gloss, hidden, padding_idx=PAD)
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
        enc_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.src_enc = nn.TransformerEncoder(enc_layer, num_layers=max(1, layers // 2))
        self.gloss_enc = nn.TransformerEncoder(enc_layer, num_layers=max(1, layers // 2))
        self.x_proj = nn.Linear(control_dim, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, control_dim),
        )

    def encode(self, source, gloss_ids, gloss_mask, len_id):
        B, T, _, _ = source.shape
        src = source.reshape(B, T, 534)
        sh = self.src_proj(src) + sinusoidal_positions(T, self.hidden, source.device)[None]
        sh = self.src_enc(sh).mean(dim=1)
        G = gloss_ids.shape[1]
        gh = self.gloss_emb(gloss_ids) + sinusoidal_positions(G, self.hidden, source.device)[None] + self.len_emb(len_id)[:, None]
        gh = self.gloss_enc(gh, src_key_padding_mask=~gloss_mask)
        denom = gloss_mask.sum(dim=1).clamp_min(1).float()[:, None]
        gh = (gh * gloss_mask.float()[:, :, None]).sum(dim=1) / denom
        return sh + gh + self.len_emb(len_id)

    def forward(self, x_t, source, t, gloss_ids, gloss_mask, len_id):
        cond = self.encode(source, gloss_ids, gloss_mask, len_id)
        tt = torch.stack([t, 1.0 - t, torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)
        h = self.x_proj(x_t) + cond + self.t_mlp(tt)
        return self.net(h)


class PoseCritic(nn.Module):
    """Polymorphic frozen pose-encoder wrapper.

    Supports two backbones:
      * `ae`: encoder of `WholeClipPoseAE` (reconstruction objective).
      * `sign_jepa`: target encoder of the Sign-JEPA model (JEPA prediction
        objective on Phoenix train; trained to be motion-style invariant).

    Exposes a uniform `embed(pose_raw_flat, len_id) -> (B, T, H)` interface.
    Pose-normalization mean/std are captured from the source checkpoint and
    applied internally.
    """

    def __init__(self, kind: str, encoder: nn.Module, mean: torch.Tensor, std: torch.Tensor,
                 unk_gloss_id: int = 1):
        super().__init__()
        self.kind = kind
        self.encoder = encoder
        self.register_buffer("sem_mean", mean)
        self.register_buffer("sem_std", std)
        self.unk_gloss_id = int(unk_gloss_id)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def embed(self, pose_raw_flat: torch.Tensor, len_id: torch.Tensor) -> torch.Tensor:
        pose_norm = (pose_raw_flat - self.sem_mean) / self.sem_std
        feat, _ = sequence_features(pose_norm, pose_raw_flat)
        if self.kind == "ae":
            return self.encoder(feat)
        if self.kind == "sign_jepa":
            B, T = pose_raw_flat.shape[:2]
            zero_mask = torch.zeros(B, T, dtype=torch.bool, device=pose_raw_flat.device)
            gloss_id = torch.full((B,), self.unk_gloss_id, dtype=torch.long, device=pose_raw_flat.device)
            return self.encoder(feat, zero_mask, gloss_id, len_id)
        raise ValueError(f"unknown PoseCritic kind: {self.kind!r}")


def load_pose_critic(path: Path, kind: str, device: torch.device) -> PoseCritic:
    """Load a frozen pose-encoder critic.

    kind=ae       -> WholeClipPoseAE encoder, hidden=256, no conditioning
    kind=sign_jepa -> SignJEPAModel.target_encoder, hidden=256, conditioned on
                     gloss_id + len_id; uses UNK gloss_id so the conditioning
                     bias is constant across anchor and refined.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    if kind == "ae":
        ae = WholeClipPoseAE(
            feat_dim=ckpt["feat_dim"],
            hidden=ckpt["hidden"],
            enc_layers=a["enc_layers"],
            dec_layers=a["dec_layers"],
            heads=a["heads"],
            dropout=0.0,
        )
        ae.load_state_dict(ckpt["model"])
        ae.to(device)
        mean = ckpt["mean"].to(device).float()
        std = ckpt["std"].to(device).float().clamp_min(1e-6)
        return PoseCritic("ae", ae.encoder, mean, std).to(device)
    if kind == "sign_jepa":
        from scripts.train_sign_jepa_slrtp178 import SignJEPAModel  # local import to avoid circular cost
        model = SignJEPAModel(
            feat_dim=ckpt["feat_dim"],
            desc_dim=ckpt["desc_dim"],
            num_gloss=len(ckpt["gloss_to_id"]),
            hidden=a["hidden"],
            layers=a["layers"],
            heads=a["heads"],
            pred_layers=a["pred_layers"],
            dropout=0.0,
            max_len=a["seg_len"],
        )
        model.load_state_dict(ckpt["model"])
        model.to(device)
        # Sign-JEPA stores its own pose mean/std (same convention as AE).
        mean = ckpt["mean"].to(device).float()
        std = ckpt["std"].to(device).float().clamp_min(1e-6)
        unk_id = int(ckpt["gloss_to_id"].get("<unk>", 1))
        return PoseCritic("sign_jepa", model.target_encoder, mean, std, unk_gloss_id=unk_id).to(device)
    raise ValueError(f"unknown jepa_sem_type: {kind!r}")


def jepa_semantic_loss(z_pred: torch.Tensor, z_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cosine + L2 semantic distance between per-frame latents.

    z_pred: (B, T, H) latent from refined pose (with grad)
    z_ref: (B, T, H) latent from anchor pose (detached)
    Returns (combined, cos_loss, l2_loss).
    """
    z_ref = z_ref.detach()
    pn = F.normalize(z_pred, dim=-1)
    rn = F.normalize(z_ref, dim=-1)
    cos_loss = (1.0 - (pn * rn).sum(dim=-1)).mean()
    l2_loss = F.mse_loss(z_pred, z_ref)
    return cos_loss + l2_loss, cos_loss, l2_loss


def _amp_smooth_batched(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Replicate-padded temporal avg-pool on a (B, T, J, 3) tensor."""
    if kernel <= 1:
        return x
    B, T = x.shape[:2]
    y = x.reshape(B, T, -1).transpose(1, 2)  # (B, F, T)
    pad = kernel // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=kernel, stride=1)
    return y.transpose(1, 2).reshape(x.shape)


def amplify_batched(pose: torch.Tensor, g_body: float, g_hand: float, g_face: float = 1.0,
                    lp_kernel: int = 5) -> torch.Tensor:
    """Batched version of `sign_jepa_motion_amplify.amplify_clip`.

    pose: (B, T, 178, 3) JEPA carrier.
    Returns motion-amplified `S+ = amp(S, g_body, g_hand, g_face, lp_kernel)`
    where body / right-hand-local / left-hand-local / face deviations from
    their respective per-clip temporal means are LP-amplified by the gains.
    Hands are re-attached to amplified wrists exactly (no joint drift).
    Closed-form, fully differentiable, no learning.
    """
    pose = pose.float()
    out = pose.clone()
    # body
    body = pose[:, :, BODY_SLICE]
    body_mean = body.mean(dim=1, keepdim=True)
    body_dev = body - body_mean
    body_dev_lp = _amp_smooth_batched(body_dev, lp_kernel)
    body_amp_dev = body_dev + (g_body - 1.0) * body_dev_lp if lp_kernel > 1 else g_body * body_dev
    body_amp = body_mean + body_amp_dev
    out[:, :, BODY_SLICE] = body_amp
    # right hand (attached to body wrist idx 2)
    rh_root = pose[:, :, RWRIST:RWRIST + 1]
    rh_local = pose[:, :, RH] - rh_root
    rh_local_mean = rh_local.mean(dim=1, keepdim=True)
    rh_local_dev = rh_local - rh_local_mean
    rh_dev_lp = _amp_smooth_batched(rh_local_dev, lp_kernel)
    rh_amp_dev = rh_local_dev + (g_hand - 1.0) * rh_dev_lp if lp_kernel > 1 else g_hand * rh_local_dev
    rh_local_amp = rh_local_mean + rh_amp_dev
    out[:, :, RH] = body_amp[:, :, RWRIST:RWRIST + 1] + rh_local_amp
    # left hand (attached to body wrist idx 5)
    lh_root = pose[:, :, LWRIST:LWRIST + 1]
    lh_local = pose[:, :, LH] - lh_root
    lh_local_mean = lh_local.mean(dim=1, keepdim=True)
    lh_local_dev = lh_local - lh_local_mean
    lh_dev_lp = _amp_smooth_batched(lh_local_dev, lp_kernel)
    lh_amp_dev = lh_local_dev + (g_hand - 1.0) * lh_dev_lp if lp_kernel > 1 else g_hand * lh_local_dev
    lh_local_amp = lh_local_mean + lh_amp_dev
    out[:, :, LH] = body_amp[:, :, LWRIST:LWRIST + 1] + lh_local_amp
    # face (default off)
    if g_face != 1.0:
        face = pose[:, :, FACE_SLICE]
        face_mean = face.mean(dim=1, keepdim=True)
        face_dev = face - face_mean
        face_dev_lp = _amp_smooth_batched(face_dev, lp_kernel)
        face_amp_dev = face_dev + (g_face - 1.0) * face_dev_lp if lp_kernel > 1 else g_face * face_dev
        out[:, :, FACE_SLICE] = face_mean + face_amp_dev
    return out


def sibling_rank_loss(z_pred: torch.Tensor, z_plus: torch.Tensor, z_minus: torch.Tensor,
                      margin: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contrastive margin loss: refined latent should be closer to the
    motion-passed sibling `S+` than to the motion-failed carrier `S-`.

    L = softplus( d(z_pred, z_plus) - d(z_pred, z_minus) + margin )

    Using d = 1 - cosine_similarity (per-frame averaged). Both anchors are
    detached. At identity (pred = carrier = S-): d_minus = 0, d_plus > 0,
    so loss is large and positive — identity is provably the worst point.
    """
    z_plus = z_plus.detach()
    z_minus = z_minus.detach()
    pn = F.normalize(z_pred, dim=-1)
    pp = F.normalize(z_plus, dim=-1)
    pm = F.normalize(z_minus, dim=-1)
    d_plus = (1.0 - (pn * pp).sum(dim=-1)).mean()
    d_minus = (1.0 - (pn * pm).sum(dim=-1)).mean()
    loss = F.softplus(d_plus - d_minus + margin)
    return loss, d_plus, d_minus


@torch.no_grad()
def batch_motion_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    sp, sr = region_speed(pred), region_speed(ref)
    return {
        "hand_speed_ratio": sp["hand"] / max(sr["hand"], 1e-9),
        "body_speed_ratio": sp["body"] / max(sr["body"], 1e-9),
        "face_speed_ratio": sp["face"] / max(sr["face"], 1e-9),
        "hand_jerk_ratio": hand_jerk(pred) / max(hand_jerk(ref), 1e-9),
        "hand_posestd_ratio": float(pred[:, :, HAND].std()) / max(float(ref[:, :, HAND].std()), 1e-9),
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_rows(ROOT / args.train_manifest)
    gloss_to_id = build_gloss_vocab(manifest)
    meta = {str(r["id"]): r for r in manifest}
    pose_rows = []
    for row in load_pose_rows(ROOT / args.train_pt):
        m = meta.get(str(row["id"]), row)
        row = dict(row)
        row["gloss"] = str(m.get("gloss", row.get("gloss", "")))
        pose_rows.append(row)
    source_map = load_pose_map(ROOT / args.train_source_pt)
    random.Random(args.seed).shuffle(pose_rows)
    n_val = max(args.min_val, int(args.val_frac * len(pose_rows)))
    val_rows, train_rows = pose_rows[:n_val], pose_rows[n_val:]
    if args.smoke:
        train_rows = train_rows[: args.smoke_train]
        val_rows = val_rows[: args.smoke_val]
    basis_len = args.T_pose - 1 if args.control_target == "velocity" else args.T_pose
    control_branches = 2 if args.control_target == "two_branch" else 1
    basis = dct_basis(basis_len, args.control_basis)
    train_ds = ControlDataset(train_rows, source_map, gloss_to_id, basis, args.T_pose, args.max_gloss_len, args.control_target, args.max_train_items, args.seed)
    val_ds = ControlDataset(val_rows, source_map, gloss_to_id, basis, args.T_pose, args.max_gloss_len, args.control_target, args.max_val_items, args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=args.drop_last)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    control_dim = control_branches * args.control_basis * 42 * 3
    model = ControlFlow(len(gloss_to_id), control_dim, hidden=args.hidden, layers=args.layers, heads=args.heads, dropout=args.dropout).to(device)
    if args.init_from:
        init_ckpt = torch.load(ROOT / args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(init_ckpt["model"], strict=False)
        print(f"[control-flow] warm-start init_from={args.init_from} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    basis_d = basis.to(device)
    ctrl_mean, ctrl_std = compute_control_stats(train_ds, args.control_stats_items)
    ctrl_mean_d = ctrl_mean.to(device)
    ctrl_std_d = ctrl_std.to(device)
    critic = None
    if args.lambda_jepa_sem > 0 or args.lambda_sibling > 0:
        critic = load_pose_critic(ROOT / args.jepa_sem_ckpt, args.jepa_sem_type, device)
        n_enc_params = sum(p.numel() for p in critic.parameters())
        print(f"[control-flow] semantic critic [{args.jepa_sem_type}] loaded from {args.jepa_sem_ckpt} "
              f"({n_enc_params/1e6:.2f}M frozen) anchor={args.jepa_sem_source} "
              f"lambda_sem={args.lambda_jepa_sem} lambda_sibling={args.lambda_sibling}", flush=True)
    sibling_lp_choices = tuple(int(k) for k in str(args.sibling_lp_choices).split(",") if k.strip())
    print(f"[control-flow] device={device} train={len(train_ds)} val={len(val_ds)} basis={args.control_basis} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    def run_epoch(loader, train_mode: bool) -> dict:
        model.train(train_mode)
        vals = {k: [] for k in ["loss", "fm", "coeff", "vel", "pose", "speed", "jerk", "gate",
                                "sem", "sem_cos", "sem_l2", "sib", "sib_dplus", "sib_dminus"]}
        ratios = []
        for step, batch in enumerate(loader, 1):
            source = batch["source"].to(device)
            gt = batch["gt"].to(device)
            target_raw = batch["coeff"].to(device).reshape(source.shape[0], -1)
            target = (target_raw - ctrl_mean_d) / ctrl_std_d
            gloss_ids = batch["gloss_ids"].to(device)
            gloss_mask = batch["gloss_mask"].to(device)
            len_id = torch.tensor([length_bucket_id(int(x)) for x in batch["raw_len"].tolist()], dtype=torch.long, device=device)
            start = torch.randn_like(target) * args.noise_scale
            t = torch.rand(target.shape[0], device=device)
            x_t = (1.0 - t[:, None]) * start + t[:, None] * target
            with torch.set_grad_enabled(train_mode):
                v = model(x_t, source, t, gloss_ids, gloss_mask, len_id)
                end = start + v
                fm = F.mse_loss(v, target - start)
                coeff_loss = F.smooth_l1_loss(end, target)
                end_raw = end * ctrl_std_d + ctrl_mean_d
                end_coeff = end_raw.reshape(source.shape[0], control_branches * args.control_basis, 42, 3)
                tgt_coeff = target_raw.reshape(source.shape[0], control_branches * args.control_basis, 42, 3)
                pred_pose = apply_controls(source, end_coeff, basis_d, control_target=args.control_target)
                tgt_pose = apply_controls(source, tgt_coeff, basis_d, control_target=args.control_target)
                pred_vel = pred_pose[:, 1:, HAND] - pred_pose[:, :-1, HAND]
                tgt_vel = tgt_pose[:, 1:, HAND] - tgt_pose[:, :-1, HAND]
                pred_j = pred_vel[:, 2:] - 2.0 * pred_vel[:, 1:-1] + pred_vel[:, :-2]
                tgt_j = tgt_vel[:, 2:] - 2.0 * tgt_vel[:, 1:-1] + tgt_vel[:, :-2]
                vel_loss = F.smooth_l1_loss(pred_vel, tgt_vel)
                pose_loss = F.smooth_l1_loss(pred_pose[:, :, HAND], tgt_pose[:, :, HAND])
                pred_speed = hand_speed_batch(pred_pose)
                gt_speed = hand_speed_batch(gt).clamp_min(1e-8)
                pred_jerk = hand_jerk_batch(pred_pose)
                gt_jerk = hand_jerk_batch(gt).clamp_min(1e-8)
                speed_loss = F.smooth_l1_loss(pred_speed, gt_speed)
                jerk_loss = F.smooth_l1_loss(pred_j, tgt_j)
                gate_loss = (
                    F.relu(args.speed_floor * gt_speed - pred_speed).div(gt_speed).pow(2).mean()
                    + F.relu(pred_jerk - args.jerk_cap * gt_jerk).div(gt_jerk).pow(2).mean()
                )
                pred_flat = pred_pose.reshape(pred_pose.shape[0], pred_pose.shape[1], 534)
                if critic is not None and args.lambda_jepa_sem > 0:
                    anchor = source if args.jepa_sem_source == "source_pose" else gt
                    anchor_flat = anchor.reshape(anchor.shape[0], anchor.shape[1], 534)
                    z_pred_sem = critic.embed(pred_flat, len_id)
                    with torch.no_grad():
                        z_anchor = critic.embed(anchor_flat, len_id)
                    sem_total, sem_cos, sem_l2 = jepa_semantic_loss(z_pred_sem, z_anchor)
                else:
                    sem_total = torch.zeros((), device=device)
                    sem_cos = torch.zeros((), device=device)
                    sem_l2 = torch.zeros((), device=device)

                if critic is not None and args.lambda_sibling > 0:
                    # Stochastic per-batch sibling synth so the renderer cannot
                    # memorise a single canonical amp config.
                    g_body_b = float(random.uniform(args.sibling_g_lo, args.sibling_g_hi))
                    g_hand_b = float(random.uniform(args.sibling_g_lo, args.sibling_g_hi))
                    lp_b = int(random.choice(sibling_lp_choices))
                    with torch.no_grad():
                        sib_plus = amplify_batched(source, g_body_b, g_hand_b, g_face=1.0, lp_kernel=lp_b)
                    sib_plus_flat = sib_plus.reshape(sib_plus.shape[0], sib_plus.shape[1], 534)
                    sib_minus_flat = source.reshape(source.shape[0], source.shape[1], 534)
                    z_pred_sib = critic.embed(pred_flat, len_id)
                    with torch.no_grad():
                        z_plus = critic.embed(sib_plus_flat, len_id)
                        z_minus = critic.embed(sib_minus_flat, len_id)
                    sib_total, sib_dplus, sib_dminus = sibling_rank_loss(
                        z_pred_sib, z_plus, z_minus, margin=args.sibling_margin
                    )
                else:
                    sib_total = torch.zeros((), device=device)
                    sib_dplus = torch.zeros((), device=device)
                    sib_dminus = torch.zeros((), device=device)
                loss = (
                    args.lambda_fm * fm
                    + args.lambda_coeff * coeff_loss
                    + args.lambda_vel * vel_loss
                    + args.lambda_pose * pose_loss
                    + args.lambda_speed * speed_loss
                    + args.lambda_jerk * jerk_loss
                    + args.lambda_gate * gate_loss
                    + args.lambda_jepa_sem * sem_total
                    + args.lambda_sibling * sib_total
                )
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
            vals["loss"].append(float(loss.detach().cpu()))
            vals["fm"].append(float(fm.detach().cpu()))
            vals["coeff"].append(float(coeff_loss.detach().cpu()))
            vals["vel"].append(float(vel_loss.detach().cpu()))
            vals["pose"].append(float(pose_loss.detach().cpu()))
            vals["speed"].append(float(speed_loss.detach().cpu()))
            vals["jerk"].append(float(jerk_loss.detach().cpu()))
            vals["gate"].append(float(gate_loss.detach().cpu()))
            vals["sem"].append(float(sem_total.detach().cpu()))
            vals["sem_cos"].append(float(sem_cos.detach().cpu()))
            vals["sem_l2"].append(float(sem_l2.detach().cpu()))
            vals["sib"].append(float(sib_total.detach().cpu()))
            vals["sib_dplus"].append(float(sib_dplus.detach().cpu()))
            vals["sib_dminus"].append(float(sib_dminus.detach().cpu()))
            if (not train_mode) and args.eval_motion_batches and len(ratios) < args.eval_motion_batches:
                ratios.append(batch_motion_metrics(pred_pose.detach(), gt.detach()))
            if train_mode and args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({"step": step, "loss": vals["loss"][-1], "fm": vals["fm"][-1], "speed": vals["speed"][-1], "grad_norm": float(grad)}), flush=True)
            if train_mode and args.max_steps and step >= args.max_steps:
                break
            if (not train_mode) and args.val_batches and step >= args.val_batches:
                break
        prefix = "train" if train_mode else "val"
        rec = {f"{prefix}_{k}": float(np.mean(v)) for k, v in vals.items()}
        if ratios:
            for k in ratios[0]:
                rec[f"{prefix}_{k}"] = float(np.mean([r[k] for r in ratios]))
        return rec

    best = None
    log = []

    def payload(metrics: dict) -> dict:
        return {
            "model": model.state_dict(),
            "args": vars(args),
            "best": metrics,
            "gloss_to_id": gloss_to_id,
            "basis": basis,
            "control_mean": ctrl_mean,
            "control_std": ctrl_std,
            "T_pose": args.T_pose,
            "control_basis": args.control_basis,
            "control_branches": control_branches,
            "control_target": args.control_target,
            "hidden": args.hidden,
        }

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1), **tr, **va}
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)
        if best is None or rec["val_loss"] < best["val_loss"]:
            best = dict(rec)
            torch.save(payload(best), out_dir / "best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[control-flow] saved best -> {out_dir / 'best.pt'}", flush=True)


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    control_dim = int(ckpt.get("control_branches", 1)) * int(ckpt["control_basis"]) * 42 * 3
    model = ControlFlow(
        len(ckpt["gloss_to_id"]),
        control_dim,
        hidden=ckpt["hidden"],
        layers=ckpt["args"]["layers"],
        heads=ckpt["args"]["heads"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def sample_manifest(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, ckpt = load_model(ROOT / args.ckpt, device)
    basis = ckpt["basis"].to(device)
    ctrl_mean = ckpt["control_mean"].to(device)
    ctrl_std = ckpt["control_std"].to(device)
    rows = load_rows(ROOT / args.manifest_json)
    source_map = load_pose_map(ROOT / args.source_pt)
    if args.reference_pt:
        ref = torch.load(ROOT / args.reference_pt, map_location="cpu", weights_only=False)
        by_id = {str(r["id"]): r for r in rows}
        rows = [by_id[str(sid)] for sid in ref.keys() if str(sid) in by_id and str(sid) in source_map]
    if args.max_clips:
        rows = rows[: args.max_clips]
    out = {}
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    control_branches = int(ckpt.get("control_branches", 1))
    control_dim = control_branches * int(ckpt["control_basis"]) * 42 * 3
    for start_i in range(0, len(rows), args.batch_size):
        batch_rows = rows[start_i:start_i + args.batch_size]
        kept, sources, gloss_seqs, lens = [], [], [], []
        for row in batch_rows:
            sid = str(row["id"])
            if sid not in source_map:
                continue
            src_raw = source_map[sid]
            L = int(src_raw.shape[0])
            src = resize_seq(src_raw.reshape(L, -1), ckpt["T_pose"]).reshape(ckpt["T_pose"], 178, 3)
            glosses = [g for g in str(row.get("gloss", "")).split() if g]
            ids = [ckpt["gloss_to_id"].get(g, UNK) for g in glosses][: ckpt["args"]["max_gloss_len"]] or [UNK]
            kept.append((sid, L, src))
            sources.append(src.to(device))
            gloss_seqs.append(ids)
            lens.append(L)
        if not kept:
            continue
        source = torch.stack(sources)
        G = max(len(ids) for ids in gloss_seqs)
        gloss_ids = torch.full((len(kept), G), PAD, dtype=torch.long, device=device)
        gloss_mask = torch.zeros(len(kept), G, dtype=torch.bool, device=device)
        for i, ids in enumerate(gloss_seqs):
            gloss_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            gloss_mask[i, : len(ids)] = True
        len_id = torch.tensor([length_bucket_id(L) for L in lens], dtype=torch.long, device=device)
        x = torch.randn((len(kept), control_dim), generator=gen, device=device) * float(ckpt["args"].get("noise_scale", 1.0))
        for s in range(args.steps):
            t = torch.full((len(kept),), s / max(1, args.steps), device=device)
            x = x + model(x, source, t, gloss_ids, gloss_mask, len_id) / max(1, args.steps)
        coeff_raw = x * ctrl_std + ctrl_mean
        coeff = coeff_raw.reshape(len(kept), control_branches * int(ckpt["control_basis"]), 42, 3)
        pose_batch = apply_controls(
            source,
            coeff,
            basis,
            control_target=ckpt.get("control_target", ckpt["args"].get("control_target", "velocity")),
        ).cpu()
        for j, (sid, L, _src) in enumerate(kept):
            pose = resize_seq(pose_batch[j].reshape(ckpt["T_pose"], -1), L).reshape(L, 178, 3).float()
            out[sid] = pose.contiguous()
        done = min(start_i + len(batch_rows), len(rows))
        if args.log_every and (done == len(batch_rows) or done % args.log_every < args.batch_size or done == len(rows)):
            print(f"[control-flow-sample] {done}/{len(rows)}", flush=True)
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--train_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    tr.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    tr.add_argument("--train_source_pt", default="external/SLRTP-Sign-Production-Evaluation/results/sign_jepa_train_scale080.pt")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_motion_control_flow")
    tr.add_argument("--T_pose", type=int, default=160)
    tr.add_argument("--control_basis", type=int, default=16)
    tr.add_argument("--control_target", choices=["velocity", "displacement", "two_branch"], default="velocity")
    tr.add_argument("--max_gloss_len", type=int, default=64)
    tr.add_argument("--hidden", type=int, default=384)
    tr.add_argument("--layers", type=int, default=4)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--eval_batch_size", type=int, default=96)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--noise_scale", type=float, default=1.0)
    tr.add_argument("--control_stats_items", type=int, default=0)
    tr.add_argument("--lambda_fm", type=float, default=1.0)
    tr.add_argument("--lambda_coeff", type=float, default=0.5)
    tr.add_argument("--lambda_vel", type=float, default=2.0)
    tr.add_argument("--lambda_pose", type=float, default=0.5)
    tr.add_argument("--lambda_speed", type=float, default=1.0)
    tr.add_argument("--lambda_jerk", type=float, default=1.0)
    tr.add_argument("--lambda_gate", type=float, default=0.0)
    tr.add_argument("--speed_floor", type=float, default=0.85)
    tr.add_argument("--jerk_cap", type=float, default=1.6)
    tr.add_argument("--lambda_jepa_sem", type=float, default=0.0,
                    help="Weight of frozen-JEPA semantic-preservation loss on refined poses (0 disables).")
    tr.add_argument("--jepa_sem_type", choices=["ae", "sign_jepa"], default="ae",
                    help="Critic backbone: 'ae' uses WholeClipPoseAE encoder (reconstruction objective), "
                         "'sign_jepa' uses SignJEPAModel.target_encoder (JEPA prediction objective).")
    tr.add_argument("--jepa_sem_ckpt", default="outputs/sota_chase/sign_jepa_ae_fullclip/best.pt",
                    help="Critic checkpoint path; format must match --jepa_sem_type.")
    tr.add_argument("--jepa_sem_source", choices=["source_pose", "gt_pose"], default="source_pose",
                    help="Anchor pose for the semantic critic: source_pose preserves the JEPA carrier's semantic; gt_pose distills the GT semantic.")
    tr.add_argument("--init_from", default="",
                    help="Optional warm-start checkpoint with matching architecture (e.g. two_branch best.pt).")
    tr.add_argument("--lambda_sibling", type=float, default=0.0,
                    help="Weight of SiblingRank contrastive loss (0 disables). Requires a critic loaded via --lambda_jepa_sem>0 OR --jepa_sem_ckpt path (we reuse the same critic for sibling embeddings).")
    tr.add_argument("--sibling_g_lo", type=float, default=2.5,
                    help="Lower bound of stochastic amp gain for sibling synthesis.")
    tr.add_argument("--sibling_g_hi", type=float, default=4.5,
                    help="Upper bound of stochastic amp gain for sibling synthesis.")
    tr.add_argument("--sibling_lp_choices", default="3,5,7",
                    help="Comma-separated LP kernel choices for sibling synthesis (sampled per batch).")
    tr.add_argument("--sibling_margin", type=float, default=0.05,
                    help="Margin in the softplus margin loss; d(P,S+) must be < d(P,S-) - margin to incur zero loss.")
    tr.add_argument("--val_frac", type=float, default=0.05)
    tr.add_argument("--min_val", type=int, default=256)
    tr.add_argument("--val_batches", type=int, default=20)
    tr.add_argument("--eval_motion_batches", type=int, default=4)
    tr.add_argument("--max_train_items", type=int, default=0)
    tr.add_argument("--max_val_items", type=int, default=1024)
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=100)
    tr.add_argument("--max_steps", type=int, default=0)
    tr.add_argument("--drop_last", action="store_true")
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--smoke_train", type=int, default=64)
    tr.add_argument("--smoke_val", type=int, default=32)

    sm = sub.add_parser("sample_manifest")
    sm.add_argument("--ckpt", required=True)
    sm.add_argument("--source_pt", required=True)
    sm.add_argument("--manifest_json", required=True)
    sm.add_argument("--reference_pt", default="")
    sm.add_argument("--out_pt", required=True)
    sm.add_argument("--steps", type=int, default=20)
    sm.add_argument("--batch_size", type=int, default=32)
    sm.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--log_every", type=int, default=100)
    sm.add_argument("--max_clips", type=int, default=0)

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample_manifest(args)


if __name__ == "__main__":
    main()
