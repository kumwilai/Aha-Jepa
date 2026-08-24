"""Paired motion-detail refiner for Sign-JEPA outputs.

This is the next step after the standalone AE-flow result:

    old low-motion pose + gloss program -> full-motion pose

Training uses real Phoenix clips with synthetic low-pass collapse and optional
old-generator carriers as the source. Inference uses the old JEPA generator
output as the semantic carrier. The model is a source-conditioned rectified
flow in normalized pose space.
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

from scripts.train_sign_jepa_ae_fullclip import accel_loss, hand_jerk, load_pose_rows, region_speed  # noqa: E402
from scripts.train_sign_jepa_flow_generator import temporal_smooth  # noqa: E402
from scripts.train_sign_jepa_generator_slrtp178 import speed_envelope_loss, std_match_loss, velocity_loss  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import LENGTH_BUCKETS, length_bucket_id, resize_seq, sinusoidal_positions  # noqa: E402

PAD = 0
UNK = 1
HAND_SLICE = slice(8, 50)


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_gloss_vocab(rows: list[dict]) -> dict[str, int]:
    glosses = sorted({g for r in rows for g in str(r.get("gloss", "")).split() if g})
    return {"<pad>": PAD, "<unk>": UNK, **{g: i + 2 for i, g in enumerate(glosses)}}


def as_pose_tensor(x) -> torch.Tensor:
    if isinstance(x, dict):
        x = x.get("poses_3d", x.get("pose"))
    if x is None:
        raise ValueError("source item does not contain a pose tensor")
    x = torch.as_tensor(x).float()
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    if x.ndim == 2:
        return x
    raise ValueError(f"expected pose rank 2 or 3, got {tuple(x.shape)}")


def load_source_maps(spec: str) -> list[dict[str, torch.Tensor]]:
    paths = [p.strip() for p in str(spec or "").split(",") if p.strip()]
    source_maps = []
    for p in paths:
        data = torch.load(ROOT / p, map_location="cpu", weights_only=False)
        cleaned = {}
        for sid, pose in data.items():
            try:
                flat = as_pose_tensor(pose)
            except Exception:
                continue
            if flat.shape[1] == 534 and torch.isfinite(flat).all():
                cleaned[str(sid)] = flat.contiguous()
        source_maps.append(cleaned)
        print(f"[motion-refiner] loaded train source {p} clips={len(cleaned)}", flush=True)
    return source_maps


class RefinerDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        gloss_to_id: dict[str, int],
        mean: torch.Tensor,
        std: torch.Tensor,
        T_pose: int,
        max_gloss_len: int,
        kernels: list[int],
        source_maps: list[dict[str, torch.Tensor]] | None = None,
        source_prob: float = 0.75,
        generated_target_mode: str = "full",
        detail_kernel: int = 9,
        detail_scale: float = 1.0,
        max_items: int = 0,
        seed: int = 0,
    ):
        rows = list(rows)
        if max_items and len(rows) > max_items:
            rows = random.Random(seed).sample(rows, max_items)
        self.rows = rows
        self.gloss_to_id = gloss_to_id
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)
        self.T_pose = int(T_pose)
        self.max_gloss_len = int(max_gloss_len)
        self.kernels = [int(k) for k in kernels if int(k) > 1 and int(k) % 2 == 1]
        if not self.kernels:
            self.kernels = [9]
        self.source_maps = source_maps or []
        self.source_prob = float(np.clip(source_prob, 0.0, 1.0))
        self.generated_target_mode = str(generated_target_mode)
        self.detail_kernel = int(detail_kernel)
        if self.detail_kernel > 1 and self.detail_kernel % 2 == 0:
            raise ValueError("--detail_kernel must be odd")
        self.detail_scale = float(detail_scale)

    def __len__(self) -> int:
        return len(self.rows)

    def make_lowpass_source(self, target_raw: torch.Tensor) -> tuple[torch.Tensor, int, str]:
        k = random.choice(self.kernels)
        src_raw = temporal_smooth(target_raw, k)
        # Random partial smoothing makes train inputs cover old-generator-like
        # under-motion and less-collapsed carriers.
        blend = random.uniform(0.65, 1.0)
        src_raw = (1.0 - blend) * target_raw + blend * src_raw
        return src_raw.reshape(self.T_pose, -1), int(target_raw.shape[0]), f"lowpass{k}"

    def make_train_source(self, sid: str, target_raw: torch.Tensor, target_len: int) -> tuple[torch.Tensor, int, str]:
        if self.source_maps and random.random() < self.source_prob:
            source_map = random.choice(self.source_maps)
            src = source_map.get(sid)
            if src is not None:
                src_len = int(src.shape[0])
                src = resize_seq(src, self.T_pose)
                return src.float(), src_len, "generated"
        src, _src_len, kind = self.make_lowpass_source(target_raw)
        return src, int(target_len), kind

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        sid = str(row["id"])
        target_len = int(row["pose"].shape[0])
        gt_raw = resize_seq(row["pose"].reshape(target_len, -1), self.T_pose).reshape(self.T_pose, 178, 3)
        src_raw, source_len, source_kind = self.make_train_source(sid, gt_raw, target_len)
        if source_kind == "generated" and self.generated_target_mode == "detail":
            gt_low = temporal_smooth(gt_raw, self.detail_kernel) if self.detail_kernel > 1 else torch.zeros_like(gt_raw)
            gt_detail = gt_raw.reshape(self.T_pose, -1) - gt_low.reshape(self.T_pose, -1)
            target_raw = src_raw.reshape(self.T_pose, -1) + self.detail_scale * gt_detail
        else:
            target_raw = gt_raw.reshape(self.T_pose, -1)
        target = (target_raw.reshape(self.T_pose, -1) - self.mean) / self.std
        src = (src_raw.reshape(self.T_pose, -1) - self.mean) / self.std
        glosses = [g for g in str(row.get("gloss", "")).split() if g]
        ids = [self.gloss_to_id.get(g, UNK) for g in glosses][: self.max_gloss_len]
        if not ids:
            ids = [UNK]
        return {
            "id": sid,
            "source": src.float(),
            "target": target.float(),
            "source_raw": src_raw.reshape(self.T_pose, -1).float(),
            "target_raw": target_raw.reshape(self.T_pose, -1).float(),
            "gt_raw": gt_raw.reshape(self.T_pose, -1).float(),
            "gloss_ids": torch.tensor(ids, dtype=torch.long),
            "raw_len": torch.tensor(source_len, dtype=torch.long),
            "target_len": torch.tensor(target_len, dtype=torch.long),
            "source_kind": source_kind,
        }


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    G = max(len(x["gloss_ids"]) for x in batch)
    gloss_ids = torch.full((B, G), PAD, dtype=torch.long)
    gloss_mask = torch.zeros(B, G, dtype=torch.bool)
    for i, item in enumerate(batch):
        ids = item["gloss_ids"]
        gloss_ids[i, : len(ids)] = ids
        gloss_mask[i, : len(ids)] = True
    return {
        "source": torch.stack([x["source"] for x in batch]),
        "target": torch.stack([x["target"] for x in batch]),
        "source_raw": torch.stack([x["source_raw"] for x in batch]),
        "target_raw": torch.stack([x["target_raw"] for x in batch]),
        "gt_raw": torch.stack([x["gt_raw"] for x in batch]),
        "gloss_ids": gloss_ids,
        "gloss_mask": gloss_mask,
        "raw_len": torch.stack([x["raw_len"] for x in batch]),
        "target_len": torch.stack([x["target_len"] for x in batch]),
        "ids": [x["id"] for x in batch],
        "source_kind": [x["source_kind"] for x in batch],
    }


class SourceFlowRefiner(nn.Module):
    def __init__(
        self,
        num_gloss: int,
        pose_dim: int = 534,
        hidden: int = 256,
        gloss_layers: int = 2,
        layers: int = 6,
        heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden = hidden
        self.x_proj = nn.Linear(pose_dim, hidden)
        self.src_proj = nn.Linear(pose_dim, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.progress_mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gloss_emb = nn.Embedding(num_gloss, hidden, padding_idx=PAD)
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
        gloss_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.gloss_encoder = nn.TransformerEncoder(gloss_layer, num_layers=gloss_layers)
        layer = nn.TransformerDecoderLayer(
            hidden, heads, hidden * 4, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerDecoder(layer, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, pose_dim))

    def encode_gloss(self, gloss_ids: torch.Tensor, gloss_mask: torch.Tensor, len_id: torch.Tensor):
        G = gloss_ids.shape[1]
        h = self.gloss_emb(gloss_ids)
        h = h + sinusoidal_positions(G, self.hidden, gloss_ids.device)[None]
        h = h + self.len_emb(len_id)[:, None]
        return self.gloss_encoder(h, src_key_padding_mask=~gloss_mask)

    def forward(self, x_t, source, t, gloss_ids, gloss_mask, len_id):
        B, T, _ = x_t.shape
        device = x_t.device
        mem = self.encode_gloss(gloss_ids, gloss_mask, len_id)
        p = torch.linspace(0.0, 1.0, T, device=device)
        prog = torch.stack([p, 1.0 - p, torch.sin(math.pi * p), torch.cos(math.pi * p)], dim=-1)
        tt = torch.stack([t, 1.0 - t, torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)
        q = (
            self.x_proj(x_t)
            + self.src_proj(source)
            + sinusoidal_positions(T, self.hidden, device)[None]
            + self.progress_mlp(prog)[None]
            + self.t_mlp(tt)[:, None]
        )
        h = self.blocks(q, mem, memory_key_padding_mask=~gloss_mask)
        return self.out(h)


@torch.no_grad()
def motion_metrics(pred_batch: torch.Tensor, ref_batch: torch.Tensor) -> dict:
    sp, sr = region_speed(pred_batch), region_speed(ref_batch)
    return {
        "hand_speed_ratio": sp["hand"] / max(sr["hand"], 1e-9),
        "body_speed_ratio": sp["body"] / max(sr["body"], 1e-9),
        "face_speed_ratio": sp["face"] / max(sr["face"], 1e-9),
        "hand_jerk_ratio": hand_jerk(pred_batch) / max(hand_jerk(ref_batch), 1e-9),
        "hand_posestd_ratio": float(pred_batch[:, :, 8:50].std()) / max(float(ref_batch[:, :, 8:50].std()), 1e-9),
    }


def endpoint_losses(pred, target):
    pv = pred[:, 1:] - pred[:, :-1]
    tv = target[:, 1:] - target[:, :-1]
    pa = pv[:, 1:] - pv[:, :-1]
    ta = tv[:, 1:] - tv[:, :-1]
    pj = pa[:, 1:] - pa[:, :-1]
    tj = ta[:, 1:] - ta[:, :-1]
    return {
        "pose": F.smooth_l1_loss(pred, target),
        "vel": velocity_loss(pred, target),
        "accel": accel_loss(pred, target),
        "jerk": F.smooth_l1_loss(pj[:, :, HAND_SLICE], tj[:, :, HAND_SLICE]),
        "speed": speed_envelope_loss(pred, target),
        "std": std_match_loss(pred, target),
    }


def hand_speed_scalar(pose: torch.Tensor) -> torch.Tensor:
    """Mean hand velocity magnitude for [T,178,3] or [B,T,178,3]."""
    if pose.ndim == 3:
        h = pose[:, HAND_SLICE]
        return (h[1:] - h[:-1]).norm(dim=-1).mean()
    h = pose[:, :, HAND_SLICE]
    return (h[:, 1:] - h[:, :-1]).norm(dim=-1).mean(dim=(1, 2))


def build_energy_stats(rows: list[dict], T_pose: int) -> dict:
    vals: dict[int, list[float]] = {i: [] for i in range(len(LENGTH_BUCKETS))}
    all_vals = []
    for row in rows:
        pose = row["pose"]
        raw_len = int(pose.shape[0])
        p = resize_seq(pose.reshape(raw_len, -1), T_pose).reshape(T_pose, 178, 3).float()
        v = float(hand_speed_scalar(p))
        bid = length_bucket_id(raw_len)
        vals.setdefault(bid, []).append(v)
        all_vals.append(v)
    global_v = float(np.median(all_vals)) if all_vals else 1.0
    by_bucket = []
    for i in range(len(LENGTH_BUCKETS)):
        bucket_vals = vals.get(i, [])
        by_bucket.append(float(np.median(bucket_vals)) if bucket_vals else global_v)
    return {"hand_speed_by_bucket": torch.tensor(by_bucket).float(), "global_hand_speed": global_v}


def spectral_source_lock(
    x: torch.Tensor,
    source: torch.Tensor,
    kernel: int,
    mode: str = "highpass",
    detail_kernel: int = 3,
) -> torch.Tensor:
    """Keep JEPA low-frequency semantics and expose only high-frequency hands.

    `x` and `source` are normalized [B,T,534] poses.  The returned pose keeps
    body/face exactly from `source`.  Hand low-pass also comes from `source`;
    only hand high-pass is allowed to come from `x`.
    """
    if kernel <= 1:
        return x
    if kernel % 2 == 0:
        raise ValueError("--spectral_source_lock_kernel must be odd")
    B, T, D = x.shape
    if D != 534:
        raise ValueError(f"expected pose dim 534, got {D}")
    x_pose = x.reshape(B, T, 178, 3)
    src_pose = source.reshape(B, T, 178, 3)
    src_lp = temporal_smooth(source.reshape(B, T, D), kernel).reshape(B, T, 178, 3)
    out = src_pose.clone()
    if mode == "highpass":
        x_lp = temporal_smooth(x.reshape(B, T, D), kernel).reshape(B, T, 178, 3)
        hand_detail = x_pose[:, :, HAND_SLICE] - x_lp[:, :, HAND_SLICE]
    elif mode == "midband":
        if detail_kernel <= 1 or detail_kernel % 2 == 0:
            raise ValueError("--spectral_detail_kernel must be odd and > 1 for midband lock")
        if detail_kernel >= kernel:
            raise ValueError("--spectral_detail_kernel must be smaller than --spectral_source_lock_kernel")
        x_lp_fine = temporal_smooth(x.reshape(B, T, D), detail_kernel).reshape(B, T, 178, 3)
        x_lp_coarse = temporal_smooth(x.reshape(B, T, D), kernel).reshape(B, T, 178, 3)
        hand_detail = x_lp_fine[:, :, HAND_SLICE] - x_lp_coarse[:, :, HAND_SLICE]
    else:
        raise ValueError(f"unknown spectral lock mode {mode!r}")
    out[:, :, HAND_SLICE] = src_lp[:, :, HAND_SLICE] + hand_detail
    return out.reshape(B, T, D)


def make_flow_start(source: torch.Tensor, args: argparse.Namespace, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Initial state for the rectified flow.

    `source` starts at the JEPA carrier.  `noise_detail` keeps the JEPA
    low-frequency pose intact but replaces the hand high-frequency component
    with Gaussian detail.  This turns the refiner into a conditional motion
    detail generator instead of a deterministic source-to-target smoother.
    """
    mode = getattr(args, "flow_start", "source")
    if mode == "source":
        return source
    if mode != "noise_detail":
        raise ValueError(f"unknown --flow_start {mode!r}")
    scale = float(getattr(args, "flow_noise_scale", 1.0))
    if generator is None:
        noise = torch.randn_like(source) * scale
    else:
        noise = torch.randn(source.shape, generator=generator, device=source.device, dtype=source.dtype) * scale
    start = source + noise
    kernel = int(getattr(args, "spectral_source_lock_kernel", 1))
    if kernel > 1:
        start = spectral_source_lock(
            start,
            source,
            kernel,
            mode=getattr(args, "spectral_lock_mode", "highpass"),
            detail_kernel=int(getattr(args, "spectral_detail_kernel", 3)),
        )
    return start


def energy_conserve_spectral_pose(
    pose: torch.Tensor,
    source: torch.Tensor,
    target_speed: float,
    kernel: int,
    energy_scale: float = 1.0,
    min_residual_scale: float = 0.25,
    max_residual_scale: float = 8.0,
) -> torch.Tensor:
    """Raw-pose spectral lock with per-clip hand-energy normalization.

    The low-pass hand trajectory is source-locked, but the learned high-pass
    residual is rescaled so the final hand velocity approaches a real-data
    bucket target.
    """
    if kernel <= 1:
        return pose
    pose3 = pose.reshape(pose.shape[0], 178, 3).float()
    src3 = source.reshape(source.shape[0], 178, 3).float()
    locked = src3.clone()
    pose_lp = temporal_smooth(pose3, kernel)
    src_lp = temporal_smooth(src3, kernel)
    high = pose3[:, HAND_SLICE] - pose_lp[:, HAND_SLICE]
    locked[:, HAND_SLICE] = src_lp[:, HAND_SLICE] + high
    cur = float(hand_speed_scalar(locked).clamp_min(1e-8))
    tgt = max(float(target_speed) * float(energy_scale), 1e-8)
    scale = float(np.clip(tgt / cur, min_residual_scale, max_residual_scale))
    locked[:, HAND_SLICE] = src_lp[:, HAND_SLICE] + scale * high
    return locked.reshape(pose.shape[0], -1)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = torch.load(ROOT / args.stats_ckpt, map_location="cpu", weights_only=False)
    mean = stats["mean"].float()
    std = stats["std"].float().clamp_min(1e-6)
    manifest = load_rows(ROOT / args.train_manifest)
    gloss_to_id = build_gloss_vocab(manifest)
    meta = {str(r["id"]): r for r in manifest}
    pose_rows = load_pose_rows(ROOT / args.train_pt)
    rows = []
    for row in pose_rows:
        m = meta.get(row["id"], row)
        row = dict(row)
        row["gloss"] = str(m.get("gloss", row.get("gloss", "")))
        row["length"] = int(m.get("length", row["pose"].shape[0]) or row["pose"].shape[0])
        rows.append(row)
    energy_stats = build_energy_stats(rows, args.T_pose)
    random.Random(args.seed).shuffle(rows)
    n_val = max(args.min_val, int(args.val_frac * len(rows)))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    if args.smoke:
        train_rows = train_rows[: args.smoke_train]
        val_rows = val_rows[: args.smoke_val]
    kernels = [int(x) for x in args.source_kernels.split(",") if x.strip()]
    source_maps = load_source_maps(args.train_source_pt)
    train_ds = RefinerDataset(
        train_rows, gloss_to_id, mean, std, args.T_pose, args.max_gloss_len, kernels,
        source_maps=source_maps, source_prob=args.source_prob,
        generated_target_mode=args.generated_target_mode,
        detail_kernel=args.detail_kernel,
        detail_scale=args.detail_scale,
        max_items=args.max_train_items, seed=args.seed,
    )
    val_ds = RefinerDataset(
        val_rows, gloss_to_id, mean, std, args.T_pose, args.max_gloss_len, kernels,
        source_maps=source_maps, source_prob=args.val_source_prob,
        generated_target_mode=args.generated_target_mode,
        detail_kernel=args.detail_kernel,
        detail_scale=args.detail_scale,
        max_items=args.max_val_items, seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=args.drop_last)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    model = SourceFlowRefiner(len(gloss_to_id), hidden=args.hidden, gloss_layers=args.gloss_layers, layers=args.layers, heads=args.heads, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[motion-refiner] device={device} train={len(train_ds)} val={len(val_ds)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    def run_epoch(loader, train_mode: bool) -> dict:
        model.train(train_mode)
        vals = {k: [] for k in ["loss", "fm", "pose", "vel", "accel", "jerk", "speed", "std"]}
        ratios = []
        for step, batch in enumerate(loader, start=1):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            gloss_ids = batch["gloss_ids"].to(device)
            gloss_mask = batch["gloss_mask"].to(device)
            len_id = torch.as_tensor([length_bucket_id(int(x)) for x in batch["raw_len"].tolist()], dtype=torch.long, device=device)
            t = torch.rand(target.shape[0], device=device)
            start = make_flow_start(source, args)
            x_t = (1.0 - t[:, None, None]) * start + t[:, None, None] * target
            if train_mode and args.noise_std > 0:
                x_t = x_t + torch.randn_like(x_t) * args.noise_std * (1.0 - t[:, None, None])
            with torch.set_grad_enabled(train_mode):
                v = model(x_t, source, t, gloss_ids, gloss_mask, len_id)
                end = start + v
                loss_target = target
                loss_start = start
                if args.spectral_source_lock_kernel > 1:
                    end = spectral_source_lock(
                        end,
                        source,
                        args.spectral_source_lock_kernel,
                        mode=args.spectral_lock_mode,
                        detail_kernel=args.spectral_detail_kernel,
                    )
                    loss_target = spectral_source_lock(
                        target,
                        source,
                        args.spectral_source_lock_kernel,
                        mode=args.spectral_lock_mode,
                        detail_kernel=args.spectral_detail_kernel,
                    )
                    loss_start = spectral_source_lock(
                        start,
                        source,
                        args.spectral_source_lock_kernel,
                        mode=args.spectral_lock_mode,
                        detail_kernel=args.spectral_detail_kernel,
                    )
                    fm = F.mse_loss(end - loss_start, loss_target - loss_start)
                else:
                    fm = F.mse_loss(v, target - start)
                el = endpoint_losses(end, loss_target)
                loss = (
                    args.lambda_fm * fm
                    + args.lambda_pose * el["pose"]
                    + args.lambda_vel * el["vel"]
                    + args.lambda_accel * el["accel"]
                    + args.lambda_jerk * el["jerk"]
                    + args.lambda_speed * el["speed"]
                    + args.lambda_std * el["std"]
                )
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    g = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
            vals["loss"].append(float(loss.detach().cpu()))
            vals["fm"].append(float(fm.detach().cpu()))
            for k in el:
                vals[k].append(float(el[k].detach().cpu()))
            if (not train_mode) and args.eval_motion_batches and len(ratios) < args.eval_motion_batches:
                mean_d = mean.to(device)
                std_d = std.to(device)
                praw = (end.detach() * std_d + mean_d).reshape(end.shape[0], end.shape[1], 178, 3)
                traw = batch["gt_raw"].to(device).reshape(end.shape[0], end.shape[1], 178, 3)
                ratios.append(motion_metrics(praw, traw))
            if train_mode and args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({"step": step, "loss": vals["loss"][-1], "fm": vals["fm"][-1], "pose": vals["pose"][-1], "speed": vals["speed"][-1], "grad_norm": float(g)}), flush=True)
            if train_mode and args.max_steps and step >= args.max_steps:
                break
            if (not train_mode) and args.val_batches and step >= args.val_batches:
                break
        prefix = "train" if train_mode else "val"
        rec = {f"{prefix}_{k}": float(np.mean(v)) for k, v in vals.items()}
        if ratios:
            for k in ratios[0].keys():
                rec[f"{prefix}_{k}"] = float(np.mean([r[k] for r in ratios]))
        return rec

    best = None
    log = []

    def checkpoint_payload(metrics: dict) -> dict:
        return {
            "model": model.state_dict(),
            "args": vars(args),
            "best": metrics,
            "mean": mean,
            "std": std,
            "gloss_to_id": gloss_to_id,
            "hidden": args.hidden,
            "T_pose": args.T_pose,
            "energy_stats": energy_stats,
        }

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1), **tr, **va}
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)
        if args.save_every_epochs and ep % args.save_every_epochs == 0:
            ckpt_path = out_dir / f"epoch_{ep:03d}.pt"
            torch.save(checkpoint_payload(dict(rec)), ckpt_path)
            (out_dir / f"epoch_{ep:03d}.json").write_text(json.dumps(rec, indent=2))
            print(f"[motion-refiner] saved epoch checkpoint -> {ckpt_path}", flush=True)
        if best is None or rec["val_loss"] < best["val_loss"]:
            best = dict(rec)
            torch.save(checkpoint_payload(best), out_dir / "best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[motion-refiner] saved best -> {out_dir / 'best.pt'}", flush=True)


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SourceFlowRefiner(
        len(ckpt["gloss_to_id"]),
        hidden=ckpt["hidden"],
        gloss_layers=ckpt["args"]["gloss_layers"],
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
    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    rows = load_rows(ROOT / args.manifest_json)
    source_pt = torch.load(ROOT / args.source_pt, map_location="cpu", weights_only=True)
    if args.reference_pt:
        ref = torch.load(ROOT / args.reference_pt, map_location="cpu", weights_only=False)
        row_by_id = {str(r["id"]): r for r in rows}
        rows = [row_by_id[str(sid)] for sid in ref.keys() if str(sid) in row_by_id and str(sid) in source_pt]
    if args.max_clips:
        rows = rows[: args.max_clips]
    out = {}
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        kept = []
        sources = []
        lens = []
        gloss_seqs = []
        for row in batch_rows:
            sid = str(row["id"])
            if sid not in source_pt:
                continue
            src_raw = as_pose_tensor(source_pt[sid])
            L = int(src_raw.shape[0])
            src = resize_seq(src_raw, ckpt["T_pose"])
            glosses = [g for g in str(row.get("gloss", "")).split() if g]
            ids = [ckpt["gloss_to_id"].get(g, UNK) for g in glosses][: ckpt["args"]["max_gloss_len"]] or [UNK]
            kept.append((sid, L, src))
            sources.append(((src.to(device) - mean) / std).float())
            lens.append(L)
            gloss_seqs.append(ids)
        if not kept:
            continue
        G = max(len(ids) for ids in gloss_seqs)
        gloss_ids = torch.full((len(kept), G), PAD, dtype=torch.long, device=device)
        gloss_mask = torch.zeros((len(kept), G), dtype=torch.bool, device=device)
        for i, ids in enumerate(gloss_seqs):
            gloss_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            gloss_mask[i, : len(ids)] = True
        source = torch.stack(sources, dim=0)
        len_id = torch.tensor([length_bucket_id(L) for L in lens], dtype=torch.long, device=device)
        flow_args = argparse.Namespace(
            flow_start=ckpt["args"].get("flow_start", "source"),
            flow_noise_scale=ckpt["args"].get("flow_noise_scale", 1.0),
            spectral_source_lock_kernel=args.spectral_source_lock_kernel,
            spectral_lock_mode=ckpt["args"].get("spectral_lock_mode", "highpass"),
            spectral_detail_kernel=ckpt["args"].get("spectral_detail_kernel", 3),
        )
        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed) + int(start))
        x = make_flow_start(source, flow_args, generator=gen)
        for s in range(args.steps):
            t = torch.full((x.shape[0],), s / max(1, args.steps), device=device)
            v = model(x, source, t, gloss_ids, gloss_mask, len_id)
            x = x + v / max(1, args.steps)
        if args.spectral_source_lock_kernel > 1:
            x = spectral_source_lock(
                x,
                source,
                args.spectral_source_lock_kernel,
                mode=ckpt["args"].get("spectral_lock_mode", "highpass"),
                detail_kernel=ckpt["args"].get("spectral_detail_kernel", 3),
            )
        pose_batch = (x * std + mean).cpu()
        for j, (sid, L, src) in enumerate(kept):
            pose = pose_batch[j]
            if args.residual_scale != 1.0:
                src_pose = src.cpu()
                pose = src_pose + args.residual_scale * (pose - src_pose)
            if args.energy_conserve_spectral and args.spectral_source_lock_kernel > 1:
                stats = ckpt.get("energy_stats") or {}
                speeds = stats.get("hand_speed_by_bucket")
                if speeds is None:
                    raise ValueError("checkpoint does not contain energy_stats; retrain or patch the checkpoint")
                target_speed = float(torch.as_tensor(speeds)[length_bucket_id(L)])
                pose = energy_conserve_spectral_pose(
                    pose,
                    src.cpu(),
                    target_speed=target_speed,
                    kernel=args.spectral_source_lock_kernel,
                    energy_scale=args.energy_scale,
                    min_residual_scale=args.energy_min_residual_scale,
                    max_residual_scale=args.energy_max_residual_scale,
                )
            pose = resize_seq(pose, L).reshape(L, 178, 3).float()
            if args.post_smooth_kernel > 1 and args.post_smooth_blend > 0:
                pose = (1 - args.post_smooth_blend) * pose + args.post_smooth_blend * temporal_smooth(pose, args.post_smooth_kernel)
            out[sid] = pose.contiguous()
        done = min(start + len(batch_rows), len(rows))
        if args.log_every and (done == len(batch_rows) or done % args.log_every < batch_size or done == len(rows)):
            print(f"[motion-refiner-sample] {done}/{len(rows)}", flush=True)
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--stats_ckpt", default="outputs/sota_chase/sign_jepa_ae_fullclip/best.pt")
    tr.add_argument("--train_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    tr.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_motion_refiner")
    tr.add_argument("--T_pose", type=int, default=160)
    tr.add_argument("--max_gloss_len", type=int, default=64)
    tr.add_argument("--source_kernels", default="7,9,11,13,15")
    tr.add_argument(
        "--train_source_pt",
        default="",
        help="Optional comma-separated .pt files of old-generator train carriers keyed by clip id.",
    )
    tr.add_argument("--source_prob", type=float, default=0.75)
    tr.add_argument("--val_source_prob", type=float, default=1.0)
    tr.add_argument("--generated_target_mode", choices=["full", "detail"], default="full")
    tr.add_argument("--detail_kernel", type=int, default=9)
    tr.add_argument("--detail_scale", type=float, default=1.0)
    tr.add_argument("--spectral_source_lock_kernel", type=int, default=1)
    tr.add_argument("--spectral_lock_mode", choices=["highpass", "midband"], default="highpass")
    tr.add_argument("--spectral_detail_kernel", type=int, default=3)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--gloss_layers", type=int, default=2)
    tr.add_argument("--layers", type=int, default=6)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--batch_size", type=int, default=32)
    tr.add_argument("--eval_batch_size", type=int, default=48)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--lambda_fm", type=float, default=1.0)
    tr.add_argument("--lambda_pose", type=float, default=1.0)
    tr.add_argument("--lambda_vel", type=float, default=2.0)
    tr.add_argument("--lambda_accel", type=float, default=2.0)
    tr.add_argument("--lambda_jerk", type=float, default=0.0)
    tr.add_argument("--lambda_speed", type=float, default=1.0)
    tr.add_argument("--lambda_std", type=float, default=0.5)
    tr.add_argument("--noise_std", type=float, default=0.02)
    tr.add_argument("--flow_start", choices=["source", "noise_detail"], default="source")
    tr.add_argument("--flow_noise_scale", type=float, default=1.0)
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
    tr.add_argument("--save_every_epochs", type=int, default=0)
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
    sm.add_argument("--residual_scale", type=float, default=1.0)
    sm.add_argument("--post_smooth_kernel", type=int, default=1)
    sm.add_argument("--post_smooth_blend", type=float, default=0.0)
    sm.add_argument("--spectral_source_lock_kernel", type=int, default=1)
    sm.add_argument("--energy_conserve_spectral", action="store_true")
    sm.add_argument("--energy_scale", type=float, default=1.0)
    sm.add_argument("--energy_min_residual_scale", type=float, default=0.25)
    sm.add_argument("--energy_max_residual_scale", type=float, default=8.0)
    sm.add_argument("--batch_size", type=int, default=16)
    sm.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sm.add_argument("--log_every", type=int, default=100)
    sm.add_argument("--max_clips", type=int, default=0)
    sm.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample_manifest(args)


if __name__ == "__main__":
    main()
