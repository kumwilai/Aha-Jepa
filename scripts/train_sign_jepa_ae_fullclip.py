"""Whole-clip reconstruction autoencoder for native SLRTP-178 poses.

This is the reconstruction latent used by the Sign-JEPA collapse fix. It is
intentionally pose-only: no gloss/text conditioning enters the autoencoder.
The downstream generator owns conditioning; the AE owns faithful motion decode.
"""
from __future__ import annotations

import argparse
import json
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

from scripts.train_sign_jepa_generator_slrtp178 import (  # noqa: E402
    JEPAPoseDecoder,
    speed_envelope_loss,
    std_match_loss,
    velocity_loss,
)
from scripts.train_sign_jepa_slrtp178 import resize_seq, sequence_features, sinusoidal_positions  # noqa: E402


def accel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pa = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    ta = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return F.smooth_l1_loss(pa, ta)


def region_speed(raw: torch.Tensor) -> dict[str, float]:
    v = raw[:, 1:] - raw[:, :-1]
    return {
        "hand": float(v[:, :, 8:50].norm(dim=-1).mean()),
        "body": float(v[:, :, 0:8].norm(dim=-1).mean()),
        "face": float(v[:, :, 50:178].norm(dim=-1).mean()),
    }


def hand_jerk(raw: torch.Tensor) -> float:
    v = raw[:, 1:] - raw[:, :-1]
    a = v[:, 1:] - v[:, :-1]
    j = a[:, 1:] - a[:, :-1]
    return float(j[:, :, 8:50].norm(dim=-1).mean())


def load_pose_rows(path: Path) -> list[dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for sid, row in data.items():
        pose = row["poses_3d"].float()
        if pose.ndim != 3 or pose.shape[1:] != (178, 3) or pose.shape[0] < 4:
            continue
        rows.append({
            "id": str(sid),
            "pose": pose,
            "text": str(row.get("text", "")),
            "gloss": str(row.get("gloss", "")),
        })
    return rows


class FullClipPoseDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        mean: torch.Tensor,
        std: torch.Tensor,
        T_pose: int,
        max_items: int = 0,
        seed: int = 0,
    ):
        rows = list(rows)
        if max_items and len(rows) > max_items:
            rows = random.Random(seed).sample(rows, max_items)
        self.rows = rows
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)
        self.T_pose = int(T_pose)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        pose_raw = resize_seq(row["pose"].reshape(row["pose"].shape[0], -1), self.T_pose)
        pose = (pose_raw - self.mean) / self.std
        return {
            "id": row["id"],
            "pose_raw": pose_raw.float(),
            "pose": pose.float(),
            "raw_len": torch.tensor(int(row["pose"].shape[0]), dtype=torch.long),
        }


class PoseOnlyEncoder(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.hidden = int(hidden)
        self.in_proj = nn.Linear(feat_dim, hidden)
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

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(feat)
        h = h + sinusoidal_positions(feat.shape[1], self.hidden, feat.device)[None]
        return self.norm(self.blocks(h))


class WholeClipPoseAE(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden: int = 256,
        enc_layers: int = 4,
        dec_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.encoder = PoseOnlyEncoder(feat_dim, hidden, enc_layers, heads, dropout)
        self.decoder = JEPAPoseDecoder(hidden, 534, dec_layers, heads, dropout)

    def forward(self, pose_norm: torch.Tensor, pose_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat, _desc = sequence_features(pose_norm, pose_raw)
        z = self.encoder(feat)
        return z, self.decoder(z)


def collate(batch: list[dict]) -> dict:
    return {
        "pose_raw": torch.stack([x["pose_raw"] for x in batch]),
        "pose": torch.stack([x["pose"] for x in batch]),
        "raw_len": torch.stack([x["raw_len"] for x in batch]),
        "ids": [x["id"] for x in batch],
    }


@torch.no_grad()
def evaluate(model, loader, mean, std, device, args, max_batches: int = 0) -> dict:
    model.eval()
    losses, pose_losses, vel_losses, accel_losses = [], [], [], []
    acc = {f"{r}_{k}": [] for r in ("hand", "body", "face") for k in ("pred", "real")}
    jerk_pred, jerk_real, std_pred, std_real = [], [], [], []
    mean = mean.to(device)
    std = std.to(device)
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        pose = batch["pose"].to(device)
        pose_raw = batch["pose_raw"].to(device)
        _z, pred = model(pose, pose_raw)
        pose_l = F.smooth_l1_loss(pred, pose)
        vel_l = velocity_loss(pred, pose)
        acc_l = accel_loss(pred, pose)
        loss = (
            args.lambda_pose * pose_l
            + args.lambda_vel * vel_l
            + args.lambda_accel * acc_l
            + args.lambda_speed * speed_envelope_loss(pred, pose)
            + args.lambda_std * std_match_loss(pred, pose)
        )
        losses.append(float(loss.cpu()))
        pose_losses.append(float(pose_l.cpu()))
        vel_losses.append(float(vel_l.cpu()))
        accel_losses.append(float(acc_l.cpu()))
        praw = (pred * std + mean).reshape(pred.shape[0], pred.shape[1], 178, 3)
        traw = pose_raw.reshape(pose_raw.shape[0], pose_raw.shape[1], 178, 3)
        sp, sr = region_speed(praw), region_speed(traw)
        for r in ("hand", "body", "face"):
            acc[f"{r}_pred"].append(sp[r])
            acc[f"{r}_real"].append(sr[r])
        jerk_pred.append(hand_jerk(praw))
        jerk_real.append(hand_jerk(traw))
        std_pred.append(float(praw[:, :, 8:50].std()))
        std_real.append(float(traw[:, :, 8:50].std()))
    ratio = lambda a, b: float(np.mean(a) / max(float(np.mean(b)), 1e-9))
    return {
        "loss": float(np.mean(losses)),
        "pose": float(np.mean(pose_losses)),
        "vel": float(np.mean(vel_losses)),
        "accel": float(np.mean(accel_losses)),
        "hand_speed_ratio": ratio(acc["hand_pred"], acc["hand_real"]),
        "body_speed_ratio": ratio(acc["body_pred"], acc["body_real"]),
        "face_speed_ratio": ratio(acc["face_pred"], acc["face_real"]),
        "hand_jerk_ratio": ratio(jerk_pred, jerk_real),
        "hand_posestd_ratio": ratio(std_pred, std_real),
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jepa_ckpt = torch.load(ROOT / args.stats_ckpt, map_location="cpu", weights_only=False)
    mean = jepa_ckpt["mean"].float()
    std = jepa_ckpt["std"].float().clamp_min(1e-6)
    rows = load_pose_rows(ROOT / args.train_pt)
    random.Random(args.seed).shuffle(rows)
    n_val = max(args.min_val, int(args.val_frac * len(rows)))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    if args.smoke:
        train_rows = train_rows[: args.smoke_train]
        val_rows = val_rows[: args.smoke_val]

    probe_ds = FullClipPoseDataset(train_rows[:1], mean, std, args.T_pose)
    feat, _ = sequence_features(probe_ds[0]["pose"][None], probe_ds[0]["pose_raw"][None])
    model = WholeClipPoseAE(
        feat_dim=feat.shape[-1],
        hidden=args.hidden,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    train_ds = FullClipPoseDataset(train_rows, mean, std, args.T_pose, args.max_train_items, args.seed)
    val_ds = FullClipPoseDataset(val_rows, mean, std, args.T_pose, args.max_val_items, args.seed + 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"[ae-fullclip] device={device} train={len(train_ds)} val={len(val_ds)} "
        f"T_pose={args.T_pose} feat_dim={feat.shape[-1]} hidden={args.hidden} "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
        flush=True,
    )

    best = None
    log = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        vals = []
        for step, batch in enumerate(train_loader, start=1):
            pose = batch["pose"].to(device)
            pose_raw = batch["pose_raw"].to(device)
            _z, pred = model(pose, pose_raw)
            pose_l = F.smooth_l1_loss(pred, pose)
            vel_l = velocity_loss(pred, pose)
            acc_l = accel_loss(pred, pose)
            speed_l = speed_envelope_loss(pred, pose)
            std_l = std_match_loss(pred, pose)
            loss = (
                args.lambda_pose * pose_l
                + args.lambda_vel * vel_l
                + args.lambda_accel * acc_l
                + args.lambda_speed * speed_l
                + args.lambda_std * std_l
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            vals.append(float(loss.detach().cpu()))
            if args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({
                    "step": step,
                    "loss": vals[-1],
                    "pose": float(pose_l.detach().cpu()),
                    "vel": float(vel_l.detach().cpu()),
                    "accel": float(acc_l.detach().cpu()),
                    "speed": float(speed_l.detach().cpu()),
                    "std": float(std_l.detach().cpu()),
                    "grad_norm": float(gnorm),
                }), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        va = evaluate(model, val_loader, mean, std, device, args, args.val_batches)
        rec = {
            "epoch": ep,
            "wall_s": round(time.time() - t0, 1),
            "train_loss": float(np.mean(vals)),
            **{f"val_{k}": v for k, v in va.items()},
        }
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)
        if best is None or rec["val_loss"] < best["val_loss"]:
            best = dict(rec)
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best": best,
                "mean": mean,
                "std": std,
                "feat_dim": feat.shape[-1],
                "hidden": args.hidden,
                "T_pose": args.T_pose,
            }, out_dir / "best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[ae-fullclip] saved best -> {out_dir / 'best.pt'}", flush=True)
    print(f"[ae-fullclip] done best={best}", flush=True)


@torch.no_grad()
def eval_ckpt(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    model = WholeClipPoseAE(
        feat_dim=ckpt["feat_dim"],
        hidden=ckpt["hidden"],
        enc_layers=ckpt["args"]["enc_layers"],
        dec_layers=ckpt["args"]["dec_layers"],
        heads=ckpt["args"]["heads"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    rows = load_pose_rows(ROOT / args.eval_pt)
    if args.max_clips:
        rows = rows[: args.max_clips]
    ds = FullClipPoseDataset(rows, ckpt["mean"], ckpt["std"], ckpt["T_pose"])
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
    dummy = argparse.Namespace(**ckpt["args"])
    metrics = evaluate(model, loader, ckpt["mean"], ckpt["std"], device, dummy, args.val_batches)
    metrics = {k: round(float(v), 6) for k, v in metrics.items()}
    print(json.dumps(metrics, indent=2), flush=True)
    if args.out_json:
        out = ROOT / args.out_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--stats_ckpt", default="outputs/sota_chase/sign_jepa_slrtp178/best.pt")
    tr.add_argument("--train_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_ae_fullclip")
    tr.add_argument("--T_pose", type=int, default=160)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--enc_layers", type=int, default=4)
    tr.add_argument("--dec_layers", type=int, default=3)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch_size", type=int, default=32)
    tr.add_argument("--eval_batch_size", type=int, default=48)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--lambda_pose", type=float, default=1.0)
    tr.add_argument("--lambda_vel", type=float, default=2.0)
    tr.add_argument("--lambda_accel", type=float, default=2.0)
    tr.add_argument("--lambda_speed", type=float, default=0.5)
    tr.add_argument("--lambda_std", type=float, default=0.5)
    tr.add_argument("--val_frac", type=float, default=0.05)
    tr.add_argument("--min_val", type=int, default=256)
    tr.add_argument("--val_batches", type=int, default=20)
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

    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--eval_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt")
    ev.add_argument("--out_json", default="")
    ev.add_argument("--eval_batch_size", type=int, default=48)
    ev.add_argument("--num_workers", type=int, default=2)
    ev.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ev.add_argument("--max_clips", type=int, default=0)
    ev.add_argument("--val_batches", type=int, default=0)

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        eval_ckpt(args)


if __name__ == "__main__":
    main()
