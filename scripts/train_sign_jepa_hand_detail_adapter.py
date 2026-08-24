"""Semantic-locked hand-detail adapter for Sign-JEPA carriers.

The old Sign-JEPA generator is treated as the semantic carrier.  This adapter
is allowed to edit only hand joints (8:50), with a low-pass preservation loss
that discourages rewriting lexical timing while restoring hand dynamics.
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

from scripts.train_sign_jepa_ae_fullclip import accel_loss, hand_jerk, region_speed  # noqa: E402
from scripts.train_sign_jepa_flow_generator import temporal_smooth  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import LENGTH_BUCKETS, length_bucket_id, resize_seq, sinusoidal_positions  # noqa: E402

PAD = 0
UNK = 1
HAND_SLICE = slice(8, 50)
HAND_DIM = 42 * 3


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_gloss_vocab(rows: list[dict]) -> dict[str, int]:
    glosses = sorted({g for r in rows for g in str(r.get("gloss", "")).split() if g})
    return {"<pad>": PAD, "<unk>": UNK, **{g: i + 2 for i, g in enumerate(glosses)}}


def as_pose_tensor(x) -> torch.Tensor:
    if isinstance(x, dict):
        x = x.get("poses_3d", x.get("pose"))
    if x is None:
        raise ValueError("pose missing")
    x = torch.as_tensor(x).float()
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    if x.ndim == 2:
        return x
    raise ValueError(f"bad pose shape {tuple(x.shape)}")


def load_gt(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, row in data.items():
        try:
            pose = as_pose_tensor(row)
        except Exception:
            continue
        if pose.shape[1] == 534 and pose.shape[0] >= 4:
            out[str(sid)] = pose.contiguous()
    return out


def load_source(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, pose in data.items():
        try:
            pose = as_pose_tensor(pose)
        except Exception:
            continue
        if pose.shape[1] == 534 and torch.isfinite(pose).all():
            out[str(sid)] = pose.contiguous()
    return out


class HandAdapterDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        gt: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        meta: dict[str, dict],
        gloss_to_id: dict[str, int],
        mean: torch.Tensor,
        std: torch.Tensor,
        T_pose: int,
        max_gloss_len: int,
        max_items: int = 0,
        seed: int = 0,
    ):
        ids = [sid for sid in ids if sid in gt and sid in source]
        if max_items and len(ids) > max_items:
            ids = random.Random(seed).sample(ids, max_items)
        self.ids = ids
        self.gt = gt
        self.source = source
        self.meta = meta
        self.gloss_to_id = gloss_to_id
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)
        self.T_pose = int(T_pose)
        self.max_gloss_len = int(max_gloss_len)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.ids[idx]
        src_len = int(self.source[sid].shape[0])
        raw_len = int(self.gt[sid].shape[0])
        source_raw = resize_seq(self.source[sid], self.T_pose)
        target_raw = resize_seq(self.gt[sid], self.T_pose)
        source = (source_raw - self.mean) / self.std
        target = (target_raw - self.mean) / self.std
        glosses = [g for g in str(self.meta.get(sid, {}).get("gloss", "")).split() if g]
        ids = [self.gloss_to_id.get(g, UNK) for g in glosses][: self.max_gloss_len] or [UNK]
        return {
            "id": sid,
            "source": source.float(),
            "target": target.float(),
            "source_raw": source_raw.float(),
            "target_raw": target_raw.float(),
            "gloss_ids": torch.tensor(ids, dtype=torch.long),
            "raw_len": torch.tensor(src_len or raw_len, dtype=torch.long),
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
        "gloss_ids": gloss_ids,
        "gloss_mask": gloss_mask,
        "raw_len": torch.stack([x["raw_len"] for x in batch]),
        "ids": [x["id"] for x in batch],
    }


class HandDetailAdapter(nn.Module):
    def __init__(
        self,
        num_gloss: int,
        pose_dim: int = 534,
        hidden: int = 192,
        layers: int = 4,
        gloss_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden = hidden
        self.src_proj = nn.Linear(pose_dim, hidden)
        self.progress_mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
        self.gloss_emb = nn.Embedding(num_gloss, hidden, padding_idx=PAD)
        gloss_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.gloss_encoder = nn.TransformerEncoder(gloss_layer, num_layers=gloss_layers)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, HAND_DIM))

    def forward(self, source: torch.Tensor, gloss_ids: torch.Tensor, gloss_mask: torch.Tensor, len_id: torch.Tensor) -> torch.Tensor:
        B, T, _ = source.shape
        device = source.device
        G = gloss_ids.shape[1]
        gh = self.gloss_emb(gloss_ids)
        gh = gh + sinusoidal_positions(G, self.hidden, device)[None] + self.len_emb(len_id)[:, None]
        gh = self.gloss_encoder(gh, src_key_padding_mask=~gloss_mask)
        denom = gloss_mask.sum(1, keepdim=True).clamp_min(1).to(gh.dtype)
        gpool = (gh * gloss_mask[:, :, None].to(gh.dtype)).sum(1) / denom
        p = torch.linspace(0.0, 1.0, T, device=device)
        prog = torch.stack([p, 1.0 - p, torch.sin(math.pi * p), torch.cos(math.pi * p)], dim=-1)
        h = (
            self.src_proj(source)
            + sinusoidal_positions(T, self.hidden, device)[None]
            + self.progress_mlp(prog)[None]
            + gpool[:, None]
            + self.len_emb(len_id)[:, None]
        )
        return self.out(self.blocks(h))


def apply_hand_delta(source_norm: torch.Tensor, hand_delta_norm: torch.Tensor) -> torch.Tensor:
    out = source_norm.clone()
    B, T, _ = out.shape
    hand = out.reshape(B, T, 178, 3)[:, :, HAND_SLICE].reshape(B, T, HAND_DIM)
    hand = hand + hand_delta_norm
    out.reshape(B, T, 178, 3)[:, :, HAND_SLICE] = hand.reshape(B, T, 42, 3)
    return out


def hand_raw(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], x.shape[1], 178, 3)[:, :, HAND_SLICE]


def hand_vel_loss(pred_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(hand_raw(pred_raw)[:, 1:] - hand_raw(pred_raw)[:, :-1], hand_raw(target_raw)[:, 1:] - hand_raw(target_raw)[:, :-1])


def hand_accel_loss(pred_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    ph = hand_raw(pred_raw)
    th = hand_raw(target_raw)
    return F.smooth_l1_loss(ph[:, 2:] - 2 * ph[:, 1:-1] + ph[:, :-2], th[:, 2:] - 2 * th[:, 1:-1] + th[:, :-2])


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
    gt = load_gt(ROOT / args.train_pt)
    source = load_source(ROOT / args.train_source_pt)
    manifest = load_rows(ROOT / args.train_manifest)
    meta = {str(r["id"]): r for r in manifest}
    gloss_to_id = build_gloss_vocab(manifest)
    ids = [sid for sid in gt if sid in source]
    random.Random(args.seed).shuffle(ids)
    n_val = max(args.min_val, int(args.val_frac * len(ids)))
    val_ids, train_ids = ids[:n_val], ids[n_val:]
    if args.smoke:
        train_ids = train_ids[: args.smoke_train]
        val_ids = val_ids[: args.smoke_val]
    train_ds = HandAdapterDataset(train_ids, gt, source, meta, gloss_to_id, mean, std, args.T_pose, args.max_gloss_len, args.max_train_items, args.seed)
    val_ds = HandAdapterDataset(val_ids, gt, source, meta, gloss_to_id, mean, std, args.T_pose, args.max_gloss_len, args.max_val_items, args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=args.drop_last)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    model = HandDetailAdapter(len(gloss_to_id), hidden=args.hidden, layers=args.layers, gloss_layers=args.gloss_layers, heads=args.heads, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[hand-adapter] device={device} train={len(train_ds)} val={len(val_ds)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    mean_d = mean.to(device)
    std_d = std.to(device)

    def run_epoch(loader, train_mode: bool) -> dict:
        model.train(train_mode)
        vals = {k: [] for k in ["loss", "pose", "vel", "accel", "lowpass", "resid"]}
        ratios = []
        for step, batch in enumerate(loader, start=1):
            source_n = batch["source"].to(device)
            target_n = batch["target"].to(device)
            source_raw = batch["source_raw"].to(device)
            target_raw = batch["target_raw"].to(device)
            gloss_ids = batch["gloss_ids"].to(device)
            gloss_mask = batch["gloss_mask"].to(device)
            len_id = torch.as_tensor([length_bucket_id(int(x)) for x in batch["raw_len"].tolist()], dtype=torch.long, device=device)
            with torch.set_grad_enabled(train_mode):
                delta = model(source_n, gloss_ids, gloss_mask, len_id)
                pred_n = apply_hand_delta(source_n, delta)
                pred_raw = pred_n * std_d + mean_d
                pose = F.smooth_l1_loss(hand_raw(pred_n), hand_raw(target_n))
                vel = hand_vel_loss(pred_raw, target_raw)
                acc = hand_accel_loss(pred_raw, target_raw)
                low_p = temporal_smooth(hand_raw(pred_raw).reshape(pred_raw.shape[0], pred_raw.shape[1], -1), args.lowpass_kernel)
                low_s = temporal_smooth(hand_raw(source_raw).reshape(source_raw.shape[0], source_raw.shape[1], -1), args.lowpass_kernel)
                lowpass = F.smooth_l1_loss(low_p, low_s)
                resid = delta.pow(2).mean()
                loss = (
                    args.lambda_pose * pose
                    + args.lambda_vel * vel
                    + args.lambda_accel * acc
                    + args.lambda_lowpass * lowpass
                    + args.lambda_resid * resid
                )
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    g = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
            for k, v in [("loss", loss), ("pose", pose), ("vel", vel), ("accel", acc), ("lowpass", lowpass), ("resid", resid)]:
                vals[k].append(float(v.detach().cpu()))
            if (not train_mode) and args.eval_motion_batches and len(ratios) < args.eval_motion_batches:
                ratios.append(motion_metrics(pred_raw.detach().reshape(pred_raw.shape[0], pred_raw.shape[1], 178, 3), target_raw.reshape(target_raw.shape[0], target_raw.shape[1], 178, 3)))
            if train_mode and args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({"step": step, "loss": vals["loss"][-1], "pose": vals["pose"][-1], "vel": vals["vel"][-1], "lowpass": vals["lowpass"][-1], "grad_norm": float(g)}), flush=True)
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
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best": best,
                "mean": mean,
                "std": std,
                "gloss_to_id": gloss_to_id,
                "hidden": args.hidden,
                "T_pose": args.T_pose,
            }, out_dir / "best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[hand-adapter] saved best -> {out_dir / 'best.pt'}", flush=True)


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = HandDetailAdapter(
        len(ckpt["gloss_to_id"]),
        hidden=ckpt["hidden"],
        layers=ckpt["args"]["layers"],
        gloss_layers=ckpt["args"]["gloss_layers"],
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
    source_pt = load_source(ROOT / args.source_pt)
    if args.reference_pt:
        ref = torch.load(ROOT / args.reference_pt, map_location="cpu", weights_only=False)
        row_by_id = {str(r["id"]): r for r in rows}
        rows = [row_by_id[str(sid)] for sid in ref.keys() if str(sid) in row_by_id and str(sid) in source_pt]
    if args.max_clips:
        rows = rows[: args.max_clips]
    out = {}
    bs = max(1, int(args.batch_size))
    for start in range(0, len(rows), bs):
        batch_rows = rows[start : start + bs]
        kept, sources, lens, gloss_seqs = [], [], [], []
        for row in batch_rows:
            sid = str(row["id"])
            if sid not in source_pt:
                continue
            src_raw = source_pt[sid]
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
        source_n = torch.stack(sources, 0)
        len_id = torch.tensor([length_bucket_id(L) for L in lens], dtype=torch.long, device=device)
        delta = model(source_n, gloss_ids, gloss_mask, len_id) * args.hand_gain
        pred_n = apply_hand_delta(source_n, delta)
        pred = (pred_n * std + mean).cpu()
        for j, (sid, L, _src) in enumerate(kept):
            pose = resize_seq(pred[j], L).reshape(L, 178, 3).float()
            if args.post_smooth_blend > 0 and args.post_smooth_kernel > 1:
                pose = (1 - args.post_smooth_blend) * pose + args.post_smooth_blend * temporal_smooth(pose, args.post_smooth_kernel)
            out[sid] = pose.contiguous()
        done = min(start + len(batch_rows), len(rows))
        if args.log_every and (done == len(batch_rows) or done % args.log_every < bs or done == len(rows)):
            print(f"[hand-adapter-sample] {done}/{len(rows)}", flush=True)
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
    tr.add_argument("--train_source_pt", default="external/SLRTP-Sign-Production-Evaluation/results/sign_jepa_train_scale080.pt")
    tr.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_hand_detail_adapter")
    tr.add_argument("--T_pose", type=int, default=160)
    tr.add_argument("--max_gloss_len", type=int, default=64)
    tr.add_argument("--hidden", type=int, default=192)
    tr.add_argument("--layers", type=int, default=4)
    tr.add_argument("--gloss_layers", type=int, default=1)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--eval_batch_size", type=int, default=96)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--lambda_pose", type=float, default=1.0)
    tr.add_argument("--lambda_vel", type=float, default=3.0)
    tr.add_argument("--lambda_accel", type=float, default=2.0)
    tr.add_argument("--lambda_lowpass", type=float, default=1.0)
    tr.add_argument("--lambda_resid", type=float, default=1e-4)
    tr.add_argument("--lowpass_kernel", type=int, default=9)
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
    sm.add_argument("--hand_gain", type=float, default=1.0)
    sm.add_argument("--post_smooth_kernel", type=int, default=1)
    sm.add_argument("--post_smooth_blend", type=float, default=0.0)
    sm.add_argument("--batch_size", type=int, default=64)
    sm.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sm.add_argument("--log_every", type=int, default=100)
    sm.add_argument("--max_clips", type=int, default=0)

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample_manifest(args)


if __name__ == "__main__":
    main()
