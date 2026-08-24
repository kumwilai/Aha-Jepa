"""Full-phrase Sign-JEPA generator for native SLRTP-178.

This is the WER-focused branch. The first Sign-JEPA generator produces each
gloss span independently and concatenates them. This script trains a single
sentence-level model:

    full gloss sequence + duration -> full pose sequence

It freezes the Sign-JEPA encoder and supervises both predictive motion latents
and native SLRTP-178 pose. No exemplar/frame retrieval is used at inference.
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
    SignJEPAModel,
    length_bucket_id,
    resize_seq,
    sequence_features,
    sinusoidal_positions,
)
from scripts.train_sign_jepa_generator_slrtp178 import (  # noqa: E402
    speed_envelope_loss,
    std_match_loss,
    velocity_loss,
)

PAD = 0
UNK = 1


def load_frozen_jepa(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
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
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def load_rows(path: Path) -> list[dict]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    rows = []
    for sid, row in data.items():
        rows.append({
            "id": str(sid),
            "text": str(row.get("text", "")),
            "gloss": str(row.get("gloss", "")),
            "pose": row["poses_3d"].float(),
        })
    return rows


class PhraseDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        gloss_to_id: dict[str, int],
        mean: torch.Tensor,
        std: torch.Tensor,
        T_pose: int,
        max_gloss_len: int,
        max_items: int = 0,
        seed: int = 0,
    ):
        rows = list(rows)
        if max_items and len(rows) > max_items:
            rows = random.Random(seed).sample(rows, max_items)
        self.rows = rows
        self.gloss_to_id = gloss_to_id
        self.mean = mean
        self.std = std
        self.T_pose = int(T_pose)
        self.max_gloss_len = int(max_gloss_len)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        pose_raw = resize_seq(row["pose"].reshape(row["pose"].shape[0], -1), self.T_pose)
        pose = (pose_raw - self.mean) / self.std
        glosses = [g for g in row["gloss"].split() if g]
        ids = [self.gloss_to_id.get(g, UNK) for g in glosses][: self.max_gloss_len]
        if not ids:
            ids = [UNK]
        return {
            "id": row["id"],
            "pose_raw": pose_raw,
            "pose": pose,
            "gloss_ids": torch.tensor(ids, dtype=torch.long),
            "raw_len": torch.tensor(int(row["pose"].shape[0]), dtype=torch.long),
        }


def collate(batch: list[dict]):
    B = len(batch)
    G = max(len(x["gloss_ids"]) for x in batch)
    gloss_ids = torch.full((B, G), PAD, dtype=torch.long)
    gloss_mask = torch.zeros(B, G, dtype=torch.bool)
    for i, item in enumerate(batch):
        g = item["gloss_ids"]
        gloss_ids[i, : len(g)] = g
        gloss_mask[i, : len(g)] = True
    return {
        "pose_raw": torch.stack([x["pose_raw"] for x in batch]),
        "pose": torch.stack([x["pose"] for x in batch]),
        "gloss_ids": gloss_ids,
        "gloss_mask": gloss_mask,
        "raw_len": torch.stack([x["raw_len"] for x in batch]),
        "ids": [x["id"] for x in batch],
    }


class PhraseJEPAGenerator(nn.Module):
    def __init__(
        self,
        num_gloss: int,
        hidden: int,
        gloss_layers: int,
        dec_layers: int,
        heads: int,
        dropout: float,
        pose_dim: int = 534,
    ):
        super().__init__()
        self.hidden = hidden
        self.gloss_emb = nn.Embedding(num_gloss, hidden, padding_idx=PAD)
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
        enc_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.gloss_encoder = nn.TransformerEncoder(enc_layer, num_layers=gloss_layers)
        dec_layer = nn.TransformerDecoderLayer(
            hidden, heads, hidden * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.z_norm = nn.LayerNorm(hidden)
        self.pose_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, pose_dim),
        )

    def encode_gloss(self, gloss_ids: torch.Tensor, gloss_mask: torch.Tensor, len_id: torch.Tensor):
        B, G = gloss_ids.shape
        h = self.gloss_emb(gloss_ids)
        h = h + sinusoidal_positions(G, self.hidden, gloss_ids.device)[None]
        h = h + self.len_emb(len_id)[:, None]
        h = self.gloss_encoder(h, src_key_padding_mask=~gloss_mask)
        return h

    def forward(self, gloss_ids: torch.Tensor, gloss_mask: torch.Tensor, len_id: torch.Tensor, T: int):
        B = gloss_ids.shape[0]
        device = gloss_ids.device
        memory = self.encode_gloss(gloss_ids, gloss_mask, len_id)
        pos = sinusoidal_positions(T, self.hidden, device)[None].expand(B, -1, -1)
        p = torch.linspace(0.0, 1.0, T, device=device)
        prog = torch.stack([p, 1.0 - p, torch.sin(math.pi * p), torch.cos(math.pi * p)], dim=-1)
        q = self.query_mlp(torch.cat([pos, prog[None].expand(B, -1, -1)], dim=-1))
        z = self.decoder(q, memory, memory_key_padding_mask=~gloss_mask)
        z = self.z_norm(z)
        pose = self.pose_head(z)
        return z, pose


@torch.no_grad()
def target_latents(jepa: SignJEPAModel, pose_norm, pose_raw):
    B, T, _ = pose_norm.shape
    feat, desc = sequence_features(pose_norm, pose_raw)
    zero_mask = torch.zeros(B, T, dtype=torch.bool, device=pose_norm.device)
    gloss_id = torch.zeros(B, dtype=torch.long, device=pose_norm.device)
    len_id = torch.full((B,), length_bucket_id(T), dtype=torch.long, device=pose_norm.device)
    z = jepa.target_encoder(feat, zero_mask, gloss_id, len_id)
    return z.detach(), desc.detach()


def run_epoch(model, jepa, loader, opt, mean, std, device, args, train: bool):
    model.train(train)
    vals = []
    cos_vals = []
    pose_vals = []
    std_vals = []
    speed_vals = []
    for step, batch in enumerate(loader, start=1):
        pose = batch["pose"].to(device)
        pose_raw = batch["pose_raw"].to(device)
        gloss_ids = batch["gloss_ids"].to(device)
        gloss_mask = batch["gloss_mask"].to(device)
        len_id = torch.as_tensor(
            [length_bucket_id(int(x)) for x in batch["raw_len"].tolist()],
            dtype=torch.long, device=device,
        )
        with torch.set_grad_enabled(train):
            z_tgt, desc_tgt = target_latents(jepa, pose, pose_raw)
            z_pred, pose_pred = model(gloss_ids, gloss_mask, len_id, pose.shape[1])
            latent = F.smooth_l1_loss(F.normalize(z_pred, dim=-1), F.normalize(z_tgt, dim=-1))
            pose_loss = F.smooth_l1_loss(pose_pred, pose)
            vel = velocity_loss(pose_pred, pose)
            pose_pred_raw = pose_pred * std.to(device) + mean.to(device)
            _feat_pred, desc_pred = sequence_features(pose_pred, pose_pred_raw)
            desc = F.smooth_l1_loss(desc_pred, desc_tgt)
            std_loss = std_match_loss(pose_pred, pose)
            speed_loss = speed_envelope_loss(pose_pred, pose)
            loss = (
                args.lambda_latent * latent
                + args.lambda_pose * pose_loss
                + args.lambda_vel * vel
                + args.lambda_desc * desc
                + args.lambda_std * std_loss
                + args.lambda_speed * speed_loss
            )
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
        vals.append(float(loss.detach().cpu()))
        cos_vals.append(float((F.normalize(z_pred, dim=-1) * F.normalize(z_tgt, dim=-1)).sum(dim=-1).mean().detach().cpu()))
        pose_vals.append(float(pose_loss.detach().cpu()))
        std_vals.append(float(std_loss.detach().cpu()))
        speed_vals.append(float(speed_loss.detach().cpu()))
        if train and args.log_every and (step == 1 or step % args.log_every == 0):
            print(json.dumps({
                "step": step,
                "loss": vals[-1],
                "cos": cos_vals[-1],
                "pose": pose_vals[-1],
                "std": std_vals[-1],
                "speed": speed_vals[-1],
                "pose_pred_std": float(pose_pred.detach().std().cpu()),
                "pose_tgt_std": float(pose.detach().std().cpu()),
            }), flush=True)
        if train and args.max_steps and step >= args.max_steps:
            break
    prefix = "train" if train else "val"
    return {
        f"{prefix}_loss": float(np.mean(vals)),
        f"{prefix}_cos": float(np.mean(cos_vals)),
        f"{prefix}_pose": float(np.mean(pose_vals)),
        f"{prefix}_std": float(np.mean(std_vals)),
        f"{prefix}_speed": float(np.mean(speed_vals)),
    }


def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jepa, ckpt = load_frozen_jepa(ROOT / args.jepa_ckpt, device)
    mean = ckpt["mean"].float()
    std = ckpt["std"].float()
    gloss_to_id = ckpt["gloss_to_id"]
    rows = load_rows(ROOT / args.train_pt)
    random.Random(args.seed).shuffle(rows)
    n_val = max(args.min_val, int(args.val_frac * len(rows)))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    if args.smoke:
        train_rows = train_rows[:args.smoke_train]
        val_rows = val_rows[:args.smoke_val]
    train_ds = PhraseDataset(train_rows, gloss_to_id, mean, std, args.T_pose, args.max_gloss_len, args.max_train_items, args.seed)
    val_ds = PhraseDataset(val_rows, gloss_to_id, mean, std, args.T_pose, args.max_gloss_len, args.max_val_items, args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    model = PhraseJEPAGenerator(
        len(gloss_to_id), int(ckpt["args"]["hidden"]), args.gloss_layers,
        args.dec_layers, args.heads, args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[phrase-jepa] device={device} train={len(train_ds)} val={len(val_ds)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    best = None
    log = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, jepa, train_loader, opt, mean, std, device, args, train=True)
        va = run_epoch(model, jepa, val_loader, opt, mean, std, device, args, train=False)
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
                "jepa_ckpt": str(args.jepa_ckpt),
                "gloss_to_id": gloss_to_id,
                "mean": mean,
                "std": std,
                "hidden": int(ckpt["args"]["hidden"]),
            }, out_dir / "best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[phrase-jepa] saved best -> {out_dir / 'best.pt'}", flush=True)


@torch.no_grad()
def sample_manifest(args: argparse.Namespace):
    device = torch.device(args.device)
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    gloss_to_id = ckpt["gloss_to_id"]
    model = PhraseJEPAGenerator(
        len(gloss_to_id), ckpt["hidden"], ckpt["args"]["gloss_layers"],
        ckpt["args"]["dec_layers"], ckpt["args"]["heads"], 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    rows = json.load(open(ROOT / args.manifest_json))
    if args.reference_pt:
        ref = torch.load(ROOT / args.reference_pt, map_location="cpu", weights_only=True)
        row_by_id = {str(r["id"]): r for r in rows}
        rows = [row_by_id[sid] for sid in ref.keys()]
    if args.max_clips:
        rows = rows[:args.max_clips]
    out = {}
    for i, row in enumerate(rows, start=1):
        glosses = [g for g in str(row.get("gloss", "")).split() if g]
        ids = [gloss_to_id.get(g, UNK) for g in glosses][: ckpt["args"]["max_gloss_len"]]
        if not ids:
            ids = [UNK]
        gloss_ids = torch.tensor(ids, dtype=torch.long, device=device)[None]
        gloss_mask = torch.ones_like(gloss_ids, dtype=torch.bool)
        L = int(row.get("length", 0) or args.T_fallback)
        len_id = torch.tensor([length_bucket_id(L)], dtype=torch.long, device=device)
        _z, pose = model(gloss_ids, gloss_mask, len_id, ckpt["args"]["T_pose"])
        pose = pose[0] * std + mean
        pose = resize_seq(pose.cpu(), L).reshape(L, 178, 3).float()
        out[str(row["id"])] = pose.contiguous()
        if args.log_every and (i == 1 or i % args.log_every == 0):
            print(f"[phrase-jepa-sample] {i}/{len(rows)}", flush=True)
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    lens = np.asarray([v.shape[0] for v in out.values()])
    print(f"saved -> {out_path}")
    print(f"T mean={lens.mean():.1f} p10/50/90={np.percentile(lens, [10,50,90]).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--jepa_ckpt", default="outputs/sota_chase/sign_jepa_slrtp178/best.pt")
    tr.add_argument("--train_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_phrase_generator_slrtp178")
    tr.add_argument("--T_pose", type=int, default=160)
    tr.add_argument("--max_gloss_len", type=int, default=64)
    tr.add_argument("--gloss_layers", type=int, default=3)
    tr.add_argument("--dec_layers", type=int, default=4)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--batch_size", type=int, default=48)
    tr.add_argument("--eval_batch_size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--lambda_latent", type=float, default=1.0)
    tr.add_argument("--lambda_pose", type=float, default=0.2)
    tr.add_argument("--lambda_vel", type=float, default=0.2)
    tr.add_argument("--lambda_desc", type=float, default=0.3)
    tr.add_argument("--lambda_std", type=float, default=0.1)
    tr.add_argument("--lambda_speed", type=float, default=0.2)
    tr.add_argument("--val_frac", type=float, default=0.05)
    tr.add_argument("--min_val", type=int, default=256)
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=100)
    tr.add_argument("--max_steps", type=int, default=0)
    tr.add_argument("--max_train_items", type=int, default=0)
    tr.add_argument("--max_val_items", type=int, default=1024)
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--smoke_train", type=int, default=128)
    tr.add_argument("--smoke_val", type=int, default=64)

    sp = sub.add_parser("sample_manifest")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--manifest_json", required=True)
    sp.add_argument("--out_pt", required=True)
    sp.add_argument("--reference_pt", default="")
    sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sp.add_argument("--T_fallback", type=int, default=80)
    sp.add_argument("--log_every", type=int, default=100)
    sp.add_argument("--max_clips", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample_manifest(args)


if __name__ == "__main__":
    main()
