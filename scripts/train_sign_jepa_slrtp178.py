"""Sign-JEPA pretraining on native SLRTP-178 Phoenix spans.

This is a retrieval-free representation prototype for sign production. It
learns to predict masked future motion latents from visible context, gloss, and
duration. The target is not raw coordinate reconstruction alone: the predictor
also has to recover local motion descriptors that carry stroke timing,
hand-speed envelopes, wrist movement, and hand bone directions.

The intended downstream use is a text/gloss-to-latent rollout generator. This
script only trains the JEPA dynamics representation and writes a checkpoint.
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

LENGTH_BUCKETS = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128]

BLOCKS = {
    "body": (0, 8),
    "rh": (8, 29),
    "lh": (29, 50),
    "face": (50, 178),
}

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def flat_pose(x) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    return x


def resize_seq(x: torch.Tensor, T: int) -> torch.Tensor:
    x = flat_pose(x)
    if x.shape[0] == T:
        return x
    return F.interpolate(
        x.T.unsqueeze(0), size=int(T), mode="linear", align_corners=False
    ).squeeze(0).T.contiguous()


def length_bucket_id(length: int) -> int:
    for i, b in enumerate(LENGTH_BUCKETS):
        if length <= b:
            return i
    return len(LENGTH_BUCKETS) - 1


def sinusoidal_positions(T: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    emb = torch.cat([torch.sin(pos[:, None] * freqs), torch.cos(pos[:, None] * freqs)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def load_bank(path: Path):
    bank = torch.load(path, map_location="cpu", weights_only=False)
    spans = []
    for gloss, rows in bank["gloss_to_exemplars"].items():
        for sid, s, e in rows:
            if sid not in bank["exemplar_poses"]:
                continue
            T = int(bank["exemplar_poses"][sid].shape[0])
            s = max(0, min(int(s), T))
            e = max(0, min(int(e), T))
            if e - s >= 3:
                spans.append((str(gloss), sid, s, e))
    return bank, spans


def compute_stats(pose_pool: dict) -> tuple[torch.Tensor, torch.Tensor]:
    s = None
    ss = None
    n = 0
    for arr in pose_pool.values():
        x = flat_pose(torch.as_tensor(arr)).double()
        if s is None:
            s = x.sum(0)
            ss = (x * x).sum(0)
        else:
            s += x.sum(0)
            ss += (x * x).sum(0)
        n += x.shape[0]
    mean = (s / max(n, 1)).float()
    var = (ss / max(n, 1)).float() - mean * mean
    return mean, var.clamp_min(1e-6).sqrt()


def build_gloss_vocab(bank: dict) -> dict[str, int]:
    glosses = sorted(str(g) for g in bank["gloss_to_exemplars"].keys())
    return {"<null>": 0, "<unk>": 1, **{g: i + 2 for i, g in enumerate(glosses)}}


class SLRTPSpanDataset(Dataset):
    def __init__(
        self,
        bank: dict,
        spans: list[tuple[str, str, int, int]],
        gloss_to_id: dict[str, int],
        mean: torch.Tensor,
        std: torch.Tensor,
        seg_len: int,
        max_items: int = 0,
        seed: int = 0,
        row_by_sid: dict | None = None,
        sent_max_len: int = 32,
    ):
        self.pose_pool = bank["exemplar_poses"]
        spans = list(spans)
        if max_items and len(spans) > max_items:
            spans = random.Random(seed).sample(spans, max_items)
        self.spans = spans
        self.gloss_to_id = gloss_to_id
        self.mean = mean
        self.std = std
        self.seg_len = seg_len
        self.row_by_sid = row_by_sid or {}
        self.sent_max_len = sent_max_len

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, idx):
        gloss, sid, s, e = self.spans[idx]
        raw_len = max(1, int(e) - int(s))
        pose_raw = flat_pose(torch.as_tensor(self.pose_pool[sid][s:e], dtype=torch.float32))
        pose_raw = resize_seq(pose_raw, self.seg_len)
        pose_norm = (pose_raw - self.mean) / self.std
        row = self.row_by_sid.get(str(sid), {})
        sent_glosses = [str(g) for g in str(row.get("gloss", "")).split() if g]
        if not sent_glosses:
            sent_glosses = [gloss]
        pos = 0
        for j, g in enumerate(sent_glosses):
            if g == gloss:
                pos = j
                break
        sent_ids = torch.zeros(self.sent_max_len, dtype=torch.long)
        sent_mask = torch.zeros(self.sent_max_len, dtype=torch.bool)
        for j, g in enumerate(sent_glosses[: self.sent_max_len]):
            sent_ids[j] = int(self.gloss_to_id.get(g, 1))
            sent_mask[j] = True
        return {
            "pose_raw": pose_raw,
            "pose": pose_norm,
            "gloss_id": torch.tensor(self.gloss_to_id.get(gloss, 1), dtype=torch.long),
            "len_id": torch.tensor(length_bucket_id(raw_len), dtype=torch.long),
            "sent_gloss_ids": sent_ids,
            "sent_mask": sent_mask,
            "sent_pos": torch.tensor(min(pos, self.sent_max_len - 1), dtype=torch.long),
        }


def sequence_features(pose_norm: torch.Tensor, pose_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frame features and compact motion descriptors.

    Features feed the JEPA encoders. Descriptors are predicted from future
    latents to make the representation attend to recognizer-relevant dynamics.
    """
    B, T, D = pose_norm.shape
    vel = torch.zeros_like(pose_norm)
    vel[:, 1:] = pose_norm[:, 1:] - pose_norm[:, :-1]
    acc = torch.zeros_like(pose_norm)
    acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

    raw = pose_raw.reshape(B, T, 178, 3)
    vel_raw = torch.zeros_like(raw)
    vel_raw[:, 1:] = raw[:, 1:] - raw[:, :-1]
    acc_raw = torch.zeros_like(raw)
    acc_raw[:, 1:] = vel_raw[:, 1:] - vel_raw[:, :-1]

    speeds = []
    accels = []
    for lo, hi in BLOCKS.values():
        speeds.append(vel_raw[:, :, lo:hi].norm(dim=-1).mean(dim=-1, keepdim=True))
        accels.append(acc_raw[:, :, lo:hi].norm(dim=-1).mean(dim=-1, keepdim=True))
    speed_desc = torch.cat(speeds, dim=-1)
    accel_desc = torch.cat(accels, dim=-1)

    hand_dirs = []
    edges = torch.as_tensor(HAND_EDGES, device=pose_raw.device, dtype=torch.long)
    for lo in (8, 29):
        hand = raw[:, :, lo:lo + 21]
        bones = hand[:, :, edges[:, 1]] - hand[:, :, edges[:, 0]]
        dirs = F.normalize(bones, dim=-1, eps=1e-6).reshape(B, T, -1)
        hand_dirs.append(dirs)
    hand_dirs = torch.cat(hand_dirs, dim=-1)

    wrists = raw[:, :, [8, 29]].reshape(B, T, 6)
    wrist_vel = vel_raw[:, :, [8, 29]].reshape(B, T, 6)

    motion_desc = torch.cat([speed_desc, accel_desc, hand_dirs, wrists, wrist_vel], dim=-1)
    enc_feat = torch.cat([pose_norm, vel, acc, motion_desc], dim=-1)
    return enc_feat, motion_desc


def sample_future_mask(
    B: int,
    T: int,
    min_context: int,
    target_len_min: int,
    target_len_max: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        hi_len = max(target_len_min, min(target_len_max, T - min_context))
        L = random.randint(target_len_min, hi_len)
        start = random.randint(min_context, max(min_context, T - L))
        mask[b, start:start + L] = True
    return mask


class SignJEPAEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_gloss: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.max_len = max_len
        self.in_proj = nn.Linear(feat_dim + 1, hidden)
        self.gloss_emb = nn.Embedding(num_gloss, hidden)
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
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

    def forward(
        self,
        feat: torch.Tensor,
        mask_flag: torch.Tensor,
        gloss_id: torch.Tensor,
        len_id: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = feat.shape
        x = torch.cat([feat, mask_flag.float().unsqueeze(-1)], dim=-1)
        h = self.in_proj(x)
        h = h + sinusoidal_positions(T, self.hidden, feat.device)[None]
        h = h + self.gloss_emb(gloss_id)[:, None] + self.len_emb(len_id)[:, None]
        return self.norm(self.blocks(h))


class SignJEPAModel(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        desc_dim: int,
        num_gloss: int,
        hidden: int,
        layers: int,
        heads: int,
        pred_layers: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.context_encoder = SignJEPAEncoder(
            feat_dim, num_gloss, hidden, layers, heads, dropout, max_len
        )
        self.target_encoder = SignJEPAEncoder(
            feat_dim, num_gloss, hidden, layers, heads, 0.0, max_len
        )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        pred_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(pred_layer, num_layers=pred_layers)
        self.pred_norm = nn.LayerNorm(hidden)
        self.desc_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, desc_dim),
        )

    @torch.no_grad()
    def update_target(self, ema: float) -> None:
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.data.mul_(ema).add_(pc.data, alpha=1.0 - ema)

    def forward(
        self,
        feat: torch.Tensor,
        gloss_id: torch.Tensor,
        len_id: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_feat = feat.masked_fill(future_mask.unsqueeze(-1), 0.0)
        context_z = self.context_encoder(context_feat, future_mask, gloss_id, len_id)
        pred_z = self.pred_norm(self.predictor(context_z))
        with torch.no_grad():
            zero_mask = torch.zeros_like(future_mask)
            target_z = self.target_encoder(feat, zero_mask, gloss_id, len_id)
        pred_desc = self.desc_head(pred_z)
        return pred_z, target_z.detach(), pred_desc


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    z = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()


def carrier_losses(model, batch, device, args, recon_head):
    """Objective-specific loss for a carrier pretraining step.

    objective='jepa'       : masked latent prediction + EMA target (default).
    objective='mae'        : masked raw-feature reconstruction (detail-preserving).
    objective='ae'         : full (unmasked) raw-feature reconstruction.

    For 'mae'/'ae' the predictor latent is decoded back to feature space by
    ``recon_head``; the EMA momentum is forced to 0 so the frozen target encoder
    used downstream is exactly the reconstruction-trained context encoder.
    """
    pose = batch["pose"].to(device)
    pose_raw = batch["pose_raw"].to(device)
    gloss_id = batch["gloss_id"].to(device)
    len_id = batch["len_id"].to(device)
    feat, desc = sequence_features(pose, pose_raw)
    obj = args.objective
    if obj == "ae":
        future_mask = torch.zeros(pose.shape[0], pose.shape[1], dtype=torch.bool, device=device)
    else:
        future_mask = sample_future_mask(
            pose.shape[0], pose.shape[1], args.min_context,
            args.target_len_min, args.target_len_max, device,
        )
    pred_z, target_z, pred_desc = model(feat, gloss_id, len_id, future_mask)

    if obj == "jepa":
        m = future_mask.unsqueeze(-1)
        pred_n = F.normalize(pred_z, dim=-1)
        target_n = F.normalize(target_z, dim=-1)
        latent = F.smooth_l1_loss(pred_n[m.expand_as(pred_n)], target_n[m.expand_as(target_n)])
        desc_loss = F.smooth_l1_loss(pred_desc[m.expand_as(pred_desc)], desc[m.expand_as(desc)])
        var = variance_loss(pred_z[future_mask]) + variance_loss(target_z[future_mask])
        loss = latent + args.lambda_desc * desc_loss + args.lambda_var * var
        cos = (pred_n[future_mask] * target_n[future_mask]).sum(dim=-1).mean()
        stats = {"latent": float(latent.detach().cpu()), "desc": float(desc_loss.detach().cpu()),
                 "var": float(var.detach().cpu()), "cos": float(cos.detach().cpu())}
        return loss, pred_z, target_z, stats

    # reconstruction objectives (mae / ae)
    recon = recon_head(pred_z)
    if obj == "mae":
        m = future_mask.unsqueeze(-1)
        recon_loss = F.smooth_l1_loss(recon[m.expand_as(recon)], feat[m.expand_as(feat)])
        var = variance_loss(pred_z[future_mask])
    else:  # ae
        recon_loss = F.smooth_l1_loss(recon, feat)
        var = variance_loss(pred_z)
    loss = recon_loss + args.lambda_var * var
    stats = {"recon": float(recon_loss.detach().cpu()), "var": float(var.detach().cpu())}
    return loss, pred_z, target_z, stats


def train_epoch(model, loader, opt, device, args, recon_head=None) -> dict:
    model.train()
    ema = args.ema if args.objective == "jepa" else 0.0
    vals = []
    stat_acc: dict[str, list] = {}
    for step, batch in enumerate(loader, start=1):
        loss, pred_z, target_z, stats = carrier_losses(model, batch, device, args, recon_head)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        params = list(model.parameters()) + (list(recon_head.parameters()) if recon_head is not None else [])
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        model.update_target(ema)
        vals.append(float(loss.detach().cpu()))
        for k, v in stats.items():
            stat_acc.setdefault(k, []).append(v)
        if args.log_every and (step == 1 or step % args.log_every == 0):
            print(
                json.dumps({
                    "step": step, "loss": vals[-1], **{k: stats[k] for k in stats},
                    "grad_norm": float(gnorm),
                    "pred_std": float(pred_z.detach().std().cpu()),
                    "target_std": float(target_z.detach().std().cpu()),
                }),
                flush=True,
            )
        if args.max_steps and step >= args.max_steps:
            break
    out = {"train_loss": float(np.mean(vals))}
    for k, v in stat_acc.items():
        out[f"train_{k}"] = float(np.mean(v))
    return out


@torch.no_grad()
def eval_epoch(model, loader, device, args, recon_head=None) -> dict:
    model.eval()
    vals = []
    stat_acc: dict[str, list] = {}
    for bi, batch in enumerate(loader):
        if args.val_batches and bi >= args.val_batches:
            break
        loss, _pred_z, _target_z, stats = carrier_losses(model, batch, device, args, recon_head)
        vals.append(float(loss.cpu()))
        for k, v in stats.items():
            stat_acc.setdefault(k, []).append(v)
    out = {"val_loss": float(np.mean(vals))}
    for k, v in stat_acc.items():
        out[f"val_{k}"] = float(np.mean(v))
    return out


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[sign-jepa] loading bank", flush=True)
    bank, spans = load_bank(ROOT / args.bank)
    gloss_to_id = build_gloss_vocab(bank)
    print(f"[sign-jepa] spans={len(spans)} glosses={len(gloss_to_id)}", flush=True)
    mean, std = compute_stats(bank["exemplar_poses"])
    random.Random(args.seed).shuffle(spans)
    n_val = max(args.min_val, int(args.val_frac * len(spans)))
    val_spans = spans[:n_val]
    train_spans = spans[n_val:]
    if args.smoke:
        train_spans = train_spans[:args.smoke_train]
        val_spans = val_spans[:args.smoke_val]

    train_ds = SLRTPSpanDataset(
        bank, train_spans, gloss_to_id, mean, std, args.seg_len,
        max_items=args.max_train_items, seed=args.seed,
    )
    val_ds = SLRTPSpanDataset(
        bank, val_spans, gloss_to_id, mean, std, args.seg_len,
        max_items=args.max_val_items, seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    sample = train_ds[0]
    feat, desc = sequence_features(sample["pose"][None], sample["pose_raw"][None])
    model = SignJEPAModel(
        feat_dim=feat.shape[-1],
        desc_dim=desc.shape[-1],
        num_gloss=len(gloss_to_id),
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        pred_layers=args.pred_layers,
        dropout=args.dropout,
        max_len=args.seg_len,
    ).to(device)
    # Reconstruction head for the mae/ae carrier-swap objectives (maps the
    # predictor latent back to feature space). None for jepa.
    recon_head = None
    param_groups = list(model.parameters())
    if args.objective in ("mae", "ae"):
        recon_head = nn.Linear(args.hidden, feat.shape[-1]).to(device)
        param_groups = param_groups + list(recon_head.parameters())
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"[sign-jepa] objective={args.objective} device={device} train={len(train_ds)} val={len(val_ds)} "
        f"feat_dim={feat.shape[-1]} desc_dim={desc.shape[-1]} "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
        flush=True,
    )

    log = []
    best = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_rec = train_epoch(model, train_loader, opt, device, args, recon_head)
        val_rec = eval_epoch(model, val_loader, device, args, recon_head)
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1), **train_rec, **val_rec}
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)
        if best is None or rec["val_loss"] < best["val_loss"]:
            best = dict(rec)
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "mean": mean,
                    "std": std,
                    "gloss_to_id": gloss_to_id,
                    "length_buckets": LENGTH_BUCKETS,
                    "feat_dim": feat.shape[-1],
                    "desc_dim": desc.shape[-1],
                    "best": best,
                },
                out_dir / "best.pt",
            )
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[sign-jepa] saved best -> {out_dir / 'best.pt'}", flush=True)
    print(f"[sign-jepa] done best={best}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt")
    ap.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_slrtp178")
    ap.add_argument("--objective", choices=["jepa", "mae", "ae"], default="jepa",
                    help="carrier pretraining objective for the carrier-swap ablation: "
                         "jepa=masked latent prediction+EMA (default), mae=masked raw "
                         "reconstruction, ae=full reconstruction")
    ap.add_argument("--seg_len", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--pred_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--lambda_desc", type=float, default=0.5)
    ap.add_argument("--lambda_var", type=float, default=0.02)
    ap.add_argument("--min_context", type=int, default=16)
    ap.add_argument("--target_len_min", type=int, default=8)
    ap.add_argument("--target_len_max", type=int, default=24)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--min_val", type=int, default=512)
    ap.add_argument("--val_batches", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--max_train_items", type=int, default=0)
    ap.add_argument("--max_val_items", type=int, default=2048)
    ap.add_argument("--drop_last", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_train", type=int, default=256)
    ap.add_argument("--smoke_val", type=int, default=96)
    args = ap.parse_args()
    if args.min_context + args.target_len_min > args.seg_len:
        raise ValueError("--min_context + --target_len_min must be <= --seg_len")
    train(args)


if __name__ == "__main__":
    main()
