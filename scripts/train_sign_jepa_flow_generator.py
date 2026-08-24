"""Conditional flow-matching generator over the whole-clip AE latent.

Pipeline:
  gloss program -> sampled rectified flow in reconstruction latent -> frozen AE decoder

The target latent comes from scripts/train_sign_jepa_ae_fullclip.py. This script
does not use the predictive JEPA latent and does not regress latents with L1/L2.
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

from scripts.train_sign_jepa_ae_fullclip import (  # noqa: E402
    FullClipPoseDataset,
    WholeClipPoseAE,
    hand_jerk,
    load_pose_rows,
    region_speed,
)
from scripts.train_sign_jepa_generator_slrtp178 import allocate_lengths  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import LENGTH_BUCKETS, length_bucket_id, resize_seq, sequence_features, sinusoidal_positions  # noqa: E402

PAD = 0
UNK = 1


def load_manifest_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_gloss_vocab(rows: list[dict]) -> dict[str, int]:
    glosses = sorted({g for r in rows for g in str(r.get("gloss", "")).split() if g})
    return {"<pad>": PAD, "<unk>": UNK, **{g: i + 2 for i, g in enumerate(glosses)}}


def mean_lengths_by_gloss(rows: list[dict]) -> dict[str, float]:
    vals: dict[str, list[float]] = {}
    for r in rows:
        glosses = [g for g in str(r.get("gloss", "")).split() if g]
        if not glosses:
            continue
        per = max(1.0, float(r.get("length", 0) or 0) / len(glosses))
        for g in glosses:
            vals.setdefault(g, []).append(per)
    return {g: float(np.mean(v)) for g, v in vals.items()}


def temporal_smooth(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Depthwise moving average along time for [B,T,C] or [T,J,C] tensors."""
    if kernel <= 1:
        return x
    if kernel % 2 == 0:
        raise ValueError("--prior_smooth/--post_smooth_kernel must be odd")
    original_shape = x.shape
    if x.ndim == 3 and original_shape[-2:] == (178, 3):
        y = x.reshape(x.shape[0], -1).T[None]
        squeeze = "pose"
    elif x.ndim == 3:
        y = x.transpose(1, 2)
        squeeze = "batch"
    else:
        raise ValueError(f"Unsupported temporal_smooth shape: {tuple(x.shape)}")
    pad = kernel // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=kernel, stride=1)
    if squeeze == "pose":
        return y.squeeze(0).T.reshape(original_shape)
    return y.transpose(1, 2)


def sample_prior(shape: tuple[int, int, int], device: torch.device, smooth_kernel: int) -> torch.Tensor:
    z = torch.randn(shape, device=device)
    z = temporal_smooth(z, smooth_kernel)
    dims = tuple(range(1, z.ndim))
    z = z - z.mean(dim=dims, keepdim=True)
    z = z / z.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return z


class FlowClipDataset(Dataset):
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
        self.pose_ds = FullClipPoseDataset(rows, mean, std, T_pose)
        self.max_gloss_len = int(max_gloss_len)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        item = self.pose_ds[idx]
        row = self.rows[idx]
        glosses = [g for g in str(row.get("gloss", "")).split() if g]
        ids = [self.gloss_to_id.get(g, UNK) for g in glosses][: self.max_gloss_len]
        if not ids:
            ids = [UNK]
        item["gloss_ids"] = torch.tensor(ids, dtype=torch.long)
        item["gloss"] = str(row.get("gloss", ""))
        return item


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
        "pose_raw": torch.stack([x["pose_raw"] for x in batch]),
        "pose": torch.stack([x["pose"] for x in batch]),
        "raw_len": torch.stack([x["raw_len"] for x in batch]),
        "gloss_ids": gloss_ids,
        "gloss_mask": gloss_mask,
        "ids": [x["id"] for x in batch],
    }


class ConditionalLatentFlow(nn.Module):
    def __init__(
        self,
        num_gloss: int,
        hidden: int = 256,
        gloss_layers: int = 3,
        flow_layers: int = 6,
        heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.gloss_emb = nn.Embedding(num_gloss, hidden, padding_idx=PAD)
        self.len_emb = nn.Embedding(len(LENGTH_BUCKETS), hidden)
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden))
        gloss_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.gloss_encoder = nn.TransformerEncoder(gloss_layer, num_layers=gloss_layers)
        self.t_mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.progress_mlp = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.x_proj = nn.Linear(hidden, hidden)
        flow_layer = nn.TransformerDecoderLayer(
            hidden, heads, hidden * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.flow = nn.TransformerDecoder(flow_layer, num_layers=flow_layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def encode_gloss(self, gloss_ids: torch.Tensor, gloss_mask: torch.Tensor, len_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, G = gloss_ids.shape
        h = self.gloss_emb(gloss_ids)
        h = h + sinusoidal_positions(G, self.hidden, gloss_ids.device)[None]
        h = h + self.len_emb(len_id)[:, None]
        null = self.null_token.expand(B, -1, -1) + self.len_emb(len_id)[:, None]
        mem = torch.cat([null, h], dim=1)
        mem_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=gloss_ids.device), gloss_mask], dim=1)
        mem = self.gloss_encoder(mem, src_key_padding_mask=~mem_mask)
        return mem, mem_mask

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        gloss_ids: torch.Tensor,
        gloss_mask: torch.Tensor,
        len_id: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x_t.shape
        device = x_t.device
        mem, mem_mask = self.encode_gloss(gloss_ids, gloss_mask, len_id)
        pos = sinusoidal_positions(T, self.hidden, device)[None]
        p = torch.linspace(0.0, 1.0, T, device=device)
        prog = torch.stack([p, 1.0 - p, torch.sin(math.pi * p), torch.cos(math.pi * p)], dim=-1)
        tt = torch.stack([t, 1.0 - t, torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)
        q = self.x_proj(x_t) + pos + self.progress_mlp(prog)[None] + self.t_mlp(tt)[:, None]
        h = self.flow(q, mem, memory_key_padding_mask=~mem_mask)
        return self.out(h)


def load_ae(path: Path, device: torch.device) -> tuple[WholeClipPoseAE, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WholeClipPoseAE(
        feat_dim=ckpt["feat_dim"],
        hidden=ckpt["hidden"],
        enc_layers=ckpt["args"]["enc_layers"],
        dec_layers=ckpt["args"]["dec_layers"],
        heads=ckpt["args"]["heads"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


@torch.no_grad()
def encode_ae_latent(ae: WholeClipPoseAE, pose_norm: torch.Tensor, pose_raw: torch.Tensor) -> torch.Tensor:
    feat, _ = sequence_features(pose_norm, pose_raw)
    return ae.encoder(feat).detach()


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ae, ae_ckpt = load_ae(ROOT / args.ae_ckpt, device)
    train_pose_rows = load_pose_rows(ROOT / args.train_pt)
    train_manifest = load_manifest_rows(ROOT / args.train_manifest)
    gloss_to_id = build_gloss_vocab(train_manifest)
    manifest_by_id = {str(r["id"]): r for r in train_manifest}
    rows = []
    for row in train_pose_rows:
        meta = manifest_by_id.get(row["id"], row)
        row = dict(row)
        row["gloss"] = str(meta.get("gloss", row.get("gloss", "")))
        row["length"] = int(meta.get("length", row["pose"].shape[0]) or row["pose"].shape[0])
        rows.append(row)
    random.Random(args.seed).shuffle(rows)
    n_val = max(args.min_val, int(args.val_frac * len(rows)))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    if args.smoke:
        train_rows = train_rows[: args.smoke_train]
        val_rows = val_rows[: args.smoke_val]

    mean = ae_ckpt["mean"].float()
    std = ae_ckpt["std"].float()
    T_pose = int(ae_ckpt["T_pose"])
    train_ds = FlowClipDataset(train_rows, gloss_to_id, mean, std, T_pose, args.max_gloss_len, args.max_train_items, args.seed)
    val_ds = FlowClipDataset(val_rows, gloss_to_id, mean, std, T_pose, args.max_gloss_len, args.max_val_items, args.seed + 1)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    model = ConditionalLatentFlow(
        num_gloss=len(gloss_to_id),
        hidden=ae_ckpt["hidden"],
        gloss_layers=args.gloss_layers,
        flow_layers=args.flow_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gloss_mean_lengths = mean_lengths_by_gloss(train_manifest)
    print(
        f"[jepa-flow] device={device} train={len(train_ds)} val={len(val_ds)} "
        f"T_pose={T_pose} hidden={ae_ckpt['hidden']} glosses={len(gloss_to_id)} "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
        flush=True,
    )

    def run_epoch(loader, train_mode: bool) -> dict:
        model.train(train_mode)
        vals, grad_norms, zstd, vstd = [], [], [], []
        for step, batch in enumerate(loader, start=1):
            pose = batch["pose"].to(device)
            pose_raw = batch["pose_raw"].to(device)
            gloss_ids = batch["gloss_ids"].to(device)
            gloss_mask = batch["gloss_mask"].to(device)
            len_id = torch.as_tensor([length_bucket_id(int(x)) for x in batch["raw_len"].tolist()], dtype=torch.long, device=device)
            if train_mode and args.cond_drop > 0.0:
                drop = torch.rand(gloss_mask.shape[0], device=device) < args.cond_drop
                gloss_mask = gloss_mask.clone()
                gloss_mask[drop] = False
            with torch.set_grad_enabled(train_mode):
                z1 = encode_ae_latent(ae, pose, pose_raw)
                z0 = sample_prior(tuple(z1.shape), device, args.prior_smooth)
                t = torch.rand(z1.shape[0], device=device)
                x_t = (1.0 - t[:, None, None]) * z0 + t[:, None, None] * z1
                target = z1 - z0
                pred = model(x_t, t, gloss_ids, gloss_mask, len_id)
                loss = F.mse_loss(pred, target)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    g = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                    grad_norms.append(float(g))
            vals.append(float(loss.detach().cpu()))
            zstd.append(float(z1.detach().std().cpu()))
            vstd.append(float(pred.detach().std().cpu()))
            if train_mode and args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({
                    "step": step,
                    "fm_loss": vals[-1],
                    "z_tgt_std": zstd[-1],
                    "v_pred_std": vstd[-1],
                    "grad_norm": grad_norms[-1] if grad_norms else 0.0,
                }), flush=True)
            if train_mode and args.max_steps and step >= args.max_steps:
                break
            if (not train_mode) and args.val_batches and step >= args.val_batches:
                break
        prefix = "train" if train_mode else "val"
        rec = {
            f"{prefix}_fm_loss": float(np.mean(vals)),
            f"{prefix}_z_tgt_std": float(np.mean(zstd)),
            f"{prefix}_v_pred_std": float(np.mean(vstd)),
        }
        if train_mode:
            rec[f"{prefix}_grad_norm"] = float(np.mean(grad_norms)) if grad_norms else 0.0
        return rec

    best = None
    log = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, True)
        with torch.no_grad():
            va = run_epoch(val_loader, False)
        rec = {"epoch": ep, "wall_s": round(time.time() - t0, 1), **tr, **va}
        log.append(rec)
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(rec), flush=True)
        if best is None or rec["val_fm_loss"] < best["val_fm_loss"]:
            best = dict(rec)
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best": best,
                "ae_ckpt": str(args.ae_ckpt),
                "gloss_to_id": gloss_to_id,
                "mean": mean,
                "std": std,
                "hidden": ae_ckpt["hidden"],
                "T_pose": T_pose,
                "gloss_mean_lengths": gloss_mean_lengths,
            }, out_dir / "best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[jepa-flow] saved best -> {out_dir / 'best.pt'}", flush=True)
    print(f"[jepa-flow] done best={best}", flush=True)


def load_flow(path: Path, device: torch.device) -> tuple[ConditionalLatentFlow, dict, WholeClipPoseAE, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ae, ae_ckpt = load_ae(ROOT / ckpt["ae_ckpt"], device)
    model = ConditionalLatentFlow(
        num_gloss=len(ckpt["gloss_to_id"]),
        hidden=ckpt["hidden"],
        gloss_layers=ckpt["args"]["gloss_layers"],
        flow_layers=ckpt["args"]["flow_layers"],
        heads=ckpt["args"]["heads"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, ae, ae_ckpt


@torch.no_grad()
def sample_latent(
    model: ConditionalLatentFlow,
    gloss_ids: torch.Tensor,
    gloss_mask: torch.Tensor,
    len_id: torch.Tensor,
    T: int,
    steps: int,
    guidance: float,
    prior_smooth: int,
) -> torch.Tensor:
    x = sample_prior((gloss_ids.shape[0], T, model.hidden), gloss_ids.device, prior_smooth)
    dt = 1.0 / max(1, steps)
    uncond_mask = torch.zeros_like(gloss_mask)
    for i in range(steps):
        t = torch.full((x.shape[0],), i / max(1, steps), device=x.device)
        v = model(x, t, gloss_ids, gloss_mask, len_id)
        if guidance != 1.0:
            vu = model(x, t, gloss_ids, uncond_mask, len_id)
            v = vu + guidance * (v - vu)
        x = x + dt * v
    return x


@torch.no_grad()
def sample_manifest(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, ckpt, ae, _ae_ckpt = load_flow(ROOT / args.ckpt, device)
    gloss_to_id = ckpt["gloss_to_id"]
    rows = load_manifest_rows(ROOT / args.manifest_json)
    if args.reference_pt:
        ref = torch.load(ROOT / args.reference_pt, map_location="cpu", weights_only=True)
        row_by_id = {str(r["id"]): r for r in rows}
        missing = [sid for sid in ref.keys() if str(sid) not in row_by_id]
        if missing:
            raise KeyError(f"{len(missing)} reference ids missing from manifest, first={missing[:3]}")
        rows = [row_by_id[str(sid)] for sid in ref.keys()]
    if args.max_clips:
        rows = rows[: args.max_clips]
    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    out = {}
    for i, row in enumerate(rows, start=1):
        glosses = [g for g in str(row.get("gloss", "")).split() if g]
        if not glosses:
            glosses = ["<unk>"]
        ids = [gloss_to_id.get(g, UNK) for g in glosses][: ckpt["args"]["max_gloss_len"]]
        if not ids:
            ids = [UNK]
        L = int(row.get("length", 0) or args.T_fallback)
        gloss_ids = torch.tensor(ids, dtype=torch.long, device=device)[None]
        gloss_mask = torch.ones_like(gloss_ids, dtype=torch.bool)
        len_id = torch.tensor([length_bucket_id(L)], dtype=torch.long, device=device)
        z = sample_latent(
            model,
            gloss_ids,
            gloss_mask,
            len_id,
            ckpt["T_pose"],
            args.steps,
            args.guidance,
            int(ckpt["args"].get("prior_smooth", 1)),
        )
        pose = ae.decoder(z)[0] * std + mean
        pose = resize_seq(pose.cpu(), L).reshape(L, 178, 3).float()
        if args.post_smooth_kernel > 1 and args.post_smooth_blend > 0.0:
            smooth = temporal_smooth(pose, args.post_smooth_kernel)
            blend = float(np.clip(args.post_smooth_blend, 0.0, 1.0))
            pose = (1.0 - blend) * pose + blend * smooth
        out[str(row["id"])] = pose.contiguous()
        if args.log_every and (i == 1 or i % args.log_every == 0):
            print(f"[jepa-flow-sample] {i}/{len(rows)}", flush=True)
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    lens = np.asarray([v.shape[0] for v in out.values()])
    print(f"saved -> {out_path}")
    print(f"clips={len(out)} T mean={lens.mean():.1f} p10/50/90={np.percentile(lens, [10,50,90]).tolist()}")


@torch.no_grad()
def motion_eval(args: argparse.Namespace) -> None:
    pred = torch.load(ROOT / args.pred_pt, map_location="cpu", weights_only=True)
    ref = torch.load(ROOT / args.reference_pt, map_location="cpu", weights_only=False)
    ids = [sid for sid in ref.keys() if sid in pred]
    if args.max_clips:
        ids = ids[: args.max_clips]
    acc = {f"{r}_{k}": [] for r in ("hand", "body", "face") for k in ("pred", "real")}
    jerk_p, jerk_r, std_p, std_r = [], [], [], []
    for sid in ids:
        p = pred[sid].float()
        rrow = ref[sid]
        r = (rrow["poses_3d"] if isinstance(rrow, dict) else rrow).float()
        if p.shape[0] != r.shape[0]:
            p = resize_seq(p.reshape(p.shape[0], -1), r.shape[0]).reshape(r.shape[0], 178, 3)
        pb = p[None]
        rb = r[None]
        sp, sr = region_speed(pb), region_speed(rb)
        for region in ("hand", "body", "face"):
            acc[f"{region}_pred"].append(sp[region])
            acc[f"{region}_real"].append(sr[region])
        jerk_p.append(hand_jerk(pb))
        jerk_r.append(hand_jerk(rb))
        std_p.append(float(pb[:, :, 8:50].std()))
        std_r.append(float(rb[:, :, 8:50].std()))
    ratio = lambda a, b: float(np.mean(a) / max(float(np.mean(b)), 1e-9))
    metrics = {
        "clips": len(ids),
        "hand_speed_ratio": ratio(acc["hand_pred"], acc["hand_real"]),
        "body_speed_ratio": ratio(acc["body_pred"], acc["body_real"]),
        "face_speed_ratio": ratio(acc["face_pred"], acc["face_real"]),
        "hand_jerk_ratio": ratio(jerk_p, jerk_r),
        "hand_posestd_ratio": ratio(std_p, std_r),
    }
    metrics = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}
    print(json.dumps(metrics, indent=2), flush=True)
    if args.out_json:
        out = ROOT / args.out_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))


@torch.no_grad()
def postprocess_pt(args: argparse.Namespace) -> None:
    pred = torch.load(ROOT / args.in_pt, map_location="cpu", weights_only=True)
    out = {}
    blend = float(np.clip(args.blend, 0.0, 1.0))
    for sid, pose in pred.items():
        pose = pose.float()
        if args.smooth_kernel > 1 and blend > 0.0:
            pose = (1.0 - blend) * pose + blend * temporal_smooth(pose, args.smooth_kernel)
        out[str(sid)] = pose.contiguous()
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--ae_ckpt", default="outputs/sota_chase/sign_jepa_ae_fullclip/best.pt")
    tr.add_argument("--train_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    tr.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_flow_generator")
    tr.add_argument("--max_gloss_len", type=int, default=64)
    tr.add_argument("--gloss_layers", type=int, default=3)
    tr.add_argument("--flow_layers", type=int, default=6)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch_size", type=int, default=32)
    tr.add_argument("--eval_batch_size", type=int, default=48)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--cond_drop", type=float, default=0.1)
    tr.add_argument(
        "--prior_smooth",
        type=int,
        default=1,
        help="Odd moving-average kernel for the Gaussian source prior. "
             "Use >1 to train a temporally correlated latent prior.",
    )
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

    sm = sub.add_parser("sample_manifest")
    sm.add_argument("--ckpt", required=True)
    sm.add_argument("--manifest_json", required=True)
    sm.add_argument("--out_pt", required=True)
    sm.add_argument("--reference_pt", default="")
    sm.add_argument("--steps", type=int, default=50)
    sm.add_argument("--guidance", type=float, default=1.0)
    sm.add_argument("--post_smooth_kernel", type=int, default=1)
    sm.add_argument("--post_smooth_blend", type=float, default=0.0)
    sm.add_argument("--T_fallback", type=int, default=80)
    sm.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sm.add_argument("--log_every", type=int, default=100)
    sm.add_argument("--max_clips", type=int, default=0)

    me = sub.add_parser("motion_eval")
    me.add_argument("--pred_pt", required=True)
    me.add_argument("--reference_pt", required=True)
    me.add_argument("--out_json", default="")
    me.add_argument("--max_clips", type=int, default=0)

    pp = sub.add_parser("postprocess_pt")
    pp.add_argument("--in_pt", required=True)
    pp.add_argument("--out_pt", required=True)
    pp.add_argument("--smooth_kernel", type=int, default=3)
    pp.add_argument("--blend", type=float, default=0.76)

    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "sample_manifest":
        sample_manifest(args)
    elif args.mode == "motion_eval":
        motion_eval(args)
    else:
        postprocess_pt(args)


if __name__ == "__main__":
    main()
