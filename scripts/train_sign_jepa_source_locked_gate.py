"""Learned source-locked hand gate for Sign-JEPA detail residuals.

The gate is deliberately constrained:

  output = old_jepa carrier
  output[hands] += gamma(t, joint) * (detail_refiner - old_jepa)[hands]

Body and face are hard-locked to the carrier.  The model only learns a bounded
hand-joint multiplier, initialized near the successful fixed-gain fusion.
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
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402

HAND_SLICE = slice(8, 50)
HAND_JOINTS = 42
HAND_DIM = HAND_JOINTS * 3


def as_pose(x) -> torch.Tensor:
    if isinstance(x, dict):
        x = x.get("poses_3d", x.get("pose"))
    if x is None:
        raise ValueError("pose missing")
    x = torch.as_tensor(x).float()
    if x.ndim == 2:
        x = x.reshape(x.shape[0], 178, 3)
    if x.ndim != 3 or x.shape[1:] != (178, 3):
        raise ValueError(f"bad pose shape {tuple(x.shape)}")
    return x


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, pose in data.items():
        try:
            p = as_pose(pose).contiguous()
        except Exception:
            continue
        if p.shape[0] >= 4 and torch.isfinite(p).all():
            out[str(sid)] = p
    return out


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def hand(x: torch.Tensor) -> torch.Tensor:
    return x[:, HAND_SLICE].reshape(x.shape[0], HAND_DIM)


def velocity(x: torch.Tensor) -> torch.Tensor:
    return x[1:] - x[:-1]


def make_features(source: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    src_h = hand(source)
    res_h = hand(detail - source)
    src_v = torch.zeros_like(src_h)
    res_v = torch.zeros_like(res_h)
    src_v[1:] = src_h[1:] - src_h[:-1]
    res_v[1:] = res_h[1:] - res_h[:-1]
    speed = res_v.reshape(res_v.shape[0], HAND_JOINTS, 3).norm(dim=-1)
    progress = torch.linspace(0, 1, source.shape[0], dtype=source.dtype)[:, None]
    prog = torch.cat([progress, 1 - progress, torch.sin(math.pi * progress), torch.cos(math.pi * progress)], dim=-1)
    return torch.cat([src_h, res_h, src_v, res_v, speed, prog], dim=-1)


class GateDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        source: dict[str, torch.Tensor],
        detail: dict[str, torch.Tensor],
        gt: dict[str, torch.Tensor] | None,
        T_pose: int,
        max_items: int = 0,
        seed: int = 0,
    ):
        ids = [str(sid) for sid in ids if str(sid) in source and str(sid) in detail]
        if gt is not None:
            ids = [sid for sid in ids if sid in gt]
        if max_items and len(ids) > max_items:
            ids = random.Random(seed).sample(ids, max_items)
        self.ids = ids
        self.source = source
        self.detail = detail
        self.gt = gt
        self.T_pose = int(T_pose)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.ids[idx]
        src_orig = self.source[sid]
        raw_len = int(src_orig.shape[0])
        source = resize_seq(src_orig.reshape(raw_len, -1), self.T_pose).reshape(self.T_pose, 178, 3)
        detail = self.detail[sid]
        detail = resize_seq(detail.reshape(detail.shape[0], -1), self.T_pose).reshape(self.T_pose, 178, 3)
        item = {
            "id": sid,
            "source": source.float(),
            "detail": detail.float(),
            "features": make_features(source, detail).float(),
            "raw_len": torch.tensor(raw_len, dtype=torch.long),
        }
        if self.gt is not None:
            gt = self.gt[sid]
            item["gt"] = resize_seq(gt.reshape(gt.shape[0], -1), self.T_pose).reshape(self.T_pose, 178, 3).float()
        return item


def collate(batch: list[dict]) -> dict:
    out = {
        "source": torch.stack([x["source"] for x in batch]),
        "detail": torch.stack([x["detail"] for x in batch]),
        "features": torch.stack([x["features"] for x in batch]),
        "raw_len": torch.stack([x["raw_len"] for x in batch]),
        "ids": [x["id"] for x in batch],
    }
    if "gt" in batch[0]:
        out["gt"] = torch.stack([x["gt"] for x in batch])
    return out


class SourceLockedGate(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, layers: int = 4, max_gain: float = 5.5, init_gain: float = 4.0):
        super().__init__()
        self.max_gain = float(max_gain)
        p = min(max(float(init_gain) / max(self.max_gain, 1e-6), 1e-4), 1 - 1e-4)
        self.register_buffer("init_bias", torch.tensor(math.log(p / (1 - p)), dtype=torch.float32))
        blocks: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)]
        for _ in range(max(0, int(layers) - 1)):
            blocks += [nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)]
        self.net = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden, HAND_JOINTS)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.out(self.net(features)) + self.init_bias
        return self.max_gain * torch.sigmoid(logits)


def apply_gate(source: torch.Tensor, detail: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    out = source.clone()
    residual = detail - source
    out[:, :, HAND_SLICE] = source[:, :, HAND_SLICE] + gamma[..., None] * residual[:, :, HAND_SLICE]
    return out


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    source = load_pose_map(ROOT / args.source_pt)
    detail = load_pose_map(ROOT / args.detail_pt)
    gt = load_pose_map(ROOT / args.gt_pt)
    ids = sorted(set(source) & set(detail) & set(gt))
    random.Random(args.seed).shuffle(ids)
    n_val = max(args.min_val, int(round(len(ids) * args.val_frac))) if len(ids) > args.min_val else max(1, len(ids) // 10)
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    train_ds = GateDataset(train_ids, source, detail, gt, args.T_pose, args.max_train_items, args.seed)
    val_ds = GateDataset(val_ids, source, detail, gt, args.T_pose, args.max_val_items, args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    input_dim = train_ds[0]["features"].shape[-1]
    model = SourceLockedGate(input_dim, args.hidden, args.layers, args.max_gain, args.init_gain).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[source-locked-gate] device={device} train={len(train_ds)} val={len(val_ds)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict:
        model.train(train_mode)
        vals = {k: [] for k in ("loss", "gt", "teacher", "vel", "accel", "gate_reg", "gate_smooth")}
        ratios = []
        for step, batch in enumerate(loader, start=1):
            features = batch["features"].to(device)
            source_b = batch["source"].to(device)
            detail_b = batch["detail"].to(device)
            gt_b = batch["gt"].to(device)
            with torch.set_grad_enabled(train_mode):
                gamma = model(features)
                pred = apply_gate(source_b, detail_b, gamma)
                teacher = apply_gate(source_b, detail_b, torch.full_like(gamma, args.teacher_gain))
                pred_h = pred[:, :, HAND_SLICE]
                gt_h = gt_b[:, :, HAND_SLICE]
                gt_loss = F.smooth_l1_loss(pred_h, gt_h)
                teacher_loss = F.smooth_l1_loss(pred_h, teacher[:, :, HAND_SLICE])
                vel_loss = F.smooth_l1_loss(velocity(pred_h), velocity(gt_h))
                acc_loss = accel_loss(pred_h.reshape(pred_h.shape[0], pred_h.shape[1], -1), gt_h.reshape(gt_h.shape[0], gt_h.shape[1], -1))
                gate_reg = (gamma - args.init_gain).abs().mean()
                gate_smooth = (gamma[:, 1:] - gamma[:, :-1]).abs().mean()
                loss = (
                    args.lambda_gt * gt_loss
                    + args.lambda_teacher * teacher_loss
                    + args.lambda_vel * vel_loss
                    + args.lambda_accel * acc_loss
                    + args.lambda_gate_reg * gate_reg
                    + args.lambda_gate_smooth * gate_smooth
                )
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
            vals["loss"].append(float(loss.detach().cpu()))
            vals["gt"].append(float(gt_loss.detach().cpu()))
            vals["teacher"].append(float(teacher_loss.detach().cpu()))
            vals["vel"].append(float(vel_loss.detach().cpu()))
            vals["accel"].append(float(acc_loss.detach().cpu()))
            vals["gate_reg"].append(float(gate_reg.detach().cpu()))
            vals["gate_smooth"].append(float(gate_smooth.detach().cpu()))
            if (not train_mode) and args.eval_motion_batches and len(ratios) < args.eval_motion_batches:
                ratios.append(motion_metrics(pred.detach(), gt_b))
            if train_mode and args.log_every and (step == 1 or step % args.log_every == 0):
                print(json.dumps({"step": step, "loss": vals["loss"][-1], "gt": vals["gt"][-1], "teacher": vals["teacher"][-1]}), flush=True)
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
        if best is None or rec[args.select_metric] < best[args.select_metric]:
            best = dict(rec)
            torch.save({"model": model.state_dict(), "args": vars(args), "best": best, "input_dim": input_dim}, out_dir / "best.pt")
            (out_dir / "best.json").write_text(json.dumps(best, indent=2))
            print(f"[source-locked-gate] saved best -> {out_dir / 'best.pt'}", flush=True)


@torch.no_grad()
def motion_metrics(pred_batch: torch.Tensor, ref_batch: torch.Tensor) -> dict:
    sp, sr = region_speed(pred_batch), region_speed(ref_batch)
    return {
        "hand_speed_ratio": sp["hand"] / max(sr["hand"], 1e-9),
        "body_speed_ratio": sp["body"] / max(sr["body"], 1e-9),
        "face_speed_ratio": sp["face"] / max(sr["face"], 1e-9),
        "hand_jerk_ratio": hand_jerk(pred_batch) / max(hand_jerk(ref_batch), 1e-9),
        "hand_posestd_ratio": float(pred_batch[:, :, HAND_SLICE].std() / ref_batch[:, :, HAND_SLICE].std().clamp_min(1e-9)),
    }


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SourceLockedGate(
        ckpt["input_dim"],
        hidden=ckpt["args"]["hidden"],
        layers=ckpt["args"]["layers"],
        max_gain=ckpt["args"]["max_gain"],
        init_gain=ckpt["args"]["init_gain"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, ckpt = load_model(ROOT / args.ckpt, device)
    source = load_pose_map(ROOT / args.source_pt)
    detail = load_pose_map(ROOT / args.detail_pt)
    ids = sorted(set(source) & set(detail))
    if args.reference_pt:
        ref = load_pose_map(ROOT / args.reference_pt)
        ids = [sid for sid in ref if sid in source and sid in detail]
    if args.manifest_json:
        row_ids = [str(r["id"]) for r in load_rows(ROOT / args.manifest_json)]
        ids = [sid for sid in row_ids if sid in source and sid in detail and (not args.reference_pt or sid in set(ids))]
    if args.max_clips:
        ids = ids[: args.max_clips]
    out = {}
    gates = {}
    T_pose = int(ckpt["args"]["T_pose"])
    for start in range(0, len(ids), args.batch_size):
        batch_ids = ids[start : start + args.batch_size]
        sources, details, features, raw_lens = [], [], [], []
        for sid in batch_ids:
            src = source[sid]
            det = detail[sid]
            raw_lens.append(int(src.shape[0]))
            src_t = resize_seq(src.reshape(src.shape[0], -1), T_pose).reshape(T_pose, 178, 3)
            det_t = resize_seq(det.reshape(det.shape[0], -1), T_pose).reshape(T_pose, 178, 3)
            sources.append(src_t)
            details.append(det_t)
            features.append(make_features(src_t, det_t))
        source_b = torch.stack(sources).to(device)
        detail_b = torch.stack(details).to(device)
        feature_b = torch.stack(features).to(device)
        gamma = model(feature_b)
        if args.gate_scale != 1.0 or args.gate_bias != 0.0:
            max_gain = float(ckpt["args"]["max_gain"])
            gamma = (gamma * args.gate_scale + args.gate_bias).clamp(0.0, max_gain)
        pred = apply_gate(source_b, detail_b, gamma).cpu()
        gamma_c = gamma.cpu()
        for i, sid in enumerate(batch_ids):
            pose = resize_seq(pred[i].reshape(T_pose, -1), raw_lens[i]).reshape(raw_lens[i], 178, 3)
            if args.post_smooth_kernel > 1 and args.post_smooth_blend > 0:
                pose = (1 - args.post_smooth_blend) * pose + args.post_smooth_blend * temporal_smooth(pose, args.post_smooth_kernel)
            out[sid] = pose.contiguous()
            gates[sid] = gamma_c[i].contiguous()
        done = min(start + len(batch_ids), len(ids))
        if args.log_every and (done == len(batch_ids) or done % args.log_every < len(batch_ids) or done == len(ids)):
            print(f"[source-locked-gate-sample] {done}/{len(ids)}", flush=True)
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")
    if args.gates_pt:
        gates_path = ROOT / args.gates_pt
        gates_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gates, gates_path)
        print(f"saved gates -> {gates_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--source_pt", required=True)
    tr.add_argument("--detail_pt", required=True)
    tr.add_argument("--gt_pt", required=True)
    tr.add_argument("--out_dir", default="outputs/sota_chase/sign_jepa_source_locked_gate")
    tr.add_argument("--T_pose", type=int, default=160)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--layers", type=int, default=4)
    tr.add_argument("--max_gain", type=float, default=5.5)
    tr.add_argument("--init_gain", type=float, default=4.0)
    tr.add_argument("--teacher_gain", type=float, default=4.0)
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--eval_batch_size", type=int, default=96)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--lambda_gt", type=float, default=0.5)
    tr.add_argument("--lambda_teacher", type=float, default=1.0)
    tr.add_argument("--lambda_vel", type=float, default=0.5)
    tr.add_argument("--lambda_accel", type=float, default=0.2)
    tr.add_argument("--lambda_gate_reg", type=float, default=0.02)
    tr.add_argument("--lambda_gate_smooth", type=float, default=0.02)
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
    tr.add_argument("--select_metric", default="val_loss")

    sm = sub.add_parser("sample")
    sm.add_argument("--ckpt", required=True)
    sm.add_argument("--source_pt", required=True)
    sm.add_argument("--detail_pt", required=True)
    sm.add_argument("--out_pt", required=True)
    sm.add_argument("--gates_pt", default="")
    sm.add_argument("--manifest_json", default="")
    sm.add_argument("--reference_pt", default="")
    sm.add_argument("--batch_size", type=int, default=96)
    sm.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sm.add_argument("--post_smooth_kernel", type=int, default=3)
    sm.add_argument("--post_smooth_blend", type=float, default=0.05)
    sm.add_argument("--gate_scale", type=float, default=1.0)
    sm.add_argument("--gate_bias", type=float, default=0.0)
    sm.add_argument("--max_clips", type=int, default=0)
    sm.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        sample(args)


if __name__ == "__main__":
    main()
