"""Source-locked Sign-JEPA hand-detail fusion.

This is the lightweight composition that worked empirically:

  output = old_jepa carrier
  output[hands] += hand_gain * (detail_refiner_base - old_jepa)[hands]

Body and face remain locked to the JEPA carrier, so the refiner can improve
hand motion without rewriting the semantic trajectory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_sign_jepa_ae_fullclip import hand_jerk, region_speed  # noqa: E402
from scripts.train_sign_jepa_flow_generator import temporal_smooth  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402


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
    out = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def fuse(args: argparse.Namespace) -> None:
    source = load_pose_map(ROOT / args.source_pt)
    detail = load_pose_map(ROOT / args.detail_pt)
    out = {}
    for sid, base in detail.items():
        if sid not in source:
            continue
        src = source[sid].float()
        det = base.float()
        if src.shape[0] != det.shape[0]:
            src = resize_seq(src.reshape(src.shape[0], -1), det.shape[0]).reshape(det.shape[0], 178, 3)
        pose = src.clone()
        residual = det - src
        if args.highpass_kernel > 1:
            residual = residual - temporal_smooth(residual, args.highpass_kernel)
        pose[:, 8:50] += args.hand_gain * residual[:, 8:50]
        if args.post_smooth_kernel > 1 and args.post_smooth_blend > 0:
            pose = (1 - args.post_smooth_blend) * pose + args.post_smooth_blend * temporal_smooth(pose, args.post_smooth_kernel)
        out[sid] = pose.contiguous()
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")


def motion_eval(args: argparse.Namespace) -> None:
    pred = load_pose_map(ROOT / args.pred_pt)
    ref_raw = torch.load(ROOT / args.reference_pt, map_location="cpu", weights_only=False)
    ids = [str(sid) for sid in ref_raw.keys() if str(sid) in pred]
    if args.max_clips:
        ids = ids[: args.max_clips]
    acc = {f"{r}_{k}": [] for r in ("hand", "body", "face") for k in ("pred", "real")}
    jerk_p, jerk_r, std_p, std_r = [], [], [], []
    for sid in ids:
        p = pred[sid].float()
        r = as_pose(ref_raw[sid]).float()
        if p.shape[0] != r.shape[0]:
            p = resize_seq(p.reshape(p.shape[0], -1), r.shape[0]).reshape(r.shape[0], 178, 3)
        pb, rb = p[None], r[None]
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
    print(json.dumps(metrics, indent=2), flush=True)
    if args.out_json:
        out_path = ROOT / args.out_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    fu = sub.add_parser("fuse")
    fu.add_argument("--source_pt", required=True)
    fu.add_argument("--detail_pt", required=True)
    fu.add_argument("--out_pt", required=True)
    fu.add_argument("--hand_gain", type=float, default=4.0)
    fu.add_argument("--highpass_kernel", type=int, default=1)
    fu.add_argument("--post_smooth_kernel", type=int, default=3)
    fu.add_argument("--post_smooth_blend", type=float, default=0.05)

    ev = sub.add_parser("motion_eval")
    ev.add_argument("--pred_pt", required=True)
    ev.add_argument("--reference_pt", required=True)
    ev.add_argument("--out_json", default="")
    ev.add_argument("--max_clips", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "fuse":
        fuse(args)
    else:
        motion_eval(args)


if __name__ == "__main__":
    main()
