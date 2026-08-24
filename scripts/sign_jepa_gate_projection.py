"""Closed-form gate projection for pure-generative Sign-JEPA outputs.

Inputs:
  - JEPA source carrier pose
  - learned two-branch control pose

Output:
  - body/face/wrists from the JEPA carrier
  - learned local hand articulation
  - extra JEPA low-frequency local-hand motion, maximized under an internal
    jerk/speed-ratio constraint

No retrieval, no real-clip splicing, and no train-corpus statistics are used.
The projection uses only the current clip's JEPA carrier and generated pose.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.sign_jepa_source_locked_fusion import as_pose, load_pose_map  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402

BODY = slice(0, 8)
RH = slice(8, 29)
LH = slice(29, 50)
FACE = slice(50, 178)
RWRIST = 2
LWRIST = 5


def smooth_time(x: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return x
    y = x.reshape(x.shape[0], -1).T[None]
    pad = kernel // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=kernel, stride=1)
    return y[0].T.reshape(x.shape)


def hand_local(pose: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        pose[:, RH] - pose[:, RWRIST:RWRIST + 1],
        pose[:, LH] - pose[:, LWRIST:LWRIST + 1],
    ], dim=1)


def attach_local(source: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    out = source.clone()
    out[:, RH] = source[:, RWRIST:RWRIST + 1] + local[:, :21]
    out[:, LH] = source[:, LWRIST:LWRIST + 1] + local[:, 21:]
    return out


def hand_speed(pose: torch.Tensor) -> float:
    v = pose[1:, 8:50] - pose[:-1, 8:50]
    return float(v.norm(dim=-1).mean())


def hand_jerk(pose: torch.Tensor) -> float:
    v = pose[1:, 8:50] - pose[:-1, 8:50]
    a = v[1:] - v[:-1]
    j = a[1:] - a[:-1]
    return float(j.norm(dim=-1).mean())


def project_clip(source: torch.Tensor, learned: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    if source.shape[0] != learned.shape[0]:
        source = resize_seq(source.reshape(source.shape[0], -1), learned.shape[0]).reshape(learned.shape[0], 178, 3)
    source = source.float()
    learned = learned.float()

    src_local = hand_local(source)
    learned_local = hand_local(learned)
    src_dev = src_local - src_local.mean(dim=0, keepdim=True)
    boost = smooth_time(src_dev, args.kernel)
    if args.endpoint_taper:
        T = boost.shape[0]
        t = torch.linspace(0.0, 1.0, T, dtype=boost.dtype, device=boost.device)[:, None, None]
        taper = torch.sin(torch.pi * t).clamp_min(0.0)
        boost = boost * taper

    base = attach_local(source, learned_local)
    base_speed = hand_speed(base)
    base_jerk = hand_jerk(base)
    base_ratio = base_jerk / max(base_speed, 1e-8)
    ratio_cap = max(base_ratio * args.max_ratio_lift, args.min_ratio_cap)

    lo, hi = 0.0, float(args.max_beta)
    best_beta = 0.0
    best = base
    for _ in range(args.search_steps):
        mid = (lo + hi) * 0.5
        cand = attach_local(source, learned_local + mid * boost)
        sp = hand_speed(cand)
        jk = hand_jerk(cand)
        ratio = jk / max(sp, 1e-8)
        drift = float((mid * boost).pow(2).mean().sqrt())
        ok = ratio <= ratio_cap and drift <= args.max_rms_drift
        if ok:
            best_beta = mid
            best = cand
            lo = mid
        else:
            hi = mid

    out = source.clone()
    out[:, BODY] = source[:, BODY]
    out[:, FACE] = source[:, FACE]
    out[:, RH] = best[:, RH]
    out[:, LH] = best[:, LH]
    rec = {
        "base_speed": base_speed,
        "base_jerk": base_jerk,
        "base_jerk_per_speed": base_ratio,
        "beta": float(best_beta),
        "proj_speed": hand_speed(out),
        "proj_jerk": hand_jerk(out),
        "proj_jerk_per_speed": hand_jerk(out) / max(hand_speed(out), 1e-8),
    }
    return out.contiguous(), rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_pt", required=True)
    ap.add_argument("--learned_pt", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--max_beta", type=float, default=1.0)
    ap.add_argument("--max_ratio_lift", type=float, default=1.05)
    ap.add_argument("--min_ratio_cap", type=float, default=0.0)
    ap.add_argument("--max_rms_drift", type=float, default=999.0)
    ap.add_argument("--endpoint_taper", action="store_true")
    ap.add_argument("--search_steps", type=int, default=18)
    args = ap.parse_args()
    if args.kernel > 1 and args.kernel % 2 == 0:
        raise ValueError("--kernel must be odd")

    source = load_pose_map(ROOT / args.source_pt)
    learned = load_pose_map(ROOT / args.learned_pt)
    out = {}
    trace = {}
    for sid, pose in learned.items():
        if sid not in source:
            continue
        proj, rec = project_clip(source[sid], pose, args)
        out[sid] = proj
        trace[sid] = rec

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")
    if args.trace_json:
        tp = ROOT / args.trace_json
        tp.parent.mkdir(parents=True, exist_ok=True)
        vals = list(trace.values())
        summary = {
            "source_pt": args.source_pt,
            "learned_pt": args.learned_pt,
            "out_pt": args.out_pt,
            "kernel": args.kernel,
            "max_beta": args.max_beta,
            "max_ratio_lift": args.max_ratio_lift,
            "mean_beta": float(sum(v["beta"] for v in vals) / max(len(vals), 1)),
            "mean_speed_lift": float(sum(v["proj_speed"] / max(v["base_speed"], 1e-8) for v in vals) / max(len(vals), 1)),
            "clips": len(vals),
            "per_clip": trace,
        }
        tp.write_text(json.dumps(summary, indent=2))
        print(f"saved trace -> {tp}")


if __name__ == "__main__":
    main()
