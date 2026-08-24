"""Oracle candidate mixer for Sign-JEPA experiments.

This is a diagnostic upper bound, not a deployable method.  It selects one
candidate output per clip using ground-truth pose alignment costs, then writes a
mixed prediction file for official SLRTP evaluation.  If this oracle improves
official metrics, candidate selection is the bottleneck and a learned selector
is worth training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.sign_jepa_motion_texture_retrieval import as_pose  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402

HAND = slice(8, 50)
WRISTS = [9, 10]


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def load_gt(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {str(k): as_pose(v).contiguous() for k, v in data.items()}


def eval_rate_pose(pose: torch.Tensor) -> torch.Tensor:
    return pose[::2].float()


def resize_to(pose: torch.Tensor, length: int) -> torch.Tensor:
    if pose.shape[0] == length:
        return pose
    return resize_seq(pose.reshape(pose.shape[0], -1), length).reshape(length, 178, 3)


def speed(x: torch.Tensor, joints: slice | list[int]) -> torch.Tensor:
    seg = x[:, joints]
    if seg.shape[0] < 2:
        return torch.tensor(0.0)
    return (seg[1:] - seg[:-1]).norm(dim=-1).mean()


def pose_cost(pred: torch.Tensor, gt: torch.Tensor, motion_weight: float, length_weight: float) -> float:
    pred_eval = eval_rate_pose(pred)
    gt_eval = eval_rate_pose(gt)
    length_ratio = abs(pred_eval.shape[0] / max(1, gt_eval.shape[0]) - 1.0)
    aligned = resize_to(pred_eval, gt_eval.shape[0])
    mje = (aligned - gt_eval).norm(dim=-1).mean()
    hand_mje = (aligned[:, HAND] - gt_eval[:, HAND]).norm(dim=-1).mean()
    wrist_speed = speed(aligned, WRISTS) / speed(gt_eval, WRISTS).clamp_min(1e-6)
    motion_penalty = (wrist_speed - 1.0).abs()
    return float(mje + 0.5 * hand_mje + motion_weight * motion_penalty + length_weight * length_ratio)


def parse_candidate(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path
    name, path = spec.split("=", 1)
    return name, Path(path)


def compose(args: argparse.Namespace) -> None:
    gt = load_gt(ROOT / args.gt_pt)
    candidates = []
    for spec in args.candidate:
        name, path = parse_candidate(spec)
        poses = load_pose_map(ROOT / path)
        candidates.append((name, poses))
    out = {}
    trace = []
    counts = Counter()
    for sid, gt_pose in gt.items():
        best = None
        for name, poses in candidates:
            if sid not in poses:
                continue
            cost = pose_cost(poses[sid], gt_pose, args.motion_weight, args.length_weight)
            if best is None or cost < best["cost"]:
                best = {"name": name, "pose": poses[sid], "cost": cost}
        if best is None:
            continue
        out[sid] = best["pose"].contiguous()
        counts[best["name"]] += 1
        trace.append({"id": sid, "choice": best["name"], "cost": best["cost"]})
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")
    print(json.dumps(dict(counts), indent=2))
    if args.trace_json:
        trace_path = ROOT / args.trace_json
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps({"counts": dict(counts), "items": trace}, indent=2))
        print(f"saved trace -> {trace_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_pt", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--candidate", action="append", required=True)
    ap.add_argument("--motion_weight", type=float, default=0.04)
    ap.add_argument("--length_weight", type=float, default=0.03)
    args = ap.parse_args()
    compose(args)


if __name__ == "__main__":
    main()
