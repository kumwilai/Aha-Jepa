"""Wrist / arm-trajectory amplitude restoration for collapsed Sign-JEPA carriers.

Diagnosis (see step2_flow_generator.md): the Sign-JEPA candidate family locks
body/face to the JEPA carrier, whose arm trajectory is collapsed.  Measured on
the multi-objective selector test output: finger travel `1.006x` GT but wrist
travel only `0.422x` GT.  The SLRTP `total_distance` metric is exactly the
wrist-travel ratio (1.0 optimal), so the family is permanently capped at ~0.4.

This operator restores arm-motion *amplitude* without rewriting *timing*:

    body'(t) = body_mean + g * (body(t) - body_mean)

`g` is a per-clip gain that scales the collapsed body deviation-from-mean so the
clip's eval-rate wrist travel matches a length-matched real-corpus target built
from the train split (no dev/test ground truth).  Because every body joint
scales around its own per-clip mean, the JEPA semantic timing (the trajectory
*shape* and its temporal correlation) is preserved — only the collapsed
magnitude is corrected.

Skeleton (verified by proximity analysis on GT):
  body          = joints 0:8   (wrists are idx 2 = RWrist, idx 5 = LWrist)
  right hand    = joints 8:29  (joint 8 is co-located with body wrist 2)
  left hand     = joints 29:50 (joint 29 is co-located with body wrist 5)
  face          = joints 50:178 (untouched: head joints barely move)

Each hand block is rigidly translated by its wrist's amplification delta, so
hands stay attached to the arm and finger texture is bit-exactly preserved.

Usage::

    python scripts/sign_jepa_wrist_restore.py \
      --in_pt   external/.../results/sign_jepa_multiobjective_selector_test.pt \
      --train_gt_pt external/.../data/train.pt \
      --out_pt  external/.../results/sign_jepa_multiobjective_selector_restored_test.pt \
      --trace_json outputs/sota_chase/sign_jepa_multiobjective_selector/test_restore_trace.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.sign_jepa_motion_texture_retrieval import as_pose  # noqa: E402
from scripts.train_sign_jepa_flow_generator import temporal_smooth  # noqa: E402

BODY = slice(0, 8)
RH = slice(8, 29)        # right hand, attached to body wrist idx 2
LH = slice(29, 50)       # left hand,  attached to body wrist idx 5
RWRIST, LWRIST = 2, 5


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def eval_wrist_travel(pose: torch.Tensor) -> float:
    """Mean over the two wrists of summed eval-rate (::2) frame displacement.

    This mirrors the SLRTP `pose_distance` metric, which downsamples 25fps
    poses by 2 before measuring wrist path length.
    """
    s = pose[::2].float()
    if s.shape[0] < 2:
        return 0.0
    d = (s[1:, [RWRIST, LWRIST]] - s[:-1, [RWRIST, LWRIST]]).norm(dim=-1)
    return float(d.sum(0).mean())


def build_length_travel_index(train_gt: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted (eval_length, wrist_travel) pairs from real train clips."""
    lens, travels = [], []
    for pose in train_gt.values():
        s = pose[::2]
        if s.shape[0] < 2:
            continue
        lens.append(int(s.shape[0]))
        travels.append(eval_wrist_travel(pose))
    order = np.argsort(lens)
    return np.asarray(lens)[order], np.asarray(travels)[order]


def knn_target(eval_len: int, lens: np.ndarray, travels: np.ndarray, k: int) -> float:
    """Median real wrist travel among the k train clips closest in length."""
    pos = int(np.searchsorted(lens, eval_len))
    lo = max(0, pos - k // 2)
    hi = min(len(lens), lo + k)
    lo = max(0, hi - k)
    return float(np.median(travels[lo:hi]))


def smooth_time(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Moving-average along time for a [T, J, 3] tensor (odd kernel)."""
    if kernel <= 1:
        return x
    y = x.reshape(x.shape[0], -1).T[None]              # [1, J*3, T]
    pad = kernel // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=kernel, stride=1)
    return y[0].T.reshape(x.shape)


def restore_clip(pose: torch.Tensor, gain: float, lp_kernel: int,
                 smooth_kernel: int) -> torch.Tensor:
    """Amplify the *low-frequency* body deviation-from-mean by `gain`.

    Only the low-pass component (gross arm swing) is amplified; the high-pass
    body component passes through at 1x.  This is the fix for jerk inflation:
    a uniform amplification scales high-frequency content too, and the hand
    blocks — translated rigidly by the wrist delta — inherit that amplified
    jerk.  Amplifying only the low-pass keeps the wrist delta smooth.
    """
    pose = pose.float()
    body = pose[:, BODY]
    bmean = body.mean(0, keepdim=True)
    dev_lp = smooth_time(body - bmean, lp_kernel)       # gross arm swing only
    boost = (gain - 1.0) * dev_lp                       # smooth, low-frequency
    out = pose.clone()
    out[:, BODY] = body + boost
    # hands follow their wrist's (smooth) boost -> attachment + texture kept
    out[:, RH] = pose[:, RH] + boost[:, RWRIST][:, None, :]
    out[:, LH] = pose[:, LH] + boost[:, LWRIST][:, None, :]
    if smooth_kernel > 1:
        sm = temporal_smooth(out, smooth_kernel)
        out[:, :50] = 0.85 * out[:, :50] + 0.15 * sm[:, :50]
    return out.contiguous()


def restore(args: argparse.Namespace) -> None:
    poses = load_pose_map(ROOT / args.in_pt)
    train_gt = load_pose_map(ROOT / args.train_gt_pt)
    lens, travels = build_length_travel_index(train_gt)
    print(f"[restore] train target index: {len(lens)} real clips, "
          f"wrist travel median={np.median(travels):.4f}", flush=True)

    out: dict[str, torch.Tensor] = {}
    trace = []
    gains, before, after = [], [], []
    for sid, pose in poses.items():
        eval_len = int(pose[::2].shape[0])
        cur = eval_wrist_travel(pose)
        tgt = args.target_ratio * knn_target(eval_len, lens, travels, args.knn)
        raw_gain = tgt / max(cur, 1e-6)
        gain = float(np.clip(raw_gain, args.gain_min, args.gain_max))
        restored = restore_clip(pose, gain, args.lp_kernel, args.smooth_kernel)
        out[sid] = restored
        gains.append(gain)
        before.append(cur)
        after.append(eval_wrist_travel(restored))
        trace.append({
            "id": sid, "eval_len": eval_len,
            "wrist_travel_in": round(cur, 4),
            "wrist_travel_target": round(tgt, 4),
            "wrist_travel_out": round(after[-1], 4),
            "gain": round(gain, 3), "gain_raw": round(raw_gain, 3),
        })

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}", flush=True)
    print(f"[restore] gain: mean={np.mean(gains):.3f} "
          f"[{np.min(gains):.3f}, {np.max(gains):.3f}] "
          f"clamped={int(np.sum((np.array(gains) <= args.gain_min + 1e-6) | (np.array(gains) >= args.gain_max - 1e-6)))}", flush=True)
    print(f"[restore] eval-rate wrist travel: in={np.mean(before):.4f} "
          f"out={np.mean(after):.4f} (train real median={np.median(travels):.4f})", flush=True)

    if args.trace_json:
        tp = ROOT / args.trace_json
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps({
            "in_pt": args.in_pt, "target_ratio": args.target_ratio,
            "gain_min": args.gain_min, "gain_max": args.gain_max,
            "smooth_kernel": args.smooth_kernel,
            "gain_mean": float(np.mean(gains)),
            "wrist_travel_in_mean": float(np.mean(before)),
            "wrist_travel_out_mean": float(np.mean(after)),
            "items": trace,
        }, indent=2))
        print(f"saved trace -> {tp}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_pt", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--train_gt_pt", default="external/SLRTP-Sign-Production-Evaluation/"
                    "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--target_ratio", type=float, default=1.0,
                    help="fraction of the real-corpus wrist travel to aim for")
    ap.add_argument("--gain_min", type=float, default=0.5)
    ap.add_argument("--gain_max", type=float, default=3.5)
    ap.add_argument("--knn", type=int, default=31,
                    help="train clips (closest by length) for the travel target")
    ap.add_argument("--lp_kernel", type=int, default=9,
                    help="odd kernel splitting gross arm swing (amplified) from "
                         "high-frequency body content (passed through at 1x)")
    ap.add_argument("--smooth_kernel", type=int, default=1,
                    help="odd kernel for light body/hand smoothing; 1 = off")
    args = ap.parse_args()
    restore(args)


if __name__ == "__main__":
    main()
