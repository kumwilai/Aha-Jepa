"""Boost low-frequency wrist trajectory while preserving local hand shape."""
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

RH = slice(8, 29)
LH = slice(29, 50)
HAND = slice(8, 50)
RWRIST, LWRIST = 2, 5


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def smooth_time(x: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return x
    y = x.reshape(x.shape[0], -1).T[None]
    pad = kernel // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=kernel, stride=1)
    return y[0].T.reshape(x.shape)


def hand_speed(pose: torch.Tensor) -> float:
    if pose.shape[0] < 2:
        return 0.0
    v = pose[1:, HAND] - pose[:-1, HAND]
    return float(v.norm(dim=-1).mean())


def boost_clip(pose: torch.Tensor, gain: float, lp_kernel: int, body_follow: float) -> tuple[torch.Tensor, dict]:
    pose = pose.float()
    out = pose.clone()
    rw = pose[:, RWRIST:RWRIST + 1]
    lw = pose[:, LWRIST:LWRIST + 1]
    rw_dev = rw - rw.mean(0, keepdim=True)
    lw_dev = lw - lw.mean(0, keepdim=True)
    rw_delta = (float(gain) - 1.0) * smooth_time(rw_dev, lp_kernel)
    lw_delta = (float(gain) - 1.0) * smooth_time(lw_dev, lp_kernel)

    out[:, RH] = out[:, RH] + rw_delta
    out[:, LH] = out[:, LH] + lw_delta
    out[:, RWRIST:RWRIST + 1] = out[:, RWRIST:RWRIST + 1] + float(body_follow) * rw_delta
    out[:, LWRIST:LWRIST + 1] = out[:, LWRIST:LWRIST + 1] + float(body_follow) * lw_delta
    return out.contiguous(), {
        "src_hand_speed": hand_speed(pose),
        "out_hand_speed": hand_speed(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_pt", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--gain", type=float, default=1.82)
    ap.add_argument("--lp_kernel", type=int, default=7)
    ap.add_argument("--body_follow", type=float, default=1.0)
    args = ap.parse_args()

    poses = load_pose_map(ROOT / args.in_pt)
    out, trace, ratios = {}, {}, []
    for sid, pose in poses.items():
        y, rec = boost_clip(pose, args.gain, args.lp_kernel, args.body_follow)
        out[sid] = y
        ratios.append(rec["out_hand_speed"] / max(rec["src_hand_speed"], 1e-8))
        trace[sid] = rec

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")
    if ratios:
        print(f"out/src hand speed p10/50/90={[float(v) for v in np.percentile(ratios, [10, 50, 90])]}")
    if args.trace_json:
        tp = ROOT / args.trace_json
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(trace, indent=2))
        print(f"trace -> {tp}")


if __name__ == "__main__":
    main()
