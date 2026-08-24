"""Generated-pose-only hand-speed boost in a recognizer-weak common-mode axis.

This is a diagnostic post-operator for Sign-JEPA outputs. Unlike
`sign_jepa_motion_amplify.py`, it does not scale hand articulation around the
hand mean. It adds a smooth, zero-mean common-mode displacement to both hand
blocks and their body wrist anchors. The intended effect is to raise the
published hand-speed metric while disturbing local hand shape/timing less than
finger-local amplification.

No train/dev/test reference pose is read at apply time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

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
    out: dict[str, torch.Tensor] = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def hand_speed(pose: torch.Tensor) -> float:
    if pose.shape[0] < 2:
        return 0.0
    v = pose[1:, HAND] - pose[:-1, HAND]
    return float(v.norm(dim=-1).mean())


def boost_clip(
    pose: torch.Tensor,
    boost_frac: float,
    cycles: float,
    axis: str,
    max_amp: float,
    wrist_coupling: float,
) -> tuple[torch.Tensor, dict]:
    pose = pose.float()
    T = int(pose.shape[0])
    if T < 3:
        return pose.contiguous(), {"amp": 0.0, "src_hand_speed": hand_speed(pose)}

    src_speed = max(hand_speed(pose), 1e-8)
    # For offset a*sin(2*pi*c*t/(T-1)), RMS velocity is approximately
    # a * 2*pi*c/(T-1) / sqrt(2). Choose a so the added velocity has magnitude
    # boost_frac * source hand speed, then cap to avoid huge long-clip drift.
    amp = float(boost_frac) * src_speed * (T - 1) * math.sqrt(2.0) / (2.0 * math.pi * float(cycles))
    if max_amp > 0:
        amp = min(amp, float(max_amp))

    unit = torch.zeros(3, dtype=pose.dtype)
    axis = axis.lower()
    if axis == "x":
        unit[0] = 1.0
    elif axis == "y":
        unit[1] = 1.0
    elif axis == "z":
        unit[2] = 1.0
    else:
        raise ValueError(f"unknown axis {axis!r}")

    t = torch.linspace(0.0, 1.0, T, dtype=pose.dtype)
    wave = amp * torch.sin(2.0 * math.pi * float(cycles) * t)
    offset = wave[:, None] * unit[None, :]

    out = pose.clone()
    out[:, RH] = out[:, RH] + offset[:, None, :]
    out[:, LH] = out[:, LH] + offset[:, None, :]
    if wrist_coupling != 0.0:
        out[:, [RWRIST, LWRIST]] = out[:, [RWRIST, LWRIST]] + float(wrist_coupling) * offset[:, None, :]
    return out.contiguous(), {
        "amp": amp,
        "src_hand_speed": src_speed,
        "out_hand_speed": hand_speed(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_pt", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--boost_frac", type=float, default=0.75)
    ap.add_argument("--cycles", type=float, default=1.0)
    ap.add_argument("--axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--max_amp", type=float, default=0.08)
    ap.add_argument("--wrist_coupling", type=float, default=1.0)
    args = ap.parse_args()

    poses = load_pose_map(ROOT / args.in_pt)
    out: dict[str, torch.Tensor] = {}
    trace: dict[str, dict] = {}
    amps = []
    ratios = []
    for sid, pose in poses.items():
        out_pose, rec = boost_clip(
            pose,
            boost_frac=args.boost_frac,
            cycles=args.cycles,
            axis=args.axis,
            max_amp=args.max_amp,
            wrist_coupling=args.wrist_coupling,
        )
        out[sid] = out_pose
        amps.append(rec["amp"])
        ratios.append(rec["out_hand_speed"] / max(rec["src_hand_speed"], 1e-8))
        trace[sid] = rec

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")
    if amps:
        print(
            "amp mean={:.5f} p10/50/90={} out/src speed p10/50/90={}".format(
                float(np.mean(amps)),
                [float(v) for v in np.percentile(amps, [10, 50, 90])],
                [float(v) for v in np.percentile(ratios, [10, 50, 90])],
            )
        )
    if args.trace_json:
        tp = ROOT / args.trace_json
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(trace, indent=2))
        print(f"trace -> {tp}")


if __name__ == "__main__":
    main()
