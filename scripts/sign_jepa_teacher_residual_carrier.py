"""JEPA-anchored teacher-residual carrier composition.

Method: JTRC, JEPA Teacher-Residual Carrier.

Start from a Sign-JEPA carrier pose and inject a bounded residual from a
high-BLEU teacher pose.  The residual can be global or restricted to a region.
The default `bodyface_jepa_hands_teacher` mode keeps body + face from JEPA and
uses only the teacher hand residual, making JEPA the majority pose carrier
(136/178 joints) while letting the teacher repair the hand channel that the
SLRTP back-translator is most sensitive to.

This is a composition/evaluation utility, not a training script.  It uses no
test labels; it only reads two prediction files with matching split ids.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
BODY = slice(0, 8)
HAND = slice(8, 50)
FACE = slice(50, 178)


def torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def as_pose(x) -> torch.Tensor:
    if isinstance(x, dict):
        x = x.get("poses_3d", x.get("pose"))
    pose = torch.as_tensor(x).float()
    if pose.ndim == 2:
        pose = pose.reshape(pose.shape[0], 178, 3)
    if pose.ndim != 3 or pose.shape[1:] != (178, 3):
        raise ValueError(f"bad pose shape {tuple(pose.shape)}")
    return pose


def resize_pose(pose: torch.Tensor, length: int) -> torch.Tensor:
    if pose.shape[0] == length:
        return pose
    y = pose.reshape(pose.shape[0], -1).T[None]
    y = F.interpolate(y, size=int(length), mode="linear", align_corners=True)
    return y[0].T.reshape(int(length), 178, 3)


def filter_keys(path: str) -> list[str] | None:
    if not path:
        return None
    data = torch_load_cpu(ROOT / path)
    return [str(k) for k in data.keys()]


def compose_clip(carrier: torch.Tensor, teacher: torch.Tensor, mode: str, alpha: float) -> torch.Tensor:
    teacher = resize_pose(teacher, carrier.shape[0])
    alpha = float(alpha)
    if mode == "global_residual":
        return (carrier + alpha * (teacher - carrier)).contiguous()
    if mode == "bodyface_jepa_hands_teacher":
        out = carrier.clone()
        out[:, HAND] = carrier[:, HAND] + alpha * (teacher[:, HAND] - carrier[:, HAND])
        return out.contiguous()
    if mode == "body_jepa_handface_teacher":
        out = carrier.clone()
        out[:, HAND] = carrier[:, HAND] + alpha * (teacher[:, HAND] - carrier[:, HAND])
        out[:, FACE] = carrier[:, FACE] + alpha * (teacher[:, FACE] - carrier[:, FACE])
        return out.contiguous()
    if mode == "hands_jepa_bodyface_teacher":
        out = carrier.clone()
        out[:, BODY] = carrier[:, BODY] + alpha * (teacher[:, BODY] - carrier[:, BODY])
        out[:, FACE] = carrier[:, FACE] + alpha * (teacher[:, FACE] - carrier[:, FACE])
        return out.contiguous()
    raise ValueError(f"unknown mode {mode!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carrier_pt", required=True,
                    help="Sign-JEPA carrier prediction .pt.")
    ap.add_argument("--teacher_pt", required=True,
                    help="High-BLEU teacher prediction .pt.")
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--mode", default="bodyface_jepa_hands_teacher",
                    choices=[
                        "global_residual",
                        "bodyface_jepa_hands_teacher",
                        "body_jepa_handface_teacher",
                        "hands_jepa_bodyface_teacher",
                    ])
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Residual strength. 1.0 copies the selected teacher region.")
    ap.add_argument("--filter_keys_pt", default="",
                    help="Optional .pt whose keys define output order/subset, e.g. local SLRTP test.pt.")
    args = ap.parse_args()

    carrier = torch_load_cpu(ROOT / args.carrier_pt)
    teacher = torch_load_cpu(ROOT / args.teacher_pt)
    keys = filter_keys(args.filter_keys_pt) or sorted(set(map(str, carrier.keys())) & set(map(str, teacher.keys())))
    out = {}
    missing = []
    for sid in keys:
        if sid not in carrier or sid not in teacher:
            missing.append(sid)
            continue
        out[sid] = compose_clip(as_pose(carrier[sid]), as_pose(teacher[sid]), args.mode, args.alpha)
    if missing:
        raise SystemExit(f"missing {len(missing)} ids, first={missing[:3]}")

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)} mode={args.mode} alpha={args.alpha}")


if __name__ == "__main__":
    main()
