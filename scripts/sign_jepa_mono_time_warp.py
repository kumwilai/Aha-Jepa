"""MonoTimeWarp: per-clip closed-form monotone time-warp of the JEPA carrier.

This operator is the principled "retiming-only" alternative to the
scalar JEPA+amp deviation amplifier (`sign_jepa_motion_amplify.py`).
Idea: rather than scaling the carrier's pose deviation around its
per-clip temporal mean, we hold the carrier's *pose set* exactly and
redistribute the time axis so that equal output time spans equal
hand-motion energy. The output is a length-preserving inverse-CDF
resampling of the carrier; every output pose is a convex combination
of two adjacent carrier poses, so the per-frame hand-pose distribution
is a subset of the carrier's.

Why this is interesting on top of JEPA+amp:

  - JEPA+amp scales the dev-from-mean of all frames uniformly. That
    raises both motion and spatial-distortion. Pure JEPA carrier
    BLEU-4 12.47 -> JEPA+amp BLEU-4 11.15 (-1.32 BLEU = the amp tax).
  - MonoTimeWarp introduces ZERO spatial distortion (every output
    pose lies on the carrier's pose manifold by construction). If
    the BT evaluator reads the per-frame hand distribution, this
    operator should not pay an amp tax.
  - Inverse-CDF re-time alone does NOT change the carrier's TOTAL
    hand-motion. It redistributes per-frame velocity (uniformizes
    motion across output time) but its mean speed equals the
    carrier's. So a pure warp typically will NOT pass the motion
    gate; it has to be paired with a mild amp.

Two CLI configurations are intended for this script:

  1. PURE-WARP (--g_hand 1.0 --g_body 1.0): tests "zero spatial
     distortion" — does retiming alone improve BLEU vs the carrier?
  2. WARP+MILD-AMP (--g_hand 2.0 --g_body 2.0 --amp_lp_kernel 5):
     tests "warp reduces the amp needed" — does pre-warping let a
     smaller `g` clear the motion gate while paying less BLEU tax
     than canonical JEPA+amp (g=3.5)?

This script is closed-form. No retrieval, no train-corpus stats, no
evaluator distillation, no learning. Pure-generative at inference.
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

from scripts.sign_jepa_motion_texture_retrieval import as_pose  # noqa: E402

BODY = slice(0, 8)
RH = slice(8, 29)
LH = slice(29, 50)
HAND = slice(8, 50)
FACE = slice(50, 178)
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


def smooth_time(x: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return x
    if x.ndim == 1:
        y = x[None, None]
    else:
        y = x.reshape(x.shape[0], -1).T[None]
    pad = kernel // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=kernel, stride=1)
    if x.ndim == 1:
        return y[0, 0]
    return y[0].T.reshape(x.shape)


def hand_motion_energy(pose: torch.Tensor, smooth_kernel: int = 3) -> torch.Tensor:
    """Per-frame hand-motion magnitude s(t) in R^T.

    Smoothed by a small replicate-padded average to suppress
    per-frame noise (we only need the macro motion envelope to drive
    the warp).
    """
    T = pose.shape[0]
    hand = pose[:, HAND]
    vel = hand[1:] - hand[:-1]  # (T-1, J, 3)
    s = vel.norm(dim=-1).mean(dim=-1)  # (T-1,)
    # Pad to length T (no velocity at t=0)
    s = torch.cat([torch.zeros(1, dtype=s.dtype, device=s.device), s], dim=0)
    return smooth_time(s, smooth_kernel)


def mono_time_warp_clip(pose: torch.Tensor, eps_floor: float, smooth_kernel: int) -> torch.Tensor:
    """Inverse-CDF resampling of `pose` by its own hand-motion energy.

    pose: (T, 178, 3) carrier pose
    eps_floor: floor added to the energy s(t) as `eps_floor * mean(s)`
               (prevents degenerate compression in zero-motion clips).
    smooth_kernel: replicate-pad avg kernel applied to s(t) before CDF.
    Returns:    (T, 178, 3) retimed pose, same length, no extrapolation.

    Construction guarantees:
      1. Output pose at every frame is a convex combination of two
         adjacent carrier frames. The output hand-pose distribution
         is therefore a subset of the carrier's.
      2. The output frame index in carrier-time grows monotonically
         (it is the inverse of a monotonically non-decreasing CDF).
      3. The mapping is identity when s(t) is constant — i.e. carriers
         with uniform motion are unchanged.
    """
    pose = pose.float()
    T = pose.shape[0]
    if T < 3:
        return pose.clone()
    s = hand_motion_energy(pose, smooth_kernel)
    s_mean = s.mean().clamp_min(1e-6)
    s = s + eps_floor * s_mean
    c = torch.cumsum(s, dim=0)
    c = c - c[0]
    c_max = c[-1].clamp_min(1e-6)
    c_norm = c / c_max  # (T,) monotonic in [0, 1]
    u_new = torch.linspace(0.0, 1.0, T, dtype=pose.dtype, device=pose.device)
    # Find for each u_new the carrier-time t s.t. c_norm(t) = u_new.
    # searchsorted returns insertion indices in [0, T].
    idx = torch.searchsorted(c_norm, u_new, right=False)
    idx = idx.clamp(1, T - 1)
    left = idx - 1
    c_l = c_norm[left]
    c_r = c_norm[idx]
    denom = (c_r - c_l).clamp_min(1e-9)
    alpha = ((u_new - c_l) / denom).clamp(0.0, 1.0)
    t_frac = left.float() + alpha
    # Sample pose at fractional time t_frac by linear interp on carrier frames.
    t_floor = t_frac.floor().long().clamp(0, T - 1)
    t_ceil = (t_floor + 1).clamp(0, T - 1)
    a = (t_frac - t_floor.float())[:, None, None]
    return (1.0 - a) * pose[t_floor] + a * pose[t_ceil]


def amp_deviation(dev: torch.Tensor, gain: float, lp_kernel: int) -> torch.Tensor:
    """Same as `sign_jepa_motion_amplify.amp_deviation` (kept local
    to keep this script standalone). lp_kernel<=1 = uniform amp;
    lp_kernel>1 = LP-amplify, HP-preserve.
    """
    if lp_kernel > 1:
        lp = smooth_time(dev, lp_kernel)
        return dev + (gain - 1.0) * lp
    return gain * dev


def amplify_post_warp(pose: torch.Tensor, g_body: float, g_hand: float, g_face: float,
                      amp_lp_kernel: int) -> torch.Tensor:
    """Apply (optionally) the JEPA+amp deviation amplifier after the
    monotone time-warp. If all gains are 1.0 this is a no-op.
    """
    if g_body == 1.0 and g_hand == 1.0 and g_face == 1.0:
        return pose
    pose = pose.float()
    out = pose.clone()
    body = pose[:, BODY]
    body_mean = body.mean(0, keepdim=True)
    body_amp = body_mean + amp_deviation(body - body_mean, g_body, amp_lp_kernel)
    out[:, BODY] = body_amp

    rh_root_orig = pose[:, RWRIST:RWRIST + 1]
    rh_local = pose[:, RH] - rh_root_orig
    rh_local_mean = rh_local.mean(0, keepdim=True)
    rh_local_amp = rh_local_mean + amp_deviation(rh_local - rh_local_mean, g_hand, amp_lp_kernel)
    out[:, RH] = body_amp[:, RWRIST:RWRIST + 1] + rh_local_amp

    lh_root_orig = pose[:, LWRIST:LWRIST + 1]
    lh_local = pose[:, LH] - lh_root_orig
    lh_local_mean = lh_local.mean(0, keepdim=True)
    lh_local_amp = lh_local_mean + amp_deviation(lh_local - lh_local_mean, g_hand, amp_lp_kernel)
    out[:, LH] = body_amp[:, LWRIST:LWRIST + 1] + lh_local_amp

    if g_face != 1.0:
        face = pose[:, FACE]
        face_mean = face.mean(0, keepdim=True)
        out[:, FACE] = face_mean + amp_deviation(face - face_mean, g_face, amp_lp_kernel)
    return out.contiguous()


def warp_clip(pose: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    warped = mono_time_warp_clip(pose, eps_floor=args.eps_floor, smooth_kernel=args.smooth_kernel)
    if args.g_body == 1.0 and args.g_hand == 1.0 and args.g_face == 1.0:
        return warped
    return amplify_post_warp(warped, args.g_body, args.g_hand, args.g_face, args.amp_lp_kernel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_pt", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--trace_json", default="")
    ap.add_argument("--eps_floor", type=float, default=0.1,
                    help="floor in CDF as `eps_floor * mean(s)` (prevents degenerate compression)")
    ap.add_argument("--smooth_kernel", type=int, default=3,
                    help="odd avg-pool kernel for hand-motion-energy smoothing")
    ap.add_argument("--g_body", type=float, default=1.0,
                    help="post-warp body amp gain (1.0 = pure warp, no amp)")
    ap.add_argument("--g_hand", type=float, default=1.0,
                    help="post-warp hand amp gain (1.0 = pure warp)")
    ap.add_argument("--g_face", type=float, default=1.0,
                    help="post-warp face amp gain (1.0 = unchanged)")
    ap.add_argument("--amp_lp_kernel", type=int, default=5,
                    help="post-warp amp LP kernel (5 matches JEPA+amp default)")
    args = ap.parse_args()

    if args.smooth_kernel > 1 and args.smooth_kernel % 2 == 0:
        raise ValueError("--smooth_kernel must be odd")
    if args.amp_lp_kernel > 1 and args.amp_lp_kernel % 2 == 0:
        raise ValueError("--amp_lp_kernel must be odd")

    poses = load_pose_map(ROOT / args.in_pt)
    out: dict[str, torch.Tensor] = {}
    for sid, pose in poses.items():
        out[sid] = warp_clip(pose, args)

    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)} "
          f"eps_floor={args.eps_floor} smooth_kernel={args.smooth_kernel} "
          f"g_body={args.g_body} g_hand={args.g_hand} g_face={args.g_face} "
          f"amp_lp_kernel={args.amp_lp_kernel}", flush=True)

    if args.trace_json:
        tp = ROOT / args.trace_json
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps({
            "in_pt": args.in_pt,
            "eps_floor": args.eps_floor,
            "smooth_kernel": args.smooth_kernel,
            "g_body": args.g_body, "g_hand": args.g_hand, "g_face": args.g_face,
            "amp_lp_kernel": args.amp_lp_kernel,
            "clips": len(out),
        }, indent=2))
        print(f"saved trace -> {tp}", flush=True)


if __name__ == "__main__":
    main()
