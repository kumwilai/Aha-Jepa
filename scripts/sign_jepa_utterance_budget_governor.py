"""Utterance-level energy-budget governor for Sign-JEPA carrier outputs.

This is deliberately not retrieval: it fits one fixed linear policy from
train pairs, then applies the policy to a generated sentence. Inference uses
only the generated source pose plus text/gloss metadata for that sentence.
"""
from __future__ import annotations

import argparse
import hashlib
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

from scripts.sign_jepa_motion_amplify import amplify_clip  # noqa: E402
from scripts.sign_jepa_source_locked_fusion import as_pose, load_pose_map  # noqa: E402


HAND = slice(8, 50)


def stable_hash(token: str, n: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little") % n


def hand_speed(pose: torch.Tensor) -> float:
    x = pose.reshape(pose.shape[0], 178, 3).float()
    if x.shape[0] < 2:
        return 0.0
    v = x[1:, HAND] - x[:-1, HAND]
    return float(v.norm(dim=-1).mean())


def hand_jerk(pose: torch.Tensor) -> float:
    x = pose.reshape(pose.shape[0], 178, 3).float()
    if x.shape[0] < 3:
        return 0.0
    v = x[1:, HAND] - x[:-1, HAND]
    a = v[1:] - v[:-1]
    return float(a.norm(dim=-1).mean())


def row_text(row: dict) -> tuple[list[str], list[str]]:
    glosses = [g for g in str(row.get("gloss", "")).split() if g]
    words = [w.lower() for w in str(row.get("text", "")).split() if w]
    return glosses, words


def features(row: dict, src: torch.Tensor, n_hash: int, feature_mode: str = "full") -> np.ndarray:
    glosses, words = row_text(row)
    sp = max(hand_speed(src), 1e-8)
    jk = max(hand_jerk(src), 1e-8)
    T = max(int(src.shape[0]), 1)
    if feature_mode == "text":
        sp = 1.0
        jk = 1.0
    nums = np.asarray(
        [
            1.0,
            math.log1p(len(glosses)),
            math.log1p(len(words)),
            math.log1p(len(str(row.get("text", "")))),
            math.log1p(T),
            math.log(sp),
            math.log(jk),
            sp,
            jk,
        ],
        dtype=np.float64,
    )
    x = np.zeros(nums.shape[0] + n_hash, dtype=np.float64)
    x[: nums.shape[0]] = nums
    toks = [f"g:{g}" for g in glosses] + [f"w:{w}" for w in words]
    if toks:
        scale = 1.0 / math.sqrt(len(toks))
        for tok in toks:
            x[nums.shape[0] + stable_hash(tok, n_hash)] += scale
    return x


def load_rows(path: Path) -> dict[str, dict]:
    rows = json.load(open(path))
    return {str(r["id"]): r for r in rows}


def fit(args: argparse.Namespace) -> None:
    rows = load_rows(ROOT / args.train_manifest)
    source = load_pose_map(ROOT / args.train_source_pt)
    ref_raw = torch.load(ROOT / args.train_reference_pt, map_location="cpu", weights_only=False)
    xs, ys, kept = [], [], 0
    ratios = []
    rng = np.random.default_rng(args.seed)
    for sid, src in source.items():
        if sid not in rows or sid not in ref_raw:
            continue
        ref = as_pose(ref_raw[sid])
        ref_speed = max(hand_speed(ref), 1e-8)
        train_sources = [src]
        for _ in range(max(0, int(args.aug_repeats))):
            gain = float(rng.uniform(args.source_amp_min, args.source_amp_max))
            train_sources.append(
                amplify_clip(
                    src,
                    g_body=1.0,
                    g_hand=gain,
                    g_face=1.0,
                    lp_kernel=1,
                    smooth_kernel=1,
                )
            )
        for train_src in train_sources:
            src_speed = max(hand_speed(train_src), 1e-8)
            if args.target_mode == "abs_speed":
                ratio = float(np.clip(ref_speed / src_speed, args.target_ratio_min, args.target_ratio_max))
                target = math.log(ref_speed)
            else:
                ratio = float(np.clip(ref_speed / src_speed, args.target_ratio_min, args.target_ratio_max))
                target = math.log(ratio)
            xs.append(features(rows[sid], train_src, args.n_hash, args.feature_mode))
            ys.append(target)
            ratios.append(ratio)
        kept += 1
    if kept < 100:
        raise RuntimeError(f"too few train pairs: {kept}")
    X = np.stack(xs)
    y = np.asarray(ys, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xn = (X - mu) / sd
    eye = np.eye(Xn.shape[1], dtype=np.float64)
    w = np.linalg.solve(Xn.T @ Xn + args.ridge * eye, Xn.T @ y)
    pred = Xn @ w
    out = {
        "n_hash": args.n_hash,
        "ridge": args.ridge,
        "mu": torch.from_numpy(mu).float(),
        "sd": torch.from_numpy(sd).float(),
        "w": torch.from_numpy(w).float(),
        "target_ratio_min": args.target_ratio_min,
        "target_ratio_max": args.target_ratio_max,
        "target_mode": args.target_mode,
        "feature_mode": args.feature_mode,
        "train_pairs": kept,
        "train_examples": int(len(xs)),
        "aug_repeats": int(args.aug_repeats),
        "source_amp_min": float(args.source_amp_min),
        "source_amp_max": float(args.source_amp_max),
        "train_ratio_mean": float(np.mean(ratios)),
        "train_ratio_p10_p50_p90": [float(v) for v in np.percentile(ratios, [10, 50, 90])],
        "train_pred_log_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
    }
    out_path = ROOT / args.out_policy
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, torch.Tensor)}, indent=2))
    print(f"saved -> {out_path}", flush=True)


def predict_log_gain(policy: dict, row: dict, src: torch.Tensor) -> float:
    n_hash = int(policy["n_hash"])
    feature_mode = str(policy.get("feature_mode", "full"))
    x = features(row, src, n_hash, feature_mode)
    mu = policy["mu"].numpy().astype(np.float64)
    sd = policy["sd"].numpy().astype(np.float64)
    w = policy["w"].numpy().astype(np.float64)
    pred = float(((x - mu) / sd) @ w)
    if str(policy.get("target_mode", "ratio")) == "abs_speed":
        return pred - math.log(max(hand_speed(src), 1e-8))
    return pred


def apply(args: argparse.Namespace) -> None:
    policy = torch.load(ROOT / args.policy, map_location="cpu", weights_only=False)
    rows = load_rows(ROOT / args.manifest_json)
    source = load_pose_map(ROOT / args.in_pt)
    out = {}
    trace = {}
    gains = []
    for sid, src in source.items():
        row = rows.get(sid, {"id": sid, "text": "", "gloss": ""})
        raw_gain = math.exp(predict_log_gain(policy, row, src))
        g_hand = float(np.clip(raw_gain, args.min_gain, args.max_gain))
        g_body = float(np.clip(1.0 + args.body_coupling * (g_hand - 1.0), 1.0, args.max_body_gain))
        pose = amplify_clip(src, g_body=g_body, g_hand=g_hand, g_face=1.0,
                            lp_kernel=args.lp_kernel, smooth_kernel=args.smooth_kernel)
        out[sid] = pose.contiguous()
        gains.append(g_hand)
        trace[sid] = {
            "raw_gain": raw_gain,
            "g_hand": g_hand,
            "g_body": g_body,
            "src_hand_speed": hand_speed(src),
            "out_hand_speed": hand_speed(pose),
        }
    out_path = ROOT / args.out_pt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved -> {out_path} clips={len(out)}")
    print(f"g_hand mean={np.mean(gains):.3f} p10/50/90={np.percentile(gains, [10,50,90]).tolist()}")
    if args.trace_json:
        tp = ROOT / args.trace_json
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(trace, indent=2))
        print(f"trace -> {tp}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    ft = sub.add_parser("fit")
    ft.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    ft.add_argument("--train_source_pt", required=True)
    ft.add_argument("--train_reference_pt", default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    ft.add_argument("--out_policy", required=True)
    ft.add_argument("--n_hash", type=int, default=512)
    ft.add_argument("--ridge", type=float, default=10.0)
    ft.add_argument("--target_ratio_min", type=float, default=0.75)
    ft.add_argument("--target_ratio_max", type=float, default=1.80)
    ft.add_argument("--target_mode", choices=["ratio", "abs_speed"], default="ratio")
    ft.add_argument("--feature_mode", choices=["full", "text"], default="full")
    ft.add_argument("--aug_repeats", type=int, default=0,
                    help="extra train-only synthetically hand-dampened sources per clip")
    ft.add_argument("--source_amp_min", type=float, default=0.5)
    ft.add_argument("--source_amp_max", type=float, default=1.0)
    ft.add_argument("--seed", type=int, default=0)

    aply = sub.add_parser("apply")
    aply.add_argument("--policy", required=True)
    aply.add_argument("--manifest_json", required=True)
    aply.add_argument("--in_pt", required=True)
    aply.add_argument("--out_pt", required=True)
    aply.add_argument("--trace_json", default="")
    aply.add_argument("--min_gain", type=float, default=1.0)
    aply.add_argument("--max_gain", type=float, default=1.55)
    aply.add_argument("--body_coupling", type=float, default=0.35)
    aply.add_argument("--max_body_gain", type=float, default=1.20)
    aply.add_argument("--lp_kernel", type=int, default=1)
    aply.add_argument("--smooth_kernel", type=int, default=1)

    args = ap.parse_args()
    if args.mode == "fit":
        fit(args)
    else:
        apply(args)


if __name__ == "__main__":
    main()
