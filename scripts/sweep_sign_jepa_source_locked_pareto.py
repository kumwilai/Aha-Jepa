"""Sweep source-locked Sign-JEPA checkpoints by motion Pareto score.

This script is intentionally limited to the cheap stage:

1. sample detail bases from epoch checkpoints;
2. fuse them back into the old Sign-JEPA carrier with hand-only gains;
3. compute motion ratios against GT;
4. write a ranked summary.

Run SLRTP only on the top few candidates from the resulting summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def checkpoint_paths(ckpt_dir: Path, include_best: bool) -> list[Path]:
    paths = sorted(ckpt_dir.glob("epoch_*.pt"))
    if include_best and (ckpt_dir / "best.pt").exists():
        paths.append(ckpt_dir / "best.pt")
    if not paths:
        raise FileNotFoundError(f"no checkpoint files found in {ckpt_dir}")
    return paths


def candidate_score(metrics: dict) -> float:
    speed = float(metrics["hand_speed_ratio"])
    jerk = float(metrics["hand_jerk_ratio"])
    std = float(metrics["hand_posestd_ratio"])
    body = float(metrics["body_speed_ratio"])
    face = float(metrics["face_speed_ratio"])

    score = abs(speed - 0.95)
    score += 0.25 * abs(jerk - 1.45)
    score += 0.10 * abs(std - 0.94)
    if speed < 0.85:
        score += 2.0 * (0.85 - speed)
    if speed > 1.05:
        score += 2.0 * (speed - 1.05)
    if jerk > 1.70:
        score += 0.75 * (jerk - 1.70)
    # Penalize accidental body/face rewrites. The source-locked recipe should
    # keep these in the old-JEPA regime.
    if body > 0.50:
        score += 0.25 * (body - 0.50)
    if face > 0.35:
        score += 0.25 * (face - 0.35)
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--gains", default="2.5,3.0,3.5,4.0,4.5")
    ap.add_argument("--include_best", action="store_true")
    ap.add_argument("--sample_device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--post_smooth_kernel", type=int, default=3)
    ap.add_argument("--post_smooth_blend", type=float, default=0.05)
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument(
        "--dev_source_pt",
        default="external/SLRTP-Sign-Production-Evaluation/results/sign_jepa_generator_manifest_dev_trainpolicy.pt",
    )
    ap.add_argument(
        "--test_source_pt",
        default="external/SLRTP-Sign-Production-Evaluation/results/sign_jepa_generator_manifest_test_trainpolicy.pt",
    )
    ap.add_argument("--dev_manifest", default="data/phoenix/phoenix_dev.json")
    ap.add_argument("--test_manifest", default="data/phoenix/phoenix_test.json")
    ap.add_argument(
        "--dev_reference_pt",
        default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt",
    )
    ap.add_argument(
        "--test_reference_pt",
        default="external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/test.pt",
    )
    args = ap.parse_args()

    ckpt_dir = ROOT / args.ckpt_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gains = [float(x) for x in args.gains.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    split_cfg = {
        "dev": (args.dev_source_pt, args.dev_manifest, args.dev_reference_pt),
        "test": (args.test_source_pt, args.test_manifest, args.test_reference_pt),
    }

    rows = []
    for ckpt in checkpoint_paths(ckpt_dir, args.include_best):
        ckpt_name = ckpt.stem
        for split in splits:
            if split not in split_cfg:
                raise ValueError(f"unknown split {split!r}")
            source_pt, manifest, reference_pt = split_cfg[split]
            base_pt = out_dir / f"{ckpt_name}_{split}_base.pt"
            if not base_pt.exists():
                sample_cmd = [
                    sys.executable,
                    "-u",
                    "scripts/train_sign_jepa_motion_refiner.py",
                    "sample_manifest",
                    "--ckpt",
                    rel(ckpt),
                    "--source_pt",
                    source_pt,
                    "--manifest_json",
                    manifest,
                    "--reference_pt",
                    reference_pt,
                    "--out_pt",
                    rel(base_pt),
                    "--steps",
                    str(args.steps),
                    "--batch_size",
                    str(args.batch_size),
                    "--device",
                    args.sample_device,
                ]
                if args.max_clips:
                    sample_cmd.extend(["--max_clips", str(args.max_clips)])
                run(sample_cmd)

            for gain in gains:
                gain_tag = str(gain).replace(".", "p")
                fused_pt = out_dir / f"{ckpt_name}_{split}_hg{gain_tag}.pt"
                metrics_json = out_dir / f"{ckpt_name}_{split}_hg{gain_tag}_motion.json"
                if not fused_pt.exists():
                    run(
                        [
                            sys.executable,
                            "-u",
                            "scripts/sign_jepa_source_locked_fusion.py",
                            "fuse",
                            "--source_pt",
                            source_pt,
                            "--detail_pt",
                            rel(base_pt),
                            "--out_pt",
                            rel(fused_pt),
                            "--hand_gain",
                            str(gain),
                            "--post_smooth_kernel",
                            str(args.post_smooth_kernel),
                            "--post_smooth_blend",
                            str(args.post_smooth_blend),
                        ]
                    )
                if not metrics_json.exists():
                    run(
                        [
                            sys.executable,
                            "-u",
                            "scripts/sign_jepa_source_locked_fusion.py",
                            "motion_eval",
                            "--pred_pt",
                            rel(fused_pt),
                            "--reference_pt",
                            reference_pt,
                            "--out_json",
                            rel(metrics_json),
                        ]
                    )
                metrics = json.loads(metrics_json.read_text())
                row = {
                    "ckpt": ckpt.name,
                    "split": split,
                    "gain": gain,
                    "base_pt": rel(base_pt),
                    "fused_pt": rel(fused_pt),
                    "metrics_json": rel(metrics_json),
                    **metrics,
                }
                row["score"] = candidate_score(row)
                rows.append(row)

    rows.sort(key=lambda x: x["score"])
    summary_json = out_dir / "pareto_motion_summary.json"
    summary_json.write_text(json.dumps(rows, indent=2))

    lines = [
        "# Source-Locked Sign-JEPA Motion Pareto Sweep",
        "",
        "| rank | ckpt | split | gain | score | hand speed | hand jerk | hand std | body | face | fused artifact |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[: args.top_k], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["ckpt"],
                    row["split"],
                    f"{row['gain']:.2f}",
                    f"{row['score']:.4f}",
                    f"{row['hand_speed_ratio']:.3f}",
                    f"{row['hand_jerk_ratio']:.3f}",
                    f"{row['hand_posestd_ratio']:.3f}",
                    f"{row['body_speed_ratio']:.3f}",
                    f"{row['face_speed_ratio']:.3f}",
                    f"`{row['fused_pt']}`",
                ]
            )
            + " |"
        )
    summary_md = out_dir / "pareto_motion_summary.md"
    summary_md.write_text("\n".join(lines) + "\n")
    print(f"saved summary -> {summary_json}")
    print(f"saved summary -> {summary_md}")


if __name__ == "__main__":
    main()
