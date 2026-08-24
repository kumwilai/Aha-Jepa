"""Train a recognizer-aware duration policy for the Sign-JEPA generator.

The policy sees only deployable manifest fields at inference time. During
training it uses already evaluated back-translation text predictions from a
small duration-scale sweep and learns to predict the per-scale edit cost. At
sampling time the generator uses a fixed text-only total-length regressor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_sign_jepa_generator_slrtp178 import duration_policy_features  # noqa: E402


DEFAULT_LEXICON = [
    "REGEN",
    "GEWITTER",
    "WIND",
    "SCHNEE",
    "SONNE",
    "WOLKE",
    "TEMPERATUR",
    "GRAD",
    "NORD",
    "SUED",
    "WEST",
    "OST",
    "MORGEN",
    "NACHT",
    "TAG",
    "WOCHENENDE",
    "FROST",
    "NEBEL",
    "STARK",
    "SCHWACH",
    "MAESSIG",
    "KOENNEN",
    "REGION",
    "IX",
]


def words(text: str) -> list[str]:
    text = re.sub(r"[^\w\säöüßÄÖÜ]", " ", text.lower())
    return [w for w in text.split() if w]


def edit_distance(hyp: list[str], ref: list[str]) -> int:
    dp = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, start=1):
        nxt = [i] + [0] * len(ref)
        for j, r in enumerate(ref, start=1):
            nxt[j] = min(dp[j] + 1, nxt[j - 1] + 1, dp[j - 1] + int(h != r))
        dp = nxt
    return int(dp[-1])


def parse_candidate(value: str) -> tuple[float, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be SCALE=PATH")
    scale_s, path_s = value.split("=", 1)
    return float(scale_s), ROOT / path_s


def ordered_rows(manifest_path: Path, reference_pt: Path) -> list[dict]:
    rows = json.load(open(manifest_path))
    ref = torch.load(reference_pt, map_location="cpu", weights_only=True)
    by_id = {str(r["id"]): r for r in rows}
    missing = [sid for sid in ref.keys() if sid not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} reference ids missing from manifest, first={missing[:3]}")
    return [by_id[sid] for sid in ref.keys()]


def build_policy_template(train_manifest: Path, rows: list[dict], lexicon: list[str], feature_mode: str) -> dict:
    train_rows = json.load(open(train_manifest))
    gloss_counts = Counter(g for r in train_rows for g in str(r.get("gloss", "")).split())
    signers = sorted({str(r.get("signer", "")) for r in train_rows + rows})
    return {
        "gloss_counts": dict(gloss_counts),
        "signers": signers,
        "lexicon": lexicon,
        "feature_mode": feature_mode,
    }


def make_xy(rows: list[dict], candidates: list[tuple[float, Path]], policy_base: dict):
    scales = [s for s, _ in candidates]
    preds = []
    for _scale, path in candidates:
        vals = torch.load(path, map_location="cpu", weights_only=True)
        if len(vals) != len(rows):
            raise ValueError(f"{path} has {len(vals)} predictions for {len(rows)} rows")
        preds.append(vals)
    x = np.asarray([duration_policy_features(row, policy_base) for row in rows], dtype=np.float32)
    y = []
    ref_lens = []
    for i, row in enumerate(rows):
        ref = words(str(row.get("text", "")))
        ref_lens.append(len(ref))
        y.append([edit_distance(words(preds[j][i]), ref) for j in range(len(scales))])
    return scales, x, np.asarray(y, dtype=np.float32), np.asarray(ref_lens, dtype=np.float32)


def train_model(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(x, y)
    return model


def train_length_model(
    rows: list[dict],
    x_len: np.ndarray,
    scale_model: RandomForestRegressor,
    x_scale: np.ndarray,
    scales: list[float],
    args: argparse.Namespace,
) -> tuple[RandomForestRegressor, dict]:
    pred_cost = np.asarray(scale_model.predict(x_scale), dtype=np.float32)
    chosen = choose_with_margin(pred_cost, scales, args.base_scale, args.final_switch_margin)
    raw_lengths = np.asarray([max(1.0, float(row.get("length", 0) or 0)) for row in rows], dtype=np.float32)
    chosen_scales = np.asarray([float(scales[int(i)]) for i in chosen], dtype=np.float32)
    target_lengths = np.maximum(1.0, raw_lengths * chosen_scales)
    length_model = RandomForestRegressor(
        n_estimators=args.length_trees,
        min_samples_leaf=args.length_min_leaf,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        random_state=args.seed + 101,
        n_jobs=args.n_jobs,
    )
    length_model.fit(x_len, np.log(target_lengths))
    fit_lengths = np.exp(np.asarray(length_model.predict(x_len), dtype=np.float32))
    abs_err = np.abs(fit_lengths - target_lengths)
    ratio = target_lengths / raw_lengths
    return length_model, {
        "target": "bt_policy_selected_scaled_total_len",
        "feature_mode": "text_only",
        "target_mean_len": float(target_lengths.mean()),
        "fit_mean_len": float(fit_lengths.mean()),
        "target_mean_ratio_to_manifest_len": float(ratio.mean()),
        "fit_mae_frames": float(abs_err.mean()),
        "fit_p50_abs_frames": float(np.percentile(abs_err, 50)),
        "fit_p90_abs_frames": float(np.percentile(abs_err, 90)),
        "chosen_scale_counts": dict(sorted(Counter(f"{scales[int(i)]:.2f}" for i in chosen).items())),
        "length_min": int(max(1, np.floor(np.percentile(target_lengths, 0.5)))),
        "length_max": int(max(4, np.ceil(np.percentile(target_lengths, 99.5)))),
    }


def choose_with_margin(pred: np.ndarray, scales: list[float], base_scale: float, margin: float) -> np.ndarray:
    best = pred.argmin(axis=1)
    if margin <= 0.0 or base_scale not in scales:
        return best
    base_idx = scales.index(base_scale)
    chosen = []
    for i, best_idx in enumerate(best):
        if float(pred[i, base_idx] - pred[i, best_idx]) >= margin:
            chosen.append(int(best_idx))
        else:
            chosen.append(base_idx)
    return np.asarray(chosen, dtype=np.int64)


def cross_validate(x: np.ndarray, y: np.ndarray, ref_lens: np.ndarray, scales: list[float], args: argparse.Namespace):
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed + 17)
    fixed = {f"{s:.2f}": float(100.0 * y[:, j].sum() / ref_lens.sum()) for j, s in enumerate(scales)}
    oof_pred = np.zeros_like(y, dtype=np.float32)
    for tr, te in kf.split(x):
        model = train_model(x[tr], y[tr], args)
        oof_pred[te] = np.asarray(model.predict(x[te]), dtype=np.float32)
    margin_grid = [float(x) for x in args.margin_grid.split(",") if x.strip()]
    if not margin_grid:
        margin_grid = [0.0]
    margin_records = []
    best_margin = margin_grid[0]
    best_err = None
    best_chosen = None
    for margin in margin_grid:
        chosen_arr = choose_with_margin(oof_pred, scales, args.base_scale, margin)
        err = float(y[np.arange(len(y)), chosen_arr].sum())
        margin_records.append({
            "margin": margin,
            "wer_proxy": float(100.0 * err / ref_lens.sum()),
            "scale_counts": dict(sorted(Counter(f"{scales[i]:.2f}" for i in chosen_arr).items())),
        })
        if best_err is None or err < best_err:
            best_err = err
            best_margin = margin
            best_chosen = chosen_arr
    counts = Counter(f"{scales[i]:.2f}" for i in best_chosen)
    return {
        "fixed_wer_proxy": fixed,
        "oracle_wer_proxy": float(100.0 * y.min(axis=1).sum() / ref_lens.sum()),
        "cv_policy_wer_proxy": float(100.0 * best_err / ref_lens.sum()),
        "best_switch_margin": best_margin,
        "cv_scale_counts": dict(sorted(counts.items())),
        "margin_sweep": margin_records,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--reference_pt", required=True)
    ap.add_argument("--train_manifest_json", default="data/phoenix/phoenix_train.json")
    ap.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    ap.add_argument("--out_policy", required=True)
    ap.add_argument("--out_report", required=True)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--min_leaf", type=int, default=10)
    ap.add_argument("--max_depth", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--base_scale", type=float, default=0.70)
    ap.add_argument(
        "--feature_mode",
        choices=["text_only", "legacy_gt_length"],
        default="text_only",
        help="Policy feature set. text_only is the deployable no-GT-length mode.",
    )
    ap.add_argument("--length_trees", type=int, default=300)
    ap.add_argument("--length_min_leaf", type=int, default=10)
    ap.add_argument(
        "--margin_grid",
        default="0,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0",
        help="Comma-separated predicted edit-cost margins for conservative switching.",
    )
    args = ap.parse_args()

    rows = ordered_rows(ROOT / args.manifest_json, ROOT / args.reference_pt)
    policy_base = build_policy_template(ROOT / args.train_manifest_json, rows, DEFAULT_LEXICON, args.feature_mode)
    candidates = sorted(args.candidate, key=lambda x: x[0])
    scales, x, y, ref_lens = make_xy(rows, candidates, policy_base)
    report = cross_validate(x, y, ref_lens, scales, args)
    args.final_switch_margin = float(report["best_switch_margin"])
    model = train_model(x, y, args)
    length_policy_base = {**policy_base, "feature_mode": "text_only"}
    x_len = np.asarray([duration_policy_features(row, length_policy_base) for row in rows], dtype=np.float32)
    length_model, length_report = train_length_model(rows, x_len, model, x, scales, args)
    report["length_model"] = length_report
    policy = {
        **policy_base,
        "model": model,
        "scales": scales,
        "base_scale": float(args.base_scale),
        "switch_margin": float(report["best_switch_margin"]),
        "feature_dim": int(x.shape[1]),
        "length_model": length_model,
        "length_feature_mode": "text_only",
        "length_feature_dim": int(x_len.shape[1]),
        "length_target": length_report["target"],
        "length_min": int(length_report["length_min"]),
        "length_max": int(length_report["length_max"]),
        "trainer": vars(args),
        "report": report,
    }
    out_policy = ROOT / args.out_policy
    out_policy.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(policy, out_policy)
    out_report = ROOT / args.out_report
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"policy -> {out_policy}", flush=True)
    print(f"report -> {out_report}", flush=True)


if __name__ == "__main__":
    main()
