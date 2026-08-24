"""Build a Sign-JEPA exemplar bank for CSL-Daily (HRNet-133).

Produces the same on-disk structure that `load_bank` in
`train_sign_jepa_slrtp178.py` expects:

    bank["gloss_to_exemplars"] = {gloss: [(sid, start, end), ...]}
    bank["exemplar_poses"]     = {sid: np.ndarray[T, 133, 3] float32}

Source data:
  * per-clip CTC gloss spans   data/csl-daily/gloss_spans_ctc_train.json
  * HRNet-133 train poses (MSKA) external/baselines/MSKA/data/CSL-Daily/CSL-Daily.train

This is TRAIN-only (the JEPA / generator never see dev or test poses).
"""
from __future__ import annotations

import json
import pickle
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
SPANS = ROOT / "data/csl-daily/gloss_spans_ctc_train.json"
TRAIN_PKL = ROOT / "external/baselines/MSKA/data/CSL-Daily/CSL-Daily.train"
OUT = ROOT / "outputs/sota_chase/csl_jepa_bank.pt"
CLAMP_PX = 400.0   # per-keypoint, per-clip robust window (kills tracking-failure outliers)


def robust_clamp(kp: np.ndarray) -> np.ndarray:
    """Clamp x,y of each keypoint to its per-clip median +/- CLAMP_PX.

    The raw MSKA HRNet-133 train poses contain tracking-failure frames with
    coordinates up to ~1e6 px (18% of clips). Real per-keypoint motion over a
    clip is < ~400 px, so clamping to the keypoint's own temporal median +/-
    400 px bounds the garbage near the plausible pose range without distorting
    real motion. The confidence channel (index 2) is left untouched.
    """
    out = kp.copy()
    med = np.median(kp[:, :, :2], axis=0, keepdims=True)   # [1, 133, 2]
    lo = med - CLAMP_PX
    hi = med + CLAMP_PX
    out[:, :, :2] = np.clip(kp[:, :, :2], lo, hi)
    return out


def main() -> None:
    spans_by_sid = json.load(open(SPANS))
    print(f"[bank] {len(spans_by_sid)} clips with CTC spans", flush=True)
    train = pickle.load(open(TRAIN_PKL, "rb"))
    print(f"[bank] {len(train)} train clips in MSKA pickle", flush=True)

    gloss_to_exemplars: dict[str, list] = {}
    exemplar_poses: dict[str, np.ndarray] = {}
    n_spans = 0
    n_skip = 0
    for sid, spans in spans_by_sid.items():
        if sid not in train:
            n_skip += 1
            continue
        kp = train[sid]["keypoint"]
        if torch.is_tensor(kp):
            kp = kp.detach().cpu().numpy()
        kp = np.asarray(kp, dtype=np.float32)
        if kp.ndim != 3 or kp.shape[1] != 133:
            n_skip += 1
            continue
        kp = robust_clamp(kp)
        kept_here = False
        for sp in spans:
            g = str(sp["gloss"])
            s = int(sp["start"])
            e = int(sp["end"])
            if e - s < 3:
                continue
            gloss_to_exemplars.setdefault(g, []).append((sid, s, e))
            n_spans += 1
            kept_here = True
        if kept_here:
            exemplar_poses[sid] = kp

    bank = {"gloss_to_exemplars": gloss_to_exemplars, "exemplar_poses": exemplar_poses}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, OUT)
    sizes = [v.shape[0] for v in exemplar_poses.values()]
    print(
        f"[bank] glosses={len(gloss_to_exemplars)} spans={n_spans} "
        f"clips={len(exemplar_poses)} skipped={n_skip}",
        flush=True,
    )
    print(
        f"[bank] clip T mean={np.mean(sizes):.1f} "
        f"p10/50/90={np.percentile(sizes, [10, 50, 90]).tolist()}",
        flush=True,
    )
    print(f"[bank] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
