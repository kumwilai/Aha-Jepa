"""Build per-gloss exemplar bank for PG-RAST.

Reuses the CTC-Viterbi alignment logic from build_dict_aligned.py but saves
PER-CLIP per-gloss spans instead of a mean template. Result is a dict
mapping each gloss to a list of (sample_id, start_frame, end_frame) tuples
plus an indexed .pt of pose segments (z-scored PT-201).

For each dev clip, we'll retrieve a real exemplar per gloss at its NATIVE
duration (no time-stretch within the gloss span — preserves motion energy).

Output:
  outputs/sota_chase/phase24_pgrast/exemplars.pt
  format: {
      "gloss_to_exemplars": {gloss: [(sid, start, end), ...]},
      "exemplar_poses":     {sid: pose_201_zscored fp16},
      "n_glosses":          int,
      "n_exemplars":        int,
  }

Usage
  python -u scripts/build_pergloss_exemplars.py
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("AHA_JEPA_ROOT",
                          Path(__file__).resolve().parents[1]))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.data.build_dict_aligned import ctc_forced_align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/phoenix/phoenix_train.json")
    ap.add_argument("--sequences_dir", default="data/phoenix/sequences/train")
    ap.add_argument("--posterior_pkl",
                    default=os.environ.get("CORRNET_POSTERIORS",
                                       "outputs/cached_posteriors_corrnet/train.pkl"))
    ap.add_argument("--normalizer_path", default="data/phoenix/normalizer.npz")
    ap.add_argument("--out_path", default="outputs/sota_chase/phase24_pgrast/exemplars.pt")
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--min_span", type=int, default=4)
    args = ap.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f"  {len(manifest)} train clips")

    print(f"Loading normalizer")
    norm = np.load(args.normalizer_path)
    n_mean = norm["mean"][:201].astype(np.float32)
    n_std = np.maximum(norm["std"][:201].astype(np.float32), 1e-6)

    print(f"Loading CorrNet posteriors (1 GB) ...")
    t0 = time.time()
    with open(args.posterior_pkl, "rb") as f:
        posteriors = pickle.load(f)
    print(f"  {len(posteriors)} entries in {time.time()-t0:.1f}s")

    gloss_to_exemplars: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    exemplar_poses: dict[str, np.ndarray] = {}
    aligned = 0; prop_fallback = 0; n_skipped = 0

    print(f"\nAligning {len(manifest)} clips ...")
    t0 = time.time()
    for i, sample in enumerate(manifest):
        sid = sample["id"]
        glosses = sample.get("gloss", "").split()
        if not glosses:
            continue
        npy = Path(args.sequences_dir) / f"{sid}.npy"
        if not npy.exists():
            n_skipped += 1; continue
        motion = np.load(npy)[:, :201].astype(np.float32)
        z_motion = (motion - n_mean) / n_std
        T_pose = z_motion.shape[0]
        K = len(glosses)

        spans = None
        entry = posteriors.get(f"train/{sid}")
        if entry is not None:
            try:
                log_probs = entry["log_probs"]
                gloss_indices = entry["gloss_indices"]
                if len(gloss_indices) == K:
                    post_spans = ctc_forced_align(log_probs, gloss_indices, blank=0)
                    T_post = log_probs.shape[0]
                    ratio = T_pose / T_post
                    raw = [(int(s * ratio), int(e * ratio)) for s, e in post_spans]
                    spans = [
                        (max(0, s - args.margin), min(T_pose, e + args.margin))
                        for s, e in raw
                    ]
                    aligned += 1
            except Exception:
                spans = None
        if spans is None:
            spans = [(int(j * T_pose / K), int((j + 1) * T_pose / K)) for j in range(K)]
            prop_fallback += 1

        # Save per-gloss spans referencing this clip
        kept = False
        for j, gloss in enumerate(glosses):
            s, e = spans[j]
            if e - s < args.min_span:
                mid = (s + e) // 2
                s = max(0, mid - args.min_span // 2)
                e = min(T_pose, mid + args.min_span // 2)
            if e - s < 2:
                continue
            gloss_to_exemplars[gloss].append((sid, int(s), int(e)))
            kept = True
        if kept:
            # Store this clip's full z-scored pose once (fp16 for compactness)
            if sid not in exemplar_poses:
                exemplar_poses[sid] = z_motion.astype(np.float16)
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(manifest)}]  aligned={aligned} prop_fb={prop_fallback}  "
                  f"({time.time()-t0:.0f}s)")

    del posteriors

    n_exemplars = sum(len(v) for v in gloss_to_exemplars.values())
    n_glosses = len(gloss_to_exemplars)
    print(f"\n=== exemplar bank ===")
    print(f"  glosses: {n_glosses}")
    print(f"  exemplars: {n_exemplars}")
    print(f"  unique source clips: {len(exemplar_poses)}")
    print(f"  aligned via CTC: {aligned}, fallback: {prop_fallback}, skipped: {n_skipped}")
    counts = sorted([len(v) for v in gloss_to_exemplars.values()])
    print(f"  per-gloss exemplar count p10/50/90: "
          f"{np.percentile(counts, [10, 50, 90])}")

    # Save
    out = {
        "gloss_to_exemplars": dict(gloss_to_exemplars),
        "exemplar_poses": exemplar_poses,
        "n_glosses": n_glosses,
        "n_exemplars": n_exemplars,
    }
    torch.save(out, out_path)
    print(f"\n  saved → {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
