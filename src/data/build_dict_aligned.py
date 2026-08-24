"""
Build a Phoenix sign dictionary with expanded CTC-aligned per-gloss spans.

Uses CoopNS-SLR's cached CorrNet CTC posteriors to Viterbi-align each
training sample's gloss sequence to its pose frames, then extracts
per-gloss spans WITH a small margin (±2 frames) to keep natural
lead-in/hold/lead-out that v2's too-tight core-extraction threw away.

Output: data/phoenix/sign_dictionary_aligned.npz
Format matches load_sign_dictionary (v1 interface).
"""

import argparse
import os
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def ctc_forced_align(
    log_probs: np.ndarray, target_indices: List[int], blank: int = 0,
) -> List[Tuple[int, int]]:
    """CTC Viterbi forced alignment. Returns [(start, end_exclusive)] per target."""
    T, V = log_probs.shape
    N = len(target_indices)
    if N == 0 or T == 0:
        return []

    extended = [blank]
    for idx in target_indices:
        extended.append(idx)
        extended.append(blank)
    L = len(extended)

    NEG_INF = -1e18
    dp = np.full((T, L), NEG_INF, dtype=np.float64)
    bp = np.zeros((T, L), dtype=np.int32)

    dp[0, 0] = log_probs[0, extended[0]]
    if L > 1:
        dp[0, 1] = log_probs[0, extended[1]]

    for t in range(1, T):
        for s in range(L):
            emit = log_probs[t, extended[s]]
            best = dp[t - 1, s]
            back = 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best = dp[t - 1, s - 1]
                back = 1
            if (s >= 2 and extended[s] != blank and extended[s] != extended[s - 2]
                    and dp[t - 1, s - 2] > best):
                best = dp[t - 1, s - 2]
                back = 2
            dp[t, s] = best + emit
            bp[t, s] = back

    s = L - 1 if (L >= 2 and dp[T - 1, L - 1] >= dp[T - 1, L - 2]) else (L - 2 if L >= 2 else 0)

    spans = [[None, None] for _ in range(N)]
    t = T - 1
    while t >= 0:
        ext_s = extended[s]
        if ext_s != blank:
            target_i = (s - 1) // 2
            if 0 <= target_i < N:
                if spans[target_i][1] is None:
                    spans[target_i][1] = t + 1  # exclusive end
                spans[target_i][0] = t
        if t == 0:
            break
        back = bp[t, s]
        s -= back
        t -= 1

    result = []
    for i, (start, end) in enumerate(spans):
        if start is None:
            start = int(i * T / N)
            end = int((i + 1) * T / N)
        result.append((start, end if end > start else start + 1))
    return result


def resample_segment(seg: np.ndarray, target_len: int) -> np.ndarray:
    if len(seg) == target_len:
        return seg.astype(np.float32)
    if len(seg) < 2:
        return np.repeat(seg, target_len, axis=0).astype(np.float32)
    src_t = np.linspace(0, 1, len(seg))
    tgt_t = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, seg.shape[1]), dtype=np.float32)
    for d in range(seg.shape[1]):
        out[:, d] = np.interp(tgt_t, src_t, seg[:, d])
    return out


def build(
    manifest_path: str,
    sequences_dir: str,
    posterior_pkl: str,
    output_path: str,
    motion_dim: int = 201,
    margin: int = 2,
    min_span: int = 6,
    normalizer_path: str = "data/phoenix/normalizer.npz",
):
    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loading normalizer")
    norm = np.load(normalizer_path)
    norm_mean = norm["mean"][:motion_dim].astype(np.float32)
    norm_std = norm["std"][:motion_dim].astype(np.float32)

    print(f"Loading CorrNet cached posteriors")
    with open(posterior_pkl, "rb") as f:
        posteriors = pickle.load(f)
    print(f"  {len(posteriors)} posterior entries")

    gloss_segments: Dict[str, List[np.ndarray]] = defaultdict(list)
    aligned = 0
    prop_fallback = 0
    missing_posterior = 0

    for sample in manifest:
        sid = sample["id"]
        glosses = sample.get("gloss", "").split()
        if not glosses:
            continue

        npy_path = Path(sequences_dir) / f"{sid}.npy"
        if not npy_path.exists():
            continue
        motion = np.load(npy_path)[:, :motion_dim].astype(np.float32)
        motion = (motion - norm_mean) / norm_std
        T_pose = len(motion)
        K = len(glosses)

        spans = None
        post_key = f"train/{sid}"
        entry = posteriors.get(post_key)
        if entry is not None:
            try:
                log_probs = entry["log_probs"]
                gloss_indices = entry["gloss_indices"]
                if len(gloss_indices) == K:
                    post_spans = ctc_forced_align(log_probs, gloss_indices, blank=0)
                    T_post = log_probs.shape[0]
                    ratio = T_pose / T_post
                    # Expand each span by ±margin frames at the native resolution
                    raw = [(int(s * ratio), int(e * ratio)) for s, e in post_spans]
                    spans = [
                        (max(0, s - margin), min(T_pose, e + margin))
                        for s, e in raw
                    ]
                    aligned += 1
            except Exception as exc:
                spans = None

        if spans is None:
            spans = [(int(i * T_pose / K), int((i + 1) * T_pose / K)) for i in range(K)]
            prop_fallback += 1
            if entry is None:
                missing_posterior += 1

        for i, gloss in enumerate(glosses):
            start, end = spans[i]
            # Ensure minimum span length
            if end - start < min_span:
                mid = (start + end) / 2
                start = max(0, int(mid - min_span / 2))
                end = min(T_pose, int(mid + min_span / 2))
            if end - start < 2:
                continue
            gloss_segments[gloss].append(np.ascontiguousarray(motion[start:end]))

    del posteriors

    print(f"\nAlignment: {aligned} CTC-aligned, {prop_fallback} proportional fallback "
          f"({missing_posterior} missing posteriors)")

    print(f"\nBuilding templates for {len(gloss_segments)} glosses...")
    dictionary: Dict[str, np.ndarray] = {}
    durations: Dict[str, int] = {}
    stats = Counter()

    for gloss, segs in gloss_segments.items():
        if len(segs) < 2:
            continue
        # Median duration across collected spans
        lens = [len(s) for s in segs]
        median_len = max(min_span, int(np.median(lens)))
        durations[gloss] = median_len

        # Resample each to median_len, then mean
        resampled = np.stack(
            [resample_segment(s, median_len) for s in segs], axis=0,
        )
        template = resampled.mean(axis=0).astype(np.float32)
        dictionary[gloss] = template
        stats["built"] += 1

    print(f"  templates built: {stats['built']}")
    dur_vals = list(durations.values())
    print(f"  durations: min={min(dur_vals)} median={int(np.median(dur_vals))} max={max(dur_vals)}")

    # Save v1-compatible format
    save_dict = {f"template_{g}": t for g, t in dictionary.items()}
    save_dict["_glosses"] = np.array(list(dictionary.keys()))
    save_dict["_durations"] = np.array(
        [durations[g] for g in dictionary.keys()], dtype=np.int32,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved → {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/phoenix/phoenix_train.json")
    p.add_argument("--sequences_dir", default="data/phoenix/sequences/train")
    p.add_argument("--posterior_pkl",
                   default=os.environ.get("CORRNET_POSTERIORS",
                                       "outputs/cached_posteriors_corrnet/train.pkl"))
    p.add_argument("--out", default="data/phoenix/sign_dictionary_aligned.npz")
    p.add_argument("--margin", type=int, default=2,
                   help="frames to expand each CTC span on each side (v2 was 0 → too tight)")
    p.add_argument("--min_span", type=int, default=6,
                   help="floor on span length to avoid 2-frame stubs for short signs")
    args = p.parse_args()

    build(
        manifest_path=args.manifest,
        sequences_dir=args.sequences_dir,
        posterior_pkl=args.posterior_pkl,
        output_path=args.out,
        margin=args.margin,
        min_span=args.min_span,
    )


if __name__ == "__main__":
    main()
