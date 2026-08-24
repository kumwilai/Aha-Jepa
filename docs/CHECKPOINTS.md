# Checkpoints

Six checkpoints ship in this repository under `release/`, covering both corpora.
They are committed directly rather than attached as a release asset, so a plain
`git clone` gets them (the clone is ~205 MB as a result).

| File | Stage | Params | Size |
|---|---|---|---|
| `phoenix_carrier.pt` | Sign-JEPA carrier, PHOENIX | 9.46 M | 37.9 MB |
| `phoenix_v25.pt` | generator, base stage | 8.12 M | 32.6 MB |
| `phoenix_v29_deployed.pt` | generator + adversarial critic | 8.12 M | 33.2 MB |
| `csl_carrier.pt` | Sign-JEPA carrier, CSL-Daily | 9.72 M | 39.0 MB |
| `csl_v25.pt` | generator, base stage | — | 35.1 MB |
| `csl_v29_deployed.pt` | generator + adversarial critic | — | 35.8 MB |

Verify with:

```bash
cd release && sha256sum -c SHA256SUMS.txt
```

## What these are, and what they are not

These are **retrained reference checkpoints**, produced from this repository's
code on 2026-08-24. They are **not** the exact weight files behind the tables in
the paper — those were not retained. Do not quote a number as "the paper's
result" on the basis of these files.

They were trained with the paper's recipe, at `--seed 0`, from an exemplar bank
rebuilt with the same forced-alignment procedure (see below). They have **not**
been scored on the SLRTP harness, because the evaluation path needs a duration
policy and a governor policy that were also not retained. Expect them to land
near the paper's operating point; treat any specific figure as unverified until
you run the evaluation yourself.

What they are good for: a working starting point. The carrier loads, the
generator loads, the adversarial stage runs, and you can warm-start or ablate
from them instead of beginning at random initialisation.

## Training cost (measured, not estimated)

On a single RTX 5070 Ti, 52 172 training spans:

| Stage | Epochs | s/epoch | Total |
|---|---|---|---|
| PHOENIX carrier | 12 | ~24 | ~7 min |
| PHOENIX v25 | 12 | ~52 | ~11 min |
| PHOENIX v29 | 3 | ~70 | ~4 min |
| CSL carrier | 12 | ~74 | ~15 min |
| CSL v25 | 12 | ~95 | ~20 min |
| CSL v29 | 6 | ~139 | ~14 min |

**The full PHOENIX chain is about 22 minutes.** Retraining is cheap enough that
these checkpoints are a convenience rather than a prerequisite — if you have the
exemplar bank, just train your own.

## Rebuilding the exemplar bank

Training reads only the exemplar bank, so this is the one artifact you must have.
It is built in two steps:

```bash
# 1. per-gloss spans, via CTC forced alignment over a recognizer's posteriors
python scripts/build_pergloss_exemplars.py \
  --out_path outputs/sota_chase/phase24_pgrast/exemplars.pt

# 2. swap the pose payload for native SLRTP-178 poses_3d
python scripts/build_slrtp178_exemplars.py \
  --source_bank outputs/sota_chase/phase24_pgrast/exemplars.pt \
  --slrtp_train external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt \
  --out outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt
```

Step 1 needs frame-level CTC posteriors over the training split. We used a
CorrNet recognizer; **any** CTC sign recognizer trained on the same corpus will
serve, since only the Viterbi span boundaries are consumed. Our posterior cache
is not redistributable, so this step requires your own recognizer. Everything
downstream of it is fully scripted.

The `exemplar_upc` field is optional — `build_slrtp178_exemplars.py` reads it
via `.get()` and the training loader ignores it entirely. A UPC codebook is
**not** required to build a usable bank.

Reference figures from our rebuild: 7096/7096 clips aligned via CTC with zero
proportional fallbacks, 1085 glosses / 55 247 exemplars, converting to 1081
glosses / 54 930 exemplars natively at 178 keypoints (2.6 GB).

## A note on generation and length leakage

`sample_manifest` refuses to use the manifest's ground-truth length by default:

```
ValueError: sample_manifest no longer uses manifest row['length'] by default.
Pass a duration policy containing a text-only length_model, or use
--allow_gt_length_base for an explicit legacy/leaky diagnostic run.
```

This guard is deliberate and you should not disable it for anything you intend
to report. Ground-truth length is a strong shortcut, and a run that uses it is
not comparable with a pure-generation result.

To generate legitimately you need a text-only duration policy
(`scripts/train_jepa_duration_policy.py --feature_mode text_only`). Fitting one
bootstraps through a `legacy_gt_length` sweep at **training** time, which is
sound — the deployed policy sees only text — but it is a separate multi-stage
job that is not included here.
