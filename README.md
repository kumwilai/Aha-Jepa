# AHA-JEPA

**Articulated Hand-Aware JEPA for Leak-Free Gloss-Conditioned Sign Pose Generation**

Official implementation. IEEE Access, 2026 (accepted).

Wuttipong Kumwilaisak, *Senior Member, IEEE* — Department of Electronic and
Telecommunication Engineering, King Mongkut's University of Technology Thonburi
(KMUTT), Bangkok, Thailand.

---

## Overview

Sign Language Production (SLP) converts text or glosses into sign-pose motion.
We study the gloss-conditioned stage on RWTH-PHOENIX-Weather-2014T under the
SLRTP-2025 harness, where retrieval/assembly systems and ground-truth-length
shortcuts can inflate back-translation scores and complicate pure-generator
comparison.

We identify **hand-articulation collapse**: framewise training can remove
hand-motion energy and trajectory variation, while standard automatic metrics
may either degrade or fail to reveal the loss.

**AHA-JEPA** is a retrieval-free, ground-truth-length-free generator that decodes
a gloss-conditioned latent program trained against a frozen Sign-JEPA teacher,
with a heteroscedastic decoder, an adversarial hand critic, hand-velocity
distribution matching (WAV-KL), text/gloss energy conditioning, and a bounded
train-fitted governor.

## What we claim, and what we do not

In a corrected length-locked audit, a near-static carrier-anchored base generator
attains similar WER/BLEU to AHA-JEPA (WER 81.80 vs. 82.26; BLEU-4 12.33 vs.
11.71) while moving only 0.391x as much as ground truth. AHA-JEPA restores total
motion to 0.839x and median hand speed/travel to 1.01x / 0.83x.

Paired bootstraps show **no statistically resolved WER/BLEU difference** between
AHA-JEPA and the low-motion carrier, whereas DTW-MJE significantly favors the
carrier. This is a **metric/motion trade-off, not a metric win**. We therefore
report lexical metrics together with motion statistics, and treat deaf-signer
perceptual validation as complementary future work.

## Reproducibility status - read this first

This repository ships **complete source code** for every stage of the pipeline,
the carrier-swap and ablation drivers, and **six trained checkpoints** covering
both corpora (see [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)).

Please read these three caveats before quoting anything.

1. **The published checkpoints are retrained, not the paper's originals.** The
   weight files behind the paper's tables were not retained. The ones here were
   produced from this code at `--seed 0` with the paper's recipe, and have
   **not** been scored on the SLRTP harness. Treat them as a working starting
   point, not as evidence for a specific number.
2. **The exemplar bank is the one artifact you must supply.** Training reads
   nothing else. Building it needs frame-level CTC posteriors from a sign
   recognizer trained on the corpus; ours are not redistributable. Any CTC
   recognizer works, since only the Viterbi span boundaries are used. Both build
   steps are scripted - see `docs/CHECKPOINTS.md`.
3. **No datasets.** PHOENIX-2014T, CSL-Daily and the SLRTP-2025 evaluation data
   are distributed by their own licensors under their own terms. See
   [docs/REPRODUCING.md](docs/REPRODUCING.md) for how to obtain each.

Retraining is cheap: **the full PHOENIX chain takes about 22 minutes** on a
single RTX 5070 Ti (measured, not estimated). Once you have a bank, training
your own is usually easier than starting from ours.

## Install

```bash
git clone git@github.com:kumwilai/Aha-Jepa.git
cd aha-jepa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 and a CUDA GPU with at least 16 GB VRAM. All results in the paper
were produced on a single RTX 5070 Ti.

Scripts resolve paths from the repository root automatically. To run against a
tree laid out elsewhere, set `AHA_JEPA_ROOT`:

```bash
export AHA_JEPA_ROOT=/path/to/your/tree
```

The SLRTP-2025 evaluation harness pins older packages (`torchtext==0.5.0`) and
must be installed in a **separate** environment — see `docs/REPRODUCING.md`.

## Pipeline

```
SLRTP-178 exemplar bank
        |
        v
[1] Sign-JEPA carrier pretraining        scripts/train_sign_jepa_slrtp178.py
        |                                 (--objective jepa | mae | ae)
        v
[2] generator, base stage (v25)          scripts/train_sign_jepa_generator_slrtp178.py
        |   heteroscedastic decoder, WAV-KL, energy conditioning,
        |   sentence budget, predictive carrier contract
        v
[3] generator, adversarial stage (v29)   same script, --init_from v25 --lambda_adv 0.05
        |
        v
[4] utterance-budget governor            scripts/sign_jepa_utterance_budget_governor.py
        |
        v
[5] SLRTP-2025 back-translation eval     external harness, main.py
```

## Quick start

Full, verified command lines for every stage — including the CSL-Daily chain and
the carrier-swap ablation — are in **[docs/REPRODUCING.md](docs/REPRODUCING.md)**.

The end-to-end PHOENIX chain for one seed is a single driver:

```bash
scripts/carrier_swap_predictive_contract.sh 0 jepa mae ae
```

This trains the carrier for each objective, trains the generator with the
predictive carrier contract, applies the governor, and scores every row with the
SLRTP harness. It is the script that produced the paper's carrier-swap table.

## Repository layout

```
scripts/
  train_sign_jepa_slrtp178.py             Sign-JEPA carrier (jepa/mae/ae objectives)
  train_sign_jepa_generator_slrtp178.py   gloss-conditioned generator + hand critic
  sign_jepa_utterance_budget_governor.py  bounded train-fitted motion governor
  aha_jepa_csl.py                         CSL-Daily chain (jepa/gen/generate)
  build_slrtp178_exemplars.py             exemplar-bank builder
  build_csl_jepa_bank.py                  CSL-Daily bank builder
  carrier_swap_*.sh                       carrier-swap ablation drivers
  jepa_latent*_figure.py                  paper figures
  ...                                     full experimental arc, negative results included
docs/
  REPRODUCING.md                          data acquisition + verified commands
```

The `scripts/` directory deliberately retains the **full experimental arc**,
including variants that did not work. The paper's negative results (bounded
post-hoc refiners, latent-noise decoders, velocity-field refiners) are all
reproducible from the code here.

## Citation

```bibtex
@article{kumwilaisak2026ahajepa,
  author  = {Kumwilaisak, Wuttipong},
  title   = {{AHA-JEPA}: Articulated Hand-Aware {JEPA} for Leak-Free
             Gloss-Conditioned Sign Pose Generation},
  journal = {IEEE Access},
  year    = {2026}
}
```

## Acknowledgment

Supported by the Mid-Career Research Fund through the National Research Council
of Thailand (NRCT) under Grant N42A670175.

Evaluation uses the [SLRTP-2025 Sign Production Evaluation
harness](https://github.com/walsharry/SLRTP-Sign-Production-Evaluation) from the
3rd SLRTP Workshop at CVPR 2025.

## Licence

MIT — see [LICENSE](LICENSE).
