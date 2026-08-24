# Reproducing AHA-JEPA

Every command below is transcribed from the driver scripts that produced the
paper's results. The source driver is named for each stage so you can check the
flags against the code.

---

## 0. Prerequisites

### 0.1 Environments

Two environments are required, because the SLRTP-2025 harness pins packages that
conflict with a modern PyTorch stack.

```bash
# (a) AHA-JEPA training environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, numpy, scikit-learn, joblib, matplotlib

# (b) SLRTP-2025 evaluation environment -- SEPARATE
conda create --name slrtp python=3.8 && conda activate slrtp
git clone https://github.com/walsharry/SLRTP-Sign-Production-Evaluation.git \
    external/SLRTP-Sign-Production-Evaluation
pip install -r external/SLRTP-Sign-Production-Evaluation/requirements.txt
```

The paper's results were produced against SLRTP harness commit `db35105`.

### 0.2 Data

None of the following is redistributed here. Obtain each from its licensor.

| Dataset | Source | Used for |
|---|---|---|
| SLRTP-2025 evaluation data + back-translation model | [SLRTP workshop](https://slrtpworkshop.github.io/) | training poses, evaluation, GT references |
| RWTH-PHOENIX-Weather-2014T | [RWTH Aachen](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) | gloss manifests |
| CSL-Daily | [USTC](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/) — requires signed agreement | second-corpus experiments |

Place the SLRTP release so that these paths resolve from the repo root:

```
external/SLRTP-Sign-Production-Evaluation/pretrained/
  SLRTP-Sign-Production-Evaluation-Data/
    data/train.pt          # ~1.7 GB
    data/dev.pt            # ~114 MB
    data/test.pt
    backTranslation_PHIX_model/
```

### 0.3 Exemplar bank (required — training reads nothing else)

Training does not read the raw corpus. `SLRTPSpanDataset` reads a single
exemplar bank, a dict with `exemplar_poses` (`{clip_id: [T, 178, 3]}`),
`gloss_to_exemplars` (`{gloss: [(clip_id, start, end), ...]}`), and
`exemplar_upc`.

```bash
python scripts/build_slrtp178_exemplars.py \
  --source_bank outputs/sota_chase/phase24_pgrast/exemplars_upc.pt \
  --slrtp_train external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt \
  --out outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt
```

> **Note.** `--source_bank` supplies the per-gloss span index. It comes from a
> CTC forced alignment over the training corpus and is **not included in this
> release**. To build a bank from scratch you need per-gloss spans from your own
> forced alignment over SLRTP train; the pose payload itself comes straight from
> `train.pt`. This is the single largest obstacle to a cold-start reproduction —
> please open an issue if you need the span index.

---

## 1. Sign-JEPA carrier pretraining

Source: `scripts/carrier_swap_ablation.sh:58`

```bash
python -u scripts/train_sign_jepa_slrtp178.py \
  --objective jepa \
  --epochs 12 --batch_size 64 \
  --bank outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt \
  --out_dir outputs/sota_chase/sign_jepa_slrtp178
```

Swap `--objective jepa` for `mae` or `ae` to build the carrier-swap baselines.
The three carriers are otherwise trained identically — that is the point of the
ablation.

---

## 2. Generator, base stage (v25)

Source: `scripts/carrier_swap_predictive_contract.sh:38-48, 69-81`

```bash
python -u scripts/train_sign_jepa_generator_slrtp178.py train \
  --out_dir outputs/sota_chase/sign_jepa_v25_stochastic \
  --jepa_ckpt outputs/sota_chase/sign_jepa_slrtp178/best.pt \
  --seed 0 --epochs 12 \
  --bank outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt \
  --batch_size 64 --lr 2e-4 \
  --lambda_pose 0.12 --lambda_vel 0.15 --lambda_desc 0.15 \
  --lambda_wavkl 0.75 --wavkl_scales 1,3,7,15 --wavkl_bins 64 \
  --wavkl_vmax 0.15 --wavkl_max_batches 30 \
  --use_energy_cond --lambda_energy_pred 0.1 --energy_desc_scale 0.01 \
  --use_sentence_budget --sentence_max_len 32 \
  --lambda_vel_dir 0.25 --vel_dir_wrist_weight 0.0 \
  --stochastic_decoder --init_log_var -4.0 --lambda_nll 1.0 \
  --latent_loss_mode both --latent_raw_weight 1.0 --latent_cov_weight 0.25 \
  --lambda_latent 1.5 \
  --lambda_jepa_pc 1.0 --jepa_pc_loss_mode both --jepa_pc_raw_weight 1.0 \
  --jepa_pc_min_context 16 --jepa_pc_target_len_min 8 --jepa_pc_target_len_max 24 \
  --num_workers 2 --log_every 400 --device cuda
```

### The predictive carrier contract

The `--lambda_jepa_pc` block is what makes the carrier objective load-bearing.
Without it, the pose/velocity/descriptor losses let an AE or MAE carrier reach
nearly the same downstream surface as JEPA, and the carrier-swap ablation
reports a tie. With it, masked future target latents must be predicted from
visible generated context *through the frozen carrier predictor* — a task JEPA
was pretrained for and AE/MAE were not.

Use `--latent_loss_mode both` (direction **and** magnitude). Direction-only
(`cosine`) discards carrier magnitude and rank, and the ablation goes flat.

---

## 3. Generator, adversarial stage (v29 — the deployed model)

Source: `scripts/carrier_swap_predictive_contract.sh`; hyperparameters confirmed
against the run header of `sign_jepa_v29_adv_train.log`
(`lambda_adv=0.05 d_lr=0.0002 d_hidden=64`, 8.12 M trainable parameters).

Identical to stage 2, plus `--init_from`, a lower LR, 3 epochs, and the critic:

```bash
python -u scripts/train_sign_jepa_generator_slrtp178.py train \
  --out_dir outputs/sota_chase/sign_jepa_v29_adv \
  --jepa_ckpt outputs/sota_chase/sign_jepa_slrtp178/best.pt \
  --init_from outputs/sota_chase/sign_jepa_v25_stochastic/best.pt \
  --seed 0 --epochs 3 --lr 5e-5 \
  --lambda_adv 0.05 --d_lr 2e-4 --d_hidden 64 \
  # ...all stage-2 flags unchanged...
```

---

## 4. Utterance-budget governor

Source: `scripts/carrier_swap_predictive_contract.sh:94-98`

```bash
python -u scripts/sign_jepa_utterance_budget_governor.py apply \
  --policy outputs/sota_chase/sign_jepa_v10_budget_governor/policy.pt \
  --manifest_json data/phoenix/phoenix_dev.json \
  --in_pt  outputs/sota_chase/sign_jepa_v29_adv_raw_dev.pt \
  --out_pt external/SLRTP-Sign-Production-Evaluation/results/aha_jepa_dev.pt \
  --min_gain 1.0 --max_gain 1.55 \
  --body_coupling 0.35 --max_body_gain 1.20 \
  --lp_kernel 1 --smooth_kernel 1
```

The governor is **train-fitted and bounded**: the policy is fit on training data
(`... governor.py fit`), and at inference it sees only the generated pose plus
that sentence's text/gloss metadata. It never sees ground-truth motion or
length. The gain bounds above are the ones used for every reported number.

---

## 5. SLRTP-2025 evaluation

Source: `scripts/carrier_swap_predictive_contract.sh:100-103`

Run in the **SLRTP environment**, not the training one:

```bash
cd external/SLRTP-Sign-Production-Evaluation
python -u main.py \
  results/aha_jepa_dev.pt \
  pretrained/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt \
  pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model \
  --tag aha_jepa_dev
```

Input is `{clip_id: [T, 178, 3]}`. The harness downsamples 25 fps to 12 fps
internally. Scores land in `results/aha_jepa_dev.json`.

Report lexical metrics (BLEU-4, WER) **together with** motion statistics. A
system can win WER by barely moving; that is the failure mode this paper is
about.

---

## 6. Carrier-swap ablation (the paper's central experiment)

One driver runs all three carriers end to end for a given seed:

```bash
scripts/carrier_swap_predictive_contract.sh 0 jepa mae ae
```

Variants, each a separate driver in `scripts/`:

| Driver | What it isolates |
|---|---|
| `carrier_swap_predictive_contract.sh` | full contract (the headline) |
| `carrier_swap_no_contract.sh` | contract removed — carriers tie |
| `carrier_swap_richcov.sh` / `carrier_swap_richlatent.sh` | latent anchor richness |
| `carrier_swap_parity.sh` | matched-budget parity check |
| `carrier_swap_min.sh` | minimal-loss surface |
| `carrier_swap_multiseed.sh` | seed variance |
| `carrier_swap_ensemble.sh` | ensemble rows |

---

## 7. CSL-Daily

Source: `scripts/aha_jepa_csl_chain.sh:25-67`

```bash
# bank
python scripts/build_csl_jepa_bank.py

# carrier
python -u scripts/aha_jepa_csl.py jepa \
  --out_dir outputs/sota_chase/sign_jepa_csl \
  --epochs 12 --batch_size 64 --objective jepa

# generator, base
python -u scripts/aha_jepa_csl.py gen \
  --jepa_ckpt outputs/sota_chase/sign_jepa_csl/best.pt \
  --out_dir outputs/sota_chase/aha_csl_v25 \
  --epochs 12 --batch_size 64 \
  --stochastic_decoder --lambda_nll 1.0 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_vmax 60 \
  --use_energy_cond --use_sentence_budget --lambda_energy_pred 0.1

# generator, adversarial
python -u scripts/aha_jepa_csl.py gen \
  --jepa_ckpt outputs/sota_chase/sign_jepa_csl/best.pt \
  --out_dir outputs/sota_chase/aha_csl_v29 \
  --init_from outputs/sota_chase/aha_csl_v25/best.pt \
  --epochs 6 --batch_size 64 --lambda_adv 0.05 \
  --stochastic_decoder --lambda_nll 1.0 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_vmax 60 \
  --use_energy_cond --use_sentence_budget --lambda_energy_pred 0.1

# generate
python -u scripts/aha_jepa_csl.py generate \
  --ckpt outputs/sota_chase/aha_csl_v29/best.pt --split dev --tag aha
```

`--wavkl_vmax 60` for CSL-Daily versus `0.15` for PHOENIX: the two corpora use
different pose scalings, and the WAV-KL histogram range must match the corpus.
Using the PHOENIX value on CSL-Daily silently collapses the histogram.

CSL-Daily is scored with gloss-WER via an MSKA-SLR recognizer, not with the
SLRTP harness. Sentence-level text BLEU is not meaningful on our CSL-Daily
setup and is not reported.

---

## Known gaps

| Artifact | Status |
|---|---|
| Trained checkpoints (carrier, v25, v29) | not retained — retrain via §1–3 |
| Governor policy `policy.pt` | not retained — refit with `governor.py fit` |
| Phase-40 exemplar bank | build via §0.3 |
| Phase-24 span index (`--source_bank`) | not released — needs your own forced alignment |
| SLRTP / PHOENIX / CSL-Daily data | obtain from licensors (§0.2) |

Issues and questions: <https://github.com/kumwilai/Aha-Jepa/issues>
