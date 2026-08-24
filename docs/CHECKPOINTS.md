# Checkpoints

Seven checkpoints ship in this repository under `release/`, covering both
corpora. They are committed directly rather than attached as a release asset, so
a plain `git clone` gets them (the clone is ~235 MB as a result).

| File | Stage | Size |
|---|---|---|
| `phoenix_carrier.pt` | Sign-JEPA carrier, PHOENIX | 37.9 MB |
| `phoenix_v24a.pt` | generator + velocity-direction (fingers) | 32.0 MB |
| `phoenix_v25.pt` | generator + heteroscedastic decoder | 32.6 MB |
| `phoenix_v29_deployed.pt` | generator + adversarial hand critic | 33.2 MB |
| `csl_carrier.pt` | Sign-JEPA carrier, CSL-Daily | 39.0 MB |
| `csl_v25.pt` | generator, base stage | 35.1 MB |
| `csl_v29_deployed.pt` | generator + adversarial hand critic | 35.8 MB |

Verify with:

```bash
cd release && sha256sum -c SHA256SUMS.txt
```

## Measured scores

PHOENIX checkpoints scored on the **SLRTP-2025 harness** (`backTranslation_PHIX_model`,
beam 3, 12 fps), dev n=515 / test n=641, seed 0. Generation is fully leak-free:
lengths come from a text-only duration policy, and ground-truth length is never
used at inference.

| arm | BLEU-4 | BLEU-1 | CHRF | ROUGE | WER | DTW-MJE | TotDist | AvgDur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev, raw | 10.513 | 29.950 | 30.299 | 34.160 | 81.852 | 0.03690 | 0.885 | 0.752 |
| dev, governed | 10.511 | 29.856 | 30.237 | 34.189 | 81.807 | 0.03710 | 0.897 | 0.752 |
| test, governed | 10.864 | 30.535 | 30.967 | 34.395 | 81.994 | 0.03708 | 0.866 | 0.745 |

Pose-standard-deviation ratio against ground truth is **0.810 / 0.830 / 0.834**
respectively. We report this alongside every score deliberately: a sign-pose
generator can improve a back-translation metric by moving less, and a score
quoted without a variance figure hides that failure mode.

The CSL-Daily checkpoints are **not** scored here. CSL-Daily is evaluated by
gloss-WER through an MSKA-SLR recognizer rather than the SLRTP harness, and that
scoring path is not included in this release.

## These do not reproduce the paper

The paper reports test BLEU-4 11.71 / WER 82.26 for AHA-JEPA, against a
low-motion carrier baseline of BLEU-4 12.33 / WER 81.80.

These checkpoints reach **test BLEU-4 10.864 / WER 81.994**. WER lands close to
the reported figure; **BLEU-4 remains about 0.85 short**, and further below the
carrier baseline. One metric agreeing is not a reproduction, and these files
should not be cited as reproducing the paper's table.

They are a close reconstruction, trained at `--seed 0` from this repository's
code, on an exemplar bank rebuilt with the same forced-alignment procedure. One
input could not be recovered: the original Sign-JEPA carrier the deployed model
was anchored to no longer exists, so the lineage here is anchored to a retrained
substitute. The generator weights dominate, but the latent anchor target is not
identical to the one behind the published numbers.

Remaining differences visible in the artifacts: our clips are shorter and
jerkier than the recorded deployed model (AvgDur 0.745 vs 0.808, hand-jerk ratio
1.67 vs 1.39). This recognizer rewards smooth, longer output, which is the most
plausible source of the residual BLEU shortfall.

## Which recipe produced these - read this before retraining

**The deployed model is the tail of a staged curriculum, not a single training
run.** The chain is:

```
generator_slrtp178 -> v8_wavkl -> v9_energy_budget -> v11_sentence_budget
                   -> v24a_vel_dir_fingers (3 ep)
                   -> v25_stochastic       (3 ep)
                   -> v29_adv              (3 ep)
```

Each stage warm-starts from the previous one and runs only a few epochs. Training
`v25` from scratch instead discards the whole curriculum.

**Do not take the generator flags from `scripts/carrier_swap_*.sh`.** Those
scripts implement the carrier-swap *ablation*, which deliberately uses a
different loss balance and adds a predictive carrier contract. The deployed
lineage uses neither. The difference is large - training with the ablation flags
instead of the deployed ones cost **4.7 WER and 2.0 BLEU-4** in our own
measurements.

| flag | deployed lineage | carrier-swap ablation |
|---|---|---|
| `lambda_pose` | 0.2 | 0.12 |
| `lambda_vel` | 0.2 | 0.15 |
| `lambda_desc` | 0.3 | 0.15 |
| `lambda_wavkl` | 1.0 | 0.75 |
| `lambda_latent` | 1.0 | 1.5 |
| `latent_loss_mode` | `cosine` | `both` |
| `latent_cov_weight` | 0.0 | 0.25 |
| `lambda_jepa_pc` | **0.0** | 1.0 |

The deployed column is not guesswork: it is read from the stored `args` inside a
surviving lineage checkpoint. The predictive contract in particular never
appears in the deployed model - its training log contains zero occurrences of
`jepa_pc`.

## Training cost (measured)

On a single RTX 5070 Ti, 52 172 training spans:

| Stage | Epochs | s/epoch |
|---|---|---|
| PHOENIX carrier | 12 | ~24 |
| PHOENIX v24a / v25 / v29 | 3 each | ~35 / ~43 / ~53 |
| CSL carrier | 12 | ~74 |
| CSL v25 | 12 | ~95 |
| CSL v29 | 6 | ~139 |

Resuming the PHOENIX lineage from `v11` costs about **7 minutes** for all three
stages. Retraining is cheap enough that these checkpoints are a convenience
rather than a prerequisite.

## Rebuilding the exemplar bank

Training reads only the exemplar bank, so this is the one artifact you must have.
It is built in two steps:

```bash
# 1. per-gloss spans, via CTC forced alignment over a recognizer's posteriors
CORRNET_POSTERIORS=/path/to/train.pkl \
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

The `exemplar_upc` field is optional - `build_slrtp178_exemplars.py` reads it via
`.get()` and the training loader ignores it entirely. A UPC codebook is **not**
required to build a usable bank.

Our rebuild aligned 7096/7096 clips with zero proportional fallbacks, giving 1081
glosses / 54 930 exemplars natively at 178 keypoints (2.6 GB). It reproduces the
original bank's vocabulary exactly: a surviving lineage checkpoint records
`n_gloss_ids = 1083`, and the rebuilt bank yields the same 1083 glosses and the
identical `train=52172 / val=2048` split.

## Generation is length-leak-guarded

`sample_manifest` refuses to use the manifest's ground-truth length by default:

```
ValueError: sample_manifest no longer uses manifest row['length'] by default.
Pass a duration policy containing a text-only length_model, or use
--allow_gt_length_base for an explicit legacy/leaky diagnostic run.
```

Do not disable this guard for anything you intend to report. Ground-truth length
is a strong shortcut and a run that uses it is not comparable with a
pure-generation result.

Every dev and test number above was produced through the text-only policy:
515/515 dev and 641/641 test clips record `length_source="text_policy"`. The leak
flag was used only on the training split, to calibrate the duration policy and to
generate the governor's fit source.

To generate you therefore need a text-only duration policy, fitted with
`scripts/train_jepa_duration_policy.py --feature_mode text_only`. It consumes
back-translation hypotheses from a small duration sweep; `scripts/dump_bt_hypotheses.py`
produces those from a pose file. Fit the governor on train poses generated at the
length scale the policy actually selects - fitting it on mismatched lengths makes
its predicted gains fall below `--min_gain` and turns it into a no-op.
