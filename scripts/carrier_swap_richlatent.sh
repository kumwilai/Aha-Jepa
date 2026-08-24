#!/usr/bin/env bash
# Rich-latent SLP demonstration. Same from-scratch pipeline as the carrier swap,
# but the generator anchors to the FULL carrier latent (--latent_loss_mode both,
# direction+magnitude) with a stronger latent weight, so representation richness
# can propagate to the output instead of being discarded by the magnitude-blind
# cosine anchor.
#
# Hypothesis: the rich anchor helps the HIGH-eff-rank carrier (JEPA, 121) more
# than the LOW one (AE, 71). The differential vs the cosine baselines
# (cswap_jepa_s0 / cswap_ae_s0) is the "JEPA's richness is exploitable" result.
#
#   usage: scripts/carrier_swap_richlatent.sh [jepa ae ...]   (default jepa ae)
# Resumable. Results tagged cswap_<OBJ>_rich_<split>.
set -uo pipefail
cd "${AHA_JEPA_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source ~/research/coopns-slr/.venv/bin/activate
PY=$HOME/research/coopns-slr/.venv/bin/python
ROOT=outputs/sota_chase
CSW=$ROOT/carrier_swap
BANK=$ROOT/phase40_slrtp178/exemplars_slrtp178_upc.pt
P=external/SLRTP-Sign-Production-Evaluation
DATA=$P/pretrained/SLRTP-Sign-Production-Evaluation-Data/data
DUR=$ROOT/jepa_duration_policy/train_bt_rf_policy.joblib
GOV=$ROOT/sign_jepa_v10_budget_governor/policy.pt
OBJS=("$@"); [ ${#OBJS[@]} -eq 0 ] && OBJS=(jepa ae)

# rich-latent recipe: both-mode anchor (dir+magnitude), stronger latent weight
RICH="--latent_loss_mode both --latent_raw_weight 1.0 --lambda_latent 2.0"
GEN_FLAGS="--bank $BANK --batch_size 64 --lr 2e-4 \
  --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_scales 1,3,7,15 --wavkl_bins 64 --wavkl_vmax 0.15 --wavkl_max_batches 30 \
  --use_energy_cond --lambda_energy_pred 0.1 --energy_desc_scale 0.01 \
  --use_sentence_budget --sentence_max_len 32 \
  --lambda_vel_dir 0.3 --vel_dir_wrist_weight 0.0 \
  --stochastic_decoder --init_log_var -4.0 --lambda_nll 1.0 \
  --train_manifest_json data/phoenix/phoenix_train.json --num_workers 2 --log_every 400 --device cuda"

carrier_path() { case "$1" in jepa) echo "$ROOT/sign_jepa_slrtp178/best.pt";; *) echo "$CSW/$1_carrier/best.pt";; esac; }

run() {
  OBJ="$1"; LOG=$CSW/${OBJ}_rich.log
  banner(){ echo; echo "==== [$(date +%H:%M:%S)] rich OBJ=$OBJ $* ===="; }
  fail(){ echo "rich OBJ=$OBJ FAILED at: $*"; return 1; }
  {
  CARRIER=$(carrier_path "$OBJ"); [ -f "$CARRIER" ] || { fail "no carrier"; return 1; }
  V25=$CSW/${OBJ}_v25rich
  banner "v25 rich-latent from scratch (12 ep, both-mode anchor)"
  if [ -f "$V25/best.pt" ]; then echo "skip (exists)"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V25" --jepa_ckpt "$CARRIER" --epochs 12 $GEN_FLAGS $RICH || { fail v25; return 1; }
  fi
  V29=$CSW/${OBJ}_v29rich
  banner "v29 rich-latent adv finetune (3 ep)"
  if [ -f "$V29/best.pt" ]; then echo "skip (exists)"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V29" --jepa_ckpt "$CARRIER" --epochs 3 --lr 5e-5 $GEN_FLAGS $RICH \
      --init_from "$V25/best.pt" --lambda_adv 0.05 --d_lr 2e-4 --d_hidden 64 || { fail v29; return 1; }
  fi
  for SP in dev test; do
    RES=$P/results/cswap_${OBJ}_rich_${SP}.json
    [ -f "$RES" ] && { echo "skip eval $SP"; continue; }
    banner "generate $SP"
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py sample_manifest \
      --ckpt "$V29/best.pt" --manifest_json data/phoenix/phoenix_${SP}.json \
      --reference_pt "$DATA/${SP}.pt" --out_pt "$CSW/${OBJ}_raw_rich_${SP}.pt" \
      --duration_policy "$DUR" --length_scale 1.0 || { fail "gen $SP"; return 1; }
    banner "govern $SP"
    $PY -u scripts/sign_jepa_utterance_budget_governor.py apply --policy "$GOV" \
      --manifest_json data/phoenix/phoenix_${SP}.json --in_pt "$CSW/${OBJ}_raw_rich_${SP}.pt" \
      --out_pt "$P/results/cswap_${OBJ}_rich_${SP}.pt" \
      --min_gain 1.0 --max_gain 1.55 --body_coupling 0.35 --max_body_gain 1.20 \
      --lp_kernel 1 --smooth_kernel 1 || { fail "gov $SP"; return 1; }
    banner "eval $SP"
    ( cd "$P" && python -u main.py "results/cswap_${OBJ}_rich_${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model" \
        --tag "cswap_${OBJ}_rich_${SP}" ) || { fail "eval $SP"; return 1; }
  done
  banner "DONE $OBJ rich"
  } 2>&1 | tee "$LOG"
}
for OBJ in "${OBJS[@]}"; do run "$OBJ"; done
echo "==== carrier_swap_richlatent complete ===="
