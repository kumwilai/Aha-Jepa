#!/usr/bin/env bash
# Warm-start-PARITY carrier swap: instead of the 12-ep from-scratch snapshot,
# train each carrier's generator to convergence with the FULL final recipe on a
# long budget that matches the deployed AHA-JEPA regime (~52 cumulative
# generator epochs), then the 3-ep adversarial stage. Identical protocol for all
# three carriers and NO cross-carrier warm-start, so it is fair AND reaches the
# WER~83 operating point (vs the under-trained 88-90 of the 12-ep run).
#
# Rationale: the deployed pipeline is a chain v_base(12)->v8(12)->v9(12)->
# v11(10)->v24a(3)->v25(3)->v29(3) = ~55 ep, recipe added incrementally, all on
# the JEPA carrier. Replaying that chain per-carrier would either contaminate
# (JEPA-trained generator weights) or require carrier-specific stages. A single
# long from-scratch run with the final recipe is the contamination-free
# convergence-parity equivalent.
#
#   usage: scripts/carrier_swap_parity.sh [jepa mae ae]   (default all)
# Resumable. Results tagged cswap_<OBJ>_parity_<split>.
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
LONG_EP=52
mkdir -p "$CSW"

OBJS=("$@"); [ ${#OBJS[@]} -eq 0 ] && OBJS=(jepa mae ae)

GEN_FLAGS="--bank $BANK --batch_size 64 --lr 2e-4 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_scales 1,3,7,15 --wavkl_bins 64 --wavkl_vmax 0.15 --wavkl_max_batches 30 \
  --use_energy_cond --lambda_energy_pred 0.1 --energy_desc_scale 0.01 \
  --use_sentence_budget --sentence_max_len 32 \
  --lambda_vel_dir 0.3 --vel_dir_wrist_weight 0.0 \
  --stochastic_decoder --init_log_var -4.0 --lambda_nll 1.0 \
  --train_manifest_json data/phoenix/phoenix_train.json --num_workers 2 --log_every 400 --device cuda"

carrier_path() {
  case "$1" in
    jepa) echo "$ROOT/sign_jepa_slrtp178/best.pt";;
    *)    echo "$CSW/$1_carrier/best.pt";;
  esac
}

run() {
  OBJ="$1"
  LOG=$CSW/${OBJ}_parity.log
  banner(){ echo; echo "==== [$(date +%H:%M:%S)] parity OBJ=$OBJ $* ===="; }
  fail(){ echo "parity OBJ=$OBJ FAILED at: $*"; return 1; }
  {
  CARRIER=$(carrier_path "$OBJ")
  banner "carrier (reuse seed-0): $CARRIER"
  [ -f "$CARRIER" ] || { fail "missing carrier $CARRIER"; return 1; }

  V25=$CSW/${OBJ}_v25long
  banner "v25long from scratch ($LONG_EP ep, full recipe)"
  if [ -f "$V25/best.pt" ]; then echo "skip v25long (exists)"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V25" --jepa_ckpt "$CARRIER" --epochs "$LONG_EP" $GEN_FLAGS || { fail v25long; return 1; }
  fi
  [ -f "$V25/best.pt" ] || { fail "no v25long best.pt"; return 1; }

  V29=$CSW/${OBJ}_v29long
  banner "v29long adv finetune (3 ep, init v25long)"
  if [ -f "$V29/best.pt" ]; then echo "skip v29long (exists)"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V29" --jepa_ckpt "$CARRIER" $GEN_FLAGS \
      --init_from "$V25/best.pt" --epochs 3 --lr 5e-5 \
      --lambda_adv 0.05 --d_lr 2e-4 --d_hidden 64 || { fail v29long; return 1; }
  fi
  [ -f "$V29/best.pt" ] || { fail "no v29long best.pt"; return 1; }

  for SP in dev test; do
    RES=$P/results/cswap_${OBJ}_parity_${SP}.json
    if [ -f "$RES" ]; then echo "skip eval $SP (exists)"; continue; fi
    banner "generate $SP"
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py sample_manifest \
      --ckpt "$V29/best.pt" --manifest_json data/phoenix/phoenix_${SP}.json \
      --reference_pt "$DATA/${SP}.pt" --out_pt "$CSW/${OBJ}_raw_parity_${SP}.pt" \
      --duration_policy "$DUR" --length_scale 1.0 || { fail "gen $SP"; return 1; }
    banner "govern $SP"
    $PY -u scripts/sign_jepa_utterance_budget_governor.py apply --policy "$GOV" \
      --manifest_json data/phoenix/phoenix_${SP}.json --in_pt "$CSW/${OBJ}_raw_parity_${SP}.pt" \
      --out_pt "$P/results/cswap_${OBJ}_parity_${SP}.pt" \
      --min_gain 1.0 --max_gain 1.55 --body_coupling 0.35 --max_body_gain 1.20 \
      --lp_kernel 1 --smooth_kernel 1 || { fail "gov $SP"; return 1; }
    banner "eval $SP"
    ( cd "$P" && python -u main.py "results/cswap_${OBJ}_parity_${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model" \
        --tag "cswap_${OBJ}_parity_${SP}" ) || { fail "eval $SP"; return 1; }
  done
  banner "DONE $OBJ parity"
  } 2>&1 | tee "$LOG"
}

for OBJ in "${OBJS[@]}"; do run "$OBJ"; done
echo "==== carrier_swap_parity complete ===="
