#!/usr/bin/env bash
# DECOMPOSITION remediation (only launched if the lambda_pc=0 counterfactual does
# NOT cleanly isolate the contract). Strips BOTH the predictive contract AND the
# rich latent anchoring: latent_loss_mode cosine (drop raw + cov) and lambda_jepa_pc 0.
# Everything else identical to the contract recipe. This gives a clean 3-way
# decomposition of the JEPA win:
#   contract  (rich anchor + contract)  : Table 9
#   pc0       (rich anchor, no contract): isolates the contract
#   min       (cosine anchor, no contract): isolates the rich anchoring
# cosine-only discards the latent geometry JEPA exploits, so carriers should tie here.
# Tags use _min_ so results never collide with _pc_ / _pc0_.
#
#   usage: scripts/carrier_swap_min.sh <SEED> [jepa mae ae]
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
SEED="${1:?need seed}"; shift
OBJS=("$@"); [ ${#OBJS[@]} -eq 0 ] && OBJS=(jepa mae ae)
mkdir -p "$CSW"

# cosine-only anchoring (NO raw, NO cov) + contract OFF. Only these two changes vs Table 9.
PC_FLAGS="--latent_loss_mode cosine \
  --lambda_latent 1.5 \
  --lambda_jepa_pc 0.0 --jepa_pc_loss_mode both --jepa_pc_raw_weight 1.0 \
  --jepa_pc_min_context 16 --jepa_pc_target_len_min 8 --jepa_pc_target_len_max 24"

GEN_FLAGS="--bank $BANK --batch_size 64 --lr 2e-4 \
  --lambda_pose 0.12 --lambda_vel 0.15 --lambda_desc 0.15 \
  --lambda_wavkl 0.75 --wavkl_scales 1,3,7,15 --wavkl_bins 64 --wavkl_vmax 0.15 --wavkl_max_batches 30 \
  --use_energy_cond --lambda_energy_pred 0.1 --energy_desc_scale 0.01 \
  --use_sentence_budget --sentence_max_len 32 \
  --lambda_vel_dir 0.25 --vel_dir_wrist_weight 0.0 \
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
  TAG=${OBJ}_min_s${SEED}
  LOG=$CSW/${TAG}.log
  banner(){ echo; echo "==== [$(date +%H:%M:%S)] MIN(cosine,lpc=0) seed=$SEED OBJ=$OBJ $* ===="; }
  fail(){ echo "min seed=$SEED OBJ=$OBJ FAILED at: $*"; return 1; }
  {
  CARRIER=$(carrier_path "$OBJ")
  banner "carrier: $CARRIER"
  [ -f "$CARRIER" ] || { fail "missing carrier"; return 1; }

  V25=$CSW/${OBJ}_v25min_s${SEED}
  banner "v25 (cosine,lpc=0) from scratch"
  if [ -f "$V25/best.pt" ]; then echo "skip v25min"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V25" --jepa_ckpt "$CARRIER" --seed "$SEED" --epochs 12 \
      $GEN_FLAGS $PC_FLAGS || { fail v25min; return 1; }
  fi
  [ -f "$V25/best.pt" ] || { fail "no v25min best.pt"; return 1; }

  V29=$CSW/${OBJ}_v29min_s${SEED}
  banner "v29 (cosine,lpc=0) adversarial finetune"
  if [ -f "$V29/best.pt" ]; then echo "skip v29min"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V29" --jepa_ckpt "$CARRIER" --seed "$SEED" --epochs 3 --lr 5e-5 \
      $GEN_FLAGS $PC_FLAGS --init_from "$V25/best.pt" \
      --lambda_adv 0.05 --d_lr 2e-4 --d_hidden 64 || { fail v29min; return 1; }
  fi
  [ -f "$V29/best.pt" ] || { fail "no v29min best.pt"; return 1; }

  for SP in dev test; do
    RES=$P/results/cswap_${TAG}_${SP}.json
    [ -f "$RES" ] && { echo "skip eval $SP"; continue; }
    banner "generate $SP"
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py sample_manifest \
      --ckpt "$V29/best.pt" --manifest_json data/phoenix/phoenix_${SP}.json \
      --reference_pt "$DATA/${SP}.pt" --out_pt "$CSW/${TAG}_raw_${SP}.pt" \
      --duration_policy "$DUR" --length_scale 1.0 || { fail "gen $SP"; return 1; }
    banner "govern $SP"
    $PY -u scripts/sign_jepa_utterance_budget_governor.py apply --policy "$GOV" \
      --manifest_json data/phoenix/phoenix_${SP}.json --in_pt "$CSW/${TAG}_raw_${SP}.pt" \
      --out_pt "$P/results/cswap_${TAG}_${SP}.pt" \
      --min_gain 1.0 --max_gain 1.55 --body_coupling 0.35 --max_body_gain 1.20 \
      --lp_kernel 1 --smooth_kernel 1 || { fail "gov $SP"; return 1; }
    banner "eval $SP"
    ( cd "$P" && python -u main.py "results/cswap_${TAG}_${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model" \
        --tag "cswap_${TAG}_${SP}" ) || { fail "eval $SP"; return 1; }
  done
  banner "DONE"
  } 2>&1 | tee "$LOG"
}

for OBJ in "${OBJS[@]}"; do run "$OBJ"; done
echo "==== carrier_swap_min (cosine-only, no contract) complete ===="
