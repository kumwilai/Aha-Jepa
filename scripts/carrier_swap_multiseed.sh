#!/usr/bin/env bash
# Multi-seed carrier-swap: vary the seed across the WHOLE pipeline
# (carrier + v25 + v29) so we can put error bars on the jepa/mae/ae BT tie.
# Seed 0 == the single-seed run already in $CSW (jepa reuses the production
# carrier). This script adds the requested extra seeds.
#   usage: scripts/carrier_swap_multiseed.sh <SEED> [jepa mae ae]
# Resumable: every stage is skip-if-exists. Results tagged cswap_<OBJ>_s<SEED>_<split>.
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
mkdir -p "$CSW"

SEED="${1:?need seed integer}"; shift
OBJS=("$@"); [ ${#OBJS[@]} -eq 0 ] && OBJS=(jepa mae ae)

GEN_FLAGS="--bank $BANK --epochs 12 --batch_size 64 --lr 2e-4 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_scales 1,3,7,15 --wavkl_bins 64 --wavkl_vmax 0.15 --wavkl_max_batches 30 \
  --use_energy_cond --lambda_energy_pred 0.1 --energy_desc_scale 0.01 \
  --use_sentence_budget --sentence_max_len 32 \
  --lambda_vel_dir 0.3 --vel_dir_wrist_weight 0.0 \
  --stochastic_decoder --init_log_var -4.0 --lambda_nll 1.0 \
  --train_manifest_json data/phoenix/phoenix_train.json --num_workers 2 --log_every 400 --device cuda"

run() {
  OBJ="$1"
  LOG=$CSW/${OBJ}_s${SEED}.log
  banner(){ echo; echo "==== [$(date +%H:%M:%S)] s${SEED} OBJ=$OBJ $* ===="; }
  fail(){ echo "s${SEED} OBJ=$OBJ FAILED at: $*"; return 1; }
  {
  CARRIER=$CSW/${OBJ}_carrier_s${SEED}/best.pt
  banner "carrier (--seed $SEED, --objective $OBJ, 12 ep)"
  if [ -f "$CARRIER" ]; then echo "skip carrier (exists)"; else
    $PY -u scripts/train_sign_jepa_slrtp178.py --objective "$OBJ" --seed "$SEED" \
      --epochs 12 --batch_size 64 --bank "$BANK" \
      --out_dir "$CSW/${OBJ}_carrier_s${SEED}" || { fail carrier; return 1; }
  fi
  [ -f "$CARRIER" ] || { fail "no carrier best.pt"; return 1; }

  V25=$CSW/${OBJ}_v25_s${SEED}
  banner "v25 from scratch (--seed $SEED, 12 ep)"
  if [ -f "$V25/best.pt" ]; then echo "skip v25 (exists)"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V25" --jepa_ckpt "$CARRIER" --seed "$SEED" $GEN_FLAGS || { fail v25; return 1; }
  fi
  [ -f "$V25/best.pt" ] || { fail "no v25 best.pt"; return 1; }

  V29=$CSW/${OBJ}_v29_s${SEED}
  banner "v29 adv finetune (--seed $SEED, 3 ep)"
  if [ -f "$V29/best.pt" ]; then echo "skip v29 (exists)"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V29" --jepa_ckpt "$CARRIER" --seed "$SEED" $GEN_FLAGS \
      --init_from "$V25/best.pt" --epochs 3 --lr 5e-5 \
      --lambda_adv 0.05 --d_lr 2e-4 --d_hidden 64 || { fail v29; return 1; }
  fi
  [ -f "$V29/best.pt" ] || { fail "no v29 best.pt"; return 1; }

  for SP in dev test; do
    RES=$P/results/cswap_${OBJ}_s${SEED}_${SP}.json
    if [ -f "$RES" ]; then echo "skip eval $SP (exists)"; continue; fi
    banner "generate $SP"
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py sample_manifest \
      --ckpt "$V29/best.pt" --manifest_json data/phoenix/phoenix_${SP}.json \
      --reference_pt "$DATA/${SP}.pt" --out_pt "$CSW/${OBJ}_raw_s${SEED}_${SP}.pt" \
      --duration_policy "$DUR" --length_scale 1.0 || { fail "gen $SP"; return 1; }
    banner "govern $SP"
    $PY -u scripts/sign_jepa_utterance_budget_governor.py apply --policy "$GOV" \
      --manifest_json data/phoenix/phoenix_${SP}.json --in_pt "$CSW/${OBJ}_raw_s${SEED}_${SP}.pt" \
      --out_pt "$P/results/cswap_${OBJ}_s${SEED}_${SP}.pt" \
      --min_gain 1.0 --max_gain 1.55 --body_coupling 0.35 --max_body_gain 1.20 \
      --lp_kernel 1 --smooth_kernel 1 || { fail "gov $SP"; return 1; }
    banner "eval $SP"
    ( cd "$P" && python -u main.py "results/cswap_${OBJ}_s${SEED}_${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model" \
        --tag "cswap_${OBJ}_s${SEED}_${SP}" ) || { fail "eval $SP"; return 1; }
  done
  banner "DONE $OBJ s${SEED}"
  } 2>&1 | tee "$LOG"
}

for OBJ in "${OBJS[@]}"; do run "$OBJ"; done
echo "==== multiseed s${SEED} complete ===="
