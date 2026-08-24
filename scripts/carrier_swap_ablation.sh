#!/usr/bin/env bash
# Carrier-swap ablation: train the SAME from-scratch downstream pipeline
# (carrier -> v25 generator -> v29 adversarial finetune -> generate/govern/eval)
# under three carrier pretraining objectives, to back the paper's "why JEPA"
# claim (§2.2 / §5.7). Identical budget & flags for all three; the only
# difference is the carrier objective.
#
#   OBJ = jepa : masked latent prediction + EMA  (= existing carrier, reused)
#   OBJ = mae  : masked raw-feature reconstruction
#   OBJ = ae   : full (unmasked) reconstruction
#
# IMPORTANT FAIRNESS NOTE: the PHOENIX headline AHA-JEPA was a warm-start chain
# (v24a->v25->v29). For a fair carrier swap, the generator is retrained FROM
# SCRATCH for all three carriers on the same budget. So the JEPA row here is a
# from-scratch retrain and may differ slightly from the warm-chained headline.
#
#   usage: scripts/carrier_swap_ablation.sh [jepa|mae|ae|all]   (default: all)
# Resumable: every stage is skip-if-exists.
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

# shared v25-stage flags (from-scratch generator base)
GEN_FLAGS="--bank $BANK --epochs 12 --batch_size 64 --lr 2e-4 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_scales 1,3,7,15 --wavkl_bins 64 --wavkl_vmax 0.15 --wavkl_max_batches 30 \
  --use_energy_cond --lambda_energy_pred 0.1 --energy_desc_scale 0.01 \
  --use_sentence_budget --sentence_max_len 32 \
  --lambda_vel_dir 0.3 --vel_dir_wrist_weight 0.0 \
  --stochastic_decoder --init_log_var -4.0 --lambda_nll 1.0 \
  --train_manifest_json data/phoenix/phoenix_train.json --num_workers 2 --log_every 200 --device cuda"

run_obj() {
  OBJ="$1"
  LOG=$CSW/${OBJ}.log
  banner(){ echo; echo "==== [$(date +%H:%M:%S)] OBJ=$OBJ $* ===="; }
  fail(){ echo "OBJ=$OBJ FAILED at: $*"; return 1; }
  {
  # ---- Step 1: carrier ----
  if [ "$OBJ" = "jepa" ]; then
    CARRIER=$ROOT/sign_jepa_slrtp178/best.pt
    banner "carrier: reuse existing JEPA carrier ($CARRIER)"
    [ -f "$CARRIER" ] || { fail "missing jepa carrier"; return 1; }
  else
    CARRIER=$CSW/${OBJ}_carrier/best.pt
    banner "carrier: train ${OBJ} (12 ep)"
    if [ -f "$CARRIER" ]; then echo "skip carrier (exists)"; else
      $PY -u scripts/train_sign_jepa_slrtp178.py --objective "$OBJ" \
        --epochs 12 --batch_size 64 --bank "$BANK" \
        --out_dir "$CSW/${OBJ}_carrier" || { fail "carrier"; return 1; }
    fi
    [ -f "$CARRIER" ] || { fail "carrier produced no best.pt"; return 1; }
  fi

  # ---- Step 2: generator base from scratch (12 ep, no adv) ----
  V25=$CSW/${OBJ}_v25
  banner "v25 generator from scratch (12 ep, anchored to $CARRIER)"
  if [ -f "$V25/best.pt" ]; then echo "skip v25 (exists)"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V25" --jepa_ckpt "$CARRIER" $GEN_FLAGS || { fail "v25"; return 1; }
  fi
  [ -f "$V25/best.pt" ] || { fail "v25 produced no best.pt"; return 1; }

  # ---- Step 3: adversarial finetune (3 ep, init from v25, +critic) ----
  V29=$CSW/${OBJ}_v29
  banner "v29 adversarial finetune (3 ep, init $V25/best.pt)"
  if [ -f "$V29/best.pt" ]; then echo "skip v29 (exists)"; else
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py train \
      --out_dir "$V29" --jepa_ckpt "$CARRIER" $GEN_FLAGS \
      --init_from "$V25/best.pt" --epochs 3 --lr 5e-5 \
      --lambda_adv 0.05 --d_lr 2e-4 --d_hidden 64 || { fail "v29"; return 1; }
  fi
  [ -f "$V29/best.pt" ] || { fail "v29 produced no best.pt"; return 1; }

  # ---- Step 4: generate + govern + eval both splits ----
  for SP in dev test; do
    RES=$P/results/cswap_${OBJ}_${SP}.json
    if [ -f "$RES" ]; then echo "skip eval $SP (exists)"; continue; fi
    banner "generate $SP"
    $PY -u scripts/train_sign_jepa_generator_slrtp178.py sample_manifest \
      --ckpt "$V29/best.pt" --manifest_json data/phoenix/phoenix_${SP}.json \
      --reference_pt "$DATA/${SP}.pt" --out_pt "$CSW/${OBJ}_raw_${SP}.pt" \
      --duration_policy "$DUR" --length_scale 1.0 || { fail "gen $SP"; return 1; }
    banner "govern $SP"
    $PY -u scripts/sign_jepa_utterance_budget_governor.py apply --policy "$GOV" \
      --manifest_json data/phoenix/phoenix_${SP}.json --in_pt "$CSW/${OBJ}_raw_${SP}.pt" \
      --out_pt "$P/results/cswap_${OBJ}_${SP}.pt" \
      --min_gain 1.0 --max_gain 1.55 --body_coupling 0.35 --max_body_gain 1.20 \
      --lp_kernel 1 --smooth_kernel 1 || { fail "gov $SP"; return 1; }
    banner "eval $SP"
    ( cd "$P" && python -u main.py "results/cswap_${OBJ}_${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model" \
        --tag "cswap_${OBJ}_${SP}" ) || { fail "eval $SP"; return 1; }
  done
  banner "DONE $OBJ"
  } 2>&1 | tee "$LOG"
}

WHICH="${1:-all}"
if [ "$WHICH" = "all" ]; then
  for OBJ in jepa mae ae; do run_obj "$OBJ"; done
else
  run_obj "$WHICH"
fi
echo "==== carrier_swap_ablation complete ===="
