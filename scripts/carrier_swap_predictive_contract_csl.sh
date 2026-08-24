#!/usr/bin/env bash
# CSL-Daily carrier swap with the predictive carrier contract (mirror of
# scripts/carrier_swap_predictive_contract.sh for PHOENIX, but on the HRNet-133
# CSL layout and scored by the MSKA-SLR gloss recognizer — CSL is gloss-WER-only,
# the Chinese SLT back-translators score BLEU~0).
#
# For each carrier objective we train a fresh carrier under IDENTICAL args except
# --objective, then a v25 + v29 generator under the SAME recipe with the
# predictive contract active (--lambda_jepa_pc), then generate dev/test at the
# NON-GT native_hclen teacher length. JEPA was pretrained for masked-latent
# prediction; MAE/AE were not, so the contract should separate them.
#
#   usage: scripts/carrier_swap_predictive_contract_csl.sh <SEED> [jepa mae ae]
set -uo pipefail
cd "${AHA_JEPA_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source ~/research/coopns-slr/.venv/bin/activate
PY=$HOME/research/coopns-slr/.venv/bin/python

ROOT=outputs/sota_chase
CSW=$ROOT/carrier_swap_csl
SEED="${1:?need seed}"; shift
OBJS=("$@"); [ ${#OBJS[@]} -eq 0 ] && OBJS=(jepa mae ae)
mkdir -p "$CSW"

# CSL generator recipe (from aha_jepa_csl_chain.sh) + predictive contract.
GEN_FLAGS="--batch_size 64 --lr 2e-4 \
  --stochastic_decoder --lambda_nll 1.0 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_vmax 60 --wavkl_max_batches 30 \
  --use_energy_cond --use_sentence_budget --lambda_energy_pred 0.1 \
  --train_manifest_json data/csl-daily/csl_daily_train.json"
PC_FLAGS="--lambda_jepa_pc 1.0 --jepa_pc_loss_mode both --jepa_pc_raw_weight 1.0 \
  --jepa_pc_min_context 16 --jepa_pc_target_len_min 8 --jepa_pc_target_len_max 24"

run() {
  OBJ="$1"
  TAG=${OBJ}_pc_s${SEED}
  LOG=$CSW/${TAG}.log
  banner(){ echo; echo "==== [$(date +%H:%M:%S)] csl-pc seed=$SEED OBJ=$OBJ $* ===="; }
  fail(){ echo "csl-pc seed=$SEED OBJ=$OBJ FAILED at: $*"; return 1; }
  {
  CARRIER=$CSW/carrier_${OBJ}_s${SEED}
  banner "1. carrier (--objective $OBJ, 12 ep)"
  if [ -f "$CARRIER/best.pt" ]; then echo "skip carrier"; else
    $PY -u scripts/aha_jepa_csl.py jepa \
      --out_dir "$CARRIER" --objective "$OBJ" --seed "$SEED" --epochs 12 || { fail carrier; return 1; }
  fi
  [ -f "$CARRIER/best.pt" ] || { fail "no carrier best.pt"; return 1; }

  V25=$CSW/${OBJ}_v25pc_s${SEED}
  banner "2. v25 predictive-contract from scratch (12 ep)"
  if [ -f "$V25/best.pt" ]; then echo "skip v25pc"; else
    $PY -u scripts/aha_jepa_csl.py gen \
      --jepa_ckpt "$CARRIER/best.pt" --out_dir "$V25" --seed "$SEED" --epochs 12 \
      $GEN_FLAGS $PC_FLAGS || { fail v25pc; return 1; }
  fi
  [ -f "$V25/best.pt" ] || { fail "no v25pc best.pt"; return 1; }

  V29=$CSW/${OBJ}_v29pc_s${SEED}
  banner "3. v29 predictive-contract adversarial finetune (init v25, 6 ep)"
  if [ -f "$V29/best.pt" ]; then echo "skip v29pc"; else
    $PY -u scripts/aha_jepa_csl.py gen \
      --jepa_ckpt "$CARRIER/best.pt" --out_dir "$V29" --seed "$SEED" --epochs 6 --lr 5e-5 \
      $GEN_FLAGS $PC_FLAGS --init_from "$V25/best.pt" \
      --lambda_adv 0.05 --d_lr 2e-4 --d_hidden 64 || { fail v29pc; return 1; }
  fi
  [ -f "$V29/best.pt" ] || { fail "no v29pc best.pt"; return 1; }

  for SP in dev test; do
    OUT=outputs/csl_msla_inputs/CSL-Daily.${SP}_${TAG}
    if [ -f "$OUT" ]; then echo "skip generate $SP"; continue; fi
    banner "4. generate $SP (tag=$TAG, NON-GT teacher length)"
    $PY -u scripts/aha_jepa_csl.py generate --ckpt "$V29/best.pt" \
      --split "$SP" --tag "$TAG" || { fail "gen $SP"; return 1; }
  done
  banner "DONE"
  } 2>&1 | tee "$LOG"
}

for OBJ in "${OBJS[@]}"; do run "$OBJ"; done

echo; echo "==== [$(date +%H:%M:%S)] MSKA-SLR gloss-WER scoring (seed $SEED) ===="
SRC_TAGS=(); for OBJ in "${OBJS[@]}"; do SRC_TAGS+=("${OBJ}_pc_s${SEED}"); done
$PY -u scripts/score_csl_mska.py --split dev  --sources "${SRC_TAGS[@]}" --batch_size 8 || true
$PY -u scripts/score_csl_mska.py --split test --sources "${SRC_TAGS[@]}" --batch_size 8 || true
echo "==== carrier_swap_predictive_contract_csl seed $SEED complete ===="
