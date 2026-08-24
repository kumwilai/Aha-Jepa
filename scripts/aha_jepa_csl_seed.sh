#!/usr/bin/env bash
# Train ONE additional AHA-JEPA CSL seed (full pipeline jepa->v25->v29) and
# generate its dev/test MSKA pickles, for the 3-seed ensemble. Seed-parameterised
# output dirs; resumable (skip-if-exists). Leak-free: NON-GT teacher length.
#   usage: scripts/aha_jepa_csl_seed.sh <SEED>
set -uo pipefail
cd "${AHA_JEPA_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source ~/research/coopns-slr/.venv/bin/activate
PY=$HOME/research/coopns-slr/.venv/bin/python
ROOT=outputs/sota_chase
SEED="${1:?need seed integer}"
JE=$ROOT/sign_jepa_csl_s${SEED}
V25=$ROOT/aha_csl_v25_s${SEED}
V29=$ROOT/aha_csl_v29_s${SEED}
mkdir -p "$ROOT/aha_csl"
LOG=$ROOT/aha_csl/chain_s${SEED}.log

banner(){ echo; echo "==== [$(date +%H:%M:%S)] seed=$SEED $* ===="; }
fail(){ echo "SEED $SEED FAILED at: $*"; exit 1; }

{
banner "0. bank (shared, skip if exists)"
[ -f "$ROOT/csl_jepa_bank.pt" ] || $PY scripts/build_csl_jepa_bank.py || fail "bank"

banner "1. Stage-1 Sign-JEPA (12 ep)"
if [ -f "$JE/best.pt" ]; then echo "skip (exists)"; else
$PY -u scripts/aha_jepa_csl.py jepa \
  --out_dir "$JE" --epochs 12 --batch_size 64 --seed "$SEED" || fail "jepa"
fi

banner "2. Stage-2 v25 (12 ep)"
if [ -f "$V25/best.pt" ]; then echo "skip (exists)"; else
$PY -u scripts/aha_jepa_csl.py gen \
  --jepa_ckpt "$JE/best.pt" --out_dir "$V25" \
  --epochs 12 --batch_size 64 --seed "$SEED" \
  --stochastic_decoder --lambda_nll 1.0 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_vmax 60 \
  --use_energy_cond --use_sentence_budget --lambda_energy_pred 0.1 || fail "v25"
fi

banner "3. Stage-2 v29 (init v25 + adversarial critic, 6 ep)"
if [ -f "$V29/best.pt" ]; then echo "skip (exists)"; else
$PY -u scripts/aha_jepa_csl.py gen \
  --jepa_ckpt "$JE/best.pt" --out_dir "$V29" \
  --init_from "$V25/best.pt" \
  --epochs 6 --batch_size 64 --seed "$SEED" \
  --stochastic_decoder --lambda_nll 1.0 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_vmax 60 --lambda_adv 0.05 \
  --use_energy_cond --use_sentence_budget --lambda_energy_pred 0.1 || fail "v29"
fi

banner "4. generate dev + test (tag=aha_s${SEED}, NON-GT teacher length)"
[ -f "outputs/csl_msla_inputs/CSL-Daily.dev_aha_s${SEED}" ]  || \
  $PY -u scripts/aha_jepa_csl.py generate --ckpt "$V29/best.pt" --split dev  --tag "aha_s${SEED}" || fail "gen dev"
[ -f "outputs/csl_msla_inputs/CSL-Daily.test_aha_s${SEED}" ] || \
  $PY -u scripts/aha_jepa_csl.py generate --ckpt "$V29/best.pt" --split test --tag "aha_s${SEED}" || fail "gen test"

banner "SEED $SEED DONE"
} 2>&1 | tee "$LOG"
