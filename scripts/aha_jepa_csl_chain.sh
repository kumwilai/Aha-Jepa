#!/usr/bin/env bash
# AHA-JEPA on CSL-Daily (HRNet-133) — full faithful training chain.
#   Stage 1 JEPA -> Stage 2 v25 (stochastic+energy+sentence+wavkl+nll)
#   -> Stage 2 v29 (init v25 + adversarial hand critic) -> generate dev/test
#   -> MSKA-SLR gloss-WER scoring.
# Leak-free: generation length = NON-GT teacher native_hclen length; no GT,
# no retrieval, no test/dev-tuned knobs.
set -uo pipefail
cd "${AHA_JEPA_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source ~/research/coopns-slr/.venv/bin/activate
PY=$HOME/research/coopns-slr/.venv/bin/python
ROOT=outputs/sota_chase
LOG=$ROOT/aha_csl/chain.log
mkdir -p "$ROOT/aha_csl"

banner(){ echo; echo "==== [$(date +%H:%M:%S)] $* ===="; }
fail(){ echo "CHAIN FAILED at: $*"; exit 1; }

{
banner "0. bank (skip if exists)"
[ -f "$ROOT/csl_jepa_bank.pt" ] || $PY scripts/build_csl_jepa_bank.py || fail "bank"

banner "1. Stage-1 Sign-JEPA (12 ep)"
if [ -f "$ROOT/sign_jepa_csl/best.pt" ]; then echo "skip (exists)"; else
$PY -u scripts/aha_jepa_csl.py jepa \
  --out_dir "$ROOT/sign_jepa_csl" --epochs 12 --batch_size 64 || fail "jepa"
fi

banner "2. Stage-2 v25 (stochastic + energy + sentence-budget + WAV-KL + NLL, 12 ep)"
if [ -f "$ROOT/aha_csl_v25/best.pt" ]; then echo "skip (exists)"; else
$PY -u scripts/aha_jepa_csl.py gen \
  --jepa_ckpt "$ROOT/sign_jepa_csl/best.pt" \
  --out_dir "$ROOT/aha_csl_v25" \
  --epochs 12 --batch_size 64 \
  --stochastic_decoder --lambda_nll 1.0 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_vmax 60 \
  --use_energy_cond --use_sentence_budget --lambda_energy_pred 0.1 || fail "v25"
fi

banner "3. Stage-2 v29 (init v25 + adversarial hand critic lambda_adv=0.05, 6 ep)"
if [ -f "$ROOT/aha_csl_v29/best.pt" ]; then echo "skip (exists)"; else
$PY -u scripts/aha_jepa_csl.py gen \
  --jepa_ckpt "$ROOT/sign_jepa_csl/best.pt" \
  --out_dir "$ROOT/aha_csl_v29" \
  --init_from "$ROOT/aha_csl_v25/best.pt" \
  --epochs 6 --batch_size 64 \
  --stochastic_decoder --lambda_nll 1.0 \
  --lambda_latent 1.0 --lambda_pose 0.2 --lambda_vel 0.2 --lambda_desc 0.3 \
  --lambda_wavkl 1.0 --wavkl_vmax 60 --lambda_adv 0.05 \
  --use_energy_cond --use_sentence_budget --lambda_energy_pred 0.1 || fail "v29"
fi

banner "4. generate dev + test (tag=aha, NON-GT teacher length)"
$PY -u scripts/aha_jepa_csl.py generate --ckpt "$ROOT/aha_csl_v29/best.pt" \
  --split dev  --tag aha || fail "gen dev"
$PY -u scripts/aha_jepa_csl.py generate --ckpt "$ROOT/aha_csl_v29/best.pt" \
  --split test --tag aha || fail "gen test"
# also export the v25 (no-adversarial) ablation for comparison
$PY -u scripts/aha_jepa_csl.py generate --ckpt "$ROOT/aha_csl_v25/best.pt" \
  --split dev  --tag aha_v25 || fail "gen dev v25"
$PY -u scripts/aha_jepa_csl.py generate --ckpt "$ROOT/aha_csl_v25/best.pt" \
  --split test --tag aha_v25 || fail "gen test v25"

banner "5. MSKA-SLR gloss-WER scoring (dev + test for aha and aha_v25)"
$PY -u scripts/score_csl_mska.py --split dev  --sources aha aha_v25 --batch_size 8 || fail "score dev"
$PY -u scripts/score_csl_mska.py --split test --sources aha aha_v25 --batch_size 8 || fail "score test"

banner "DONE — results in outputs/csl_mska_results/summary.json"
} 2>&1 | tee "$LOG"
