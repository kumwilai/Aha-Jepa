#!/usr/bin/env bash
# 3-seed ensemble (avg raw poses across seeds 0,1,2) -> govern -> SLRTP eval,
# for all three carriers equally. Cheap: only the back-translation eval uses GPU.
#   usage: scripts/carrier_swap_ensemble.sh [jepa mae ae]   (default all)
# Resumable. Results tagged cswap_<OBJ>_ens_<split>.
set -uo pipefail
cd "${AHA_JEPA_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source ~/research/coopns-slr/.venv/bin/activate
PY=$HOME/research/coopns-slr/.venv/bin/python
ROOT=outputs/sota_chase
CSW=$ROOT/carrier_swap
P=external/SLRTP-Sign-Production-Evaluation
GOV=$ROOT/sign_jepa_v10_budget_governor/policy.pt
OBJS=("$@"); [ ${#OBJS[@]} -eq 0 ] && OBJS=(jepa mae ae)

for OBJ in "${OBJS[@]}"; do
  LOG=$CSW/${OBJ}_ens.log
  {
  echo "==== [$(date +%H:%M:%S)] ENSEMBLE OBJ=$OBJ ===="
  for SP in dev test; do
    RES=$P/results/cswap_${OBJ}_ens_${SP}.json
    [ -f "$RES" ] && { echo "skip $SP (exists)"; continue; }
    $PY scripts/carrier_swap_ensemble.py "$OBJ" "$SP" || { echo "AVG FAIL $OBJ $SP"; continue; }
    $PY -u scripts/sign_jepa_utterance_budget_governor.py apply --policy "$GOV" \
      --manifest_json data/phoenix/phoenix_${SP}.json \
      --in_pt "$CSW/${OBJ}_raw_ens_${SP}.pt" \
      --out_pt "$P/results/cswap_${OBJ}_ens_${SP}.pt" \
      --min_gain 1.0 --max_gain 1.55 --body_coupling 0.35 --max_body_gain 1.20 \
      --lp_kernel 1 --smooth_kernel 1 || { echo "GOV FAIL $OBJ $SP"; continue; }
    ( cd "$P" && python -u main.py "results/cswap_${OBJ}_ens_${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/${SP}.pt" \
        "pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model" \
        --tag "cswap_${OBJ}_ens_${SP}" ) || { echo "EVAL FAIL $OBJ $SP"; continue; }
  done
  echo "==== [$(date +%H:%M:%S)] DONE ENSEMBLE $OBJ ===="
  } 2>&1 | tee "$LOG"
done
echo "==== carrier_swap_ensemble complete ===="
