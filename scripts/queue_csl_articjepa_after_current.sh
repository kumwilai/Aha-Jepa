#!/usr/bin/env bash
# Queue the CSL rich-articulator JEPA recovery experiment behind the currently
# running standard CSL carrier-swap driver. Intended for tmux/nohup execution.
set -uo pipefail
cd "${AHA_JEPA_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source ~/research/coopns-slr/.venv/bin/activate

WAIT_PID="${CURRENT_CSL_DRIVER_PID:-596411}"
OUT=outputs/sota_chase/carrier_swap_csl_articjepa
mkdir -p "$OUT"

echo "===== CSL artic-JEPA queue start $(date) ====="
if [ -n "$WAIT_PID" ] && ps -p "$WAIT_PID" >/dev/null 2>&1; then
  echo "waiting for existing CSL driver pid=$WAIT_PID"
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do
    sleep 300
  done
  echo "existing CSL driver pid=$WAIT_PID finished at $(date)"
else
  echo "existing CSL driver pid=$WAIT_PID not running; starting artic-JEPA now"
fi

for s in 0 1 2; do
  echo "===== ARTIC-JEPA seed $s start $(date) ====="
  bash scripts/carrier_swap_predictive_contract_csl_articjepa.sh "$s"
  echo "===== ARTIC-JEPA seed $s diagnostic $(date) ====="
  CARRIER_ROOT=outputs/sota_chase/carrier_swap_csl_articjepa \
    CARRIER_PATTERN='carrier_{obj}_artic_s{seed}/best.pt' \
    DIAG_SEED="$s" \
    python scripts/diag_carrier_latent_csl.py > "$OUT/artic_latent_s${s}.txt" 2>&1 || true
  echo "===== ARTIC-JEPA seed $s done $(date) ====="
done

echo "===== ARTIC-JEPA ALL SEEDS DONE $(date) ====="
