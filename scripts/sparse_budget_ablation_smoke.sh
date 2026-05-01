#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Smoke test for sparse_budget_ablation.sh
#
# Single-run sanity check: patch INDEX_BUDGET_PCT, build, run ONE benchmark on
# heavy_key (size-balanced), parse log, restore engine. Verifies the entire
# sweep pipeline end-to-end in ~5 minutes on a 50 GiB dataset.
#
# Pre-flight checks before running:
#   - engine.rs INDEX_BUDGET_PCT detected
#   - heavy_key dataset present
#
# Post-flight checks after running:
#   - engine.rs restored to original constant
#   - summary.tsv has exactly one data row
#   - all 12 metric columns are populated (not "-")
#
# Usage: ./scripts/sparse_budget_ablation_smoke.sh [datasets_dir]
#   datasets_dir defaults to ./datasets
# -----------------------------------------------------------------------------
set -euo pipefail

DATASETS_DIR="${1:-datasets}"
SMOKE_OUT="logs/sparse_ablation_smoke_$(date +%H-%M-%S)"
# Override via env var if you want a different smoke budget.
# Default is the smallest integer in our sweep range (1%), the most aggressive
# legal setting without engine source modification. Common alternatives:
#   SMOKE_PCT=1   smallest integer (page-floor-adjacent)
#   SMOKE_PCT=10  large (2x current default of 5%)
#   SMOKE_PCT=20  very large (engine will not fully use this)
SMOKE_PCT="${SMOKE_PCT:-1}"

echo "============================================================"
echo "  Sparse-index ablation: SMOKE TEST"
echo "============================================================"
echo "  Datasets dir: $DATASETS_DIR"
echo "  Output dir:   $SMOKE_OUT"
echo "  Test budget:  ${SMOKE_PCT}%"
echo ""

# ---- Pre-flight ----
echo ">>> Pre-flight checks"
ORIGINAL_PCT=$(grep -oP 'const INDEX_BUDGET_PCT: usize = \K[0-9]+' src/sort/core/engine.rs)
echo "    engine.rs INDEX_BUDGET_PCT = $ORIGINAL_PCT (will verify restore)"

if [[ ! -f "${DATASETS_DIR}/heavy_key.kvbin" ]]; then
  echo "    ERROR: ${DATASETS_DIR}/heavy_key.kvbin not found" >&2
  exit 1
fi

DS_GB=$(du -BG "${DATASETS_DIR}/heavy_key.kvbin" | cut -f1)
echo "    heavy_key dataset: $DS_GB"
echo ""

# ---- Run smoke sweep (1 run) ----
echo ">>> Running 1-cell smoke sweep (~5 min)..."
BUDGETS="$SMOKE_PCT" \
  DATASETS="heavy_key" \
  POLICIES="size-balanced" \
  COUNT_BASELINE_PCT=none \
  COOLDOWN_S=0 \
  ./scripts/sparse_budget_ablation.sh "$DATASETS_DIR" "$SMOKE_OUT"

echo ""
echo ">>> Post-flight checks"

# Check 1: engine.rs restored
RESTORED_PCT=$(grep -oP 'const INDEX_BUDGET_PCT: usize = \K[0-9]+' src/sort/core/engine.rs)
if [[ "$RESTORED_PCT" != "$ORIGINAL_PCT" ]]; then
  echo "    [FAIL] engine.rs not restored: was $ORIGINAL_PCT, now $RESTORED_PCT"
  exit 1
else
  echo "    [PASS] engine.rs restored: INDEX_BUDGET_PCT = $RESTORED_PCT"
fi

# Check 2: summary.tsv exists with one data row
SUMMARY="$SMOKE_OUT/summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
  echo "    [FAIL] summary.tsv missing at $SUMMARY"
  exit 1
fi
DATA_ROWS=$(($(wc -l < "$SUMMARY") - 1))
if [[ "$DATA_ROWS" != "1" ]]; then
  echo "    [FAIL] summary.tsv has $DATA_ROWS data rows (expected 1)"
  cat "$SUMMARY"
  exit 1
else
  echo "    [PASS] summary.tsv has 1 data row"
fi

# Check 3: status column is OK
STATUS=$(awk -F'\t' 'NR==2 {print $4}' "$SUMMARY")
if [[ "$STATUS" != "OK" ]]; then
  echo "    [FAIL] run status is '$STATUS' (expected OK)"
  exit 1
else
  echo "    [PASS] run completed successfully"
fi

# Check 4: all 12 metric columns populated (no "-" placeholders)
EMPTY_COLS=$(awk -F'\t' 'NR==2 {for (i=5; i<=16; i++) if ($i == "-") print i}' "$SUMMARY")
if [[ -n "$EMPTY_COLS" ]]; then
  echo "    [FAIL] metric columns missing: $EMPTY_COLS"
  cat "$SUMMARY"
  exit 1
else
  echo "    [PASS] all 12 metric columns populated"
fi

# Check 5: byte_imb should be ~1.00 for size-balanced (sanity)
BYTE_IMB=$(awk -F'\t' 'NR==2 {print $8}' "$SUMMARY")
echo "    [INFO] byte_imb = $BYTE_IMB  (expected ~1.00 for size-balanced)"
if (( $(echo "$BYTE_IMB > 1.5" | bc -l) )); then
  echo "    [WARN] byte_imb unexpectedly high (size-balanced should be <1.05)"
fi

# ---- Pretty-print ----
echo ""
echo "============================================================"
echo "  SMOKE TEST PASSED"
echo "============================================================"
echo ""
echo "Summary row:"
column -t -s $'\t' "$SUMMARY"
echo ""
echo "Full log:    $SMOKE_OUT/ablation_*.log"
echo "Summary TSV: $SUMMARY"
echo ""
echo "If all 4 PASS lines above are green, you can run the full sweep:"
echo "    ./scripts/sparse_budget_ablation.sh $DATASETS_DIR"
