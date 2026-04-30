#!/usr/bin/env bash
set -euo pipefail

# Copy gensort_200GiB.data from HDD (/tank) to local NVMe, run the Myung
# baseline sweep, then clean up the local copy. Mirrors scripts/run_both_bench.sh.
#
# Run from the repo root.
#
# Usage:
#   baselines/myung/scripts/run_myung_bench.sh [output_csv]
#
# Env overrides (passed through to myung_bench.sh):
#   MEM_MIB_LIST   space-separated memory budgets in MiB
#   THREADS_LIST   space-separated thread counts
#   ARB_LIST       "on off" — toggles the rw-lock arbitration
#   SAMPLE_RATIO   default 0.001
#
# Override the source dataset path with GENSORT_SRC=...

# ── paths ────────────────────────────────────────────────────────────────────
DATASETS_DIR="./datasets"
SCRATCH_DIR="./scratch_myung"

GENSORT_SRC="${GENSORT_SRC:-/tank/local/riki/datasets/gensort_200GiB.data}"
GENSORT_LOCAL="${DATASETS_DIR}/$(basename "${GENSORT_SRC}")"

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_CSV="${1:-./logs/myung_bench_${TS}.csv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/myung_bench.sh"
if [[ ! -x "${INNER_SCRIPT}" ]]; then
  echo "ERROR: inner script not executable: ${INNER_SCRIPT}" >&2
  exit 1
fi

# ── helpers ──────────────────────────────────────────────────────────────────
print_disk_info() {
  echo "=== Local SSD / disk info (${DATASETS_DIR}) ==="
  df -h "${DATASETS_DIR}" || true
  echo ""
}

cleanup() {
  if [[ -f "${GENSORT_LOCAL}" && "${KEEP_LOCAL_COPY:-0}" != "1" ]]; then
    echo "[cleanup] Removing $(basename "${GENSORT_LOCAL}")..."
    rm -f "${GENSORT_LOCAL}"
  fi
  if [[ -d "${SCRATCH_DIR}" ]]; then
    echo "[cleanup] Removing scratch dir ${SCRATCH_DIR}..."
    rm -rf "${SCRATCH_DIR}"
  fi
  sync
}
# Run cleanup on normal exit AND on Ctrl-C / errors.
trap cleanup EXIT

# ── setup ────────────────────────────────────────────────────────────────────
mkdir -p "${DATASETS_DIR}"
mkdir -p "$(dirname "${OUT_CSV}")"
print_disk_info

# ── copy ─────────────────────────────────────────────────────────────────────
echo "=== myung baseline benchmark ==="

if [[ ! -f "${GENSORT_SRC}" ]]; then
  echo "ERROR: gensort source not found: ${GENSORT_SRC}" >&2
  exit 1
fi

if [[ -f "${GENSORT_LOCAL}" ]]; then
  echo "[skip] $(basename "${GENSORT_LOCAL}") already present locally, reusing."
  # Tell cleanup not to delete a file the user staged ahead of time.
  KEEP_LOCAL_COPY=1
else
  echo "[copy] Copying $(basename "${GENSORT_SRC}") from $(dirname "${GENSORT_SRC}")..."
  cp "${GENSORT_SRC}" "${GENSORT_LOCAL}"
  echo "[copy] Done."
fi

print_disk_info

# ── run ──────────────────────────────────────────────────────────────────────
echo "[run] ${INNER_SCRIPT} ${GENSORT_LOCAL} ${SCRATCH_DIR} ${OUT_CSV}"
"${INNER_SCRIPT}" "${GENSORT_LOCAL}" "${SCRATCH_DIR}" "${OUT_CSV}"

echo "=== myung benchmark complete; results: ${OUT_CSV} ==="
