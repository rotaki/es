#!/usr/bin/env bash
set -euo pipefail

# Copy gensort_200GiB.data from HDD (/tank) to local NVMe (skipped if already
# present), run the Myung baseline sweep. The dataset copy is RETAINED on
# exit so subsequent runs skip the slow HDD→NVMe copy. Only the scratch
# directory (intermediate run files) is cleaned up.
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
#
# To force cleanup of the local dataset (e.g. to free disk space):
#   KEEP_DATASET=0 baselines/myung/scripts/run_myung_bench.sh
# Or just delete it manually when you're done benchmarking:
#   rm ./datasets/gensort_200GiB.data

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
  # Keep the local dataset by default — the HDD→NVMe copy is slow and a
  # repeated sweep should reuse what's already on the SSD. Set KEEP_DATASET=0
  # to force removal (e.g. to free space when finished).
  if [[ -f "${GENSORT_LOCAL}" && "${KEEP_DATASET:-1}" != "1" ]]; then
    echo "[cleanup] KEEP_DATASET=0 → removing $(basename "${GENSORT_LOCAL}")..."
    rm -f "${GENSORT_LOCAL}"
  elif [[ -f "${GENSORT_LOCAL}" ]]; then
    echo "[cleanup] keeping $(basename "${GENSORT_LOCAL}") at ${GENSORT_LOCAL} (KEEP_DATASET=1)"
  fi
  # Scratch is always nuked — it's just intermediate run files, useless after.
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
