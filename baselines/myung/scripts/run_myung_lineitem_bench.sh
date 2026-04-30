#!/usr/bin/env bash
set -euo pipefail

# Copy lineitem .kvbin + .kvbin.idx from HDD (/tank) to local NVMe, run the
# Myung kvbin sweep, then clean up. Sibling of run_myung_bench.sh.
#
# Run from the repo root.
#
# Usage:
#   baselines/myung/scripts/run_myung_lineitem_bench.sh [output_csv]
#
# Env overrides:
#   LINEITEM_SRC_GLOB   default: /tank/local/riki/datasets/lineitem_sf500.k-8-9-13-14-15.v-0-3.kvbin*
#                       (must match BOTH the data file and the .idx sidecar)
#   plus all env vars consumed by myung_kvbin_bench.sh:
#     MEM_MIB_LIST, THREADS_LIST, ARB_LIST, SAMPLE_RATIO,
#     SAMPLE_RECORDS_PER_ANCHOR, CGROUP_MODE, RCLONE_REMOTE,
#     SLEEP_BETWEEN_CONFIGS_SECONDS

# ── paths ────────────────────────────────────────────────────────────────────
DATASETS_DIR="./datasets"
SCRATCH_DIR="./scratch_myung_lineitem"

LINEITEM_SRC_GLOB="${LINEITEM_SRC_GLOB:-/tank/local/riki/datasets/lineitem_sf500.k-8-9-13-14-15.v-0-3.kvbin*}"

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_CSV="${1:-./logs/myung_lineitem_bench_${TS}.csv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/myung_kvbin_bench.sh"
if [[ ! -x "${INNER_SCRIPT}" ]]; then
  echo "ERROR: inner script not executable: ${INNER_SCRIPT}" >&2
  echo "  hint: chmod +x ${INNER_SCRIPT}" >&2
  exit 1
fi

# ── helpers ──────────────────────────────────────────────────────────────────
print_disk_info() {
  echo "=== Local SSD / disk info (${DATASETS_DIR}) ==="
  df -h "${DATASETS_DIR}" || true
  echo ""
}

glob_exists() {
  compgen -G "$1" > /dev/null 2>&1
}

# Track which dest files we copied (so we only delete those, not user-staged ones).
declare -a COPIED_FILES=()

cleanup() {
  if [[ ${#COPIED_FILES[@]} -gt 0 ]]; then
    echo "[cleanup] Removing ${#COPIED_FILES[@]} copied lineitem file(s)..."
    rm -f "${COPIED_FILES[@]}"
  fi
  if [[ -d "${SCRATCH_DIR}" ]]; then
    echo "[cleanup] Removing scratch dir ${SCRATCH_DIR}..."
    rm -rf "${SCRATCH_DIR}"
  fi
  sync
}
# Clean up on Ctrl-C / errors as well as normal exit.
trap cleanup EXIT

# ── setup ────────────────────────────────────────────────────────────────────
mkdir -p "${DATASETS_DIR}"
mkdir -p "$(dirname "${OUT_CSV}")"
print_disk_info

# ── pre-flight ───────────────────────────────────────────────────────────────
echo "=== myung lineitem (kvbin) baseline benchmark ==="

if ! glob_exists "${LINEITEM_SRC_GLOB}"; then
  echo "ERROR: No lineitem files match: ${LINEITEM_SRC_GLOB}" >&2
  exit 1
fi

mapfile -t SRC_FILES < <(compgen -G "${LINEITEM_SRC_GLOB}")
if [[ ${#SRC_FILES[@]} -eq 0 ]]; then
  echo "ERROR: glob expanded to zero files: ${LINEITEM_SRC_GLOB}" >&2
  exit 1
fi

# Identify the data file (no .idx suffix) and the sidecar (.idx) explicitly.
# We need the exact paths to pass to the inner bench.
DATA_FILE=""
IDX_FILE=""
for f in "${SRC_FILES[@]}"; do
  case "$f" in
    *.idx) IDX_FILE="$f" ;;
    *.kvbin) DATA_FILE="$f" ;;
  esac
done

if [[ -z "$DATA_FILE" || -z "$IDX_FILE" ]]; then
  echo "ERROR: glob matched files but couldn't identify both .kvbin and .kvbin.idx:" >&2
  printf '  %s\n' "${SRC_FILES[@]}" >&2
  exit 1
fi

# ── copy ─────────────────────────────────────────────────────────────────────
LOCAL_DATA="${DATASETS_DIR}/$(basename "${DATA_FILE}")"
LOCAL_IDX="${DATASETS_DIR}/$(basename "${IDX_FILE}")"

copy_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ -f "$dst" ]]; then
    echo "[skip] $(basename "$dst") already present locally."
  else
    echo "[copy] $(basename "$src") -> ${DATASETS_DIR}/ ..."
    cp "$src" "$dst"
    COPIED_FILES+=("$dst")
  fi
}

copy_if_missing "$DATA_FILE" "$LOCAL_DATA"
copy_if_missing "$IDX_FILE" "$LOCAL_IDX"

print_disk_info

# ── run ──────────────────────────────────────────────────────────────────────
echo "[run] ${INNER_SCRIPT} ${LOCAL_DATA} ${LOCAL_IDX} ${SCRATCH_DIR} ${OUT_CSV}"
"${INNER_SCRIPT}" "${LOCAL_DATA}" "${LOCAL_IDX}" "${SCRATCH_DIR}" "${OUT_CSV}"

echo "=== myung lineitem benchmark complete; results: ${OUT_CSV} ==="
