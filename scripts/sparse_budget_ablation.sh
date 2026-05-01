#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Sparse-index budget ablation (R2 reviewer question)
#
# Sweeps INDEX_BUDGET_PCT over {1, 2, 3, 4, 5} (configurable) and runs the
# kvbin sort benchmark on freq_key + heavy_key with size-balanced partitioning.
# Optionally adds count-balanced runs at a single budget for the policy
# baseline (count-balanced is structurally invariant to budget; one run is
# enough).
#
# What we measure per (dataset, policy, budget):
#   - end-to-end wall-clock                    (overall time)
#   - run-gen / merge phase split              (overall time breakdown)
#   - byte imbalance max/avg                   (boundary accuracy / byte)
#   - entry imbalance max/avg                  (boundary accuracy / dup)
#   - time imbalance max/avg                   (straggler effect)
#   - read amplification + excess reads %      (seek-vs-read tradeoff)
#   - sparse-index pages used + MB             (actual budget usage)
#
# Implementation: the sparse-index budget is a compile-time `const` in
# `src/sort/core/engine.rs`. We sed-patch it before each build, build the
# example binary, and run the benchmark. A trap ensures we restore the
# original constant even on failure.
#
# Usage:
#   ./scripts/sparse_budget_ablation.sh <datasets_dir> [output_dir]
#
# Env-var overrides:
#   THREADS              merge + run-gen thread count     (default: 8)
#   MEM_GB               memory budget in GiB              (default: 2)
#   BUDGETS              CSV of percentages to sweep       (default: "1 2 3 5 10")
#   DATASETS             space-list of dataset names       (default: "heavy_key")
#                          options: freq_key heavy_key heavy_range
#   POLICIES             space-list of partition types     (default: "size-balanced")
#   COUNT_BASELINE_PCT   single budget for count-balanced  (default: 5; "none" to skip)
#   COOLDOWN_S           sleep between runs                (default: 30)
#   MIN_FREE_GB          required free GB on OUT_DIR vol.  (default: 500)
#   SSD_CLEANUP_SLEEP_S  settle time after rm + sync       (default: 3)
#   DO_FSTRIM            issue fstrim after cleanup        (default: 1)
#   FREQ_KEY_ROWS        row count for freq_key gen        (default: 200 GiB worth)
#   HEAVY_KEY_ROWS       row count for heavy_key gen       (default: 200 GiB worth)
#   HEAVY_RANGE_ROWS     row count for heavy_range gen     (default: 200 GiB worth)
#   ABLATION_FORCE       skip disk-space pre-flight check  (default: 0)
#
# Examples:
#   # Minimum useful sweep for the R2 reviewer (~1.5–2 hours)
#   ./scripts/sparse_budget_ablation.sh datasets logs/r2_minimal
#
#   # Wider sweep across all three workloads (~5–6 hours)
#   DATASETS="freq_key heavy_key heavy_range" \
#       ./scripts/sparse_budget_ablation.sh datasets logs/r2_full
#
# -----------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# CLI / environment
# ---------------------------------------------------------------------------
if [[ ${1-} == "" ]]; then
  echo "Usage: $0 <datasets_dir> [output_dir]" >&2
  exit 1
fi

DATASETS_DIR="$1"
if [[ ! -d "$DATASETS_DIR" ]]; then
  echo "Datasets dir not found: $DATASETS_DIR" >&2
  exit 1
fi

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/sparse_ablation_${TS}"}
mkdir -p "$OUT_DIR"

THREADS="${THREADS:-8}"
MEM_GB="${MEM_GB:-2}"
PAGE_SIZE_KB=64
BUDGETS="${BUDGETS:-1 2 3 5 10}"
# Default: heavy_key only.  It has the only kind of skew that exercises sparse-
# index sampling hardest (frequency + size skew at the same time), so it is the
# right and sufficient workload for the budget-sensitivity ablation.  Override
# with DATASETS="freq_key heavy_key heavy_range" for a wider sweep.
DATASETS="${DATASETS:-heavy_key}"
POLICIES="${POLICIES:-size-balanced}"
COUNT_BASELINE_PCT="${COUNT_BASELINE_PCT:-5}"
COOLDOWN_S="${COOLDOWN_S:-30}"

# 200 GiB row counts (matched to scripts/kvbin_sort_bench_new.sh)
#   freq_key:    528 B/row             → 200×1024³ / 528  ≈ 406_720_388
#   heavy_key:   ~3408 B/row (10% dup) → 200×1024³ / 3408 ≈  63_013_489
#   heavy_range: ~6672 B/row (20% hot) → 200×1024³ / 6672 ≈  32_186_505
FREQ_KEY_ROWS="${FREQ_KEY_ROWS:-406720388}"
HEAVY_KEY_ROWS="${HEAVY_KEY_ROWS:-63013489}"
HEAVY_RANGE_ROWS="${HEAVY_RANGE_ROWS:-32186505}"

ENGINE_FILE="src/sort/core/engine.rs"
ENGINE_BACKUP="${ENGINE_FILE}.ablation_bak"
BINARY="./target/release/examples/kvbin_benchmark_cli"
MIN_FREE_GB="${MIN_FREE_GB:-500}"   # require this many GB free at start

# ---------------------------------------------------------------------------
# Safety: detect current INDEX_BUDGET_PCT, save a backup, and ensure we
# restore on normal exit AND on signals (Ctrl-C / SIGTERM).
# ---------------------------------------------------------------------------
if [[ ! -f "$ENGINE_FILE" ]]; then
  echo "Engine file not found: $ENGINE_FILE (run from repo root)" >&2
  exit 1
fi

ORIGINAL_PCT=$(grep -oP 'const INDEX_BUDGET_PCT: usize = \K[0-9]+' "$ENGINE_FILE")
if [[ -z "$ORIGINAL_PCT" ]]; then
  echo "Could not find INDEX_BUDGET_PCT in $ENGINE_FILE" >&2
  exit 1
fi

# Backup the engine source so we can recover even if sed-restore fails
cp -f "$ENGINE_FILE" "$ENGINE_BACKUP"
echo "Detected INDEX_BUDGET_PCT = $ORIGINAL_PCT"
echo "Backup saved to $ENGINE_BACKUP"

restore_engine() {
  local exit_code=$?
  local current
  current=$(grep -oP 'const INDEX_BUDGET_PCT: usize = \K[0-9]+' "$ENGINE_FILE" 2>/dev/null || echo "?")
  if [[ "$current" != "$ORIGINAL_PCT" ]]; then
    echo ""
    echo ">>> Restoring INDEX_BUDGET_PCT to $ORIGINAL_PCT (was $current)..." >&2
    if ! sed -i "s/const INDEX_BUDGET_PCT: usize = [0-9]\+;/const INDEX_BUDGET_PCT: usize = ${ORIGINAL_PCT};/" "$ENGINE_FILE" 2>/dev/null; then
      echo ">>> sed failed; falling back to backup restore" >&2
      cp -f "$ENGINE_BACKUP" "$ENGINE_FILE"
    fi
    # Verify restoration
    current=$(grep -oP 'const INDEX_BUDGET_PCT: usize = \K[0-9]+' "$ENGINE_FILE" 2>/dev/null || echo "?")
    if [[ "$current" == "$ORIGINAL_PCT" ]]; then
      echo ">>> Restored OK (now $current)"
    else
      echo "!!! ERROR: $ENGINE_FILE may be corrupted. Backup at $ENGINE_BACKUP" >&2
      echo "!!! Manually restore: cp $ENGINE_BACKUP $ENGINE_FILE" >&2
    fi
  fi
  # Best-effort cleanup of any leftover ablation tmp dirs in case a run was killed
  if [[ -n "${OUT_DIR:-}" && -d "$OUT_DIR" ]]; then
    find "$OUT_DIR" -maxdepth 1 -name 'ablation_*_tmp' -type d -exec rm -rf {} + 2>/dev/null || true
  fi
  return $exit_code
}
# Trap on normal exit AND on Ctrl-C / SIGTERM
trap restore_engine EXIT
trap 'echo ""; echo ">>> Interrupted, restoring engine..."; exit 130' INT TERM

# ---------------------------------------------------------------------------
# Pre-flight: clean stale tmp dirs and check disk space
# ---------------------------------------------------------------------------
# Clean any leftover ablation tmp dirs from a previously-killed run.  We use
# a plain rm here because robust_cleanup is defined further down; stale dirs
# at startup are typically empty or partial, so the speed hit is minor.
mkdir -p "$OUT_DIR"
find . -maxdepth 4 -name 'ablation_*_tmp' -type d -exec rm -rf {} + 2>/dev/null || true
sync

# Disk space check (best-effort).  We check both the OUT_DIR volume (where temp
# dirs and logs live) and the DATASETS_DIR volume (where ~200 GiB of data per
# missing dataset will be generated).  The two may be on different mounts.
mkdir -p "$DATASETS_DIR"
free_gb=$(df -BG --output=avail "$OUT_DIR" 2>/dev/null | tail -1 | tr -d ' G' || echo 0)
free_data_gb=$(df -BG --output=avail "$DATASETS_DIR" 2>/dev/null | tail -1 | tr -d ' G' || echo 0)

# Estimate how many 200 GiB datasets are missing → drives the data-volume need
missing_count=0
for d in $DATASETS; do
  if [[ ! -f "${DATASETS_DIR}/${d}.kvbin" || ! -f "${DATASETS_DIR}/${d}.kvbin.idx" ]]; then
    missing_count=$(( missing_count + 1 ))
  fi
done
data_required_gb=$(( missing_count * 210 ))   # 210 GB per dataset (200 GiB + slack)

if [[ "$free_gb" =~ ^[0-9]+$ ]] && (( free_gb < MIN_FREE_GB )); then
  echo "!!! WARNING: only ${free_gb} GB free on OUT_DIR volume (< MIN_FREE_GB=${MIN_FREE_GB} GB)" >&2
  echo "!!! Each 200 GiB run needs ~400 GB of temp space (run files + sparse index)." >&2
  if [[ "${ABLATION_FORCE:-0}" != "1" ]]; then
    echo "!!! Set ABLATION_FORCE=1 to override." >&2
    exit 1
  fi
fi
if (( missing_count > 0 )) && [[ "$free_data_gb" =~ ^[0-9]+$ ]] && (( free_data_gb < data_required_gb )); then
  echo "!!! WARNING: only ${free_data_gb} GB free on DATASETS_DIR volume" >&2
  echo "!!! Need ~${data_required_gb} GB to generate ${missing_count} missing dataset(s)." >&2
  if [[ "${ABLATION_FORCE:-0}" != "1" ]]; then
    echo "!!! Set ABLATION_FORCE=1 to override." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Per-thread parameters (computed once)
# ---------------------------------------------------------------------------
RG_BUF_MB=$(echo "scale=2; ($MEM_GB * 1024) / $THREADS" | bc)
MAX_FANIN=$(echo "($MEM_GB * 1024 * 1024 * 95 / 100) / ($THREADS * $PAGE_SIZE_KB) - 1" | bc)

# ---------------------------------------------------------------------------
# Cache clearing (best-effort)
# ---------------------------------------------------------------------------
clear_cache_if_available() {
  if [[ -x /usr/local/sbin/clearcache3.sh ]]; then
    if [[ $(id -u) -eq 0 ]]; then
      /usr/local/sbin/clearcache3.sh || echo "Warning: clearcache3.sh failed" >&2
    elif command -v sudo >/dev/null 2>&1; then
      sudo /usr/local/sbin/clearcache3.sh 2>/dev/null || echo "Warning: cache clear failed (sudo)" >&2
    fi
  fi
}

# ---------------------------------------------------------------------------
# Robust SSD cleanup of a directory tree.
#
# `rm -rf` on a 400 GB tree of run files is slow and the FS may not reclaim
# the space accounting before the next experiment starts.  We:
#   1. truncate every regular file to size 0 first  → releases extents fast
#      (much faster than rm's per-file extent tree walk on big files)
#   2. rm -rf the (now empty) tree                  → removes inodes / dirs
#   3. sync                                         → flush journal commits
#   4. optional fstrim                              → ask SSD to physically
#                                                     reclaim blocks (TRIM)
#   5. sleep                                        → let kernel settle
#
# This makes the disk-space accounting accurate before we start the next run,
# which matters on SSD-only servers where the temp dir alone can be hundreds
# of GB and back-to-back runs would otherwise stack up.
# ---------------------------------------------------------------------------
SSD_CLEANUP_SLEEP_S="${SSD_CLEANUP_SLEEP_S:-3}"
DO_FSTRIM="${DO_FSTRIM:-1}"   # set DO_FSTRIM=0 to skip (default: try)

robust_cleanup() {
  local dir="$1"
  if [[ -z "$dir" || ! -e "$dir" ]]; then return 0; fi

  local before_str before_kb after_kb
  before_str=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "?")
  before_kb=$(df -k "$dir" 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
  echo "  Cleaning ${dir} (size=${before_str}) ..."

  # 1. Truncate large files in place first — much faster than rm -rf for >GB files.
  find "$dir" -type f -size +1M -exec truncate -s 0 {} + 2>/dev/null || true
  sync

  # 2. Remove the tree (now mostly empty inodes + small files).
  rm -rf "$dir"
  sync

  # 3. Optional TRIM on the volume that held this dir, so the SSD physically
  #    reclaims blocks rather than just the FS counter.  Best-effort, silent
  #    if fstrim is unavailable, requires no privileges, or already discarded.
  if [[ "$DO_FSTRIM" == "1" ]] && command -v fstrim >/dev/null 2>&1; then
    local mountpoint
    mountpoint=$(df --output=target "$(dirname "$dir")" 2>/dev/null | tail -1)
    if [[ -n "$mountpoint" && -d "$mountpoint" ]]; then
      if [[ $(id -u) -eq 0 ]]; then
        fstrim "$mountpoint" >/dev/null 2>&1 || true
      elif command -v sudo >/dev/null 2>&1; then
        sudo -n fstrim "$mountpoint" >/dev/null 2>&1 || true   # -n: no prompt
      fi
    fi
  fi

  # 4. Brief settle window so the kernel finishes any deferred work.
  sleep "$SSD_CLEANUP_SLEEP_S"

  after_kb=$(df -k "$(dirname "$dir")" 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
  if [[ "$after_kb" =~ ^[0-9]+$ && "$before_kb" =~ ^[0-9]+$ ]]; then
    local freed_gb=$(( (after_kb - before_kb) / 1024 / 1024 ))
    echo "  Cleanup done. Free space delta: +${freed_gb} GB"
  fi
}

# ---------------------------------------------------------------------------
# Build generator binaries once (independent of INDEX_BUDGET_PCT) and use them
# to materialize datasets if they are missing on disk.  Mirrors the pattern in
# scripts/kvbin_sort_bench_new.sh so dataset paths stay consistent.
# ---------------------------------------------------------------------------
GEN_FREQ="./target/release/examples/gen_freq_key_kvbin"
GEN_HEAVY_KEY="./target/release/examples/gen_heavy_key_kvbin"
GEN_HEAVY_RANGE="./target/release/examples/gen_heavy_range_kvbin"

build_generators() {
  echo ""
  echo "=========================================================="
  echo "  Building dataset generators (release)..."
  echo "=========================================================="
  cargo build --release \
    --example gen_freq_key_kvbin \
    --example gen_heavy_key_kvbin \
    --example gen_heavy_range_kvbin >/dev/null
}

generate_if_missing() {
  local name="$1"
  local kvbin="${DATASETS_DIR}/${name}.kvbin"
  local idx="${DATASETS_DIR}/${name}.kvbin.idx"

  if [[ -f "$kvbin" && -f "$idx" ]]; then
    local size
    size=$(du -sh "$kvbin" 2>/dev/null | cut -f1)
    echo "  ${name}: already present (${size}), skipping generation"
    return 0
  fi

  local rows rec_desc gen_bin
  case "$name" in
    freq_key)
      rows=$FREQ_KEY_ROWS
      rec_desc="528 B/row (fixed)"
      gen_bin="$GEN_FREQ"
      ;;
    heavy_key)
      rows=$HEAVY_KEY_ROWS
      rec_desc="~3,408 B/row (10% heavy, variable)"
      gen_bin="$GEN_HEAVY_KEY"
      ;;
    heavy_range)
      rows=$HEAVY_RANGE_ROWS
      rec_desc="~6,672 B/row (20% hot range, variable)"
      gen_bin="$GEN_HEAVY_RANGE"
      ;;
    *)
      echo "Unknown dataset: $name (cannot auto-generate)" >&2
      echo "Pre-generate ${name}.kvbin + .kvbin.idx in $DATASETS_DIR yourself." >&2
      return 1
      ;;
  esac

  if [[ ! -x "$gen_bin" ]]; then
    echo "Generator missing: $gen_bin — did the cargo build succeed?" >&2
    return 1
  fi

  mkdir -p "$DATASETS_DIR"

  echo ""
  echo "================================================================"
  echo "  Generating: $name (target ~200 GiB)"
  echo "  Rows:         $(printf "%'d" "$rows")"
  echo "  Record size:  $rec_desc"
  echo "  Output:       $kvbin"
  echo "  Index:        $idx"
  echo "================================================================"

  local start_ts end_ts elapsed_s
  start_ts=$(date +%s)
  "$gen_bin" --out "$kvbin" --idx "$idx" --rows "$rows"
  end_ts=$(date +%s)
  elapsed_s=$(( end_ts - start_ts ))
  local mins=$(( elapsed_s / 60 ))
  local secs=$(( elapsed_s % 60 ))

  echo ""
  echo "  Done generating $name in ${mins}m ${secs}s"
  ls -lh "$kvbin" "$idx"
  echo ""
}

# ---------------------------------------------------------------------------
# Build one variant of the binary at INDEX_BUDGET_PCT = $1
# ---------------------------------------------------------------------------
patch_and_build() {
  local pct="$1"
  echo ""
  echo "=========================================================="
  echo "  Patching INDEX_BUDGET_PCT = ${pct} and rebuilding..."
  echo "=========================================================="
  sed -i "s/const INDEX_BUDGET_PCT: usize = [0-9]\+;/const INDEX_BUDGET_PCT: usize = ${pct};/" "$ENGINE_FILE"

  # Verify the patch landed
  local current
  current=$(grep -oP 'const INDEX_BUDGET_PCT: usize = \K[0-9]+' "$ENGINE_FILE")
  if [[ "$current" != "$pct" ]]; then
    echo "ERROR: failed to patch INDEX_BUDGET_PCT to $pct (got $current)" >&2
    exit 1
  fi

  cargo build --release --example kvbin_benchmark_cli >/dev/null
  if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: build did not produce $BINARY" >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Run one (dataset, policy, budget) combination
# ---------------------------------------------------------------------------
run_one() {
  local pct="$1"
  local dataset="$2"
  local partition_type="$3"

  local kvbin="${DATASETS_DIR}/${dataset}.kvbin"
  local idx="${DATASETS_DIR}/${dataset}.kvbin.idx"
  if [[ ! -f "$kvbin" || ! -f "$idx" ]]; then
    echo "Skipping ${dataset} (missing $kvbin or $idx)"
    echo -e "${pct}\t${dataset}\t${partition_type}\tMISSING\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-" >> "$RESULTS_FILE"
    return 0
  fi

  local name="ablation_${dataset}_${partition_type}_pct${pct}_T${THREADS}_Mem${MEM_GB}GB"
  local log_file="${OUT_DIR}/${name}.log"
  local temp_dir="${OUT_DIR}/${name}_tmp"
  mkdir -p "$temp_dir"

  CURRENT_CELL=$(( CURRENT_CELL + 1 ))
  local elapsed=$(( $(date +%s) - SWEEP_START_TS ))
  local eta_str="--"
  if (( CURRENT_CELL > 1 && elapsed > 0 )); then
    local avg=$(( elapsed / (CURRENT_CELL - 1) ))
    local remaining=$(( avg * (TOTAL_CELLS - CURRENT_CELL + 1) ))
    eta_str=$(printf "%dh%02dm" $((remaining/3600)) $(((remaining%3600)/60)))
  fi

  echo ""
  echo "----------------------------------------------------------"
  echo "  [Cell ${CURRENT_CELL}/${TOTAL_CELLS}, elapsed=$(printf '%dh%02dm' $((elapsed/3600)) $(((elapsed%3600)/60))), ETA=${eta_str}]"
  echo "  RUN: budget=${pct}%, dataset=${dataset}, policy=${partition_type}"
  echo "  Started at: $(date '+%H:%M:%S')"
  echo "----------------------------------------------------------"

  clear_cache_if_available

  local status="OK"
  if "$BINARY" \
      -n "$name" \
      -i "$kvbin" \
      --index "$idx" \
      --run-gen-threads "$THREADS" \
      --merge-threads "$THREADS" \
      --rg-buf-mb "$RG_BUF_MB" \
      --merge-fanin "$MAX_FANIN" \
      --warmup-runs 0 \
      --benchmark-runs 1 \
      --cooldown-seconds "$COOLDOWN_S" \
      --partition-type "$partition_type" \
      --dir "$temp_dir" \
      2>&1 | tee "$log_file"; then
    status="OK"
  else
    status="FAIL"
  fi

  # Robust SSD cleanup so the next experiment starts on a clean volume
  robust_cleanup "$temp_dir"
  parse_log "$pct" "$dataset" "$partition_type" "$status" "$log_file"
}

# ---------------------------------------------------------------------------
# Parse one log file and append a CSV row to RESULTS_FILE.
# Extracts: total_s, runGen_s, merge_s, byte_imbalance, entry_imbalance,
#           time_imbalance, io_imbalance, read_amp, excess_pct, sparse_pages,
#           sparse_mb, sparse_pct_of_merge_mem
# ---------------------------------------------------------------------------
parse_log() {
  local pct="$1" dataset="$2" partition_type="$3" status="$4" log="$5"

  if [[ ! -f "$log" ]]; then
    echo -e "${pct}\t${dataset}\t${partition_type}\t${status}\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-" >> "$RESULTS_FILE"
    return
  fi

  # Total wallclock from "Run 1" row in summary table (first numeric after RG col)
  local total_s rg_s merge_s
  total_s=$(grep -oP '^Run 1\s+[0-9.]+\s+[0-9]+\s+[0-9]+\s+[0-9]+\s+\K[0-9.]+' "$log" | head -1)
  rg_s=$(grep -oP '\(R\) time: \K[0-9]+' "$log" | head -1)
  merge_s=$(grep -oP '\(M\) time: \K[0-9]+' "$log" | head -1)

  # Convert ms -> s for run-gen and merge
  if [[ -n "$rg_s" ]]; then rg_s=$(echo "scale=1; $rg_s / 1000" | bc); fi
  if [[ -n "$merge_s" ]]; then merge_s=$(echo "scale=1; $merge_s / 1000" | bc); fi

  # Imbalance factors (max/avg)
  local byte_imb entry_imb time_imb io_imb
  byte_imb=$(grep -oP 'Byte imbalance factor \(max/avg\): \K[0-9.]+' "$log" | head -1)
  entry_imb=$(grep -oP 'Entry imbalance factor \(max/avg\): \K[0-9.]+' "$log" | head -1)
  time_imb=$(grep -oP 'Time imbalance factor \(max/avg\): \K[0-9.]+' "$log" | head -1)
  io_imb=$(grep -oP 'I/O imbalance factor \(max/avg\): \K[0-9.]+' "$log" | head -1)

  # Read amplification
  local read_amp excess_pct
  read_amp=$(grep -oP 'Read amplification factor: \K[0-9.]+' "$log" | head -1)
  excess_pct=$(grep -oP 'Excess reads: \K[0-9.]+' "$log" | head -1)

  # Sparse-index actual usage
  # "15743 pages (61.50 MB)"
  local sparse_pages sparse_mb
  sparse_pages=$(grep -oP 'Merge input sparse index:.*?\| \K[0-9]+(?= pages)' "$log" | head -1)
  sparse_mb=$(grep -oP 'Merge input sparse index:.*?pages \(\K[0-9.]+(?= MB)' "$log" | head -1)

  # Compute sparse % of merge memory
  local sparse_pct merge_mem_mb
  merge_mem_mb=$(grep -oP 'Merge memory \(MB\): \K[0-9.]+' "$log" | head -1)
  if [[ -n "$sparse_mb" && -n "$merge_mem_mb" ]]; then
    sparse_pct=$(echo "scale=3; $sparse_mb * 100 / $merge_mem_mb" | bc)
  fi

  echo -e "${pct}\t${dataset}\t${partition_type}\t${status}\t${total_s:--}\t${rg_s:--}\t${merge_s:--}\t${byte_imb:--}\t${entry_imb:--}\t${time_imb:--}\t${io_imb:--}\t${read_amp:--}\t${excess_pct:--}\t${sparse_pages:--}\t${sparse_mb:--}\t${sparse_pct:--}" >> "$RESULTS_FILE"
}

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
RESULTS_FILE="${OUT_DIR}/summary.tsv"
echo -e "budget_pct\tdataset\tpolicy\tstatus\ttotal_s\trun_gen_s\tmerge_s\tbyte_imb\tentry_imb\ttime_imb\tio_imb\tread_amp\texcess_pct\tsparse_pages\tsparse_mb\tsparse_pct" > "$RESULTS_FILE"

# Compute total cells for progress reporting
NUM_BUDGETS=$(echo $BUDGETS | wc -w)
NUM_DATASETS=$(echo $DATASETS | wc -w)
NUM_POLICIES=$(echo $POLICIES | wc -w)
TOTAL_CELLS=$(( NUM_BUDGETS * NUM_DATASETS * NUM_POLICIES ))
if [[ "$COUNT_BASELINE_PCT" != "none" ]]; then
  TOTAL_CELLS=$(( TOTAL_CELLS + NUM_DATASETS ))
fi
CURRENT_CELL=0
SWEEP_START_TS=$(date +%s)

echo ""
echo "============================================================"
echo "  Sparse-index budget ablation"
echo "  Output dir:   $OUT_DIR"
echo "  Datasets dir: $DATASETS_DIR"
echo "  Datasets:     $DATASETS"
echo "  Policies:     $POLICIES"
echo "  Budgets (%):  $BUDGETS"
echo "  Threads:      $THREADS"
echo "  Memory:       ${MEM_GB} GB"
echo "  rg_buf:       $RG_BUF_MB MB / thread"
echo "  max_fanin:    $MAX_FANIN"
echo "  Total cells:  $TOTAL_CELLS"
echo "  Free disk:    ${free_gb} GB"
echo "============================================================"

# Build generators (no-op if cached) and materialize any missing datasets
build_generators
echo ""
echo "=========================================================="
echo "  Checking / generating datasets in $DATASETS_DIR"
echo "=========================================================="
for d in $DATASETS; do
  generate_if_missing "$d"
done

# Show final dataset sizes
echo ""
echo "Final dataset sizes:"
for d in $DATASETS; do
  if [[ -f "${DATASETS_DIR}/${d}.kvbin" ]]; then
    sz=$(du -BG "${DATASETS_DIR}/${d}.kvbin" | cut -f1)
    echo "  ${d}: ${sz}"
  fi
done
echo ""

# Sweep budgets × datasets × size-balanced policy
for pct in $BUDGETS; do
  patch_and_build "$pct"
  for dataset in $DATASETS; do
    for policy in $POLICIES; do
      run_one "$pct" "$dataset" "$policy"
    done
  done
done

# Optional count-balanced baseline at one budget (structurally invariant)
if [[ "$COUNT_BASELINE_PCT" != "none" ]]; then
  echo ""
  echo "=========================================================="
  echo "  Count-balanced baseline at INDEX_BUDGET_PCT = ${COUNT_BASELINE_PCT}"
  echo "=========================================================="
  patch_and_build "$COUNT_BASELINE_PCT"
  for dataset in $DATASETS; do
    run_one "$COUNT_BASELINE_PCT" "$dataset" "count-balanced"
  done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  ABLATION COMPLETE"
echo "============================================================"
echo "  Logs:          $OUT_DIR/*.log"
echo "  Summary TSV:   $RESULTS_FILE"
echo ""

# Pretty-print the summary
if command -v column >/dev/null 2>&1; then
  echo "Summary:"
  column -t -s $'\t' "$RESULTS_FILE"
else
  cat "$RESULTS_FILE"
fi

echo ""
echo "Restoring $ENGINE_FILE to INDEX_BUDGET_PCT = $ORIGINAL_PCT..."
# (trap will run too, this is idempotent)
