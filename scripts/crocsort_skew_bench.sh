#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# CrocSort skew benchmark: same workload, two distributions.
#
# Exp 2.2 — HeavyKey-Extreme
#   Generator: gen_heavy_key_extreme_kvbin (99.9% rows on one key, 0.1% uniform).
#   Run CrocSort partitioning (--partition-type=size-balanced) and report the
#   measured boundary imbalance (byte / entry / time / I/O imbalance).
#
# Exp 2.3 — Multimodal
#   Generator: gen_multimodal_kvbin (mixture of K Gaussians, default K=3).
#   Same CrocSort run, same metrics.
#
# Disk discipline
#   At 200 GiB/dataset + ~200 GiB sort temp, two datasets can't coexist on a
#   typical bench NVMe. Default flow is sequential:
#       generate Exp 2.2 dataset → run Exp 2.2 → robust_cleanup dataset
#       generate Exp 2.3 dataset → run Exp 2.3 → robust_cleanup dataset
#   Use KEEP_DATASETS=1 to retain datasets between experiments (assumes you
#   have ~600 GiB free). The robust_cleanup pattern (truncate → rm → sync →
#   fstrim → settle) is borrowed from sparse_budget_ablation.sh — needed
#   because plain `rm -rf` on hundred-GB trees doesn't release blocks fast
#   enough for back-to-back runs to see correct free-space accounting.
#
# Usage:
#   ./scripts/crocsort_skew_bench.sh <datasets_dir> [output_dir]
#
# Env-var overrides:
#   THREADS             merge + run-gen thread count       (default: 8)
#   MEM_GB              memory budget in GiB                (default: 2)
#   COOLDOWN_S          sleep between runs                  (default: 30)
#   EXTREME_ROWS        row count for heavy_key_extreme     (default: 200 GiB worth)
#   MULTIMODAL_ROWS     row count for multimodal            (default: 200 GiB worth)
#   PAYLOAD_BYTES       per-row payload bytes               (default: 512)
#   MIX_MEANS           CSV means for multimodal mixture    (default: "1.0e18,9.2e18,1.74e19")
#   MIX_STDDEVS         CSV stddevs                         (default: "5.0e16,5.0e16,5.0e16")
#   MIX_WEIGHTS         CSV weights                         (default: "1,1,1")
#   SKIP_EXP_22         set to 1 to skip HeavyKey-Extreme   (default: 0)
#   SKIP_EXP_23         set to 1 to skip Multimodal         (default: 0)
#   KEEP_DATASETS       set to 1 to NOT delete datasets between experiments
#                       (requires ~600 GiB free)            (default: 0)
#   MIN_FREE_GB         required free GB on DATASETS_DIR    (default: 450)
#   MIN_FREE_OUT_GB     required free GB on OUT_DIR         (default: 250)
#   SSD_CLEANUP_SLEEP_S settle time after rm + sync         (default: 3)
#   DO_FSTRIM           issue fstrim after cleanup          (default: 1)
#   FORCE               skip disk-space pre-flight checks   (default: 0)
#   RCLONE_REMOTE       upload OUT_DIR to this rclone path  (default: gdrive:bench_results/crocsort_skew)
#
# Boundary metrics come from the runner's existing imbalance report:
#   "Byte imbalance factor (max/avg)"
#   "Entry imbalance factor (max/avg)"
#   "Time imbalance factor (max/avg)"
#   "I/O imbalance factor (max/avg)"
# -----------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# CLI / env
# ---------------------------------------------------------------------------
if [[ ${1-} == "" ]]; then
  echo "Usage: $0 <datasets_dir> [output_dir]" >&2
  exit 1
fi

DATASETS_DIR="$1"
mkdir -p "$DATASETS_DIR"

TS=$(date +"%Y-%m-%d_%H-%M-%S")
OUT_DIR=${2:-"logs/crocsort_skew_${TS}"}
mkdir -p "$OUT_DIR"

THREADS="${THREADS:-8}"
MEM_GB="${MEM_GB:-2}"
PAGE_SIZE_KB=64
COOLDOWN_S="${COOLDOWN_S:-30}"

PAYLOAD_BYTES="${PAYLOAD_BYTES:-512}"
# 200 GiB defaults assuming 528 B / row (4 + 8 + 4 + 512 payload).
EXTREME_ROWS="${EXTREME_ROWS:-406720387}"
MULTIMODAL_ROWS="${MULTIMODAL_ROWS:-406720387}"

MIX_MEANS="${MIX_MEANS:-1.0e18,9.2e18,1.74e19}"
MIX_STDDEVS="${MIX_STDDEVS:-5.0e16,5.0e16,5.0e16}"
MIX_WEIGHTS="${MIX_WEIGHTS:-1,1,1}"

SKIP_EXP_22="${SKIP_EXP_22:-0}"
SKIP_EXP_23="${SKIP_EXP_23:-0}"
KEEP_DATASETS="${KEEP_DATASETS:-0}"

MIN_FREE_GB="${MIN_FREE_GB:-450}"
MIN_FREE_OUT_GB="${MIN_FREE_OUT_GB:-250}"
SSD_CLEANUP_SLEEP_S="${SSD_CLEANUP_SLEEP_S:-3}"
DO_FSTRIM="${DO_FSTRIM:-1}"
FORCE="${FORCE:-0}"

RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:bench_results/crocsort_skew}"

GEN_EXTREME="./target/release/examples/gen_heavy_key_extreme_kvbin"
GEN_MULTIMODAL="./target/release/examples/gen_multimodal_kvbin"
BINARY="./target/release/examples/kvbin_benchmark_cli"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
upload_logs() {
  if ! command -v rclone >/dev/null 2>&1; then
    return 0
  fi
  if [[ -z "${RCLONE_REMOTE:-}" ]]; then
    return 0
  fi
  local dest="${RCLONE_REMOTE}/$(basename "$OUT_DIR")"
  echo "Uploading $OUT_DIR -> $dest ..."
  rclone copy "$OUT_DIR" "$dest" --progress \
    || echo "Warning: rclone upload failed" >&2
}

clear_cache_if_available() {
  if [[ -x /usr/local/sbin/clearcache3.sh ]]; then
    if [[ $(id -u) -eq 0 ]]; then
      /usr/local/sbin/clearcache3.sh || true
    elif command -v sudo >/dev/null 2>&1; then
      sudo -n /usr/local/sbin/clearcache3.sh 2>/dev/null || true
    fi
  fi
}

cooldown() { sleep "$COOLDOWN_S"; }

# Robust SSD cleanup of a directory tree (truncate → rm → sync → fstrim →
# sleep). Borrowed from sparse_budget_ablation.sh: plain `rm -rf` on hundred-GB
# trees of run files is slow and the FS may not reclaim space accounting before
# the next experiment starts; truncate-first releases extents fast, fstrim asks
# the SSD to physically reclaim blocks, the sleep lets the kernel settle so the
# subsequent free-space check is accurate.
robust_cleanup() {
  local target="$1"
  if [[ -z "$target" || ! -e "$target" ]]; then return 0; fi

  local before_str before_kb after_kb
  before_str=$(du -sh "$target" 2>/dev/null | cut -f1 || echo "?")
  before_kb=$(df -k "$target" 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
  echo "  [cleanup] removing ${target} (size=${before_str}) ..."

  if [[ -d "$target" ]]; then
    find "$target" -type f -size +1M -exec truncate -s 0 {} + 2>/dev/null || true
  elif [[ -f "$target" && $(stat -c%s "$target" 2>/dev/null || echo 0) -gt $((1024*1024)) ]]; then
    truncate -s 0 "$target" 2>/dev/null || true
  fi
  sync

  rm -rf "$target"
  sync

  if [[ "$DO_FSTRIM" == "1" ]] && command -v fstrim >/dev/null 2>&1; then
    local mountpoint
    mountpoint=$(df --output=target "$(dirname "$target")" 2>/dev/null | tail -1)
    if [[ -n "$mountpoint" && -d "$mountpoint" ]]; then
      if [[ $(id -u) -eq 0 ]]; then
        fstrim "$mountpoint" >/dev/null 2>&1 || true
      elif command -v sudo >/dev/null 2>&1; then
        sudo -n fstrim "$mountpoint" >/dev/null 2>&1 || true
      fi
    fi
  fi

  sleep "$SSD_CLEANUP_SLEEP_S"

  after_kb=$(df -k "$(dirname "$target")" 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
  if [[ "$after_kb" =~ ^[0-9]+$ && "$before_kb" =~ ^[0-9]+$ ]]; then
    local freed_gb=$(( (after_kb - before_kb) / 1024 / 1024 ))
    echo "  [cleanup] done. Free space delta: +${freed_gb} GB"
  fi
}

# Robust cleanup of an individual dataset (kvbin + idx + any sidecars).
remove_dataset() {
  local name="$1"
  local kvbin="${DATASETS_DIR}/${name}.kvbin"
  local idx="${DATASETS_DIR}/${name}.kvbin.idx"
  echo ""
  echo "=========================================================="
  echo "  Removing dataset: $name (sequential disk-discipline)"
  echo "=========================================================="
  robust_cleanup "$kvbin"
  robust_cleanup "$idx"
}

free_gb_on() {
  local path="$1"
  if [[ ! -e "$path" ]]; then echo 0; return; fi
  df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -d ' G' || echo 0
}

# Per-thread parameters (computed once; same convention as kvbin_sort_bench_new.sh)
RG_BUF_MB=$(echo "scale=2; ($MEM_GB * 1024) / $THREADS" | bc)
MAX_FANIN=$(echo "($MEM_GB * 1024 * 1024 * 95 / 100) / ($THREADS * $PAGE_SIZE_KB) - 1" | bc)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "Building generators + benchmark CLI (release)..."
cargo build --release \
  --example gen_heavy_key_extreme_kvbin \
  --example gen_multimodal_kvbin \
  --example kvbin_benchmark_cli >/dev/null

for b in "$GEN_EXTREME" "$GEN_MULTIMODAL" "$BINARY"; do
  if [[ ! -x "$b" ]]; then
    echo "Build did not produce $b" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Disk pre-flight
# ---------------------------------------------------------------------------
free_data_gb=$(free_gb_on "$DATASETS_DIR")
free_out_gb=$(free_gb_on "$OUT_DIR")

needed_data_gb=210                              # one 200 GiB dataset + slack
if [[ "$KEEP_DATASETS" == "1" ]]; then
  needed_data_gb=420                            # both datasets co-resident
fi

echo ""
echo "[preflight] DATASETS_DIR free: ${free_data_gb} GB (need ~${needed_data_gb} GB)"
echo "[preflight] OUT_DIR      free: ${free_out_gb} GB (need ~${MIN_FREE_OUT_GB} GB for sort temp)"

if [[ "$free_data_gb" =~ ^[0-9]+$ ]] && (( free_data_gb < MIN_FREE_GB )); then
  echo "!!! WARNING: only ${free_data_gb} GB free on DATASETS_DIR (< MIN_FREE_GB=${MIN_FREE_GB})." >&2
  echo "!!! At 200 GiB/dataset, sequential mode (default) needs ~${needed_data_gb} GB free." >&2
  if [[ "$FORCE" != "1" ]]; then
    echo "!!! Set FORCE=1 to override." >&2
    exit 1
  fi
fi
if [[ "$free_out_gb" =~ ^[0-9]+$ ]] && (( free_out_gb < MIN_FREE_OUT_GB )); then
  echo "!!! WARNING: only ${free_out_gb} GB free on OUT_DIR (< MIN_FREE_OUT_GB=${MIN_FREE_OUT_GB})." >&2
  echo "!!! Each sort writes ~200 GB of intermediate run files to OUT_DIR." >&2
  if [[ "$FORCE" != "1" ]]; then
    echo "!!! Set FORCE=1 to override." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
generate_extreme_if_missing() {
  local kvbin="${DATASETS_DIR}/heavy_key_extreme.kvbin"
  local idx="${DATASETS_DIR}/heavy_key_extreme.kvbin.idx"
  if [[ -f "$kvbin" && -f "$idx" ]]; then
    echo "✓ heavy_key_extreme already present ($(du -sh "$kvbin" | cut -f1)), skipping."
    return 0
  fi
  echo ""
  echo "================================================================"
  echo "  Generating: heavy_key_extreme (dup_frac=0.999, payload=${PAYLOAD_BYTES} B)"
  echo "  Rows:       $(printf "%'d" "$EXTREME_ROWS")"
  echo "  Output:     $kvbin / $idx"
  echo "================================================================"
  local start_ts end_ts
  start_ts=$(date +%s)
  "$GEN_EXTREME" \
    --out "$kvbin" \
    --idx "$idx" \
    --rows "$EXTREME_ROWS" \
    --payload "$PAYLOAD_BYTES" \
    --dup-frac 0.999
  end_ts=$(date +%s)
  echo "  Done in $(( (end_ts - start_ts) / 60 ))m $(( (end_ts - start_ts) % 60 ))s"
  ls -lh "$kvbin" "$idx"
}

generate_multimodal_if_missing() {
  local kvbin="${DATASETS_DIR}/multimodal.kvbin"
  local idx="${DATASETS_DIR}/multimodal.kvbin.idx"
  if [[ -f "$kvbin" && -f "$idx" ]]; then
    echo "✓ multimodal already present ($(du -sh "$kvbin" | cut -f1)), skipping."
    return 0
  fi
  echo ""
  echo "================================================================"
  echo "  Generating: multimodal (K-Gaussian mixture, payload=${PAYLOAD_BYTES} B)"
  echo "  Rows:       $(printf "%'d" "$MULTIMODAL_ROWS")"
  echo "  Means:      $MIX_MEANS"
  echo "  Stddevs:    $MIX_STDDEVS"
  echo "  Weights:    $MIX_WEIGHTS"
  echo "  Output:     $kvbin / $idx"
  echo "================================================================"
  local start_ts end_ts
  start_ts=$(date +%s)
  "$GEN_MULTIMODAL" \
    --out "$kvbin" \
    --idx "$idx" \
    --rows "$MULTIMODAL_ROWS" \
    --payload "$PAYLOAD_BYTES" \
    --means "$MIX_MEANS" \
    --stddevs "$MIX_STDDEVS" \
    --weights "$MIX_WEIGHTS"
  end_ts=$(date +%s)
  echo "  Done in $(( (end_ts - start_ts) / 60 ))m $(( (end_ts - start_ts) % 60 ))s"
  ls -lh "$kvbin" "$idx"
}

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
RESULTS_FILE="${OUT_DIR}/summary.tsv"
echo -e "experiment\tdataset\tpartition\tstatus\ttotal_s\trun_gen_s\tmerge_s\tbyte_imb\tentry_imb\ttime_imb\tio_imb\tlog" > "$RESULTS_FILE"

parse_log_into_results() {
  local exp="$1" dataset="$2" partition="$3" status="$4" log="$5"

  if [[ ! -f "$log" ]]; then
    echo -e "${exp}\t${dataset}\t${partition}\t${status}\t-\t-\t-\t-\t-\t-\t-\t${log}" >> "$RESULTS_FILE"
    return
  fi

  local total_s rg_s merge_s byte_imb entry_imb time_imb io_imb
  total_s=$(grep -oP '^Run 1\s+[0-9.]+\s+[0-9]+\s+[0-9]+\s+[0-9]+\s+\K[0-9.]+' "$log" | head -1)
  rg_s=$(grep -oP '\(R\) time: \K[0-9]+' "$log" | head -1)
  merge_s=$(grep -oP '\(M\) time: \K[0-9]+' "$log" | head -1)
  if [[ -n "$rg_s" ]]; then rg_s=$(echo "scale=1; $rg_s / 1000" | bc); fi
  if [[ -n "$merge_s" ]]; then merge_s=$(echo "scale=1; $merge_s / 1000" | bc); fi
  byte_imb=$(grep -oP 'Byte imbalance factor \(max/avg\): \K[0-9.]+' "$log" | head -1)
  entry_imb=$(grep -oP 'Entry imbalance factor \(max/avg\): \K[0-9.]+' "$log" | head -1)
  time_imb=$(grep -oP 'Time imbalance factor \(max/avg\): \K[0-9.]+' "$log" | head -1)
  io_imb=$(grep -oP 'I/O imbalance factor \(max/avg\): \K[0-9.]+' "$log" | head -1)

  echo -e "${exp}\t${dataset}\t${partition}\t${status}\t${total_s:--}\t${rg_s:--}\t${merge_s:--}\t${byte_imb:--}\t${entry_imb:--}\t${time_imb:--}\t${io_imb:--}\t${log}" >> "$RESULTS_FILE"
}

run_one() {
  # exp_label dataset partition
  local exp="$1" dataset="$2" partition="$3"

  local kvbin="${DATASETS_DIR}/${dataset}.kvbin"
  local idx="${DATASETS_DIR}/${dataset}.kvbin.idx"
  if [[ ! -f "$kvbin" || ! -f "$idx" ]]; then
    echo "Skipping ${dataset} (missing $kvbin or $idx)"
    parse_log_into_results "$exp" "$dataset" "$partition" "MISSING" "/dev/null"
    return 0
  fi

  local name="${exp}_${dataset}_${partition}_T${THREADS}_Mem${MEM_GB}GB"
  local log_file="${OUT_DIR}/${name}.log"
  local temp_dir="${OUT_DIR}/${name}_tmp"
  mkdir -p "$temp_dir"

  echo ""
  echo "----------------------------------------------------------"
  echo "  RUN: experiment=${exp}, dataset=${dataset}, partition=${partition}"
  echo "  Started: $(date '+%H:%M:%S')"
  echo "  OUT_DIR free before run: $(free_gb_on "$OUT_DIR") GB"
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
      --partition-type "$partition" \
      --dir "$temp_dir" \
      2>&1 | tee "$log_file"; then
    status="OK"
  else
    status="FAIL"
  fi

  # Robust cleanup of the sort temp dir — large run files on SSD need
  # truncate+sync+fstrim, not just rm.
  robust_cleanup "$temp_dir"
  parse_log_into_results "$exp" "$dataset" "$partition" "$status" "$log_file"
  upload_logs
}

# ---------------------------------------------------------------------------
# Best-effort cleanup of any leftover temp dirs from a killed prior run
# ---------------------------------------------------------------------------
find "$OUT_DIR" -maxdepth 1 -name '*_tmp' -type d -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  CrocSort skew bench (HeavyKey-Extreme + Multimodal)"
echo "  Output dir:    $OUT_DIR"
echo "  Datasets dir:  $DATASETS_DIR"
echo "  Threads:       $THREADS"
echo "  Memory:        ${MEM_GB} GB"
echo "  rg_buf:        $RG_BUF_MB MB / thread"
echo "  max_fanin:     $MAX_FANIN"
echo "  Skip 2.2:      $SKIP_EXP_22"
echo "  Skip 2.3:      $SKIP_EXP_23"
echo "  Keep datasets: $KEEP_DATASETS  (0 = sequential generate→run→delete)"
echo "  fstrim:        $DO_FSTRIM, settle=${SSD_CLEANUP_SLEEP_S}s"
echo "============================================================"

# ---------------------------------------------------------------------------
# Experiment 2.2 — HeavyKey-Extreme on CrocSort (size-balanced)
# Sequential disk discipline: generate just-in-time, run, then (unless
# KEEP_DATASETS=1) reclaim the 200 GiB before generating the next dataset.
# ---------------------------------------------------------------------------
if [[ "$SKIP_EXP_22" != "1" ]]; then
  echo ""
  echo "=== EXP 2.2: HeavyKey-Extreme  (CrocSort = size-balanced) ==="
  generate_extreme_if_missing
  run_one "Exp22" "heavy_key_extreme" "size-balanced"
  cooldown
  if [[ "$KEEP_DATASETS" != "1" && "$SKIP_EXP_23" != "1" ]]; then
    remove_dataset "heavy_key_extreme"
    echo "  DATASETS_DIR free after cleanup: $(free_gb_on "$DATASETS_DIR") GB"
  fi
fi

# ---------------------------------------------------------------------------
# Experiment 2.3 — Multimodal on CrocSort (size-balanced)
# ---------------------------------------------------------------------------
if [[ "$SKIP_EXP_23" != "1" ]]; then
  echo ""
  echo "=== EXP 2.3: Multimodal  (CrocSort = size-balanced) ==="
  generate_multimodal_if_missing
  run_one "Exp23" "multimodal" "size-balanced"
  cooldown
  if [[ "$KEEP_DATASETS" != "1" ]]; then
    remove_dataset "multimodal"
    echo "  DATASETS_DIR free after cleanup: $(free_gb_on "$DATASETS_DIR") GB"
  fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  DONE.  Summary:"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$RESULTS_FILE"
else
  cat "$RESULTS_FILE"
fi
echo ""
echo "Logs:        $OUT_DIR/*.log"
echo "Summary TSV: $RESULTS_FILE"

upload_logs
