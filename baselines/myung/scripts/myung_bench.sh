#!/usr/bin/env bash
# Sweep Myung baseline across memory limits on a GenSort file (canonical
# config: 200 GiB data, 16 threads, arbitration on). Produces a CSV matching
# the format used by CrocSort's bench scripts so the two can be plotted
# together by scripts/plot_myung_comparison.py.
#
# Mirrors gensort_sort_bench_new.sh Exp1: each entry in MEM_MIB_LIST is the
# cgroup hard cap. The application's --memory-mb is set to HEADROOM_FRAC of
# that (default 60%); the remaining 40% is OS overhead headroom (FS
# metadata, alignment bounce buffers, libc heap, etc.). cgroup is unused
# only when CGROUP_MODE=off.
#
# Usage:
#   scripts/myung_bench.sh <input_gensort> <scratch_dir> <output_csv> \
#       [interleave_floor_bytes] [num_runs_per_config]
#
# Env overrides (set any of these to fan out beyond the 1D memory sweep):
#   MEM_MIB_LIST (space-separated)   default: "1024 2048 4096 8192 16384 32768"
#                                    interpreted as cgroup hard cap (MiB)
#   HEADROOM_FRAC                    default: 0.6   (app gets 60% of cgroup)
#   THREADS_LIST (space-separated)   default: "16"
#   ARB_LIST     (space-separated)   default: "on"  (set "on off" to A/B)
#   SAMPLE_RATIO                     default: 0.001
#   CGROUP_MODE                      default: "on"  ("on"|"strict"|"off")
#   RCLONE_REMOTE                    default: "gdrive:bench_results/myung_gensort"
#                                    (set to "" to disable upload)

set -euo pipefail

INPUT="${1:?usage: myung_bench.sh <input> <scratch_dir> <output_csv> [C_bytes] [runs]}"
SCRATCH="${2:?scratch dir}"
OUT_CSV="${3:?output csv path}"
C_BYTES="${4:-65536}"
RUNS="${5:-3}"

MEM_MIB_LIST="${MEM_MIB_LIST:-1024 2048 4096 8192 16384 32768}"
HEADROOM_FRAC="${HEADROOM_FRAC:-0.6}"
THREADS_LIST="${THREADS_LIST:-16}"
ARB_LIST="${ARB_LIST:-on}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.001}"
CGROUP_MODE="${CGROUP_MODE:-on}"
SLEEP_BETWEEN_CONFIGS_SECONDS="${SLEEP_BETWEEN_CONFIGS_SECONDS:-30}"
# Auto-upload results to Drive (matches gensort_sort_bench_new.sh:16). Set to
# "" to disable, or override the remote path. rclone is silently skipped if
# not installed.
RCLONE_REMOTE="${RCLONE_REMOTE-gdrive:bench_results/myung_gensort}"

BIN="$(dirname "$0")/../../../target/release/examples/myung_sort_cli"
if [[ ! -x "$BIN" ]]; then
  echo "building myung_sort_cli..." >&2
  (cd "$(dirname "$0")/../../.." && cargo build --release -p myung --example myung_sort_cli)
fi

mkdir -p "$(dirname "$OUT_CSV")"

DATASET_LABEL="$(basename "$INPUT")"
DATASET_NAME_SAFE=$(echo "$DATASET_LABEL" | tr -cs 'a-zA-Z0-9_-' '_')

# ── helpers (ported from scripts/gensort_sort_bench_new.sh) ──────────────────

cooldown() {
  sleep "$SLEEP_BETWEEN_CONFIGS_SECONDS"
}

clear_cache_if_available() {
  if [[ -x /usr/local/sbin/clearcache3.sh ]]; then
    if [[ $(id -u) -eq 0 ]]; then
      /usr/local/sbin/clearcache3.sh || echo "Warning: clearcache3.sh failed" >&2
    elif command -v sudo >/dev/null 2>&1; then
      sudo /usr/local/sbin/clearcache3.sh || echo "Warning: clearcache3.sh failed (sudo)" >&2
    fi
  elif [[ -w /proc/sys/vm/drop_caches ]]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches || true
  fi
}

# Sync the CSV + log file to a Drive folder named after the CSV's stem.
# Idempotent — safe to call after every row, picks up only changed files.
# Silently no-ops if RCLONE_REMOTE is empty or `rclone` isn't installed.
upload_logs() {
  if [[ -z "${RCLONE_REMOTE:-}" ]]; then
    return 0
  fi
  if ! command -v rclone >/dev/null 2>&1; then
    return 0
  fi
  local dest="${RCLONE_REMOTE}/$(basename "$OUT_CSV" .csv)"
  rclone copy "$OUT_CSV"   "$dest" --quiet 2>/dev/null || \
    echo "Warning: rclone upload of CSV failed" >&2
  if [[ -f "$LOG_FILE" ]]; then
    rclone copy "$LOG_FILE" "$dest" --quiet 2>/dev/null || \
      echo "Warning: rclone upload of LOG failed" >&2
  fi
}

# Run "$@" under a transient systemd cgroup with hard cap = $1 MiB. Soft
# (MemoryHigh) and hard (MemoryMax) caps are both set to the full amount
# (matches gensort_sort_bench_new.sh:130-147). Headroom for the app is
# taken from --memory-mb being set below this value, not from MemoryHigh.
run_with_cgroup_limits() {
  local cgroup_mib="$1"
  local scope_name="$2"
  shift 2

  if [[ "$CGROUP_MODE" == "off" ]]; then
    "$@"
    return $?
  fi

  if [[ "$(uname -s)" != "Linux" ]] || ! command -v systemd-run >/dev/null 2>&1; then
    echo "[cgroup] Warning: systemd-run unavailable; running without cgroup enforcement." >&2
    if [[ "$CGROUP_MODE" == "strict" ]]; then
      return 1
    fi
    "$@"
    return $?
  fi

  local limit_bytes=$((cgroup_mib * 1024 * 1024))
  local unit_name="myung-${scope_name}-${DATASET_NAME_SAFE}-$$-$(date +%s)"
  echo "[cgroup] ${scope_name}: MemoryMax=${cgroup_mib}MiB, swap=0"
  systemd-run --user --scope --quiet --collect \
      --unit "${unit_name}" \
      --property "MemoryAccounting=yes" \
      --property "MemoryHigh=${limit_bytes}" \
      --property "MemoryMax=${limit_bytes}" \
      --property "MemorySwapMax=0" \
      -- "$@"
}

# ── header (note: memory_mib = app budget, cgroup_mib = OS hard cap) ────────
echo "system,dataset,cgroup_mib,memory_mib,t_user,t_eff,arbitration,run,status,total_ms,rf_ms,merge_ms,num_runs,num_ranges" > "$OUT_CSV"

# Optional per-run log file (full stdout/stderr of every binary invocation,
# delimited so failures can be inspected).
LOG_FILE="${OUT_CSV%.csv}.log"
: > "$LOG_FILE"

# Sort MEM_MIB_LIST descending so the largest cgroup runs first. If a level
# fails (OOM kill, planner unhappy, etc.), smaller levels would only fare
# worse — we abort the rest of the sweep.
MEM_MIB_LIST_DESC=$(printf '%s\n' $MEM_MIB_LIST | sort -rn | tr '\n' ' ')
echo "[sweep] memory order (descending): $MEM_MIB_LIST_DESC"

abort_remaining=0
# ── sweep ───────────────────────────────────────────────────────────────────
for CGROUP_MIB in $MEM_MIB_LIST_DESC; do
  if [[ "$abort_remaining" -eq 1 ]]; then
    echo "[sweep] skipping cgroup=${CGROUP_MIB}MiB after earlier failure." >&2
    break
  fi

  # App memory = HEADROOM_FRAC of cgroup (e.g. 0.6 * 8 GiB = 4.8 GiB).
  # Use awk for the float multiply, then truncate to int MiB.
  APP_MIB=$(awk -v c="$CGROUP_MIB" -v f="$HEADROOM_FRAC" 'BEGIN { printf "%d\n", c * f }')
  if [[ "$APP_MIB" -le 0 ]]; then
    echo "WARN: APP_MIB resolved to 0 for cgroup=$CGROUP_MIB; skipping." >&2
    continue
  fi

  for T in $THREADS_LIST; do
    for ARB in $ARB_LIST; do
      ARB_FLAG=""
      [[ "$ARB" == "off" ]] && ARB_FLAG="--no-arbitration"

      for RUN in $(seq 1 "$RUNS"); do
        rm -rf "$SCRATCH" && mkdir -p "$SCRATCH"
        clear_cache_if_available

        SCOPE_NAME="gensort_C${CGROUP_MIB}_M${APP_MIB}_T${T}_A${ARB}_R${RUN}"
        echo "----- ${SCOPE_NAME} $(date -Iseconds) -----" >> "$LOG_FILE"

        # Run, capture full output AND exit code. `|| RC=$?` keeps set -e
        # from aborting on a non-zero exit (we want to record and decide).
        RC=0
        # shellcheck disable=SC2086  # ARB_FLAG must word-split
        RAW=$(run_with_cgroup_limits "$CGROUP_MIB" "$SCOPE_NAME" \
          "$BIN" \
            --input "$INPUT" \
            --dir "$SCRATCH" \
            --output "$SCRATCH/out" \
            --threads "$T" \
            --memory-mb "$APP_MIB" \
            --interleave-floor-bytes "$C_BYTES" \
            --sample-ratio "$SAMPLE_RATIO" \
            --discard-final-output \
            $ARB_FLAG 2>&1) || RC=$?
        echo "$RAW" >> "$LOG_FILE"

        # Always wipe intermediates — even on failure — so the next run
        # (or next memory level) starts from a clean scratch dir.
        rm -rf "$SCRATCH"

        if [[ $RC -ne 0 ]]; then
          echo "FAIL: ${SCOPE_NAME} exit=$RC (last line: $(echo "$RAW" | tail -1))" >&2
          echo "myung,$DATASET_LABEL,$CGROUP_MIB,$APP_MIB,,,$ARB,$RUN,exit_${RC},,,,," \
            | tee -a "$OUT_CSV"
          upload_logs
          abort_remaining=1
          break  # don't try remaining trials at this memory level
        fi

        OUT=$(echo "$RAW" | tail -1)
        # Parse the final line:
        # "myung_sort: D=..., M=... MiB, T(user)=U, T(eff)=E, runs=R, ranges=G, total=T_ms (rf=R_ms, merge=M_ms)"
        T_USER=$(sed -nE 's/.*T\(user\)=([0-9]+).*/\1/p' <<<"$OUT")
        T_EFF=$(sed -nE 's/.*T\(eff\)=([0-9]+).*/\1/p' <<<"$OUT")
        RUNSCOUNT=$(sed -nE 's/.*runs=([0-9]+).*/\1/p' <<<"$OUT")
        RANGES=$(sed -nE 's/.*ranges=([0-9]+).*/\1/p' <<<"$OUT")
        TOTAL=$(sed -nE 's/.*total=([0-9]+) ms.*/\1/p' <<<"$OUT")
        RF=$(sed -nE 's/.*rf=([0-9]+).*/\1/p' <<<"$OUT")
        MG=$(sed -nE 's/.*merge=([0-9]+).*/\1/p' <<<"$OUT")

        if [[ -z "$TOTAL" ]]; then
          echo "FAIL: ${SCOPE_NAME} parse_fail (last line: $OUT)" >&2
          echo "myung,$DATASET_LABEL,$CGROUP_MIB,$APP_MIB,,,$ARB,$RUN,parse_fail,,,,," \
            | tee -a "$OUT_CSV"
          upload_logs
          abort_remaining=1
          break
        fi

        echo "myung,$DATASET_LABEL,$CGROUP_MIB,$APP_MIB,$T_USER,$T_EFF,$ARB,$RUN,ok,$TOTAL,$RF,$MG,$RUNSCOUNT,$RANGES" \
          | tee -a "$OUT_CSV"
        upload_logs

        cooldown
      done
      [[ "$abort_remaining" -eq 1 ]] && break
    done
    [[ "$abort_remaining" -eq 1 ]] && break
  done
done

if [[ "$abort_remaining" -eq 1 ]]; then
  echo "[sweep] aborted at smaller memory levels because the previous failed." >&2
fi
echo "[sweep] CSV: $OUT_CSV"
echo "[sweep] log: $LOG_FILE"
