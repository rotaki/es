#!/usr/bin/env bash
# Measure C — the NVMe chunk-size floor at which the device reaches ~80% of
# peak read bandwidth. Replicates Myung §3.2 fio protocol exactly:
#   ioengine=psync, direct=1, rw=read, rand_repeat=0, norandommap
# We sweep bs across 4 KiB .. 256 KiB and report bandwidth per size.
#
# The operator picks the smallest bs at which throughput saturates (80% of
# peak observed in the sweep) and plugs that into the CLI:
#   myung_sort_cli --interleave-floor-bytes <C>
#
# Usage:
#   scripts/fio_myung_calibrate.sh <dataset_path> [size_mib] [threads]
#
# <dataset_path> must live on the NVMe device being characterized. A
# multi-GiB file is ideal; the script will use `size=<size_mib>M` for fio.

set -euo pipefail

FILE="${1:?usage: fio_myung_calibrate.sh <file_on_nvme> [size_mib] [threads]}"
SIZE_MIB="${2:-4096}"
NUMJOBS="${3:-16}"

if ! command -v fio >/dev/null 2>&1; then
  echo "fio is not installed. sudo apt-get install fio" >&2
  exit 1
fi

if [[ ! -e "$FILE" ]]; then
  echo "error: $FILE does not exist. Pre-create a large test file on the NVMe." >&2
  echo "Hint: dd if=/dev/urandom of=$FILE bs=1M count=$SIZE_MIB status=progress" >&2
  exit 1
fi

OUT="fio_myung_calibrate_$(date +%Y%m%d_%H%M%S).txt"
echo "# Myung Eq. 2 calibration sweep on $FILE" | tee "$OUT"
echo "# size=${SIZE_MIB}M, numjobs=$NUMJOBS" | tee -a "$OUT"
printf "%-10s %-15s %-15s\n" "bs" "bw_mib_s" "iops" | tee -a "$OUT"

for BS in 4k 8k 16k 32k 64k 128k 256k; do
  # Drop page cache so buffered-I/O effects don't pollute (we still use
  # direct=1, but belt-and-braces).
  sync
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches || true
  fi

  LINE=$(fio \
    --name=myung_cal \
    --filename="$FILE" \
    --ioengine=psync \
    --direct=1 \
    --rw=read \
    --rand_repeat=0 \
    --norandommap \
    --size="${SIZE_MIB}M" \
    --bs="$BS" \
    --numjobs="$NUMJOBS" \
    --group_reporting \
    --minimal 2>/dev/null | tail -1)

  # fio minimal format: fields separated by ';'. Read BW and IOPS.
  #   Field 7 = bw in KiB/s (for read), Field 8 = IOPS (read).
  # Robust parsing: use awk with ';' delimiter.
  BW_KIB=$(awk -F';' '{print $7}' <<<"$LINE")
  IOPS=$(awk -F';' '{print $8}' <<<"$LINE")
  # Convert KiB/s to MiB/s.
  BW_MIB=$(awk -v kib="$BW_KIB" 'BEGIN{printf "%.1f", kib/1024.0}')
  printf "%-10s %-15s %-15s\n" "$BS" "$BW_MIB" "$IOPS" | tee -a "$OUT"
done

echo "" | tee -a "$OUT"
echo "# Pick the smallest bs at which bw_mib_s reaches ~80% of the max." | tee -a "$OUT"
echo "# That value in bytes is your C for --interleave-floor-bytes." | tee -a "$OUT"
