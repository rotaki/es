#!/bin/bash

# Compare sorting performance with both systems writing output to files
# Usage: ./compare_sort_with_output.sh <csv_file> [threads] [memory]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <csv_file> [threads] [memory]"
    echo "  csv_file: Path to lineitem CSV file"
    echo "  threads: Number of threads (default: 4)"
    echo "  memory: Memory limit (default: 4GB)"
    echo ""
    echo "Example: $0 lineitem_sf10.csv 4 4GB"
    exit 1
fi

CSV_FILE="$1"
THREADS="${2:-4}"
MEMORY="${3:-4GB}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="comparison_output_${TIMESTAMP}"

# Check if file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: File '$CSV_FILE' not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/comparison_results.txt"

# Get file info
FILE_SIZE_BYTES=$(stat -c%s "$CSV_FILE" 2>/dev/null || stat -f%z "$CSV_FILE" 2>/dev/null)
FILE_SIZE_GB=$(echo "scale=2; $FILE_SIZE_BYTES / 1024 / 1024 / 1024" | bc)

echo "======================================================================"  | tee "$LOG_FILE"
echo "SORTING PERFORMANCE COMPARISON (With File Output)"                      | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "Date: $(date)"                                                          | tee -a "$LOG_FILE"
echo "File: $CSV_FILE"                                                        | tee -a "$LOG_FILE"
echo "File size: ${FILE_SIZE_GB} GB"                                         | tee -a "$LOG_FILE"
echo "Threads: $THREADS"                                                      | tee -a "$LOG_FILE"
echo "Memory: $MEMORY"                                                        | tee -a "$LOG_FILE"
echo ""                                                                       | tee -a "$LOG_FILE"

# Detect delimiter based on file extension or content
DELIMITER=","  # Default CSV

# Test 1: External Sorter (with output file)
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "TEST 1: External Sorter (Rust) - Writing to File"                      | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "Running external sorter with:"                                          | tee -a "$LOG_FILE"
echo "  - Key columns: 10,5,1 (l_shipdate, l_extendedprice, l_partkey)"      | tee -a "$LOG_FILE"
echo "  - Value columns: 0,3,11 (l_orderkey, l_linenumber, l_commitdate)"    | tee -a "$LOG_FILE"
echo ""                                                                       | tee -a "$LOG_FILE"
# 
ES_START=$(date +%s.%N)
# 
# Note: The external sorter writes output by default
cargo run --release --example lineitem_benchmark_cli -- \
    -t "$THREADS" \
    -m "$MEMORY" \
    -k 10,5,1 \
    -v 0,3,11 \
    "$CSV_FILE" 2>&1 | tee "${OUTPUT_DIR}/external_sorter_output.log" | grep -E "Sort completed|Throughput|runs generated|Total rows|output"
# 
ES_END=$(date +%s.%N)
ES_DURATION=$(echo "$ES_END - $ES_START" | bc)
# 
echo ""                                                                       | tee -a "$LOG_FILE"
echo "External Sorter total time: $(printf "%.2f" $ES_DURATION) seconds"     | tee -a "$LOG_FILE"
echo ""                                                                       | tee -a "$LOG_FILE"

# Test 2: DuckDB (with output file)
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "TEST 2: DuckDB - Writing to File"                                      | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "Running DuckDB sort with:"                                              | tee -a "$LOG_FILE"
echo "  - Sort columns: l_shipdate, l_extendedprice, l_partkey" | tee -a "$LOG_FILE"
echo "  - Output columns: l_orderkey, l_linenumber, l_commitdate" | tee -a "$LOG_FILE"
echo ""                                                                       | tee -a "$LOG_FILE"

cat > "${OUTPUT_DIR}/duckdb_sort.sql" << EOF
SET threads=$THREADS;
SET memory_limit='$MEMORY';

-- Measure sort time
.timer on

-- Write sorted results to file (all columns, sorted by specified keys)
SELECT COUNT(l_orderkey), COUNT(l_linenumber), COUNT(l_commitdate)
FROM (
    SELECT l_orderkey, l_linenumber, l_commitdate
    FROM read_csv_auto('$CSV_FILE', delim='$DELIMITER', header=true)
    ORDER BY l_shipdate, l_extendedprice, l_partkey
    OFFSET 1
);

EOF

DUCKDB_START=$(date +%s.%N)

duckdb :memory: < "${OUTPUT_DIR}/duckdb_sort.sql" 2>&1 | tee "${OUTPUT_DIR}/duckdb_output.log"

DUCKDB_END=$(date +%s.%N)
DUCKDB_DURATION=$(echo "$DUCKDB_END - $DUCKDB_START" | bc)

echo ""                                                                       | tee -a "$LOG_FILE"
echo "DuckDB total time: $(printf "%.2f" $DUCKDB_DURATION) seconds"          | tee -a "$LOG_FILE"
echo ""                                                                       | tee -a "$LOG_FILE"

# Summary
echo ""                                                                       | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "SUMMARY"                                                               | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"
echo "External Sorter time: $(printf "%.2f" $ES_DURATION) seconds"           | tee -a "$LOG_FILE"
echo "DuckDB time: $(printf "%.2f" $DUCKDB_DURATION) seconds"                | tee -a "$LOG_FILE"

# Calculate speedup
if (( $(echo "$DUCKDB_DURATION > 0" | bc -l) )); then
    SPEEDUP=$(echo "scale=2; $DUCKDB_DURATION / $ES_DURATION" | bc)
    echo "Speedup: ${SPEEDUP}x"                                              | tee -a "$LOG_FILE"
fi

echo ""                                                                       | tee -a "$LOG_FILE"
echo "Output files:"                                                         | tee -a "$LOG_FILE"
echo "  - DuckDB: $DUCKDB_OUTPUT_FILE"                                       | tee -a "$LOG_FILE"
echo "  - External Sorter: Check logs for output location"                   | tee -a "$LOG_FILE"
echo ""                                                                       | tee -a "$LOG_FILE"
echo "Detailed logs saved to:"                                               | tee -a "$LOG_FILE"
echo "  - External Sorter: ${OUTPUT_DIR}/external_sorter_output.log"         | tee -a "$LOG_FILE"
echo "  - DuckDB: ${OUTPUT_DIR}/duckdb_output.log"                           | tee -a "$LOG_FILE"
echo "======================================================================"  | tee -a "$LOG_FILE"