#!/bin/bash

# Heatmap Benchmark Script for External Sorter
# Runs lineitem_benchmark_cli across different thread/memory combinations
# Logs run generation times for heatmap visualization

set -e

# Configuration
INPUT_FILE="${1:-lineitem_sf0.1.csv}"
OUTPUT_CSV="${2:-benchmark_results_$(date +%Y%m%d_%H%M%S).csv}"

# Thread counts to test (x-axis)
THREADS=(2 4 6 8)

# Memory sizes in MB (y-axis) - start with smaller sizes for testing
MEMORY_MB=(1 2 3 4)

# Build the benchmark binary
echo "Building lineitem_benchmark_cli..."
cargo build --release --example lineitem_benchmark_cli

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

echo "Starting heatmap benchmark..."
echo "Input file: $INPUT_FILE"
echo "Output CSV: $OUTPUT_CSV"
echo "Threads: ${THREADS[*]}"
echo "Memory sizes (MB): ${MEMORY_MB[*]}"
echo

# Create log file for raw benchmark outputs
LOG_FILE="${OUTPUT_CSV%.csv}.log"
echo "Raw benchmark outputs will be logged to: $LOG_FILE"

# Total combinations
TOTAL_COMBINATIONS=$((${#THREADS[@]} * ${#MEMORY_MB[@]}))
CURRENT_COMBINATION=0

# Run benchmarks for each combination
for memory in "${MEMORY_MB[@]}"; do
    for threads in "${THREADS[@]}"; do
        CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
        echo "Running benchmark [$CURRENT_COMBINATION/$TOTAL_COMBINATIONS]: $threads threads, ${memory}MB memory"
        
        # Log benchmark configuration
        echo "=== BENCHMARK START: threads=$threads, memory_mb=$memory ===" >> "$LOG_FILE"
        
        # Run the benchmark and capture output (increased timeout)
        ./target/release/examples/lineitem_benchmark_cli \
            -t "$threads" \
            -m "$memory" \
            "$INPUT_FILE" >> "$LOG_FILE" 2>&1 || {
            echo "  ERROR: Benchmark failed or timed out"
            echo "ERROR: Benchmark failed or timed out" >> "$LOG_FILE"
        }
        
        echo "=== BENCHMARK END: threads=$threads, memory_mb=$memory ===" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        echo "  Completed"
    done
done

echo "Benchmark completed!"
echo "Raw benchmark outputs saved to: $LOG_FILE"
echo
echo "Summary statistics:"
echo "- Total combinations tested: $TOTAL_COMBINATIONS"
echo "- Thread range: ${THREADS[0]} to ${THREADS[-1]}"
echo "- Memory range: ${MEMORY_MB[0]}MB to ${MEMORY_MB[-1]}MB"
echo
echo "To parse results and create heatmaps, run: python3 create_heatmaps.py $LOG_FILE"