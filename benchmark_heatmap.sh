#!/bin/bash

# Heatmap Benchmark Script for External Sorter
# Runs lineitem_benchmark_cli across different thread/memory combinations
# Logs run generation times for heatmap visualization

set -e

# Configuration
INPUT_FILE="${1:-lineitem_sf10.csv}"
OUTPUT_DIR="${2:-benchmark_results_$(date +%Y%m%d_%H%M%S)}"
NUM_RUNS=3  # Number of times to run each experiment

# Thread configurations to test
# Format: "run_gen_threads,merge_threads"
THREAD_CONFIGS=("2,2" "4,4" "6,6" "8,8" "4,2" "4,8" "8,4")

# Memory sizes in MB (y-axis) - start with smaller sizes for testing
MEMORY_MB=(32 64 96 128)

# Create output directory
mkdir -p "$OUTPUT_DIR"

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
echo "Output directory: $OUTPUT_DIR"
echo "Number of runs per configuration: $NUM_RUNS"
echo "Thread configurations (run_gen,merge): ${THREAD_CONFIGS[*]}"
echo "Memory sizes (MB): ${MEMORY_MB[*]}"
echo

# Total combinations (including multiple runs)
TOTAL_COMBINATIONS=$((${#THREAD_CONFIGS[@]} * ${#MEMORY_MB[@]} * $NUM_RUNS))
CURRENT_COMBINATION=0

# Run benchmarks for each combination
for memory in "${MEMORY_MB[@]}"; do
    for thread_config in "${THREAD_CONFIGS[@]}"; do
        # Parse thread configuration
        IFS=',' read -r run_gen_threads merge_threads <<< "$thread_config"
        
        for run in $(seq 1 $NUM_RUNS); do
            CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
            echo "Running benchmark [$CURRENT_COMBINATION/$TOTAL_COMBINATIONS]: run_gen=$run_gen_threads, merge=$merge_threads threads, ${memory}MB memory (run $run/$NUM_RUNS)"
            
            # Create log file for this specific run
            LOG_FILE="$OUTPUT_DIR/benchmark_r${run_gen_threads}_g${merge_threads}_m${memory}_run${run}.log"
            
            # Log benchmark configuration
            echo "=== BENCHMARK START: run_gen_threads=$run_gen_threads, merge_threads=$merge_threads, memory_mb=$memory, run=$run ===" > "$LOG_FILE"
            
            # Run the benchmark and capture output
            ./target/release/examples/lineitem_benchmark_cli \
                -r "$run_gen_threads" \
                -g "$merge_threads" \
                -m "$memory" \
                "$INPUT_FILE" >> "$LOG_FILE" 2>&1 || {
                echo "  ERROR: Benchmark failed or timed out"
                echo "ERROR: Benchmark failed or timed out" >> "$LOG_FILE"
            }
            
            echo "=== BENCHMARK END: run_gen_threads=$run_gen_threads, merge_threads=$merge_threads, memory_mb=$memory, run=$run ===" >> "$LOG_FILE"
            echo "  Completed run $run"
        done
        echo ""
    done
done

echo "Benchmark completed!"
echo "All benchmark outputs saved to directory: $OUTPUT_DIR"
echo
echo "Summary statistics:"
echo "- Total benchmark runs: $TOTAL_COMBINATIONS"
echo "- Configurations tested: $((${#THREAD_CONFIGS[@]} * ${#MEMORY_MB[@]}))"
echo "- Runs per configuration: $NUM_RUNS"
echo "- Thread configurations: ${#THREAD_CONFIGS[@]}"
echo "- Memory range: ${MEMORY_MB[0]}MB to ${MEMORY_MB[-1]}MB"
echo
echo "To parse results and create heatmaps, run: python3 create_heatmaps.py $OUTPUT_DIR"