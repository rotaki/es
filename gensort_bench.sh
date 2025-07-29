#!/bin/bash

# External Sort Benchmark Script
# Uses gensort data from 10GiB to 100GiB

# Configuration
DATA_DIR="./gensort_data"
RESULTS_DIR="./benchmark_results"
TEMP_DIR="./temp_sort"
WARMUP_RUNS=2
BENCHMARK_RUNS=5
# THREAD_SIZES=(8 16 32 64)  # Thread counts to test
THREAD_SIZES=(8)  # Thread counts to test
# MEMORY_SIZES=(1024 2048 4096 8192)  # Memory sizes in MB (1GB, 2GB, 4GB, 8GB)
MEMORY_SIZES=(128)  # Memory sizes in MB (1GB, 2GB, 4GB, 8GB)
# DATA_SIZES=(10 20 40 60 80 100)  # Data sizes in GiB
DATA_SIZES=(2 4 6 8 10)  # Data sizes in GiB

# Create results directory
mkdir -p "$RESULTS_DIR"

# Build the gen_sort_cli if not already built
echo "Building gen_sort_cli..."
cargo build --release --example gen_sort_cli

# Path to the executable
GEN_SORT_CLI="./target/release/examples/gen_sort_cli"

# Function to run benchmark for a specific configuration
run_benchmark() {
    local input_file=$1
    local memory_mb=$2
    local threads=$3
    local dataset_size=$4
    local output_file="$RESULTS_DIR/benchmark_${dataset_size}_${memory_mb}MB_${threads}T_$(date +%Y%m%d_%H%M%S).log"
    
    echo "==================================================================="
    echo "Benchmarking: $dataset_size dataset with ${memory_mb}MB memory and ${threads} threads"
    echo "Input: $input_file"
    echo "Output: $output_file"
    echo "==================================================================="
    
    # Create temp directory for this run
    local run_temp_dir="$TEMP_DIR/${dataset_size}_${memory_mb}MB_${threads}T"
    mkdir -p "$run_temp_dir"
    
    # Run the benchmark with tee to show output on terminal and save to file
    "$GEN_SORT_CLI" \
        --input "$input_file" \
        --threads $threads \
        --memory-mb $memory_mb \
        --dir "$run_temp_dir" \
        --warmup-runs $WARMUP_RUNS \
        --benchmark-runs $BENCHMARK_RUNS \
        2>&1 | tee "$output_file"
    
    # Clean up temp directory
    rm -rf "$run_temp_dir"
    
    echo "Benchmark completed. Results saved to: $output_file"
    echo ""
}

# Main benchmark loop
echo "Starting External Sort Benchmarks"
echo "================================="
echo "Configuration:"
echo "  Thread configurations: ${THREAD_SIZES[*]}"
echo "  Warmup runs: $WARMUP_RUNS"
echo "  Benchmark runs: $BENCHMARK_RUNS"
echo "  Memory configurations: ${MEMORY_SIZES[*]} MB"
echo "  Data sizes: ${DATA_SIZES[*]} GiB"
echo ""

# Check if data files exist
echo "Checking for data files..."
missing_files=0
for size in "${DATA_SIZES[@]}"; do
    input_file="$DATA_DIR/gensort_${size}GiB.data"
    if [ ! -f "$input_file" ]; then
        echo "  Missing: $input_file"
        missing_files=$((missing_files + 1))
    else
        echo "  Found: $input_file"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo ""
    echo "ERROR: Missing $missing_files data file(s)."
    echo "Please run ./generate_datasets.sh first to generate the required data."
    exit 1
fi

echo ""
echo "All data files found. Starting benchmarks..."
echo ""

# Run benchmarks for each dataset size, memory configuration, and thread count
start_time=$(date +%s)

for size in "${DATA_SIZES[@]}"; do
    input_file="$DATA_DIR/gensort_${size}GiB.data"
    
    for memory_mb in "${MEMORY_SIZES[@]}"; do
        # Skip if memory is too large compared to dataset
        # (e.g., 8GB memory for 10GB dataset doesn't make much sense for external sort)
        dataset_mb=$((size * 1024))
        if [ $memory_mb -ge $((dataset_mb / 2)) ]; then
            echo "Skipping ${size}GiB with ${memory_mb}MB memory (memory too large for meaningful external sort)"
            continue
        fi
        
        for threads in "${THREAD_SIZES[@]}"; do
            run_benchmark "$input_file" $memory_mb $threads "${size}GiB"
        done
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "==================================================================="
echo "All benchmarks completed!"
echo "Total time: $((total_time / 60)) minutes $((total_time % 60)) seconds"
echo "Results saved in: $RESULTS_DIR"
echo ""

# Generate summary report
summary_file="$RESULTS_DIR/benchmark_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "Generating summary report: $summary_file"

{
    echo "External Sort Benchmark Summary"
    echo "==============================="
    echo "Date: $(date)"
    echo "Configuration:"
    echo "  Thread configurations: ${THREAD_SIZES[*]}"
    echo "  Warmup runs: $WARMUP_RUNS"
    echo "  Benchmark runs: $BENCHMARK_RUNS"
    echo "  Memory configurations: ${MEMORY_SIZES[*]} MB"
    echo ""
    echo "Results:"
    echo ""
    
    # Extract key metrics from each log file
    for log_file in "$RESULTS_DIR"/benchmark_*.log; do
        if [ -f "$log_file" ]; then
            echo "File: $(basename "$log_file")"
            grep -E "(Policy|Throughput|Total time)" "$log_file" | tail -20
            echo ""
        fi
    done
} > "$summary_file"

echo "Summary report generated."
echo ""
echo "To view results:"
echo "  - Individual logs: ls -la $RESULTS_DIR/benchmark_*.log"
echo "  - Summary report: cat $summary_file"