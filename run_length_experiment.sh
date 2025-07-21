#!/bin/bash

# Experiment: Effect of run length (controlled by memory) on sorting performance
# Tests lineitem CSV files at scale factors SF10 (10GB), SF100 (100GB), SF1000 (1TB), SF10000 (10TB)
# Uses 40 threads and varies memory allocation to create different numbers of runs
#
# Sort configuration:
#   Key columns: 10,5,1 (l_shipdate, l_extendedprice, l_partkey)
#   Value columns: 0,3,11 (l_orderkey, l_linenumber, l_commitdate)

# Configuration
THREADS=40
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="experiments/run_length_${TIMESTAMP}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create summary file
SUMMARY_FILE="${OUTPUT_DIR}/experiment_summary.txt"
echo "=====================================" | tee "$SUMMARY_FILE"
echo "RUN LENGTH EXPERIMENT SUMMARY" | tee -a "$SUMMARY_FILE"
echo "=====================================" | tee -a "$SUMMARY_FILE"
echo "Date: $(date)" | tee -a "$SUMMARY_FILE"
echo "Testing effect of memory allocation on run generation and sorting performance" | tee -a "$SUMMARY_FILE"
echo "Fixed thread count: $THREADS" | tee -a "$SUMMARY_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Test 1: SF10 - 10GB file
LOG_FILE_SF10="${OUTPUT_DIR}/sf10_experiment.log"
echo "Running SF10 experiment..." | tee -a "$SUMMARY_FILE"
echo "Log file: $LOG_FILE_SF10" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

echo "==================================================================================" | tee "$LOG_FILE_SF10"
echo "TEST 1: SF10 - 10GB Lineitem File" | tee -a "$LOG_FILE_SF10"
echo "==================================================================================" | tee -a "$LOG_FILE_SF10"
if [ -f "lineitem_sf10.csv" ]; then
    cargo run --release --example lineitem_benchmark_cli -- \
        -t $THREADS \
        -m "512MB,1GB,2GB,4GB,8GB" \
        -k 10,5,1 \
        -v 0,3,11 \
        lineitem_sf10.csv 2>&1 | tee -a "$LOG_FILE_SF10"
    
    echo "SF10 test completed" | tee -a "$SUMMARY_FILE"
else
    echo "WARNING: lineitem_sf10.csv not found, skipping SF10 tests" | tee -a "$LOG_FILE_SF10" | tee -a "$SUMMARY_FILE"
fi

echo -e "\n" | tee -a "$SUMMARY_FILE"

# Test 2: SF100 - 100GB file
LOG_FILE_SF100="${OUTPUT_DIR}/sf100_experiment.log"
echo "Running SF100 experiment..." | tee -a "$SUMMARY_FILE"
echo "Log file: $LOG_FILE_SF100" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

echo "==================================================================================" | tee "$LOG_FILE_SF100"
echo "TEST 2: SF100 - 100GB Lineitem File" | tee -a "$LOG_FILE_SF100"
echo "==================================================================================" | tee -a "$LOG_FILE_SF100"
if [ -f "lineitem_sf100.csv" ]; then
    cargo run --release --example lineitem_benchmark_cli -- \
        -t $THREADS \
        -m "5GB,10GB,20GB,40GB,80GB" \
        -k 10,5,1 \
        -v 0,3,11 \
        lineitem_sf100.csv 2>&1 | tee -a "$LOG_FILE_SF100"
    
    echo "SF100 test completed" | tee -a "$SUMMARY_FILE"
else
    echo "WARNING: lineitem_sf100.csv not found, skipping SF100 tests" | tee -a "$LOG_FILE_SF100" | tee -a "$SUMMARY_FILE"
fi

echo -e "\n" | tee -a "$SUMMARY_FILE"

# # Test 3: SF1000 - 1TB file
# LOG_FILE_SF1000="${OUTPUT_DIR}/sf1000_experiment.log"
# echo "Running SF1000 experiment..." | tee -a "$SUMMARY_FILE"
# echo "Log file: $LOG_FILE_SF1000" | tee -a "$SUMMARY_FILE"
# echo "" | tee -a "$SUMMARY_FILE"
# 
# echo "==================================================================================" | tee "$LOG_FILE_SF1000"
# echo "TEST 3: SF1000 - 1TB Lineitem File" | tee -a "$LOG_FILE_SF1000"
# echo "==================================================================================" | tee -a "$LOG_FILE_SF1000"
# if [ -f "lineitem_sf1000.csv" ]; then
#     cargo run --release --example lineitem_benchmark_cli -- \
#         -t $THREADS \
#         -m "2GB,4GB,8GB" \
#         -k 10,5,1 \
#         -v 0,3,11 \
#         lineitem_sf1000.csv 2>&1 | tee -a "$LOG_FILE_SF1000"
#     
#     echo "SF1000 test completed" | tee -a "$SUMMARY_FILE"
# else
#     echo "WARNING: lineitem_sf1000.csv not found, skipping SF1000 tests" | tee -a "$LOG_FILE_SF1000" | tee -a "$SUMMARY_FILE"
# fi
# 
# echo -e "\n" | tee -a "$SUMMARY_FILE"
# 
# # Test 4: SF10000 - 10TB file
# LOG_FILE_SF10000="${OUTPUT_DIR}/sf10000_experiment.log"
# echo "Running SF10000 experiment..." | tee -a "$SUMMARY_FILE"
# echo "Log file: $LOG_FILE_SF10000" | tee -a "$SUMMARY_FILE"
# echo "" | tee -a "$SUMMARY_FILE"
# 
# echo "==================================================================================" | tee "$LOG_FILE_SF10000"
# echo "TEST 4: SF10000 - 10TB Lineitem File" | tee -a "$LOG_FILE_SF10000"
# echo "==================================================================================" | tee -a "$LOG_FILE_SF10000"
# if [ -f "lineitem_sf10000.csv" ]; then
#     cargo run --release --example lineitem_benchmark_cli -- \
#         -t $THREADS \
#         -m "1GB,2GB,4GB,8GB" \
#         -k 10,5,1 \
#         -v 0,3,11 \
#         lineitem_sf10000.csv 2>&1 | tee -a "$LOG_FILE_SF10000"
#     
#     echo "SF10000 test completed" | tee -a "$SUMMARY_FILE"
# else
#     echo "WARNING: lineitem_sf10000.csv not found, skipping SF10000 tests" | tee -a "$LOG_FILE_SF10000" | tee -a "$SUMMARY_FILE"
# fi

# Final summary
echo -e "\n" | tee -a "$SUMMARY_FILE"
echo "==================================================================================" | tee -a "$SUMMARY_FILE"
echo "EXPERIMENT COMPLETE" | tee -a "$SUMMARY_FILE"
echo "==================================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Results saved in directory: $OUTPUT_DIR" | tee -a "$SUMMARY_FILE"
echo "Individual log files:" | tee -a "$SUMMARY_FILE"
echo "  - SF10: sf10_experiment.log" | tee -a "$SUMMARY_FILE"
echo "  - SF100: sf100_experiment.log" | tee -a "$SUMMARY_FILE"
echo "  - SF1000: sf1000_experiment.log" | tee -a "$SUMMARY_FILE"
echo "  - SF10000: sf10000_experiment.log" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Key metrics to analyze:" | tee -a "$SUMMARY_FILE"
echo "  - Number of runs generated for each memory configuration" | tee -a "$SUMMARY_FILE"
echo "  - Sort time as a function of memory allocation" | tee -a "$SUMMARY_FILE"
echo "  - Throughput (M entries/sec) for different run counts" | tee -a "$SUMMARY_FILE"
echo "  - Run generation time vs merge time trade-offs" | tee -a "$SUMMARY_FILE"