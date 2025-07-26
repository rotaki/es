#!/bin/bash

# Script to generate GenSort datasets of various sizes
# Each record is 100 bytes

# Create output directory if it doesn't exist
OUTPUT_DIR="./gensort_data"
mkdir -p "$OUTPUT_DIR"

# Path to gensort binary
GENSORT="./dataset_generator/gensort"

# Number of threads to use (default to number of CPU cores)
THREADS=$(nproc)

# Function to calculate number of records for given size in GiB
calculate_records() {
    local size_gib=$1
    local bytes=$((size_gib * 1024 * 1024 * 1024))
    local records=$((bytes / 100))
    echo $records
}

# Generate datasets from 10 to 100 GiB in 10 GiB increments
for size in 10 20 30 40 50 60 70 80 90 100; do
    records=$(calculate_records $size)
    output_file="$OUTPUT_DIR/gensort_${size}GiB.data"
    
    echo "Generating ${size} GiB dataset..."
    echo "  Records: $records"
    echo "  Output: $output_file"
    echo "  Threads: $THREADS"
    echo "  Command: $GENSORT -t$THREADS $records $output_file"
    echo ""
    
    # Uncomment the line below to actually run the generation
    # $GENSORT -n$THREADS $records "$output_file"
done

echo "Dataset generation commands prepared."
echo "Using $THREADS threads for generation (detected from CPU cores)."
echo "To generate the datasets, uncomment the execution line in the script and run again."
echo ""
echo "You can also override the thread count by setting THREADS environment variable:"
echo "  THREADS=8 ./generate_datasets.sh"