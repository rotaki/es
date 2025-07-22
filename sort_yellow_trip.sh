#!/bin/bash

# Script to sort CSV by 2nd and 3rd columns

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_csv_file> [output_csv_file]"
    echo "If output file is not specified, sorted output will be printed to stdout"
    exit 1
fi

input_file="$1"
output_file="$2"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found"
    exit 1
fi

# Sort the CSV by 2nd column, then 3rd column
# -t',' sets comma as field separator
# -k2,2 sorts by 2nd field
# -k3,3 sorts by 3rd field (secondary sort)
if [ -z "$output_file" ]; then
    # Output to stdout if no output file specified
    sort -t',' -k2,2 -k3,3 "$input_file"
else
    # Output to file
    sort -t',' -k2,2 -k3,3 "$input_file" > "$output_file"
    echo "Sorted CSV saved to: $output_file"
fi