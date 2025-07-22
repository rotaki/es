#!/bin/bash

# Download yellow trip data for a specific year and convert to CSV
# Usage: ./download_yellow_trip.sh YEAR [OUTPUT_DIR]

if [ $# -lt 1 ]; then
    echo "Usage: $0 YEAR [OUTPUT_DIR]"
    echo "Example: $0 2023"
    echo "Example: $0 2023 /path/to/output"
    exit 1
fi

YEAR=$1
OUTPUT_DIR=${2:-"yellow_trip_data"}
URLS_FILE="yellow_trip_post_2011.txt"

# Validate year
if [ "$YEAR" -lt 2011 ]; then
    echo "Error: Year must be 2011 or later"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Yellow Trip Data Downloader"
echo "=========================="
echo "Year: $YEAR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Extract URLs for the specified year
YEAR_URLS=$(grep "yellow_tripdata_${YEAR}-" "$URLS_FILE")

if [ -z "$YEAR_URLS" ]; then
    echo "No yellow trip data found for year $YEAR"
    exit 1
fi

# Count files
FILE_COUNT=$(echo "$YEAR_URLS" | wc -l)
echo "Found $FILE_COUNT files for year $YEAR"
echo ""

# Download files
DOWNLOADED=0
echo "Downloading files..."
echo "$YEAR_URLS" | while IFS= read -r url; do
    if [ -n "$url" ]; then
        FILENAME=$(basename "$url")
        OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "  ✓ $FILENAME already exists, skipping..."
        else
            echo "  ↓ Downloading $FILENAME..."
            if wget -q --show-progress "$url" -O "$OUTPUT_FILE"; then
                echo "  ✓ $FILENAME downloaded successfully"
                ((DOWNLOADED++))
            else
                echo "  ✗ Failed to download $FILENAME"
                rm -f "$OUTPUT_FILE"
            fi
        fi
    fi
done

echo ""
echo "Download complete!"
echo ""

# Convert to CSV
CSV_OUTPUT="yellow_tripdata_${YEAR}.csv"
echo "Converting to CSV..."
echo "Output file: $CSV_OUTPUT"
echo ""

# Change to output directory and run conversion
cd "$OUTPUT_DIR" || exit 1

# Run the conversion script
python3 ../yellow_trip_parquet_to_csv.py -o "../$CSV_OUTPUT" -y "$YEAR"

cd ..

# Check if conversion was successful
if [ -f "$CSV_OUTPUT" ]; then
    SIZE=$(du -h "$CSV_OUTPUT" | cut -f1)
    echo ""
    echo "✅ Successfully created $CSV_OUTPUT (size: $SIZE)"
else
    echo ""
    echo "❌ Conversion failed - $CSV_OUTPUT not found"
    exit 1
fi