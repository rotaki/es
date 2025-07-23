# Unified Sort CLI

A unified command-line tool for sorting various file formats using external sorting with configurable thread counts for run generation and merge phases.

## Installation

```bash
cargo build --release --bin unified-sort
```

The binary will be available at `target/release/unified-sort`.

## Supported Formats

- **CSV Files**: TPC-H lineitem and NYC yellow trip data schemas
- **GenSort Binary Files**: 10-byte keys with 90-byte payloads

## Usage

### CSV Sorting

Sort CSV files with specific schemas:

```bash
# Sort lineitem data
unified-sort csv -i lineitem.tbl --schema lineitem -r 4 -g 4 -m 1024

# Sort yellow trip data
unified-sort csv -i yellow_tripdata.csv --schema yellow-trip -r 8 -g 8 -m 2048

# Run benchmark mode (multiple runs)
unified-sort csv -i lineitem.tbl --schema lineitem -r 4 -g 4 -m 1024 --benchmark --benchmark-runs 5

# Comprehensive benchmark with multiple configurations
unified-sort csv -i lineitem.tbl --schema lineitem -b --thread-counts 1,2,4,8 --memory-sizes 512MB,1GB,2GB,4GB
```

Options:
- `-i, --input`: Input CSV file path
- `--schema`: CSV schema type (`lineitem` or `yellow-trip`)
- `-r, --run-gen-threads`: Number of threads for run generation (default: 4)
- `-g, --merge-threads`: Number of threads for merge phase (default: 4)
- `-m, --memory-mb`: Maximum memory usage in MB (default: 1024)
- `-d, --delimiter`: CSV delimiter (default: |)
- `-t, --temp-dir`: Directory for temporary files (default: .)
- `-b, --benchmark`: Run in benchmark mode
- `--benchmark-runs`: Number of benchmark runs (default: 3)
- `--thread-counts`: Comma-separated thread counts for comprehensive benchmark (e.g., 1,2,4,8)
- `--memory-sizes`: Comma-separated memory sizes for comprehensive benchmark (e.g., 512MB,1GB,2GB or 512,1024,2048)

### GenSort Binary Sorting

Sort GenSort format binary files:

```bash
# Basic sorting
unified-sort gensort -i data.gensort -r 4 -g 4 -m 1024

# Sort with verification
unified-sort gensort -i data.gensort -r 8 -g 8 -m 2048 --verify
```

Options:
- `-i, --input`: Input GenSort file path
- `-r, --run-gen-threads`: Number of threads for run generation (default: 4)
- `-g, --merge-threads`: Number of threads for merge phase (default: 4)
- `-m, --memory-mb`: Maximum memory usage in MB (default: 1024)
- `-t, --temp-dir`: Directory for temporary files (default: .)
- `-v, --verify`: Verify sorted output

### Auto-Detection Mode

Automatically detect file format based on extension or content:

```bash
# Auto-detect and sort
unified-sort auto -i data.csv -r 4 -g 4 -m 1024
unified-sort auto -i data.gensort -r 4 -g 4 -m 1024
```

The tool will:
- Detect CSV files by `.csv` or `.tbl` extensions
- Detect GenSort files by `.gensort` or `.dat` extensions
- For CSV files, attempt to detect schema from filename (e.g., "lineitem" or "yellow_trip")
- Fall back to content analysis if extension is ambiguous

## Performance Tips

1. **Thread Configuration**:
   - Set run generation threads based on I/O capability (typically 4-8)
   - Set merge threads based on CPU cores (can be higher than run gen threads)
   - Example: `-r 4 -g 16` for I/O-bound run generation and CPU-bound merge

2. **Memory Configuration**:
   - More memory reduces the number of runs, improving performance
   - Typical values: 1024 MB to 8192 MB
   - Balance with available system memory

3. **Direct I/O**:
   - The tool uses Direct I/O for efficient file reading
   - Temporary directory should be on a fast disk
   - SSD recommended for best performance

## Examples

```bash
# Sort 1GB lineitem file with balanced threads
unified-sort csv -i lineitem.tbl --schema lineitem -r 4 -g 4 -m 2048

# Sort GenSort file with high merge parallelism
unified-sort gensort -i test_1m.gensort -r 4 -g 16 -m 4096 --verify

# Benchmark CSV sorting with 5 runs
unified-sort csv -i lineitem.tbl --schema lineitem -r 8 -g 8 -m 4096 -b --benchmark-runs 5

# Comprehensive benchmark comparing multiple configurations
unified-sort csv -i lineitem.tbl --schema lineitem -b --thread-counts 4,8,16 --memory-sizes 512MB,1GB,2GB,4GB

# Auto-detect and sort with custom temp directory
unified-sort auto -i data.csv -r 4 -g 8 -m 2048 -t /tmp/sort_temp
```

## Output

The tool provides detailed statistics including:
- Number of runs generated
- Run generation and merge times
- Throughput in M entries/second
- I/O statistics (read/write operations and bytes)
- Verification results (if requested)