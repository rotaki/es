# External Sorting Library

This library provides high-performance external sorting capabilities with Direct I/O support for processing large datasets that don't fit in memory.

## Benchmarking with TPC-H Lineitem Table

The library includes benchmark utilities for testing external sorting performance using TPC-H lineitem data in both CSV and Parquet formats.

### Generating TPC-H Data

To generate TPC-H lineitem data for benchmarking, you can use [tpchgen-rs](https://github.com/clflushopt/tpchgen-rs):

### CSV Lineitem Benchmark

The CSV benchmark tool allows you to test external sorting performance on TPC-H lineitem data in CSV format.

#### Usage

```bash
cargo run --release --example csv_lineitem_benchmark_cli -- [OPTIONS] <CSV_FILE>
```

#### Options

- `-k, --key-columns <COLS>` - Comma-separated list of key column indices (default: 0,3)
- `-v, --value-columns <COLS>` - Comma-separated list of value column indices (default: 10,5)
- `-t, --threads <LIST>` - Comma-separated thread counts to test (default: 1,2,4,8,16,32)
- `-m, --memory <LIST>` - Comma-separated memory sizes (e.g., 1GB,2GB,4GB or 1024,2048,4096)
  - Default: 1GB,2GB,4GB,8GB,16GB,32GB
- `-d, --delimiter <CHAR>` - CSV delimiter character (default: |)
- `-b, --buffer-size <KB>` - Direct I/O buffer size in KB (default: 64)
- `--headers` - CSV file has headers
- `--help` - Show help message

#### TPC-H Lineitem Column Indices

| Index | Column Name       | Data Type |
|-------|------------------|-----------|
| 0     | l_orderkey       | Int64     |
| 1     | l_partkey        | Int64     |
| 2     | l_suppkey        | Int64     |
| 3     | l_linenumber     | Int32     |
| 4     | l_quantity       | Float64   |
| 5     | l_extendedprice  | Float64   |
| 6     | l_discount       | Float64   |
| 7     | l_tax            | Float64   |
| 8     | l_returnflag     | String    |
| 9     | l_linestatus     | String    |
| 10    | l_shipdate       | Date32    |
| 11    | l_commitdate     | Date32    |
| 12    | l_receiptdate    | Date32    |
| 13    | l_shipinstruct   | String    |
| 14    | l_shipmode       | String    |
| 15    | l_comment        | String    |

#### Examples

```bash
# Sort by orderkey and linenumber (default)
cargo run --release --example csv_lineitem_benchmark_cli -- lineitem.csv

# Sort by shipdate
cargo run --release --example csv_lineitem_benchmark_cli -- -k 10 -v 0,3 lineitem.csv

# Sort by extended price with custom thread counts
cargo run --release --example csv_lineitem_benchmark_cli -- -k 5 -v 0,3,10 -t 1,4,8,16 lineitem.csv

# Test with specific memory sizes
cargo run --release --example csv_lineitem_benchmark_cli -- -m 2GB,4GB,8GB lineitem.csv

# Custom thread and memory combinations
cargo run --release --example csv_lineitem_benchmark_cli -- -t 1,8,32 -m 1024,8192,32768 lineitem.csv

# Use pipe delimiter (TPC-H default)
cargo run --release --example csv_lineitem_benchmark_cli -- -d "|" lineitem.csv
```

### Parquet Lineitem Benchmark

The Parquet benchmark tool provides similar functionality for Parquet format files.

#### Usage

```bash
cargo run --release --example parquet_lineitem_benchmark_cli -- [OPTIONS] <PARQUET_FILE>
```

#### Examples

```bash
# Sort by orderkey and linenumber (default)
cargo run --release --example parquet_lineitem_benchmark_cli -- lineitem.parquet

# Sort by shipdate
cargo run --release --example parquet_lineitem_benchmark_cli -- -k 10 -v 0,3 lineitem.parquet

# Sort by extended price with custom thread counts
cargo run --release --example parquet_lineitem_benchmark_cli -- -k 5 -v 0,3,10 -t 1,4,8,16 lineitem.parquet

# Test with specific memory sizes
cargo run --release --example parquet_lineitem_benchmark_cli -- -m 2GB,4GB,8GB lineitem.parquet

# Custom thread and memory combinations
cargo run --release --example parquet_lineitem_benchmark_cli -- -t 1,8,32 -m 1024,8192,32768 lineitem.parquet
```

### Benchmark Output

Both benchmark tools provide detailed performance metrics including:

- Sort time and throughput (million entries per second)
- Number of runs generated during external sorting
- I/O statistics (read/write operations and bytes)
- Speedup analysis for different thread counts
- First and last sorted entries for verification

The benchmarks automatically test different combinations of thread counts and memory sizes, making it easy to find the optimal configuration for your hardware.

### Performance Tips

1. **Direct I/O Buffer Size**: The default 64KB buffer size works well for most cases, but you may want to experiment with larger sizes (128KB, 256KB) for very large files.

2. **Memory Allocation**: More memory generally means fewer runs and better performance. The benchmark will show how many runs were generated for each configuration.

3. **Thread Count**: Performance typically improves with more threads up to the number of CPU cores, but I/O bandwidth may become the bottleneck.

4. **File System**: For best performance, use a fast SSD with a file system that supports Direct I/O (ext4, XFS).