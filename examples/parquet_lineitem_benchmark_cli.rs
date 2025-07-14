//! Flexible benchmark for sorting TPC-H lineitem table in Parquet format with CLI arguments using Direct I/O
//!
//! Usage: parquet_lineitem_benchmark_cli [OPTIONS] <PARQUET_FILE>
//!
//! Options:
//!   -k, --key-columns <COLS>     Comma-separated list of key column indices (default: 0,3)
//!   -v, --value-columns <COLS>   Comma-separated list of value column indices (default: 10,5)
//!   -t, --threads <LIST>         Comma-separated thread counts to test (default: 1,2,4,8,16,32)
//!   -m, --memory <LIST>          Comma-separated memory sizes (e.g., 1GB,2GB,4GB or 1024,2048,4096)
//!                                Default: 1GB,2GB,4GB,8GB,16GB,32GB
//!   -b, --buffer-size <KB>       Direct I/O buffer size in KB (default: 64)
//!   --help                       Show this help message

use es::{
    order_preserving_encoding::decode_bytes, ExternalSorter, IoStatsTracker, ParquetDirectConfig,
    ParquetInputDirect, Sorter,
};
use std::env;
use std::path::Path;
use std::process;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Config {
    parquet_file: String,
    key_columns: Vec<usize>,
    value_columns: Vec<usize>,
    thread_counts: Vec<usize>,
    memory_sizes: Vec<usize>,
    buffer_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            parquet_file: String::new(),
            key_columns: vec![0, 3],    // l_orderkey, l_linenumber
            value_columns: vec![10, 5], // l_shipdate, l_extendedprice
            // thread_counts: vec![1, 2, 4, 8, 16, 32],
            thread_counts: vec![4, 8],
            // memory_sizes: vec![1024, 2048, 4096, 8192, 16384, 32768], // MB
            memory_sizes: vec![4096, 8192], // MB
            buffer_size: 64 * 1024,         // 64KB default
        }
    }
}

#[derive(Clone)]
struct BenchmarkResult {
    threads: usize,
    memory_mb: usize,
    sort_time: Duration,
    count_time: Duration,
    total_entries: usize,
    throughput_meps: f64, // Million entries per second
    num_runs: usize,      // Number of runs generated
    read_ops: u64,
    read_bytes: u64,
    write_ops: u64,
    write_bytes: u64,
}

impl BenchmarkResult {
    fn new(
        threads: usize,
        memory_mb: usize,
        sort_time: Duration,
        count_time: Duration,
        total_entries: usize,
        num_runs: usize,
        io_stats: &IoStatsTracker,
    ) -> Self {
        let throughput_meps = total_entries as f64 / sort_time.as_secs_f64() / 1_000_000.0;
        let (read_ops, read_bytes) = io_stats.get_read_stats();
        let (write_ops, write_bytes) = io_stats.get_write_stats();
        Self {
            threads,
            memory_mb,
            sort_time,
            count_time,
            total_entries,
            throughput_meps,
            num_runs,
            read_ops,
            read_bytes,
            write_ops,
            write_bytes,
        }
    }
}

// Column names for TPC-H lineitem table
const COLUMN_NAMES: &[&str] = &[
    "l_orderkey",      // 0
    "l_partkey",       // 1
    "l_suppkey",       // 2
    "l_linenumber",    // 3
    "l_quantity",      // 4
    "l_extendedprice", // 5
    "l_discount",      // 6
    "l_tax",           // 7
    "l_returnflag",    // 8
    "l_linestatus",    // 9
    "l_shipdate",      // 10
    "l_commitdate",    // 11
    "l_receiptdate",   // 12
    "l_shipinstruct",  // 13
    "l_shipmode",      // 14
    "l_comment",       // 15
];

fn main() {
    let config = parse_args();

    if let Err(e) = run_benchmark(config) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let mut config = Config::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-k" | "--key-columns" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --key-columns requires an argument");
                    show_usage();
                }
                config.key_columns = parse_column_list(&args[i]);
            }
            "-v" | "--value-columns" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --value-columns requires an argument");
                    show_usage();
                }
                config.value_columns = parse_column_list(&args[i]);
            }
            "-t" | "--threads" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --threads requires an argument");
                    show_usage();
                }
                config.thread_counts = parse_size_list(&args[i]);
                if config.thread_counts.is_empty() {
                    eprintln!("Error: Invalid thread counts");
                    show_usage();
                }
            }
            "-m" | "--memory" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --memory requires an argument");
                    show_usage();
                }
                config.memory_sizes = parse_memory_list(&args[i]);
                if config.memory_sizes.is_empty() {
                    eprintln!("Error: Invalid memory sizes");
                    show_usage();
                }
            }
            "-b" | "--buffer-size" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --buffer-size requires an argument");
                    show_usage();
                }
                let kb: usize = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Error: Invalid buffer size");
                    show_usage();
                });
                config.buffer_size = kb * 1024;
            }
            "--help" | "-h" => {
                show_usage();
            }
            _ => {
                if args[i].starts_with('-') {
                    eprintln!("Error: Unknown option: {}", args[i]);
                    show_usage();
                } else {
                    config.parquet_file = args[i].clone();
                }
            }
        }
        i += 1;
    }

    if config.parquet_file.is_empty() {
        eprintln!("Error: Parquet file path is required");
        show_usage();
    }

    config
}

fn parse_column_list(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

fn parse_size_list(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

fn parse_memory_list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|x| {
            let trimmed = x.trim();
            if trimmed.ends_with("GB") || trimmed.ends_with("gb") {
                trimmed[..trimmed.len() - 2]
                    .parse::<usize>()
                    .ok()
                    .map(|v| v * 1024)
            } else if trimmed.ends_with("MB") || trimmed.ends_with("mb") {
                trimmed[..trimmed.len() - 2].parse::<usize>().ok()
            } else {
                trimmed.parse::<usize>().ok()
            }
        })
        .collect()
}

fn show_usage() -> ! {
    println!("TPC-H Lineitem Parquet Sort Benchmark with configurable columns");
    println!(
        "\nUsage: {} [OPTIONS] <PARQUET_FILE>",
        env::args().next().unwrap()
    );
    println!("\nOptions:");
    println!(
        "  -k, --key-columns <COLS>     Comma-separated list of key column indices (default: 0,3)"
    );
    println!("  -v, --value-columns <COLS>   Comma-separated list of value column indices (default: 10,5)");
    println!("  -t, --threads <LIST>         Comma-separated thread counts to test (default: 1,2,4,8,16,32)");
    println!("  -m, --memory <LIST>          Comma-separated memory sizes (e.g., 1GB,2GB,4GB or 1024,2048,4096)");
    println!("                               Default: 1GB,2GB,4GB,8GB,16GB,32GB");
    println!("  -b, --buffer-size <KB>       Direct I/O buffer size in KB (default: 64)");
    println!("  --help                       Show this help message");
    println!("\nColumn indices for TPC-H lineitem table:");
    for (i, name) in COLUMN_NAMES.iter().enumerate() {
        println!("  {}: {}", i, name);
    }
    println!("\nExamples:");
    println!("  # Sort by orderkey and linenumber (default)");
    println!("  {} lineitem.parquet", env::args().next().unwrap());
    println!("\n  # Sort by shipdate");
    println!(
        "  {} -k 10 -v 0,3 lineitem.parquet",
        env::args().next().unwrap()
    );
    println!("\n  # Sort by extended price with custom thread counts");
    println!(
        "  {} -k 5 -v 0,3,10 -t 1,4,8,16 lineitem.parquet",
        env::args().next().unwrap()
    );
    println!("\n  # Test with specific memory sizes");
    println!(
        "  {} -m 2GB,4GB,8GB lineitem.parquet",
        env::args().next().unwrap()
    );
    println!("\n  # Custom thread and memory combinations");
    println!(
        "  {} -t 1,8,32 -m 1024,8192,32768 lineitem.parquet",
        env::args().next().unwrap()
    );
    process::exit(1);
}

fn run_benchmark(config: Config) -> Result<(), String> {
    println!("TPC-H Lineitem Parquet Sort Benchmark");
    println!("======================================");

    if !Path::new(&config.parquet_file).exists() {
        return Err(format!("File {} not found", config.parquet_file));
    }

    // File information
    let file_metadata = std::fs::metadata(&config.parquet_file)
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let file_size = file_metadata.len();

    println!("\nFile Information:");
    println!("  Path: {}", config.parquet_file);
    println!(
        "  Size: {:.2} GB ({} bytes)",
        file_size as f64 / (1024.0 * 1024.0 * 1024.0),
        file_size
    );

    // Read Parquet metadata to get row count and schema info
    let parquet_reader =
        ParquetInputDirect::new(&config.parquet_file, ParquetDirectConfig::default())?;
    println!("  Total rows: {}", parquet_reader.len());

    println!("\nConfiguration:");
    println!(
        "  Key columns: {:?} ({})",
        config.key_columns,
        format_column_names(&config.key_columns)
    );
    println!(
        "  Value columns: {:?} ({})",
        config.value_columns,
        format_column_names(&config.value_columns)
    );
    println!("  Direct I/O buffer: {} KB", config.buffer_size / 1024);
    println!("  Thread counts to test: {:?}", config.thread_counts);
    println!("  Memory sizes to test (MB): {:?}", config.memory_sizes);

    // Run all benchmarks
    let mut results = Vec::new();

    for &memory_mb in &config.memory_sizes {
        for &threads in &config.thread_counts {
            println!("\n{}", "=".repeat(70));
            println!(
                "Running benchmark: {} threads, {} MB memory",
                threads, memory_mb
            );
            println!("{}", "=".repeat(70));

            match run_single_benchmark(&config, threads, memory_mb) {
                Ok(result) => results.push(result),
                Err(e) => eprintln!("Failed to run benchmark: {}", e),
            }
        }
    }

    // Display results table
    display_results_table(&results);

    Ok(())
}

fn format_column_names(indices: &[usize]) -> String {
    indices
        .iter()
        .map(|&i| {
            if i < COLUMN_NAMES.len() {
                COLUMN_NAMES[i]
            } else {
                "unknown"
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn run_single_benchmark(
    config: &Config,
    threads: usize,
    memory_mb: usize,
) -> Result<BenchmarkResult, String> {
    // Create IO tracker
    let io_tracker = IoStatsTracker::new();

    let mut parquet_config = ParquetDirectConfig::default();
    parquet_config.key_columns = config.key_columns.clone();
    parquet_config.value_columns = config.value_columns.clone();
    parquet_config.buffer_size = config.buffer_size;

    println!("Configuration:");
    println!("  Threads: {}", threads);
    println!("  Total memory budget: {} MB", memory_mb);
    println!("  Buffer per thread: {} MB", memory_mb / threads);
    println!("  Direct I/O buffer: {} KB", config.buffer_size / 1024);

    let mut parquet_input = ParquetInputDirect::new(&config.parquet_file, parquet_config)?;
    parquet_input.io_stats = Some(io_tracker.clone());
    let mut sorter = ExternalSorter::new(threads, memory_mb * 1024 * 1024);

    println!("\nStarting sort...");
    let start = Instant::now();

    let output = sorter.sort(Box::new(parquet_input))?;

    let sort_time = start.elapsed();
    println!("Sort completed in {:.2} seconds", sort_time.as_secs_f64());

    // Get sort statistics
    let stats = output.stats();
    println!("Sort statistics: {} runs generated", stats.num_runs);

    // Count entries for verification (not included in performance metrics)
    println!("\nCounting entries for verification...");
    let count_start = Instant::now();

    let mut count = 0;
    let mut first_entries = Vec::new();
    let mut last_entries = Vec::new();

    for (key, value) in output.iter() {
        if count < 5 {
            first_entries.push((key.clone(), value.clone()));
        }
        count += 1;
        if count > 0 && count % 1_000_000 == 0 {
            print!(".");
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
        // Keep last 5 entries
        last_entries.push((key, value));
        if last_entries.len() > 5 {
            last_entries.remove(0);
        }
    }
    println!();

    let count_time = count_start.elapsed();
    let total_entries = count as usize;
    println!("Verified {} entries", total_entries);

    println!("\nFirst 5 entries:");
    for (i, (key, value)) in first_entries.iter().enumerate() {
        print_entry(i, key, value, &config.key_columns, &config.value_columns);
    }

    println!("\nLast 5 entries:");
    let start_idx = total_entries.saturating_sub(5);
    for (i, (key, value)) in last_entries.iter().enumerate() {
        print_entry(
            start_idx + i,
            key,
            value,
            &config.key_columns,
            &config.value_columns,
        );
    }

    Ok(BenchmarkResult::new(
        threads,
        memory_mb,
        sort_time,
        count_time,
        total_entries,
        stats.num_runs,
        &io_tracker,
    ))
}

fn display_results_table(results: &[BenchmarkResult]) {
    if results.is_empty() {
        println!("\nNo results to display");
        return;
    }

    println!("\n{}", "=".repeat(100));
    println!("Benchmark Results Summary");
    println!("{}", "=".repeat(100));

    // Header
    println!(
        "{:<10} {:<12} {:<8} {:<12} {:<15} {:<18} {:<15}",
        "Threads", "Memory (MB)", "Runs", "Sort (s)", "Entries", "Throughput", "Read MB/s"
    );
    println!(
        "{:<10} {:<12} {:<8} {:<12} {:<15} {:<18} {:<15}",
        "", "", "", "", "", "(M entries/s)", ""
    );
    println!("{}", "-".repeat(100));

    // Results rows
    for result in results {
        let read_mb_per_sec =
            (result.read_bytes as f64 / (1024.0 * 1024.0)) / result.sort_time.as_secs_f64();

        println!(
            "{:<10} {:<12} {:<8} {:<12.2} {:<15} {:<18.2} {:<15.2}",
            result.threads,
            result.memory_mb,
            result.num_runs,
            result.sort_time.as_secs_f64(),
            result.total_entries,
            result.throughput_meps,
            read_mb_per_sec
        );
    }

    println!("{}", "=".repeat(100));

    // Calculate and display speedup table for same memory size
    let memory_sizes: Vec<usize> = results
        .iter()
        .map(|r| r.memory_mb)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for &mem_size in &memory_sizes {
        let mem_results: Vec<&BenchmarkResult> =
            results.iter().filter(|r| r.memory_mb == mem_size).collect();
        if mem_results.len() > 1 {
            println!("\nSpeedup Analysis for {} MB:", mem_size);
            println!(
                "{:<10} {:<15} {:<15} {:<15}",
                "Threads", "Sort Time (s)", "Speedup", "Efficiency (%)"
            );
            println!("{}", "-".repeat(55));

            if let Some(single_thread) = mem_results.iter().find(|r| r.threads == 1) {
                for result in &mem_results {
                    let speedup =
                        single_thread.sort_time.as_secs_f64() / result.sort_time.as_secs_f64();
                    let efficiency = (speedup / result.threads as f64) * 100.0;
                    println!(
                        "{:<10} {:<15.2} {:<15.2} {:<15.1}",
                        result.threads,
                        result.sort_time.as_secs_f64(),
                        speedup,
                        efficiency
                    );
                }
            }
        }
    }

    // I/O Statistics Summary
    println!("\nI/O Statistics Summary:");
    println!(
        "{:<10} {:<12} {:<15} {:<15} {:<15} {:<15}",
        "Threads", "Memory (MB)", "Read Ops", "Read MB", "Write Ops", "Write MB"
    );
    println!("{}", "-".repeat(82));

    for result in results {
        let read_mb = result.read_bytes as f64 / (1024.0 * 1024.0);
        let write_mb = result.write_bytes as f64 / (1024.0 * 1024.0);

        println!(
            "{:<10} {:<12} {:<15} {:<15.2} {:<15} {:<15.2}",
            result.threads, result.memory_mb, result.read_ops, read_mb, result.write_ops, write_mb
        );
    }
}

fn print_entry(index: usize, key: &[u8], value: &[u8], key_cols: &[usize], value_cols: &[usize]) {
    print!("  [{}] ", index);

    // For Parquet, we need to know the data types to decode properly
    // This is a simplified version - in real usage, you might want to get the schema
    // from the Parquet file and use it here

    // Decode and print key columns
    let mut offset = 0;
    for (i, &col_idx) in key_cols.iter().enumerate() {
        if offset >= key.len() {
            break;
        }

        // Get the expected size for this column type based on TPC-H schema
        let (field_type, expected_size) = match col_idx {
            0 | 1 | 2 => ("Int64", 8),       // orderkey, partkey, suppkey
            3 => ("Int32", 4),               // linenumber
            4 | 5 | 6 | 7 => ("Float64", 8), // quantity, extendedprice, discount, tax
            10 | 11 | 12 => ("Date32", 4),   // shipdate, commitdate, receiptdate
            _ => ("Utf8", 0),                // Variable length strings
        };

        if expected_size > 0 {
            let field_end = offset + expected_size;
            if field_end > key.len() {
                break;
            }

            let field_bytes = &key[offset..field_end];
            let decoded = decode_bytes(field_bytes, field_type)
                .unwrap_or_else(|_| String::from_utf8_lossy(field_bytes).to_string());

            if col_idx < COLUMN_NAMES.len() {
                print!("{}={}", COLUMN_NAMES[col_idx], decoded);
            } else {
                print!("col{}={}", col_idx, decoded);
            }

            if i < key_cols.len() - 1 {
                print!(", ");
            }

            offset = field_end;
            if offset < key.len() && key[offset] == 0 {
                offset += 1;
            }
        } else {
            // Variable length - find null terminator
            let field_end = key[offset..]
                .iter()
                .position(|&b| b == 0)
                .map(|pos| offset + pos)
                .unwrap_or(key.len());

            let field_bytes = &key[offset..field_end];
            let decoded = String::from_utf8_lossy(field_bytes);

            if col_idx < COLUMN_NAMES.len() {
                print!("{}={}", COLUMN_NAMES[col_idx], decoded);
            } else {
                print!("col{}={}", col_idx, decoded);
            }

            if i < key_cols.len() - 1 {
                print!(", ");
            }

            offset = field_end;
            if offset < key.len() && key[offset] == 0 {
                offset += 1;
            }
        }
    }

    print!(" -> ");

    // Decode and print value columns (similar logic)
    offset = 0;
    for (i, &col_idx) in value_cols.iter().enumerate() {
        if offset >= value.len() {
            break;
        }

        let (field_type, expected_size) = match col_idx {
            0 | 1 | 2 => ("Int64", 8),
            3 => ("Int32", 4),
            4 | 5 | 6 | 7 => ("Float64", 8),
            10 | 11 | 12 => ("Date32", 4),
            _ => ("Utf8", 0),
        };

        if expected_size > 0 {
            let field_end = offset + expected_size;
            if field_end > value.len() {
                break;
            }

            let field_bytes = &value[offset..field_end];
            let decoded = decode_bytes(field_bytes, field_type)
                .unwrap_or_else(|_| String::from_utf8_lossy(field_bytes).to_string());

            if col_idx < COLUMN_NAMES.len() {
                print!("{}={}", COLUMN_NAMES[col_idx], decoded);
            } else {
                print!("col{}={}", col_idx, decoded);
            }

            if i < value_cols.len() - 1 {
                print!(", ");
            }

            offset = field_end;
            if offset < value.len() && value[offset] == 0 {
                offset += 1;
            }
        } else {
            let field_end = value[offset..]
                .iter()
                .position(|&b| b == 0)
                .map(|pos| offset + pos)
                .unwrap_or(value.len());

            let field_bytes = &value[offset..field_end];
            let decoded = String::from_utf8_lossy(field_bytes);

            if col_idx < COLUMN_NAMES.len() {
                print!("{}={}", COLUMN_NAMES[col_idx], decoded);
            } else {
                print!("col{}={}", col_idx, decoded);
            }

            if i < value_cols.len() - 1 {
                print!(", ");
            }

            offset = field_end;
            if offset < value.len() && value[offset] == 0 {
                offset += 1;
            }
        }
    }

    println!();
}
