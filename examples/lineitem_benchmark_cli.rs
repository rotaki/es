//! Unified benchmark for sorting TPC-H lineitem table in CSV or Parquet format with CLI arguments using Direct I/O
//!
//! Usage: lineitem_benchmark_cli [OPTIONS] <FILE>
//!
//! Options:
//!   -k, --key-columns <COLS>     Comma-separated list of key column indices (default: 0,3)
//!   -v, --value-columns <COLS>   Comma-separated list of value column indices (default: 10,5)
//!   -t, --threads <LIST>         Comma-separated thread counts to test (default: 1,2,4,8,16,32)
//!   -m, --memory <LIST>          Comma-separated memory sizes (e.g., 1GB,2GB,4GB or 1024,2048,4096)
//!                                Default: 1GB,2GB,4GB,8GB,16GB,32GB
//!   -b, --buffer-size <KB>       Direct I/O buffer size in KB (default: 64)
//!   -d, --delimiter <CHAR>       CSV delimiter character (default: |) - ignored for Parquet files
//!   --headers                    CSV file has headers - ignored for Parquet files
//!   --help                       Show this help message

use arrow::datatypes::{DataType, Field, Schema};
use es::constants::DEFAULT_BUFFER_SIZE;
use es::{
    order_preserving_encoding::decode_bytes, CsvDirectConfig, CsvInputDirect, ExternalSorter, ParquetDirectConfig, ParquetInputDirect, SortInput, Sorter,
};
use std::env;
use std::path::Path;
use std::process;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Config {
    file_path: String,
    key_columns: Vec<usize>,
    value_columns: Vec<usize>,
    thread_counts: Vec<usize>,
    memory_sizes: Vec<usize>,
    delimiter: u8,
    has_headers: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            key_columns: vec![0, 3],    // l_orderkey, l_linenumber
            value_columns: vec![10, 5], // l_shipdate, l_extendedprice
            // thread_counts: vec![1, 2, 4, 8, 16, 32],
            thread_counts: vec![4, 8],
            // memory_sizes: vec![1024, 2048, 4096, 8192, 16384, 32768], // MB
            memory_sizes: vec![4096, 8192], // MB
            delimiter: b',',
            has_headers: true,
        }
    }
}

#[derive(Clone)]
struct BenchmarkResult {
    threads: usize,
    memory_mb: usize,
    sort_time: Duration,
    run_gen_time: Duration,
    merge_time: Duration,
    total_entries: usize,
    throughput_meps: f64, // Million entries per second
    num_runs: usize,      // Number of runs generated
    // Aggregated IO stats
    #[allow(dead_code)]
    read_ops: u64,
    read_bytes: u64,
    #[allow(dead_code)]
    write_ops: u64,
    write_bytes: u64,
    // Detailed IO stats
    run_gen_read_ops: u64,
    run_gen_read_bytes: u64,
    run_gen_write_ops: u64,
    run_gen_write_bytes: u64,
    merge_read_ops: u64,
    merge_read_bytes: u64,
    merge_write_ops: u64,
    merge_write_bytes: u64,
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
            "-d" | "--delimiter" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --delimiter requires an argument");
                    show_usage();
                }
                if args[i].len() != 1 {
                    eprintln!("Error: Delimiter must be a single character");
                    show_usage();
                }
                config.delimiter = args[i].as_bytes()[0];
            }
            "--headers" => {
                config.has_headers = true;
            }
            "--help" | "-h" => {
                show_usage();
            }
            _ => {
                if args[i].starts_with('-') {
                    eprintln!("Error: Unknown option: {}", args[i]);
                    show_usage();
                } else {
                    config.file_path = args[i].clone();
                }
            }
        }
        i += 1;
    }

    if config.file_path.is_empty() {
        eprintln!("Error: File path is required");
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

fn bytes_to_human_readable(bytes: usize) -> String {
    const GB: usize = 1024 * 1024 * 1024;
    const MB: usize = 1024 * 1024;
    const KB: usize = 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn show_usage() -> ! {
    println!("TPC-H Lineitem Sort Benchmark with configurable columns");
    println!("\nUsage: {} [OPTIONS] <FILE>", env::args().next().unwrap());
    println!("\nSupported file formats: .csv, .parquet");
    println!("\nOptions:");
    println!(
        "  -k, --key-columns <COLS>     Comma-separated list of key column indices (default: 0,3)"
    );
    println!("  -v, --value-columns <COLS>   Comma-separated list of value column indices (default: 10,5)");
    println!("  -t, --threads <LIST>         Comma-separated thread counts to test (default: 1,2,4,8,16,32)");
    println!("  -m, --memory <LIST>          Comma-separated memory sizes (e.g., 1GB,2GB,4GB or 1024,2048,4096)");
    println!("                               Default: 1GB,2GB,4GB,8GB,16GB,32GB");
    println!(
        "  -d, --delimiter <CHAR>       CSV delimiter character (default: |) - ignored for Parquet"
    );
    println!("  --headers                    CSV file has headers - ignored for Parquet");
    println!("  --help                       Show this help message");
    println!("\nColumn indices for TPC-H lineitem table:");
    for (i, name) in COLUMN_NAMES.iter().enumerate() {
        println!("  {}: {}", i, name);
    }
    println!("\nExamples:");
    println!("  # Sort by orderkey and linenumber (default)");
    println!("  {} lineitem.csv", env::args().next().unwrap());
    println!("  {} lineitem.parquet", env::args().next().unwrap());
    println!("\n  # Sort by shipdate");
    println!(
        "  {} -k 10 -v 0,3 lineitem.csv",
        env::args().next().unwrap()
    );
    println!("\n  # Sort by extended price with custom thread counts");
    println!(
        "  {} -k 5 -v 0,3,10 -t 1,4,8,16 lineitem.parquet",
        env::args().next().unwrap()
    );
    println!("\n  # Test with specific memory sizes");
    println!(
        "  {} -m 2GB,4GB,8GB lineitem.csv",
        env::args().next().unwrap()
    );
    println!("\n  # Custom thread and memory combinations");
    println!(
        "  {} -t 1,8,32 -m 1024,8192,32768 lineitem.parquet",
        env::args().next().unwrap()
    );
    process::exit(1);
}

fn print_first_lines(path: &str, num_lines: usize) -> Result<(), String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);

    for (i, line) in reader.lines().enumerate() {
        if i >= num_lines {
            break;
        }
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        println!("{}", line);
    }

    Ok(())
}

fn run_benchmark(config: Config) -> Result<(), String> {
    println!("TPC-H Lineitem Sort Benchmark");
    println!("=============================");

    if !Path::new(&config.file_path).exists() {
        return Err(format!("File {} not found", config.file_path));
    }

    // Determine file type
    let is_parquet = config.file_path.to_lowercase().ends_with(".parquet");
    let is_csv = config.file_path.to_lowercase().ends_with(".csv");

    if !is_parquet && !is_csv {
        return Err("File must have .csv or .parquet extension".to_string());
    }

    // File information
    let file_metadata = std::fs::metadata(&config.file_path)
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;
    let file_size = file_metadata.len();

    println!("\nFile Information:");
    println!("  Path: {}", config.file_path);
    println!("  Format: {}", if is_parquet { "Parquet" } else { "CSV" });
    println!(
        "  Size: {} GB ({} bytes)",
        bytes_to_human_readable(file_size as usize),
        file_size
    );

    // Print preview for CSV files
    if is_csv {
        println!("\nFirst few lines of input file:");
        println!("{}", "-".repeat(80));
        print_first_lines(&config.file_path, 5)?;
        println!("{}", "-".repeat(80));
    }

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
    if is_csv {
        println!(
            "  Delimiter: '{}' (ASCII {})",
            config.delimiter as char, config.delimiter
        );
        println!("  Has headers: {}", config.has_headers);
    }
    println!("  Direct I/O buffer: {} KB", DEFAULT_BUFFER_SIZE / 1024);
    println!("  Thread counts to test: {:?}", config.thread_counts);
    println!(
        "  Memory sizes to test: {}",
        config
            .memory_sizes
            .iter()
            .map(|&mb| bytes_to_human_readable(mb * 1024 * 1024))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Run all benchmarks
    let mut results = Vec::new();

    for &memory_mb in &config.memory_sizes {
        for &threads in &config.thread_counts {
            println!("\n{}", "=".repeat(70));
            println!(
                "Running benchmark: {} threads, {} memory",
                threads,
                bytes_to_human_readable(memory_mb * 1024 * 1024)
            );
            println!("{}", "=".repeat(70));

            match run_single_benchmark(&config, threads, memory_mb, is_parquet) {
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
    is_parquet: bool,
) -> Result<BenchmarkResult, String> {
    println!("Configuration:");
    println!("  Threads: {}", threads);
    println!(
        "  Total memory budget: {}",
        bytes_to_human_readable(memory_mb * 1024 * 1024)
    );
    println!(
        "  Buffer per thread: {}",
        bytes_to_human_readable(memory_mb * 1024 * 1024 / threads)
    );
    println!(
        "  Direct I/O buffer: {} KB",
        bytes_to_human_readable(DEFAULT_BUFFER_SIZE)
    );

    // Create input based on file type
    let input: Box<dyn SortInput> = if is_parquet {
        let mut parquet_config = ParquetDirectConfig::default();
        parquet_config.key_columns = config.key_columns.clone();
        parquet_config.value_columns = config.value_columns.clone();

        let parquet_input = ParquetInputDirect::new(&config.file_path, parquet_config)?;
        println!("  Total rows: {}", parquet_input.len());
        Box::new(parquet_input)
    } else {
        // Create TPC-H lineitem schema for CSV
        let schema = Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false), // 0
            Field::new("l_partkey", DataType::Int64, false),  // 1
            Field::new("l_suppkey", DataType::Int64, false),  // 2
            Field::new("l_linenumber", DataType::Int32, false), // 3
            Field::new("l_quantity", DataType::Float64, false), // 4
            Field::new("l_extendedprice", DataType::Float64, false), // 5
            Field::new("l_discount", DataType::Float64, false), // 6
            Field::new("l_tax", DataType::Float64, false),    // 7
            Field::new("l_returnflag", DataType::Utf8, false), // 8
            Field::new("l_linestatus", DataType::Utf8, false), // 9
            Field::new("l_shipdate", DataType::Date32, false), // 10
            Field::new("l_commitdate", DataType::Date32, false), // 11
            Field::new("l_receiptdate", DataType::Date32, false), // 12
            Field::new("l_shipinstruct", DataType::Utf8, false), // 13
            Field::new("l_shipmode", DataType::Utf8, false),  // 14
            Field::new("l_comment", DataType::Utf8, false),   // 15
        ]));

        let mut csv_config = CsvDirectConfig::new(schema.clone());
        csv_config.delimiter = config.delimiter;
        csv_config.key_columns = config.key_columns.clone();
        csv_config.value_columns = config.value_columns.clone();
        csv_config.has_headers = config.has_headers;

        Box::new(CsvInputDirect::new(&config.file_path, csv_config)?)
    };

    let mut sorter = ExternalSorter::new(threads, memory_mb * 1024 * 1024);

    println!("\nStarting sort...");
    let start = Instant::now();

    let output = sorter.sort(input)?;

    let sort_time = start.elapsed();
    println!("Sort completed in {:.2} seconds", sort_time.as_secs_f64());

    // Get sort statistics
    let stats = output.stats();
    println!("Sort statistics: {} runs generated", stats.num_runs);
    let mut run_sizes = String::new();
    for (i, run_info) in stats.runs_info.iter().enumerate() {
        if i > 5 {
            break;
        }
        run_sizes.push_str(&format!(
            "[#{}, {}], ",
            run_info.entries,
            bytes_to_human_readable(run_info.file_size as usize)
        ));
    }
    println!("Run sizes: {}...", run_sizes);

    // Count entries for verification (not included in performance metrics)
    println!("\nCounting entries for verification...");

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

    let total_entries = count as usize;
    println!("Verified {} entries", total_entries);

    println!("\nFirst 5 entries:");
    for (i, (key, value)) in first_entries.iter().enumerate() {
        print_entry(
            i,
            key,
            value,
            &config.key_columns,
            &config.value_columns,
            is_parquet,
        );
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
            is_parquet,
        );
    }

    // Extract detailed timing stats
    let run_gen_time = Duration::from_millis(stats.run_generation_time_ms.unwrap_or(0) as u64);
    let merge_time = Duration::from_millis(stats.merge_time_ms.unwrap_or(0) as u64);

    // Extract detailed IO stats
    let (run_gen_read_ops, run_gen_read_bytes, run_gen_write_ops, run_gen_write_bytes) =
        if let Some(io_stats) = &stats.run_generation_io_stats {
            (
                io_stats.read_ops,
                io_stats.read_bytes,
                io_stats.write_ops,
                io_stats.write_bytes,
            )
        } else {
            (0, 0, 0, 0)
        };

    let (merge_read_ops, merge_read_bytes, merge_write_ops, merge_write_bytes) =
        if let Some(io_stats) = &stats.merge_io_stats {
            (
                io_stats.read_ops,
                io_stats.read_bytes,
                io_stats.write_ops,
                io_stats.write_bytes,
            )
        } else {
            (0, 0, 0, 0)
        };

    // Calculate totals
    let total_read_ops = run_gen_read_ops + merge_read_ops;
    let total_read_bytes = run_gen_read_bytes + merge_read_bytes;
    let total_write_ops = run_gen_write_ops + merge_write_ops;
    let total_write_bytes = run_gen_write_bytes + merge_write_bytes;

    Ok(BenchmarkResult {
        threads,
        memory_mb,
        sort_time,
        run_gen_time,
        merge_time,
        total_entries,
        throughput_meps: total_entries as f64 / sort_time.as_secs_f64() / 1_000_000.0,
        num_runs: stats.num_runs,
        read_ops: total_read_ops,
        read_bytes: total_read_bytes,
        write_ops: total_write_ops,
        write_bytes: total_write_bytes,
        run_gen_read_ops,
        run_gen_read_bytes,
        run_gen_write_ops,
        run_gen_write_bytes,
        merge_read_ops,
        merge_read_bytes,
        merge_write_ops,
        merge_write_bytes,
    })
}

fn display_results_table(results: &[BenchmarkResult]) {
    if results.is_empty() {
        println!("\nNo results to display");
        return;
    }

    println!("\n{}", "=".repeat(120));
    println!("Benchmark Results Summary");
    println!("{}", "=".repeat(120));

    // Header
    println!(
        "{:<8} {:<10} {:<6} {:<12} {:<12} {:<12} {:<12} {:<15} {:<10} {:<10}",
        "Threads",
        "Memory",
        "Runs",
        "Total (s)",
        "RunGen (s)",
        "Merge (s)",
        "Entries",
        "Throughput",
        "Read MB",
        "Write MB"
    );
    println!(
        "{:<8} {:<10} {:<6} {:<12} {:<12} {:<12} {:<12} {:<15} {:<10} {:<10}",
        "", "", "", "", "", "", "", "(M entries/s)", "", ""
    );
    println!("{}", "-".repeat(120));

    // Results rows
    for result in results {
        println!(
            "{:<8} {:<10} {:<6} {:<12.2} {:<12.2} {:<12.2} {:<12} {:<15.2} {:<10.1} {:<10.1}",
            result.threads,
            bytes_to_human_readable(result.memory_mb * 1024 * 1024),
            result.num_runs,
            result.sort_time.as_secs_f64(),
            result.run_gen_time.as_secs_f64(),
            result.merge_time.as_secs_f64(),
            result.total_entries,
            result.throughput_meps,
            result.read_bytes as f64 / (1024.0 * 1024.0),
            result.write_bytes as f64 / (1024.0 * 1024.0)
        );
    }

    println!("{}", "=".repeat(120));

    // Display detailed IO statistics summary
    println!("\nDetailed I/O Statistics Summary:");
    println!("{}", "-".repeat(100));
    println!(
        "{:<8} {:<10} {:<20} {:<20} {:<20} {:<20}",
        "Threads", "Memory", "Run Gen Reads", "Run Gen Writes", "Merge Reads", "Merge Writes"
    );
    println!(
        "{:<8} {:<10} {:<20} {:<20} {:<20} {:<20}",
        "", "", "(ops / MB)", "(ops / MB)", "(ops / MB)", "(ops / MB)"
    );
    println!("{}", "-".repeat(100));

    for result in results {
        println!(
            "{:<8} {:<10} {:<20} {:<20} {:<20} {:<20}",
            result.threads,
            bytes_to_human_readable(result.memory_mb * 1024 * 1024),
            format!(
                "{} / {:.1}",
                result.run_gen_read_ops,
                result.run_gen_read_bytes as f64 / (1024.0 * 1024.0)
            ),
            format!(
                "{} / {:.1}",
                result.run_gen_write_ops,
                result.run_gen_write_bytes as f64 / (1024.0 * 1024.0)
            ),
            format!(
                "{} / {:.1}",
                result.merge_read_ops,
                result.merge_read_bytes as f64 / (1024.0 * 1024.0)
            ),
            format!(
                "{} / {:.1}",
                result.merge_write_ops,
                result.merge_write_bytes as f64 / (1024.0 * 1024.0)
            )
        );
    }
    println!("{}", "-".repeat(100));

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
            println!(
                "\nSpeedup Analysis for {}:",
                bytes_to_human_readable(mem_size * 1024 * 1024)
            );
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
}

fn print_entry(
    index: usize,
    key: &[u8],
    value: &[u8],
    key_cols: &[usize],
    value_cols: &[usize],
    is_parquet: bool,
) {
    print!("  [{}] ", index);

    // Decode and print key columns
    let mut offset = 0;
    for (i, &col_idx) in key_cols.iter().enumerate() {
        if offset >= key.len() {
            break;
        }

        // Get the expected size and type for this column
        let (field_type, expected_size) = if is_parquet {
            // For Parquet, we need to know the data types to decode properly
            match col_idx {
                0..=2 => ("Int64", 8),       // orderkey, partkey, suppkey
                3 => ("Int32", 4),               // linenumber
                4..=7 => ("Float64", 8), // quantity, extendedprice, discount, tax
                10..=12 => ("Date32", 4),   // shipdate, commitdate, receiptdate
                _ => ("Utf8", 0),                // Variable length strings
            }
        } else {
            // For CSV, schema is defined in run_single_benchmark
            match col_idx {
                0..=2 => ("Int64", 8),
                3 => ("Int32", 4),
                4..=7 => ("Float64", 8),
                10..=12 => ("Date32", 4),
                _ => ("Utf8", 0),
            }
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

        let (field_type, expected_size) = if is_parquet {
            match col_idx {
                0..=2 => ("Int64", 8),
                3 => ("Int32", 4),
                4..=7 => ("Float64", 8),
                10..=12 => ("Date32", 4),
                _ => ("Utf8", 0),
            }
        } else {
            match col_idx {
                0..=2 => ("Int64", 8),
                3 => ("Int32", 4),
                4..=7 => ("Float64", 8),
                10..=12 => ("Date32", 4),
                _ => ("Utf8", 0),
            }
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
