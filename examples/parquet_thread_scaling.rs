//! Benchmark Parquet Direct I/O performance with different thread counts
//!
//! This example tests how performance scales with thread count when using Direct I/O
//!
//! Usage: parquet_thread_scaling [OPTIONS] <PARQUET_FILE>
//!
//! Options:
//!   -k, --key-columns <COLS>     Comma-separated list of key column indices (default: 0)
//!   -v, --value-columns <COLS>   Comma-separated list of value column indices (default: 1)
//!   -t, --threads <LIST>         Comma-separated list of thread counts to test (default: 1,2,4,8,16)
//!   -b, --buffer-size <KB>       Buffer size in KB for direct I/O (default: 256)
//!   -h, --help                   Show this help message

use crossbeam::channel;
use es::{IoStatsTracker, ParquetDirectConfig, ParquetInputDirect, SortInput};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::env;
use std::fs::File;
use std::process;
use std::time::Instant;

#[derive(Clone)]
struct Config {
    parquet_file: String,
    key_columns: Vec<usize>,
    value_columns: Vec<usize>,
    thread_counts: Vec<usize>,
    buffer_size_kb: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            parquet_file: String::new(),
            key_columns: vec![0],
            value_columns: vec![1],
            thread_counts: vec![1, 2, 4, 8, 16],
            buffer_size_kb: 256,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("\nUse --help for usage information");
            process::exit(1);
        }
    };

    let filename = &config.parquet_file;
    let buffer_size = config.buffer_size_kb * 1024;

    // First, get some info about the Parquet file
    let file = File::open(filename)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let metadata = builder.metadata();
    let num_rows = metadata.file_metadata().num_rows();
    let num_row_groups = metadata.num_row_groups();
    let schema = builder.schema();

    println!("Parquet file: {}", filename);
    println!("  - Total rows: {}", num_rows);
    println!("  - Row groups: {}", num_row_groups);
    println!("  - Schema:");
    for (i, field) in schema.fields().iter().enumerate() {
        println!("    [{}] {}: {:?}", i, field.name(), field.data_type());
    }

    // Validate column indices
    for &col in &config.key_columns {
        if col >= schema.fields().len() {
            eprintln!(
                "Error: key_column {} is out of bounds (file has {} columns)",
                col,
                schema.fields().len()
            );
            process::exit(1);
        }
    }
    for &col in &config.value_columns {
        if col >= schema.fields().len() {
            eprintln!(
                "Error: value_column {} is out of bounds (file has {} columns)",
                col,
                schema.fields().len()
            );
            process::exit(1);
        }
    }

    println!("\nConfiguration:");
    println!("Key columns: {:?}", config.key_columns);
    println!("Value columns: {:?}", config.value_columns);
    println!("Thread counts: {:?}", config.thread_counts);
    println!("Direct I/O buffer: {} KB", config.buffer_size_kb);
    println!();

    // Print first few rows as preview (using first key and value columns)
    if !config.key_columns.is_empty() && !config.value_columns.is_empty() {
        print_preview(filename, config.key_columns[0], config.value_columns[0])?;
    }

    println!("Direct I/O Parquet Input Scaling:");
    println!(
        "{:<10} {:>12} {:>12} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "Threads", "Time (s)", "Rows/s (M)", "MB/s", "Speedup", "Total Rows", "IO MB/s", "IOPS"
    );
    println!("{}", "-".repeat(100));

    let mut baseline_time = 0.0;
    let mut total_rows_for_check = 0;

    for &num_threads in &config.thread_counts {
        let parquet_config = ParquetDirectConfig {
            key_columns: config.key_columns.clone(),
            value_columns: config.value_columns.clone(),
            buffer_size,
        };

        let parquet_direct = ParquetInputDirect::new(filename, parquet_config)?;

        // Create I/O tracker
        let io_tracker = IoStatsTracker::new();

        let (elapsed, total_rows, total_bytes, _actual_partitions) =
            benchmark_thread_scaling(parquet_direct, num_threads, &io_tracker);

        let duration = elapsed.as_secs_f64();
        let rows_per_sec_millions = (total_rows as f64 / elapsed.as_secs_f64()) / 1_000_000.0;
        let mb_per_sec = (total_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();

        // Get I/O statistics
        let (io_ops, io_bytes) = io_tracker.get_read_stats();
        let io_mb_per_sec = (io_bytes as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();
        let iops = io_ops as f64 / elapsed.as_secs_f64();

        // Calculate speedup
        if baseline_time == 0.0 {
            baseline_time = duration;
            total_rows_for_check = total_rows;
        }
        let speedup = baseline_time / duration;

        println!(
            "{:<10} {:>12.2} {:>12.2} {:>10.2} {:>10.2}x {:>12} {:>10.2} {:>10.0}",
            num_threads,
            duration,
            rows_per_sec_millions,
            mb_per_sec,
            speedup,
            total_rows,
            io_mb_per_sec,
            iops
        );

        // Check that all runs process the same number of rows
        if total_rows != total_rows_for_check {
            eprintln!(
                "WARNING: Row count mismatch! Expected {} but got {}",
                total_rows_for_check, total_rows
            );
        }
    }
    println!();

    // Show I/O efficiency summary
    println!("I/O Efficiency Analysis:");
    println!("- Direct I/O buffer size: {} KB", config.buffer_size_kb);
    println!("- Note: IO MB/s > MB/s due to Direct I/O page alignment (4KB boundaries)");

    Ok(())
}

fn benchmark_thread_scaling(
    parquet_direct: ParquetInputDirect,
    num_threads: usize,
    io_tracker: &IoStatsTracker,
) -> (std::time::Duration, usize, usize, usize) {
    let start = Instant::now();

    let partitions = parquet_direct.create_parallel_scanners(num_threads, Some(io_tracker.clone()));
    let actual_partitions = partitions.len();

    let (sender, receiver) = channel::unbounded();

    std::thread::scope(|s| {
        for partition in partitions {
            let sender = sender.clone();
            s.spawn(move || {
                let mut row_count = 0;
                let mut byte_count = 0;

                for (key, value) in partition {
                    row_count += 1;
                    byte_count += key.len() + value.len();
                }

                sender.send((row_count, byte_count)).unwrap();
            });
        }
        drop(sender);
    });

    let mut total_rows = 0;
    let mut total_bytes = 0;
    while let Ok((rows, bytes)) = receiver.recv() {
        total_rows += rows;
        total_bytes += bytes;
    }

    let elapsed = start.elapsed();
    (elapsed, total_rows, total_bytes, actual_partitions)
}

fn print_preview(
    filename: &str,
    key_column: usize,
    value_column: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Preview of first 3 rows:");
    println!("{}", "-".repeat(50));

    let file = File::open(filename)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.build()?;

    let mut count = 0;
    if let Some(batch) = reader.next() {
        let batch = batch?;
        let key_array = batch.column(key_column);
        let value_array = batch.column(value_column);

        for i in 0..3.min(batch.num_rows()) {
            let key = format_value(key_array, i);
            let value = format_value(value_array, i);
            println!("Row {}: key='{}', value='{}'", count + 1, key, value);
            count += 1;
        }
    }

    println!("{}", "-".repeat(50));
    println!();
    Ok(())
}

fn format_value(array: &dyn arrow::array::Array, row: usize) -> String {
    use arrow::array::AsArray;
    use arrow::datatypes::DataType;

    match array.data_type() {
        DataType::Utf8 => {
            let string_array = array.as_string::<i32>();
            string_array.value(row).to_string()
        }
        DataType::Int32 => {
            let int_array = array.as_primitive::<arrow::datatypes::Int32Type>();
            int_array.value(row).to_string()
        }
        DataType::Int64 => {
            let int_array = array.as_primitive::<arrow::datatypes::Int64Type>();
            int_array.value(row).to_string()
        }
        DataType::Decimal128(_, scale) => {
            let decimal_array = array.as_primitive::<arrow::datatypes::Decimal128Type>();
            let value = decimal_array.value(row);
            let scale_factor = 10_i128.pow(*scale as u32);
            format!(
                "{}.{:02}",
                value / scale_factor,
                (value % scale_factor).abs()
            )
        }
        DataType::Date32 => {
            let date_array = array.as_primitive::<arrow::datatypes::Date32Type>();
            let days_since_epoch = date_array.value(row);
            // Simple date display: just show days since epoch
            format!("Date32({})", days_since_epoch)
        }
        _ => format!("{:?}", array.data_type()),
    }
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = env::args().collect();
    let mut config = Config::default();

    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage(&args[0]);
                process::exit(0);
            }
            "-k" | "--key-columns" => {
                if i + 1 >= args.len() {
                    return Err("Missing value for key-columns".to_string());
                }
                i += 1;
                config.key_columns = parse_column_list(&args[i])?;
            }
            "-v" | "--value-columns" => {
                if i + 1 >= args.len() {
                    return Err("Missing value for value-columns".to_string());
                }
                i += 1;
                config.value_columns = parse_column_list(&args[i])?;
            }
            "-t" | "--threads" => {
                if i + 1 >= args.len() {
                    return Err("Missing value for threads".to_string());
                }
                i += 1;
                config.thread_counts = parse_thread_list(&args[i])?;
            }
            "-b" | "--buffer-size" => {
                if i + 1 >= args.len() {
                    return Err("Missing value for buffer-size".to_string());
                }
                i += 1;
                config.buffer_size_kb = args[i]
                    .parse()
                    .map_err(|_| format!("Invalid buffer size: {}", args[i]))?;
            }
            _ => {
                if args[i].starts_with('-') {
                    return Err(format!("Unknown option: {}", args[i]));
                }
                if !config.parquet_file.is_empty() {
                    return Err("Multiple input files specified".to_string());
                }
                config.parquet_file = args[i].clone();
            }
        }
        i += 1;
    }

    if config.parquet_file.is_empty() {
        return Err("No input file specified".to_string());
    }

    if config.key_columns.is_empty() {
        return Err("At least one key column must be specified".to_string());
    }

    if config.value_columns.is_empty() {
        return Err("At least one value column must be specified".to_string());
    }

    if config.thread_counts.is_empty() {
        return Err("At least one thread count must be specified".to_string());
    }

    Ok(config)
}

fn print_usage(program_name: &str) {
    println!("Usage: {} [OPTIONS] <PARQUET_FILE>", program_name);
    println!();
    println!("Options:");
    println!(
        "  -k, --key-columns <COLS>     Comma-separated list of key column indices (default: 0)"
    );
    println!(
        "  -v, --value-columns <COLS>   Comma-separated list of value column indices (default: 1)"
    );
    println!("  -t, --threads <LIST>         Comma-separated list of thread counts to test (default: 1,2,4,8,16)");
    println!("  -b, --buffer-size <KB>       Buffer size in KB for direct I/O (default: 256)");
    println!("  -h, --help                   Show this help message");
    println!();
    println!("Example:");
    println!(
        "  {} -k 0,3 -v 5,10 -t 1,4,8 lineitem.parquet",
        program_name
    );
}

fn parse_column_list(s: &str) -> Result<Vec<usize>, String> {
    let cols: Result<Vec<usize>, _> = s
        .split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<usize>())
        .collect();

    cols.map_err(|_| format!("Invalid column list: {}", s))
}

fn parse_thread_list(s: &str) -> Result<Vec<usize>, String> {
    let threads: Result<Vec<usize>, _> = s
        .split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<usize>())
        .collect();

    match threads {
        Ok(mut t) => {
            t.sort_unstable();
            t.dedup();
            if t.iter().any(|&x| x == 0) {
                Err("Thread count must be greater than 0".to_string())
            } else {
                Ok(t)
            }
        }
        Err(_) => Err(format!("Invalid thread list: {}", s)),
    }
}
