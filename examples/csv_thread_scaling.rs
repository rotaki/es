//! Benchmark to demonstrate CSV reading scalability with multiple threads
//! Using Direct I/O implementation
//!
//! Usage: csv_thread_scaling [OPTIONS] <CSV_FILE>
//!
//! Options:
//!   -k, --key-columns <COLS>     Comma-separated list of key column indices (default: 0)
//!   -v, --value-columns <COLS>   Comma-separated list of value column indices (default: 1)
//!   -t, --threads <LIST>         Comma-separated list of thread counts to test (default: 1,2,4,8,16)
//!   -d, --delimiter <CHAR>       CSV delimiter character (default: ,)
//!   -b, --buffer-size <KB>       Buffer size in KB for direct I/O (default: 256)
//!   --headers                    CSV file has headers
//!   --help                       Show this help message

use arrow::datatypes::{DataType, Field, Schema};
use es::{CsvDirectConfig, CsvInputDirect, IoStatsTracker, SortInput};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[derive(Clone)]
struct Config {
    csv_file: String,
    key_columns: Vec<usize>,
    value_columns: Vec<usize>,
    thread_counts: Vec<usize>,
    delimiter: u8,
    buffer_size_kb: usize,
    has_headers: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            csv_file: String::new(),
            key_columns: vec![0],
            value_columns: vec![1],
            thread_counts: vec![1, 2, 4, 8, 16],
            delimiter: b',',
            buffer_size_kb: 256,
            has_headers: false,
        }
    }
}

fn main() -> Result<(), String> {
    let config = parse_args()?;

    println!("CSV Thread Scaling Benchmark");
    println!("============================\n");

    // Check if file exists
    if !std::path::Path::new(&config.csv_file).exists() {
        return Err(format!("File does not exist: {}", config.csv_file));
    }

    let file_size = std::fs::metadata(&config.csv_file)
        .map_err(|e| format!("Failed to get metadata: {}", e))?
        .len();
    println!("Input file: {}", config.csv_file);
    println!("File size: {:.2} MB", file_size as f64 / (1024.0 * 1024.0));

    // Print first two lines to help identify delimiter and headers
    print_first_lines(&config.csv_file, 2)?;

    println!("Key columns: {:?}", config.key_columns);
    println!("Value columns: {:?}", config.value_columns);
    println!("Thread counts: {:?}", config.thread_counts);
    println!("Delimiter: '{}'", config.delimiter as char);
    println!("Direct I/O buffer: {} KB", config.buffer_size_kb);
    println!();

    // Create a generic schema for the CSV file
    // Since we don't know the exact schema, we'll create one based on the file's structure
    let schema = create_generic_schema(&config)?;

    // Direct I/O CSV configuration
    let mut direct_config = CsvDirectConfig::new(schema);
    direct_config.delimiter = config.delimiter;
    direct_config.key_columns = config.key_columns.clone();
    direct_config.value_columns = config.value_columns.clone();
    direct_config.has_headers = config.has_headers;
    direct_config.buffer_size = config.buffer_size_kb * 1024;

    // Run scaling tests
    println!("Direct I/O CSV Input Scaling:");
    println!(
        "{:<10} {:>12} {:>12} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "Threads", "Time (s)", "Rows/s (M)", "MB/s", "Speedup", "Total Rows", "IO MB/s", "IOPS"
    );
    println!("{}", "-".repeat(100));

    let mut baseline_time = 0.0;
    let mut total_rows_for_check = 0;
    for &num_threads in &config.thread_counts {
        let csv_direct =
            CsvInputDirect::new(&config.csv_file, direct_config.clone())?.with_io_stats(); // Enable I/O tracking

        let io_tracker = csv_direct.io_stats.clone();
        let (duration, throughput_rows, throughput_mb, row_count, io_mb_s, iops) =
            benchmark_parallel_read(Box::new(csv_direct), num_threads, io_tracker.as_ref())?;

        if baseline_time == 0.0 {
            baseline_time = duration;
            total_rows_for_check = row_count;
        }
        let speedup = baseline_time / duration;

        println!(
            "{:<10} {:>12.2} {:>12.2} {:>10.2} {:>10.2}x {:>12} {:>10.2} {:>10.0}",
            num_threads,
            duration,
            throughput_rows,
            throughput_mb,
            speedup,
            row_count,
            io_mb_s,
            iops
        );

        // Check that all runs process the same number of rows
        if row_count != total_rows_for_check {
            eprintln!(
                "WARNING: Row count mismatch! Expected {} but got {}",
                total_rows_for_check, row_count
            );
        }
    }
    println!();

    // Show I/O efficiency summary
    println!("I/O Efficiency Analysis:");
    println!("- Direct I/O buffer size: {} KB", config.buffer_size_kb);
    println!("- Note: IO MB/s > MB/s due to Direct I/O page alignment (4KB boundaries)");

    // // Detailed thread efficiency analysis (only for thread counts <= 8)
    // let analysis_threads: Vec<usize> = config.thread_counts.iter()
    //     .filter(|&&t| t <= 8)
    //     .copied()
    //     .collect();

    // if !analysis_threads.is_empty() {
    //     println!("Detailed Thread Efficiency Analysis:");
    //     println!("{}", "=".repeat(75));
    //
    //     println!("\nDirect I/O - Per-partition statistics:");
    //     println!("{:<10} {:>20} {:>20} {:>20}",
    //         "Threads", "Avg rows/partition", "Std dev", "Load imbalance %");
    //     println!("{}", "-".repeat(75));
    //
    //     for &num_threads in &analysis_threads {
    //         let input = Box::new(CsvInputDirect::new(&config.csv_file, direct_config.clone())?);
    //         let stats = analyze_partition_balance(input, num_threads)?;
    //         println!("{:<10} {:>20.0} {:>20.0} {:>20.1}%",
    //             num_threads, stats.avg, stats.std_dev, stats.imbalance_pct);
    //     }
    //     println!();
    // }

    // // CPU utilization test (only if 8 is in thread counts)
    // if config.thread_counts.contains(&8) {
    //     println!("CPU Utilization Test (8 threads):");
    //     println!("{}", "=".repeat(75));
    //
    //     let csv_direct = CsvInputDirect::new(&config.csv_file, direct_config.clone())?;
    //     let (cpu_stats_direct, _) = benchmark_with_cpu_monitoring(Box::new(csv_direct), 8)?;
    //     println!("Direct I/O: {:.1}% CPU utilization", cpu_stats_direct);
    //     println!();
    // }

    Ok(())
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
            "-d" | "--delimiter" => {
                if i + 1 >= args.len() {
                    return Err("Missing value for delimiter".to_string());
                }
                i += 1;
                config.delimiter = parse_delimiter(&args[i])?;
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
            "--headers" => {
                config.has_headers = true;
            }
            _ => {
                if args[i].starts_with('-') {
                    return Err(format!("Unknown option: {}", args[i]));
                }
                if !config.csv_file.is_empty() {
                    return Err("Multiple input files specified".to_string());
                }
                config.csv_file = args[i].clone();
            }
        }
        i += 1;
    }

    if config.csv_file.is_empty() {
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
    println!("Usage: {} [OPTIONS] <CSV_FILE>", program_name);
    println!();
    println!("Options:");
    println!(
        "  -k, --key-columns <COLS>     Comma-separated list of key column indices (default: 0)"
    );
    println!(
        "  -v, --value-columns <COLS>   Comma-separated list of value column indices (default: 1)"
    );
    println!("  -t, --threads <LIST>         Comma-separated list of thread counts to test (default: 1,2,4,8,16)");
    println!("  -d, --delimiter <CHAR>       CSV delimiter character (default: ,)");
    println!("  -b, --buffer-size <KB>       Buffer size in KB for direct I/O (default: 256)");
    println!("  --headers                    CSV file has headers");
    println!("  -h, --help                   Show this help message");
    println!();
    println!("Example:");
    println!(
        "  {} -k 0,3 -v 5,10 -t 1,4,8 --delimiter '|' lineitem.tbl",
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

fn print_first_lines(filename: &str, num_lines: usize) -> Result<(), String> {
    println!("\nFirst {} lines of the CSV file:", num_lines);
    println!("{}", "-".repeat(50));

    let file = File::open(filename).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);

    let mut count = 0;
    for line in reader.lines() {
        if count >= num_lines {
            break;
        }

        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;

        // Show the line with visible delimiters
        println!("Line {}: {}", count + 1, line);

        // Also show common delimiters found in the line
        if count == 0 {
            let mut delimiters = Vec::new();
            if line.contains(',') {
                delimiters.push("comma (,)");
            }
            if line.contains('|') {
                delimiters.push("pipe (|)");
            }
            if line.contains('\t') {
                delimiters.push("tab (\\t)");
            }
            if line.contains(';') {
                delimiters.push("semicolon (;)");
            }

            if !delimiters.is_empty() {
                println!("  → Detected delimiters: {}", delimiters.join(", "));
            }
        }

        count += 1;
    }

    println!("{}\n", "-".repeat(50));
    Ok(())
}

fn parse_delimiter(s: &str) -> Result<u8, String> {
    if s.len() == 1 {
        Ok(s.as_bytes()[0])
    } else if s == "\\t" || s == "tab" {
        Ok(b'\t')
    } else if s == "pipe" {
        Ok(b'|')
    } else if s == "comma" {
        Ok(b',')
    } else {
        Err(format!(
            "Invalid delimiter: '{}' (must be a single character)",
            s
        ))
    }
}

fn benchmark_parallel_read(
    input: Box<dyn SortInput>,
    num_threads: usize,
    io_tracker: Option<&IoStatsTracker>,
) -> Result<(f64, f64, f64, usize, f64, f64), String> {
    // Capture initial I/O stats if tracker provided
    let io_before = io_tracker.map(|t| t.get_read_stats()).unwrap_or((0, 0));

    let start = Instant::now();

    let partitions = input.partition(num_threads);
    let total_rows = Arc::new(AtomicUsize::new(0));
    let total_bytes = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = partitions
        .into_iter()
        .map(|partition| {
            let rows_counter = Arc::clone(&total_rows);
            let bytes_counter = Arc::clone(&total_bytes);

            thread::spawn(move || {
                // let thread_id = thread::current().id();
                // println!("[DEBUG] Thread {:?} started", thread_id);
                let mut local_rows = 0;
                let mut local_bytes = 0;

                for (key, value) in partition {
                    local_rows += 1;
                    local_bytes += key.len() + value.len();

                    // if local_rows % 100000 == 0 {
                    //     println!("[DEBUG] Thread {:?} processed {} rows", thread_id, local_rows);
                    // }
                }

                // println!("[DEBUG] Thread {:?} finished with {} rows", thread_id, local_rows);
                rows_counter.fetch_add(local_rows, Ordering::Relaxed);
                bytes_counter.fetch_add(local_bytes, Ordering::Relaxed);
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start.elapsed().as_secs_f64();
    let rows = total_rows.load(Ordering::Relaxed);
    let bytes = total_bytes.load(Ordering::Relaxed);

    // Capture final I/O stats if tracker provided
    let (io_ops_final, io_bytes_final) = io_tracker.map(|t| t.get_read_stats()).unwrap_or((0, 0));
    let io_ops = io_ops_final.saturating_sub(io_before.0);
    let io_bytes = io_bytes_final.saturating_sub(io_before.1);

    let throughput_rows = (rows as f64 / 1_000_000.0) / duration;
    let throughput_mb = (bytes as f64 / (1024.0 * 1024.0)) / duration;
    let io_throughput_mb = (io_bytes as f64 / (1024.0 * 1024.0)) / duration;
    let iops = io_ops as f64 / duration;

    Ok((
        duration,
        throughput_rows,
        throughput_mb,
        rows,
        io_throughput_mb,
        iops,
    ))
}

/*
struct PartitionStats {
    avg: f64,
    std_dev: f64,
    imbalance_pct: f64,
}

fn analyze_partition_balance(
    input: Box<dyn SortInput>,
    num_threads: usize
) -> Result<PartitionStats, String> {
    let partitions = input.partition(num_threads);
    let partition_sizes = Arc::new(std::sync::Mutex::new(Vec::new()));

    let handles: Vec<_> = partitions
        .into_iter()
        .map(|partition| {
            let sizes = Arc::clone(&partition_sizes);

            thread::spawn(move || {
                let count = partition.count();
                sizes.lock().unwrap().push(count);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let sizes = partition_sizes.lock().unwrap();
    let total: usize = sizes.iter().sum();
    let avg = total as f64 / sizes.len() as f64;

    // Calculate standard deviation
    let variance = sizes.iter()
        .map(|&size| {
            let diff = size as f64 - avg;
            diff * diff
        })
        .sum::<f64>() / sizes.len() as f64;

    let std_dev = variance.sqrt();

    // Calculate load imbalance
    let max_size = *sizes.iter().max().unwrap_or(&0) as f64;
    let imbalance_pct = if avg > 0.0 {
        ((max_size - avg) / avg) * 100.0
    } else {
        0.0
    };

    Ok(PartitionStats { avg, std_dev, imbalance_pct })
}

fn benchmark_with_cpu_monitoring(
    input: Box<dyn SortInput>,
    num_threads: usize
) -> Result<(f64, f64), String> {
    // Simple CPU monitoring - in production you'd use proper CPU metrics
    let start = Instant::now();
    let start_cpu = std::time::SystemTime::now();

    let partitions = input.partition(num_threads);
    let handles: Vec<_> = partitions
        .into_iter()
        .map(|partition| {
            thread::spawn(move || {
                let mut count = 0;
                for _ in partition {
                    count += 1;
                }
                count
            })
        })
        .collect();

    let mut total = 0;
    for handle in handles {
        total += handle.join().unwrap();
    }

    let wall_time = start.elapsed().as_secs_f64();
    let cpu_time = start_cpu.elapsed().unwrap().as_secs_f64();

    // Approximate CPU utilization
    let cpu_utilization = (cpu_time / wall_time) * 100.0 / num_threads as f64;

    Ok((cpu_utilization.min(100.0), total as f64))
}
*/

fn create_generic_schema(config: &Config) -> Result<Arc<Schema>, String> {
    // Try to read the first line to determine number of columns
    let file = File::open(&config.csv_file).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);

    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .map_err(|e| format!("Failed to read first line: {}", e))?;

    if config.has_headers {
        // Skip header line and read first data line
        first_line.clear();
        reader
            .read_line(&mut first_line)
            .map_err(|e| format!("Failed to read second line: {}", e))?;
    }

    // Count fields
    let fields_count = first_line.trim().split(config.delimiter as char).count();

    if fields_count == 0 {
        return Err("No fields found in CSV file".to_string());
    }

    // Validate column indices
    for &col in &config.key_columns {
        if col >= fields_count {
            return Err(format!(
                "Key column index {} is out of bounds (file has {} columns)",
                col, fields_count
            ));
        }
    }
    for &col in &config.value_columns {
        if col >= fields_count {
            return Err(format!(
                "Value column index {} is out of bounds (file has {} columns)",
                col, fields_count
            ));
        }
    }

    // Create a simple schema with all columns as strings
    let fields: Vec<Field> = (0..fields_count)
        .map(|i| Field::new(&format!("col_{}", i), DataType::Utf8, false))
        .collect();

    Ok(Arc::new(Schema::new(fields)))
}
