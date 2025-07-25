use arrow::datatypes::{DataType, Field, Schema};
use clap::{Parser, Subcommand, ValueEnum};
use es::{
    order_preserving_encoding::decode_bytes, CsvDirectConfig, CsvInputDirect, ExternalSorter,
    GenSortInputDirect, SortInput, Sorter,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "unified-sort")]
#[command(about = "Unified external sorter for various file formats", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Sort CSV files (lineitem or yellow trip data)
    Csv {
        /// Input CSV file path
        #[arg(short, long)]
        input: PathBuf,

        /// CSV schema type
        #[arg(short, long, value_enum)]
        schema: CsvSchema,

        /// Number of threads for run generation
        #[arg(short = 'r', long, default_value = "4")]
        run_gen_threads: usize,

        /// Number of threads for merge phase
        #[arg(short = 'g', long, default_value = "4")]
        merge_threads: usize,

        /// Maximum memory usage in MB
        #[arg(short, long, default_value = "1024")]
        memory_mb: usize,

        /// CSV delimiter
        #[arg(short, long, default_value = ",")]
        delimiter: char,

        /// Key columns for sorting (comma-separated column indices, e.g., 0,3)
        #[arg(short = 'k', long, value_delimiter = ',')]
        key_columns: Option<Vec<usize>>,

        /// Payload columns (comma-separated column indices, e.g., 1,2,4)
        #[arg(short = 'p', long, value_delimiter = ',')]
        payload_columns: Option<Vec<usize>>,

        /// Directory for temporary files
        #[arg(short = 't', long, default_value = ".")]
        temp_dir: PathBuf,

        /// Number of benchmark runs per configuration
        #[arg(long, default_value = "1")]
        benchmark_runs: usize,

        /// Comma-separated thread counts to test (e.g., 1,2,4,8)
        #[arg(long, value_delimiter = ',')]
        thread_counts: Option<Vec<usize>>,

        /// Comma-separated memory sizes to test (e.g., 512MB,1GB,2GB or 512,1024,2048)
        #[arg(long, value_delimiter = ',')]
        memory_sizes: Option<Vec<String>>,
    },

    /// Sort GenSort binary files
    Gensort {
        /// Input GenSort file path
        #[arg(short, long)]
        input: PathBuf,

        /// Number of threads for run generation
        #[arg(short = 'r', long, default_value = "4")]
        run_gen_threads: usize,

        /// Number of threads for merge phase
        #[arg(short = 'g', long, default_value = "4")]
        merge_threads: usize,

        /// Maximum memory usage in MB
        #[arg(short, long, default_value = "1024")]
        memory_mb: usize,

        /// Directory for temporary files
        #[arg(short = 't', long, default_value = ".")]
        temp_dir: PathBuf,

        /// Verify sorted output
        #[arg(short, long)]
        verify: bool,

        /// Number of benchmark runs per configuration
        #[arg(long, default_value = "1")]
        benchmark_runs: usize,

        /// Comma-separated thread counts to test (e.g., 1,2,4,8)
        #[arg(long, value_delimiter = ',')]
        thread_counts: Option<Vec<usize>>,

        /// Comma-separated memory sizes to test (e.g., 512MB,1GB,2GB or 512,1024,2048)
        #[arg(long, value_delimiter = ',')]
        memory_sizes: Option<Vec<String>>,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CsvSchema {
    /// TPC-H lineitem table schema
    Lineitem,
    /// NYC yellow trip data schema
    YellowTrip,
}

// Benchmark-related structs
#[derive(Default)]
struct RunStats {
    total_time: f64,
    run_gen_time: f64,
    merge_time: f64,
    runs_count: usize,
    run_gen_read_ops: u64,
    run_gen_read_mb: f64,
    run_gen_write_ops: u64,
    run_gen_write_mb: f64,
    merge_read_ops: u64,
    merge_read_mb: f64,
    merge_write_ops: u64,
    merge_write_mb: f64,
}

#[derive(Clone)]
struct BenchmarkResult {
    threads: usize,
    memory_mb: usize,
    memory_str: String,
    runs: usize,
    total_time: f64,
    run_gen_time: f64,
    merge_time: f64,
    entries: usize,
    throughput: f64,
    read_mb: f64,
    write_mb: f64,
    run_gen_read_ops: u64,
    run_gen_read_mb: f64,
    run_gen_write_ops: u64,
    run_gen_write_mb: f64,
    merge_read_ops: u64,
    merge_read_mb: f64,
    merge_write_ops: u64,
    merge_write_mb: f64,
}

fn lineitem_schema(
    key_columns: Vec<usize>,
    value_columns: Vec<usize>,
) -> (Arc<Schema>, Vec<usize>, Vec<usize>) {
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
    (schema, key_columns, value_columns)
}

fn yellow_taxi_schema(
    key_columns: Vec<usize>,
    value_columns: Vec<usize>,
) -> (Arc<Schema>, Vec<usize>, Vec<usize>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("vendorid", DataType::Int64, true), // 0
        Field::new("tpep_pickup_datetime", DataType::Utf8, true), // 1
        Field::new("tpep_dropoff_datetime", DataType::Utf8, true), // 2
        Field::new("passenger_count", DataType::Int64, true), // 3
        Field::new("trip_distance", DataType::Float64, true), // 4
        Field::new("ratecodeid", DataType::Int64, true), // 5
        Field::new("store_and_fwd_flag", DataType::Utf8, true), // 6
        Field::new("pulocationid", DataType::Int64, true), // 7
        Field::new("dolocationid", DataType::Int64, true), // 8
        Field::new("payment_type", DataType::Int64, true), // 9
        Field::new("fare_amount", DataType::Float64, true), // 10
        Field::new("extra", DataType::Float64, true),  // 11
        Field::new("mta_tax", DataType::Float64, true), // 12
        Field::new("tip_amount", DataType::Float64, true), // 13
        Field::new("tolls_amount", DataType::Float64, true), // 14
        Field::new("improvement_surcharge", DataType::Float64, true), // 15
        Field::new("total_amount", DataType::Float64, true), // 16
        Field::new("congestion_surcharge", DataType::Float64, true), // 17
        Field::new("airport_fee", DataType::Float64, true), // 18
    ]));
    (schema, key_columns, value_columns)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Csv {
            input,
            schema,
            run_gen_threads,
            merge_threads,
            memory_mb,
            delimiter,
            key_columns,
            payload_columns,
            temp_dir,
            benchmark_runs,
            thread_counts,
            memory_sizes,
        } => {
            sort_csv(
                &input,
                schema,
                run_gen_threads,
                merge_threads,
                memory_mb,
                delimiter,
                key_columns,
                payload_columns,
                &temp_dir,
                benchmark_runs,
                thread_counts,
                memory_sizes,
            )?;
        }
        Commands::Gensort {
            input,
            run_gen_threads,
            merge_threads,
            memory_mb,
            temp_dir,
            verify,
            benchmark_runs,
            thread_counts,
            memory_sizes,
        } => {
            sort_gensort(
                &input,
                run_gen_threads,
                merge_threads,
                memory_mb,
                &temp_dir,
                verify,
                benchmark_runs,
                thread_counts,
                memory_sizes,
            )?;
        }
    }

    Ok(())
}

fn sort_csv(
    input: &Path,
    schema: CsvSchema,
    run_gen_threads: usize,
    merge_threads: usize,
    memory_mb: usize,
    delimiter: char,
    key_columns: Option<Vec<usize>>,
    payload_columns: Option<Vec<usize>>,
    temp_dir: &Path,
    benchmark_runs: usize,
    thread_counts: Option<Vec<usize>>,
    memory_sizes: Option<Vec<String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Sorting CSV file with schema: {:?}", schema);
    println!("Input: {:?}", input);
    println!("Run generation threads: {}", run_gen_threads);
    println!("Merge threads: {}", merge_threads);
    println!("Memory limit: {} MB", memory_mb);
    println!("Delimiter: '{}'", delimiter);

    // Use provided key/payload columns or defaults (key=0, payload=1)
    let key_cols = key_columns.unwrap_or_else(|| vec![0]);
    let value_cols = payload_columns.unwrap_or_else(|| vec![1]);

    println!("Key columns: {:?}", key_cols);
    println!("Payload columns: {:?}", value_cols);

    // Create schema and configure based on type
    let (arrow_schema, final_key_columns, final_value_columns) = match schema {
        CsvSchema::Lineitem => {
            println!("Using lineitem schema");
            lineitem_schema(key_cols.clone(), value_cols.clone())
        }
        CsvSchema::YellowTrip => {
            println!("Using yellow trip schema");
            yellow_taxi_schema(key_cols.clone(), value_cols.clone())
        }
    };

    let mut config = CsvDirectConfig::new(arrow_schema);
    config.delimiter = delimiter as u8;
    config.key_columns = final_key_columns;
    config.value_columns = final_value_columns;
    config.has_headers = true;

    // Use provided thread counts and memory sizes, or defaults
    let threads = thread_counts.unwrap_or_else(|| vec![run_gen_threads]);
    let mem_sizes = memory_sizes.unwrap_or_else(|| vec![format!("{}MB", memory_mb)]);

    run_comprehensive_benchmark(
        input,
        schema,
        delimiter,
        key_cols,
        value_cols,
        &threads,
        &mem_sizes,
        temp_dir,
        benchmark_runs,
    )
}

fn run_comprehensive_benchmark(
    input: &Path,
    schema: CsvSchema,
    delimiter: char,
    key_columns: Vec<usize>,
    value_columns: Vec<usize>,
    thread_counts: &[usize],
    memory_sizes: &[String],
    temp_dir: &Path,
    num_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse memory sizes
    let memory_configs: Vec<(usize, String)> = memory_sizes
        .iter()
        .map(|s| {
            let s = s.trim();
            let (num, unit) = if s.to_uppercase().ends_with("GB") {
                let num_str = s[..s.len() - 2].trim();
                (
                    num_str.parse::<f64>().unwrap_or(1.0) * 1024.0,
                    s.to_string(),
                )
            } else if s.to_uppercase().ends_with("MB") {
                let num_str = s[..s.len() - 2].trim();
                (num_str.parse::<f64>().unwrap_or(1024.0), s.to_string())
            } else {
                // Assume MB if no unit
                (s.parse::<f64>().unwrap_or(1024.0), format!("{}MB", s))
            };
            (num as usize, unit)
        })
        .collect();

    let mut all_results = Vec::new();

    // Get one record count for display
    let (arrow_schema, final_key_columns, final_value_columns) = match schema {
        CsvSchema::Lineitem => lineitem_schema(key_columns.clone(), value_columns.clone()),
        CsvSchema::YellowTrip => yellow_taxi_schema(key_columns.clone(), value_columns.clone()),
    };

    let mut test_config = CsvDirectConfig::new(arrow_schema.clone());
    test_config.delimiter = delimiter as u8;
    test_config.key_columns = final_key_columns.clone();
    test_config.value_columns = final_value_columns.clone();
    test_config.has_headers = true;

    let test_input = CsvInputDirect::new(input, test_config)?;
    let total_entries = count_csv_records(&test_input)?;

    println!("\n=== COMPREHENSIVE BENCHMARK MODE ===");
    println!("Input file: {:?}", input);
    println!("Total entries: {}", total_entries);
    println!("Thread counts: {:?}", thread_counts);
    println!("Memory sizes: {:?}", memory_sizes);
    println!("Runs per configuration: {}", num_runs);
    println!();

    // Run benchmarks for each configuration
    for &threads in thread_counts {
        for (memory_mb, memory_str) in &memory_configs {
            println!("Testing {} threads, {} memory...", threads, memory_str);

            let mut accumulated_stats = RunStats::default();
            let mut valid_runs = 0;

            for run in 1..=num_runs {
                print!("  Run {}/{}: ", run, num_runs);

                let mut config = CsvDirectConfig::new(arrow_schema.clone());
                config.delimiter = delimiter as u8;
                config.key_columns = final_key_columns.clone();
                config.value_columns = final_value_columns.clone();
                config.has_headers = true;

                let csv_input = CsvInputDirect::new(input, config)?;

                let mut sorter = ExternalSorter::new_with_threads_and_dir(
                    threads,
                    threads,
                    memory_mb * 1024 * 1024,
                    temp_dir,
                );

                let start = Instant::now();
                let output = sorter.sort(Box::new(csv_input))?;
                let elapsed = start.elapsed();
                let stats = output.stats();

                // Accumulate stats instead of pushing to vectors
                accumulated_stats.total_time += elapsed.as_secs_f64();
                accumulated_stats.runs_count += stats.num_runs;

                if let Some(rg_time) = stats.run_generation_time_ms {
                    accumulated_stats.run_gen_time += rg_time as f64 / 1000.0;
                }
                if let Some(m_time) = stats.merge_time_ms {
                    accumulated_stats.merge_time += m_time as f64 / 1000.0;
                }
                if let Some(ref io) = stats.run_generation_io_stats {
                    accumulated_stats.run_gen_read_ops += io.read_ops;
                    accumulated_stats.run_gen_read_mb += io.read_bytes as f64 / 1_000_000.0;
                    accumulated_stats.run_gen_write_ops += io.write_ops;
                    accumulated_stats.run_gen_write_mb += io.write_bytes as f64 / 1_000_000.0;
                }
                if let Some(ref io) = stats.merge_io_stats {
                    accumulated_stats.merge_read_ops += io.read_ops;
                    accumulated_stats.merge_read_mb += io.read_bytes as f64 / 1_000_000.0;
                    accumulated_stats.merge_write_ops += io.write_ops;
                    accumulated_stats.merge_write_mb += io.write_bytes as f64 / 1_000_000.0;
                }

                valid_runs += 1;
                println!("{:.2}s", elapsed.as_secs_f64());
            }

            // Calculate averages from accumulated stats
            let runs_f64 = valid_runs as f64;
            let avg_total = accumulated_stats.total_time / runs_f64;
            let avg_run_gen = accumulated_stats.run_gen_time / runs_f64;
            let avg_merge = accumulated_stats.merge_time / runs_f64;
            let avg_runs = accumulated_stats.runs_count / valid_runs;

            // Average I/O stats
            let rg_read_ops = (accumulated_stats.run_gen_read_ops as f64 / runs_f64) as u64;
            let rg_read_mb = accumulated_stats.run_gen_read_mb / runs_f64;
            let rg_write_ops = (accumulated_stats.run_gen_write_ops as f64 / runs_f64) as u64;
            let rg_write_mb = accumulated_stats.run_gen_write_mb / runs_f64;
            let m_read_ops = (accumulated_stats.merge_read_ops as f64 / runs_f64) as u64;
            let m_read_mb = accumulated_stats.merge_read_mb / runs_f64;
            let m_write_ops = (accumulated_stats.merge_write_ops as f64 / runs_f64) as u64;
            let m_write_mb = accumulated_stats.merge_write_mb / runs_f64;

            let total_read_mb = rg_read_mb + m_read_mb;
            let total_write_mb = rg_write_mb + m_write_mb;

            all_results.push(BenchmarkResult {
                threads,
                memory_mb: *memory_mb,
                memory_str: memory_str.clone(),
                runs: avg_runs,
                total_time: avg_total,
                run_gen_time: avg_run_gen,
                merge_time: avg_merge,
                entries: total_entries,
                throughput: total_entries as f64 / avg_total / 1_000_000.0,
                read_mb: total_read_mb,
                write_mb: total_write_mb,
                run_gen_read_ops: rg_read_ops,
                run_gen_read_mb: rg_read_mb,
                run_gen_write_ops: rg_write_ops,
                run_gen_write_mb: rg_write_mb,
                merge_read_ops: m_read_ops,
                merge_read_mb: m_read_mb,
                merge_write_ops: m_write_ops,
                merge_write_mb: m_write_mb,
            });
        }
    }

    // Print summary table
    print_benchmark_summary(&all_results);

    Ok(())
}

fn sort_gensort(
    input: &Path,
    run_gen_threads: usize,
    merge_threads: usize,
    memory_mb: usize,
    temp_dir: &Path,
    verify: bool,
    benchmark_runs: usize,
    thread_counts: Option<Vec<usize>>,
    memory_sizes: Option<Vec<String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Sorting GenSort binary file");
    println!("Input: {:?}", input);
    println!("Run generation threads: {}", run_gen_threads);
    println!("Merge threads: {}", merge_threads);
    println!("Memory limit: {} MB", memory_mb);
    println!();

    // Use provided thread counts and memory sizes, or defaults
    let threads = thread_counts.unwrap_or_else(|| vec![run_gen_threads]);
    let mem_sizes = memory_sizes.unwrap_or_else(|| vec![format!("{}MB", memory_mb)]);

    run_gensort_benchmark(
        input,
        &threads,
        &mem_sizes,
        temp_dir,
        benchmark_runs,
        verify,
    )
}

fn run_gensort_benchmark(
    input: &Path,
    thread_counts: &[usize],
    memory_sizes: &[String],
    temp_dir: &Path,
    num_runs: usize,
    verify: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse memory sizes
    let memory_configs: Vec<(usize, String)> = memory_sizes
        .iter()
        .map(|s| {
            let s = s.trim();
            let (num, unit) = if s.to_uppercase().ends_with("GB") {
                let num_str = s[..s.len() - 2].trim();
                (
                    num_str.parse::<f64>().unwrap_or(1.0) * 1024.0,
                    s.to_string(),
                )
            } else if s.to_uppercase().ends_with("MB") {
                let num_str = s[..s.len() - 2].trim();
                (num_str.parse::<f64>().unwrap_or(1024.0), s.to_string())
            } else {
                // Assume MB if no unit
                (s.parse::<f64>().unwrap_or(1024.0), format!("{}MB", s))
            };
            (num as usize, unit)
        })
        .collect();

    let mut all_results = Vec::new();

    // Get record count
    let gensort_input = GenSortInputDirect::new(input)?;
    let total_entries = gensort_input.len();

    println!("\n=== GENSORT BENCHMARK MODE ===");
    println!("Input file: {:?}", input);
    println!("Total entries: {}", total_entries);
    println!("Thread counts: {:?}", thread_counts);
    println!("Memory sizes: {:?}", memory_sizes);
    println!("Runs per configuration: {}", num_runs);
    println!("Verify output: {}", verify);
    println!();

    // Run benchmarks for each configuration
    for &threads in thread_counts {
        for (memory_mb, memory_str) in &memory_configs {
            println!("Testing {} threads, {} memory...", threads, memory_str);

            let mut accumulated_stats = RunStats::default();
            let mut valid_runs = 0;

            for run in 1..=num_runs {
                print!("  Run {}/{}: ", run, num_runs);

                let gensort_input = GenSortInputDirect::new(input)?;

                let mut sorter = ExternalSorter::new_with_threads_and_dir(
                    threads,
                    threads,
                    memory_mb * 1024 * 1024,
                    temp_dir,
                );

                let start = Instant::now();
                let output = sorter.sort(Box::new(gensort_input))?;
                let elapsed = start.elapsed();
                let stats = output.stats();

                // Accumulate stats
                accumulated_stats.total_time += elapsed.as_secs_f64();
                accumulated_stats.runs_count += stats.num_runs;

                if let Some(rg_time) = stats.run_generation_time_ms {
                    accumulated_stats.run_gen_time += rg_time as f64 / 1000.0;
                }
                if let Some(m_time) = stats.merge_time_ms {
                    accumulated_stats.merge_time += m_time as f64 / 1000.0;
                }
                if let Some(ref io) = stats.run_generation_io_stats {
                    accumulated_stats.run_gen_read_ops += io.read_ops;
                    accumulated_stats.run_gen_read_mb += io.read_bytes as f64 / 1_000_000.0;
                    accumulated_stats.run_gen_write_ops += io.write_ops;
                    accumulated_stats.run_gen_write_mb += io.write_bytes as f64 / 1_000_000.0;
                }
                if let Some(ref io) = stats.merge_io_stats {
                    accumulated_stats.merge_read_ops += io.read_ops;
                    accumulated_stats.merge_read_mb += io.read_bytes as f64 / 1_000_000.0;
                    accumulated_stats.merge_write_ops += io.write_ops;
                    accumulated_stats.merge_write_mb += io.write_bytes as f64 / 1_000_000.0;
                }

                valid_runs += 1;
                println!("{:.2}s", elapsed.as_secs_f64());

                // Verify if requested (only on first run to save time)
                if verify && run == 1 {
                    println!("    Verifying sorted output...");
                    verify_sorted_output(&output)?;
                    println!("    Verification passed!");
                }
            }

            // Calculate averages
            let runs_f64 = valid_runs as f64;
            let avg_total = accumulated_stats.total_time / runs_f64;
            let avg_run_gen = accumulated_stats.run_gen_time / runs_f64;
            let avg_merge = accumulated_stats.merge_time / runs_f64;
            let avg_runs = accumulated_stats.runs_count / valid_runs;

            // Average I/O stats
            let rg_read_ops = (accumulated_stats.run_gen_read_ops as f64 / runs_f64) as u64;
            let rg_read_mb = accumulated_stats.run_gen_read_mb / runs_f64;
            let rg_write_ops = (accumulated_stats.run_gen_write_ops as f64 / runs_f64) as u64;
            let rg_write_mb = accumulated_stats.run_gen_write_mb / runs_f64;
            let m_read_ops = (accumulated_stats.merge_read_ops as f64 / runs_f64) as u64;
            let m_read_mb = accumulated_stats.merge_read_mb / runs_f64;
            let m_write_ops = (accumulated_stats.merge_write_ops as f64 / runs_f64) as u64;
            let m_write_mb = accumulated_stats.merge_write_mb / runs_f64;

            let total_read_mb = rg_read_mb + m_read_mb;
            let total_write_mb = rg_write_mb + m_write_mb;

            all_results.push(BenchmarkResult {
                threads,
                memory_mb: *memory_mb,
                memory_str: memory_str.clone(),
                runs: avg_runs,
                total_time: avg_total,
                run_gen_time: avg_run_gen,
                merge_time: avg_merge,
                entries: total_entries,
                throughput: total_entries as f64 / avg_total / 1_000_000.0,
                read_mb: total_read_mb,
                write_mb: total_write_mb,
                run_gen_read_ops: rg_read_ops,
                run_gen_read_mb: rg_read_mb,
                run_gen_write_ops: rg_write_ops,
                run_gen_write_mb: rg_write_mb,
                merge_read_ops: m_read_ops,
                merge_read_mb: m_read_mb,
                merge_write_ops: m_write_ops,
                merge_write_mb: m_write_mb,
            });
        }
    }

    // Print summary table
    print_benchmark_summary(&all_results);

    Ok(())
}

fn print_benchmark_summary(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(120));
    println!("Benchmark Results Summary");
    println!("{}", "=".repeat(120));
    println!(
        "{:<8} {:<12} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10}",
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
        "{:<8} {:<12} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10}",
        "", "", "", "", "", "", "", "(M entries/s)", "", ""
    );
    println!("{}", "-".repeat(120));

    for result in results {
        println!(
            "{:<8} {:<12} {:<6} {:<10.2} {:<12.2} {:<10.2} {:<12} {:<16.2} {:<10.1} {:<10.1}",
            result.threads,
            result.memory_str,
            result.runs,
            result.total_time,
            result.run_gen_time,
            result.merge_time,
            result.entries,
            result.throughput,
            result.read_mb,
            result.write_mb,
        );
    }
    println!("{}", "=".repeat(120));

    // Print detailed I/O statistics
    println!("\nDetailed I/O Statistics Summary:");
    println!("{}", "-".repeat(100));
    println!(
        "{:<8} {:<12} {:<20} {:<20} {:<20} {:<20}",
        "Threads", "Memory", "Run Gen Reads", "Run Gen Writes", "Merge Reads", "Merge Writes"
    );
    println!(
        "{:<8} {:<12} {:<20} {:<20} {:<20} {:<20}",
        "", "", "(ops / MB)", "(ops / MB)", "(ops / MB)", "(ops / MB)"
    );
    println!("{}", "-".repeat(100));

    for result in results {
        println!(
            "{:<8} {:<12} {:<20} {:<20} {:<20} {:<20}",
            result.threads,
            result.memory_str,
            format!(
                "{} / {:.1}",
                result.run_gen_read_ops, result.run_gen_read_mb
            ),
            format!(
                "{} / {:.1}",
                result.run_gen_write_ops, result.run_gen_write_mb
            ),
            format!("{} / {:.1}", result.merge_read_ops, result.merge_read_mb),
            format!("{} / {:.1}", result.merge_write_ops, result.merge_write_mb),
        );
    }
    println!("{}", "-".repeat(100));
}

fn count_csv_records(input: &dyn SortInput) -> Result<usize, String> {
    let scanners = input.create_parallel_scanners(1, None);
    let mut count = 0;
    for scanner in scanners {
        for _ in scanner {
            count += 1;
        }
    }
    Ok(count)
}

fn print_sort_results(
    record_count: usize,
    elapsed: std::time::Duration,
    output: &Box<dyn es::SortOutput>,
) {
    println!("\nSort completed in {:.2} seconds", elapsed.as_secs_f64());

    let stats = output.stats();
    println!("\nSort Statistics:");
    println!("  Number of runs: {}", stats.num_runs);
    if let Some(run_gen_time) = stats.run_generation_time_ms {
        println!("  Run generation time: {} ms", run_gen_time);
    }
    if let Some(merge_time) = stats.merge_time_ms {
        println!("  Merge time: {} ms", merge_time);
    }

    let throughput_meps = record_count as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!("  Throughput: {:.2} M entries/s", throughput_meps);

    // Print I/O statistics if available
    if let Some(ref io_stats) = stats.run_generation_io_stats {
        println!("\nRun Generation I/O:");
        println!(
            "  Reads: {} ops, {:.2} MB",
            io_stats.read_ops,
            io_stats.read_bytes as f64 / 1_000_000.0
        );
        println!(
            "  Writes: {} ops, {:.2} MB",
            io_stats.write_ops,
            io_stats.write_bytes as f64 / 1_000_000.0
        );
    }

    if let Some(ref io_stats) = stats.merge_io_stats {
        println!("\nMerge Phase I/O:");
        println!(
            "  Reads: {} ops, {:.2} MB",
            io_stats.read_ops,
            io_stats.read_bytes as f64 / 1_000_000.0
        );
        println!(
            "  Writes: {} ops, {:.2} MB",
            io_stats.write_ops,
            io_stats.write_bytes as f64 / 1_000_000.0
        );
    }
}

fn verify_sorted_output(
    output: &Box<dyn es::SortOutput>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut prev_key: Option<Vec<u8>> = None;
    let mut count = 0;

    for (key, _value) in output.iter() {
        if let Some(ref prev) = prev_key {
            if key < *prev {
                eprintln!("ERROR: Sort order violation at record {}", count);
                eprintln!("  Previous key: {:?}", prev);
                eprintln!("  Current key: {:?}", key);
                return Err("Sort order violation".into());
            }
        }
        prev_key = Some(key);
        count += 1;
    }

    println!("Verified {} records - all correctly sorted!", count);
    Ok(())
}

fn print_sample_records(
    output: &Box<dyn es::SortOutput>,
    key_columns: &[usize],
    value_columns: &[usize],
    schema: CsvSchema,
) {
    let mut first_records = Vec::new();
    let mut all_records = Vec::new();

    // Collect all records to get first and last
    for (key, value) in output.iter() {
        if first_records.len() < 5 {
            first_records.push((key.clone(), value.clone()));
        }
        all_records.push((key, value));
    }

    let total = all_records.len();
    if total == 0 {
        return;
    }

    // Get column names based on schema
    let column_names = match schema {
        CsvSchema::Lineitem => vec![
            "l_orderkey",
            "l_partkey",
            "l_suppkey",
            "l_linenumber",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
            "l_shipinstruct",
            "l_shipmode",
            "l_comment",
        ],
        CsvSchema::YellowTrip => vec![
            "vendorid",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "ratecodeid",
            "store_and_fwd_flag",
            "pulocationid",
            "dolocationid",
            "payment_type",
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "improvement_surcharge",
            "total_amount",
            "congestion_surcharge",
            "airport_fee",
        ],
    };

    println!("\nFirst 5 records:");
    for (i, (key, value)) in first_records.iter().enumerate() {
        print!("  Record {}: ", i);
        print_csv_key_value(
            key,
            value,
            key_columns,
            value_columns,
            &column_names,
            schema,
        );
    }

    if total > 5 {
        println!("\nLast 5 records:");
        let start_idx = total.saturating_sub(5);
        for i in start_idx..total {
            print!("  Record {}: ", i);
            let (key, value) = &all_records[i];
            print_csv_key_value(
                key,
                value,
                key_columns,
                value_columns,
                &column_names,
                schema,
            );
        }
    }
}

fn print_csv_key_value(
    key: &[u8],
    value: &[u8],
    key_columns: &[usize],
    value_columns: &[usize],
    column_names: &[&str],
    schema: CsvSchema,
) {
    // Decode the key columns
    print!("Key: ");
    let mut offset = 0;
    for (i, &col_idx) in key_columns.iter().enumerate() {
        if offset >= key.len() {
            break;
        }

        // Determine the data type based on schema and column index
        let field_type = match schema {
            CsvSchema::Lineitem => match col_idx {
                0..=2 => "Int64",    // orderkey, partkey, suppkey
                3 => "Int32",        // linenumber
                4..=7 => "Float64",  // quantity, extendedprice, discount, tax
                10..=12 => "Date32", // shipdate, commitdate, receiptdate
                _ => "Utf8",         // strings
            },
            CsvSchema::YellowTrip => match col_idx {
                0 | 3 | 5 | 7..=9 => "Int64", // vendorid, passenger_count, ratecodeid, location ids, payment_type
                4 | 10..=18 => "Float64",     // trip_distance, fare amounts
                1 | 2 => "Utf8",              // datetime strings
                _ => "Utf8",                  // other strings
            },
        };

        // Get expected size
        let expected_size = match field_type {
            "Int64" => 8,
            "Int32" => 4,
            "Float64" => 8,
            "Date32" => 4,
            _ => 0, // Variable length
        };

        if expected_size > 0 {
            let field_end = offset + expected_size;
            if field_end > key.len() {
                break;
            }

            let field_bytes = &key[offset..field_end];
            let decoded = decode_bytes(field_bytes, field_type)
                .unwrap_or_else(|_| format!("<{} bytes>", field_bytes.len()));

            if col_idx < column_names.len() {
                print!("{}={}", column_names[col_idx], decoded);
            } else {
                print!("col{}={}", col_idx, decoded);
            }

            offset = field_end;
            // Skip null terminator if present
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

            if col_idx < column_names.len() {
                print!("{}={}", column_names[col_idx], decoded);
            } else {
                print!("col{}={}", col_idx, decoded);
            }

            offset = field_end;
            if offset < key.len() && key[offset] == 0 {
                offset += 1;
            }
        }

        if i < key_columns.len() - 1 {
            print!(", ");
        }
    }

    print!(" | Value: ");
    if value_columns.is_empty() {
        print!("<empty>");
    } else {
        // For value, just show a preview
        let preview = if value.len() > 50 {
            format!("<{} bytes>", value.len())
        } else {
            format!("<{} bytes>", value.len())
        };
        print!("{}", preview);
    }
    println!();
}

fn print_gensort_sample_records(output: &Box<dyn es::SortOutput>) {
    let mut first_records = Vec::new();
    let mut all_records = Vec::new();

    // Collect records
    for (key, payload) in output.iter() {
        if first_records.len() < 5 {
            first_records.push((key.clone(), payload.clone()));
        }
        all_records.push((key, payload));
    }

    let total = all_records.len();
    if total == 0 {
        return;
    }

    println!("\nFirst 5 records:");
    for (i, (key, _payload)) in first_records.iter().enumerate() {
        print!("  Record {}: Key = ", i);
        match std::str::from_utf8(key) {
            Ok(s) => println!("\"{}\"", s),
            Err(_) => println!("{:?}", key),
        }
    }

    if total > 5 {
        println!("\nLast 5 records:");
        let start_idx = total.saturating_sub(5);
        for i in start_idx..total {
            let (key, _payload) = &all_records[i];
            print!("  Record {}: Key = ", i);
            match std::str::from_utf8(key) {
                Ok(s) => println!("\"{}\"", s),
                Err(_) => println!("{:?}", key),
            }
        }
    }
}
