use clap::{Parser, Subcommand, ValueEnum};
use es::{
    CsvDirectConfig, CsvInputDirect, ExternalSorter, GenSortInputDirect, Sorter, SortInput,
    order_preserving_encoding::decode_bytes,
};
use arrow::datatypes::{DataType, Field, Schema};
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
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CsvSchema {
    /// TPC-H lineitem table schema
    Lineitem,
    /// NYC yellow trip data schema
    YellowTrip,
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
            temp_dir,
            benchmark,
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
                &temp_dir,
                benchmark,
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
        } => {
            sort_gensort(
                &input,
                run_gen_threads,
                merge_threads,
                memory_mb,
                &temp_dir,
                verify,
            )?;
        }
        Commands::Auto {
            input,
            run_gen_threads,
            merge_threads,
            memory_mb,
            temp_dir,
            delimiter,
        } => {
            auto_detect_and_sort(
                &input,
                run_gen_threads,
                merge_threads,
                memory_mb,
                &temp_dir,
                delimiter,
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
    temp_dir: &Path,
    benchmark: bool,
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
    
    // Create schema and configure based on type
    let (arrow_schema, key_columns, value_columns) = match schema {
        CsvSchema::Lineitem => {
            println!("Using lineitem schema - sorting by l_orderkey, l_linenumber");
            let schema = Arc::new(Schema::new(vec![
                Field::new("l_orderkey", DataType::Int64, false),       // 0
                Field::new("l_partkey", DataType::Int64, false),        // 1
                Field::new("l_suppkey", DataType::Int64, false),        // 2
                Field::new("l_linenumber", DataType::Int32, false),     // 3
                Field::new("l_quantity", DataType::Float64, false),     // 4
                Field::new("l_extendedprice", DataType::Float64, false),// 5
                Field::new("l_discount", DataType::Float64, false),     // 6
                Field::new("l_tax", DataType::Float64, false),         // 7
                Field::new("l_returnflag", DataType::Utf8, false),      // 8
                Field::new("l_linestatus", DataType::Utf8, false),      // 9
                Field::new("l_shipdate", DataType::Date32, false),      // 10
                Field::new("l_commitdate", DataType::Date32, false),    // 11
                Field::new("l_receiptdate", DataType::Date32, false),   // 12
                Field::new("l_shipinstruct", DataType::Utf8, false),    // 13
                Field::new("l_shipmode", DataType::Utf8, false),        // 14
                Field::new("l_comment", DataType::Utf8, false),         // 15
            ]));
            let key_cols = vec![0, 3]; // l_orderkey, l_linenumber
            let value_cols = vec![1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            (schema, key_cols, value_cols)
        }
        CsvSchema::YellowTrip => {
            println!("Using yellow trip schema - sorting by pickup_datetime");
            let schema = Arc::new(Schema::new(vec![
                Field::new("vendorid", DataType::Int64, true),                // 0
                Field::new("tpep_pickup_datetime", DataType::Utf8, true),     // 1
                Field::new("tpep_dropoff_datetime", DataType::Utf8, true),    // 2
                Field::new("passenger_count", DataType::Int64, true),         // 3
                Field::new("trip_distance", DataType::Float64, true),         // 4
                Field::new("ratecodeid", DataType::Int64, true),             // 5
                Field::new("store_and_fwd_flag", DataType::Utf8, true),      // 6
                Field::new("pulocationid", DataType::Int64, true),           // 7
                Field::new("dolocationid", DataType::Int64, true),           // 8
                Field::new("payment_type", DataType::Int64, true),           // 9
                Field::new("fare_amount", DataType::Float64, true),          // 10
                Field::new("extra", DataType::Float64, true),                // 11
                Field::new("mta_tax", DataType::Float64, true),              // 12
                Field::new("tip_amount", DataType::Float64, true),           // 13
                Field::new("tolls_amount", DataType::Float64, true),         // 14
                Field::new("improvement_surcharge", DataType::Float64, true),// 15
                Field::new("total_amount", DataType::Float64, true),         // 16
                Field::new("congestion_surcharge", DataType::Float64, true), // 17
                Field::new("airport_fee", DataType::Float64, true),          // 18
            ]));
            let key_cols = vec![1]; // tpep_pickup_datetime
            let value_cols = vec![0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
            (schema, key_cols, value_cols)
        }
    };

    let mut config = CsvDirectConfig::new(arrow_schema);
    config.delimiter = delimiter as u8;
    config.key_columns = key_columns;
    config.value_columns = value_columns;
    config.has_headers = true;

    if benchmark {
        // Use provided thread counts and memory sizes, or defaults
        let threads = thread_counts.unwrap_or_else(|| vec![run_gen_threads]);
        let mem_sizes = memory_sizes.unwrap_or_else(|| vec![format!("{}MB", memory_mb)]);
        
        run_comprehensive_benchmark(
            input,
            schema,
            delimiter,
            &threads,
            &mem_sizes,
            temp_dir,
            benchmark_runs,
        )
    } else {
        // Save column info before moving config
        let key_cols = config.key_columns.clone();
        let value_cols = config.value_columns.clone();
        
        let csv_input = CsvInputDirect::new(input, config)?;
        let record_count = count_csv_records(&csv_input)?;
        
        let mut sorter = ExternalSorter::new_with_threads_and_dir(
            run_gen_threads,
            merge_threads,
            memory_mb * 1024 * 1024,
            temp_dir,
        );

        let start = Instant::now();
        let output = sorter.sort(Box::new(csv_input))?;
        let elapsed = start.elapsed();

        print_sort_results(record_count, elapsed, &output);
        print_sample_records(&output, &key_cols, &value_cols, schema);
        Ok(())
    }
}

fn run_comprehensive_benchmark(
    input: &Path,
    schema: CsvSchema,
    delimiter: char,
    thread_counts: &[usize],
    memory_sizes: &[String],
    temp_dir: &Path,
    num_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
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

    // Parse memory sizes
    let memory_configs: Vec<(usize, String)> = memory_sizes
        .iter()
        .map(|s| {
            let s = s.trim();
            let (num, unit) = if s.to_uppercase().ends_with("GB") {
                let num_str = s[..s.len() - 2].trim();
                (num_str.parse::<f64>().unwrap_or(1.0) * 1024.0, s.to_string())
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
    let arrow_schema = match schema {
        CsvSchema::Lineitem => Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false),
            Field::new("l_partkey", DataType::Int64, false),
            Field::new("l_suppkey", DataType::Int64, false),
            Field::new("l_linenumber", DataType::Int32, false),
            Field::new("l_quantity", DataType::Float64, false),
            Field::new("l_extendedprice", DataType::Float64, false),
            Field::new("l_discount", DataType::Float64, false),
            Field::new("l_tax", DataType::Float64, false),
            Field::new("l_returnflag", DataType::Utf8, false),
            Field::new("l_linestatus", DataType::Utf8, false),
            Field::new("l_shipdate", DataType::Date32, false),
            Field::new("l_commitdate", DataType::Date32, false),
            Field::new("l_receiptdate", DataType::Date32, false),
            Field::new("l_shipinstruct", DataType::Utf8, false),
            Field::new("l_shipmode", DataType::Utf8, false),
            Field::new("l_comment", DataType::Utf8, false),
        ])),
        CsvSchema::YellowTrip => Arc::new(Schema::new(vec![
            Field::new("vendorid", DataType::Int64, true),
            Field::new("tpep_pickup_datetime", DataType::Utf8, true),
            Field::new("tpep_dropoff_datetime", DataType::Utf8, true),
            Field::new("passenger_count", DataType::Int64, true),
            Field::new("trip_distance", DataType::Float64, true),
            Field::new("ratecodeid", DataType::Int64, true),
            Field::new("store_and_fwd_flag", DataType::Utf8, true),
            Field::new("pulocationid", DataType::Int64, true),
            Field::new("dolocationid", DataType::Int64, true),
            Field::new("payment_type", DataType::Int64, true),
            Field::new("fare_amount", DataType::Float64, true),
            Field::new("extra", DataType::Float64, true),
            Field::new("mta_tax", DataType::Float64, true),
            Field::new("tip_amount", DataType::Float64, true),
            Field::new("tolls_amount", DataType::Float64, true),
            Field::new("improvement_surcharge", DataType::Float64, true),
            Field::new("total_amount", DataType::Float64, true),
            Field::new("congestion_surcharge", DataType::Float64, true),
            Field::new("airport_fee", DataType::Float64, true),
        ])),
    };

    let (key_columns, value_columns) = match schema {
        CsvSchema::Lineitem => (vec![0, 3], vec![1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        CsvSchema::YellowTrip => (vec![1], vec![0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    };

    let mut test_config = CsvDirectConfig::new(arrow_schema.clone());
    test_config.delimiter = delimiter as u8;
    test_config.key_columns = key_columns.clone();
    test_config.value_columns = value_columns.clone();
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
            
            let mut run_times = Vec::new();
            let mut run_gen_times = Vec::new();
            let mut merge_times = Vec::new();
            let mut runs_counts = Vec::new();
            let mut run_gen_io_stats = Vec::new();
            let mut merge_io_stats = Vec::new();

            for run in 1..=num_runs {
                print!("  Run {}/{}: ", run, num_runs);
                
                let mut config = CsvDirectConfig::new(arrow_schema.clone());
                config.delimiter = delimiter as u8;
                config.key_columns = key_columns.clone();
                config.value_columns = value_columns.clone();
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

                run_times.push(elapsed.as_secs_f64());
                runs_counts.push(stats.num_runs);
                
                if let Some(rg_time) = stats.run_generation_time_ms {
                    run_gen_times.push(rg_time as f64 / 1000.0);
                }
                if let Some(m_time) = stats.merge_time_ms {
                    merge_times.push(m_time as f64 / 1000.0);
                }
                if let Some(ref io) = stats.run_generation_io_stats {
                    run_gen_io_stats.push(io.clone());
                }
                if let Some(ref io) = stats.merge_io_stats {
                    merge_io_stats.push(io.clone());
                }

                println!("{:.2}s", elapsed.as_secs_f64());
            }

            // Calculate averages
            let avg_total = run_times.iter().sum::<f64>() / run_times.len() as f64;
            let avg_run_gen = if !run_gen_times.is_empty() {
                run_gen_times.iter().sum::<f64>() / run_gen_times.len() as f64
            } else { 0.0 };
            let avg_merge = if !merge_times.is_empty() {
                merge_times.iter().sum::<f64>() / merge_times.len() as f64
            } else { 0.0 };
            let avg_runs = if !runs_counts.is_empty() {
                runs_counts.iter().sum::<usize>() / runs_counts.len()
            } else { 0 };

            // Average I/O stats
            let mut total_read_mb = 0.0;
            let mut total_write_mb = 0.0;
            let mut rg_read_ops = 0u64;
            let mut rg_read_mb = 0.0;
            let mut rg_write_ops = 0u64;
            let mut rg_write_mb = 0.0;
            let mut m_read_ops = 0u64;
            let mut m_read_mb = 0.0;
            let mut m_write_ops = 0u64;
            let mut m_write_mb = 0.0;

            for io in &run_gen_io_stats {
                rg_read_ops += io.read_ops;
                rg_read_mb += io.read_bytes as f64 / 1_000_000.0;
                rg_write_ops += io.write_ops;
                rg_write_mb += io.write_bytes as f64 / 1_000_000.0;
            }
            for io in &merge_io_stats {
                m_read_ops += io.read_ops;
                m_read_mb += io.read_bytes as f64 / 1_000_000.0;
                m_write_ops += io.write_ops;
                m_write_mb += io.write_bytes as f64 / 1_000_000.0;
            }

            let num_io_runs = run_gen_io_stats.len().max(1) as f64;
            rg_read_ops = (rg_read_ops as f64 / num_io_runs) as u64;
            rg_read_mb /= num_io_runs;
            rg_write_ops = (rg_write_ops as f64 / num_io_runs) as u64;
            rg_write_mb /= num_io_runs;
            m_read_ops = (m_read_ops as f64 / num_io_runs) as u64;
            m_read_mb /= num_io_runs;
            m_write_ops = (m_write_ops as f64 / num_io_runs) as u64;
            m_write_mb /= num_io_runs;

            total_read_mb = rg_read_mb + m_read_mb;
            total_write_mb = rg_write_mb + m_write_mb;

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
    println!("\n{}", "=".repeat(120));
    println!("Benchmark Results Summary");
    println!("{}", "=".repeat(120));
    println!("{:<8} {:<12} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10}",
        "Threads", "Memory", "Runs", "Total (s)", "RunGen (s)", "Merge (s)", 
        "Entries", "Throughput", "Read MB", "Write MB");
    println!("{:<8} {:<12} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10}",
        "", "", "", "", "", "", "", "(M entries/s)", "", "");
    println!("{}", "-".repeat(120));

    for result in &all_results {
        println!("{:<8} {:<12} {:<6} {:<10.2} {:<12.2} {:<10.2} {:<12} {:<16.2} {:<10.1} {:<10.1}",
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
    println!("{:<8} {:<12} {:<20} {:<20} {:<20} {:<20}",
        "Threads", "Memory", "Run Gen Reads", "Run Gen Writes", 
        "Merge Reads", "Merge Writes");
    println!("{:<8} {:<12} {:<20} {:<20} {:<20} {:<20}",
        "", "", "(ops / MB)", "(ops / MB)", "(ops / MB)", "(ops / MB)");
    println!("{}", "-".repeat(100));

    for result in &all_results {
        println!("{:<8} {:<12} {:<20} {:<20} {:<20} {:<20}",
            result.threads,
            result.memory_str,
            format!("{} / {:.1}", result.run_gen_read_ops, result.run_gen_read_mb),
            format!("{} / {:.1}", result.run_gen_write_ops, result.run_gen_write_mb),
            format!("{} / {:.1}", result.merge_read_ops, result.merge_read_mb),
            format!("{} / {:.1}", result.merge_write_ops, result.merge_write_mb),
        );
    }
    println!("{}", "-".repeat(100));

    Ok(())
}

fn sort_gensort(
    input: &Path,
    run_gen_threads: usize,
    merge_threads: usize,
    memory_mb: usize,
    temp_dir: &Path,
    verify: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Sorting GenSort binary file");
    println!("Input: {:?}", input);
    println!("Run generation threads: {}", run_gen_threads);
    println!("Merge threads: {}", merge_threads);
    println!("Memory limit: {} MB", memory_mb);
    println!();

    let gensort_input = GenSortInputDirect::new(input)?;
    let num_records = gensort_input.len();
    println!("Total records: {}", num_records);

    let mut sorter = ExternalSorter::new_with_threads_and_dir(
        run_gen_threads,
        merge_threads,
        memory_mb * 1024 * 1024,
        temp_dir,
    );

    let start = Instant::now();
    let output = sorter.sort(Box::new(gensort_input))?;
    let elapsed = start.elapsed();

    print_sort_results(num_records, elapsed, &output);
    print_gensort_sample_records(&output);

    if verify {
        println!("\nVerifying sorted output...");
        verify_sorted_output(&output)?;
    }

    Ok(())
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
        println!("  Reads: {} ops, {:.2} MB", io_stats.read_ops, io_stats.read_bytes as f64 / 1_000_000.0);
        println!("  Writes: {} ops, {:.2} MB", io_stats.write_ops, io_stats.write_bytes as f64 / 1_000_000.0);
    }
    
    if let Some(ref io_stats) = stats.merge_io_stats {
        println!("\nMerge Phase I/O:");
        println!("  Reads: {} ops, {:.2} MB", io_stats.read_ops, io_stats.read_bytes as f64 / 1_000_000.0);
        println!("  Writes: {} ops, {:.2} MB", io_stats.write_ops, io_stats.write_bytes as f64 / 1_000_000.0);
    }
}

fn verify_sorted_output(output: &Box<dyn es::SortOutput>) -> Result<(), Box<dyn std::error::Error>> {
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
            "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber",
            "l_quantity", "l_extendedprice", "l_discount", "l_tax",
            "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate",
            "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"
        ],
        CsvSchema::YellowTrip => vec![
            "vendorid", "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "passenger_count", "trip_distance", "ratecodeid", "store_and_fwd_flag",
            "pulocationid", "dolocationid", "payment_type", "fare_amount",
            "extra", "mta_tax", "tip_amount", "tolls_amount",
            "improvement_surcharge", "total_amount", "congestion_surcharge", "airport_fee"
        ],
    };
    
    println!("\nFirst 5 records:");
    for (i, (key, value)) in first_records.iter().enumerate() {
        print!("  Record {}: ", i);
        print_csv_key_value(key, value, key_columns, value_columns, &column_names, schema);
    }
    
    if total > 5 {
        println!("\nLast 5 records:");
        let start_idx = total.saturating_sub(5);
        for i in start_idx..total {
            print!("  Record {}: ", i);
            let (key, value) = &all_records[i];
            print_csv_key_value(key, value, key_columns, value_columns, &column_names, schema);
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
                0..=2 => "Int64",         // orderkey, partkey, suppkey
                3 => "Int32",             // linenumber
                4..=7 => "Float64",       // quantity, extendedprice, discount, tax
                10..=12 => "Date32",      // shipdate, commitdate, receiptdate
                _ => "Utf8",              // strings
            },
            CsvSchema::YellowTrip => match col_idx {
                0 | 3 | 5 | 7..=9 => "Int64",  // vendorid, passenger_count, ratecodeid, location ids, payment_type
                4 | 10..=18 => "Float64",       // trip_distance, fare amounts
                1 | 2 => "Utf8",                // datetime strings
                _ => "Utf8",                    // other strings
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

fn auto_detect_and_sort(
    input: &Path,
    run_gen_threads: usize,
    merge_threads: usize,
    memory_mb: usize,
    temp_dir: &Path,
    delimiter: char,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Auto-detecting file format for: {:?}", input);
    
    // First try to detect by extension
    let extension = input.extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase());
    
    match extension.as_deref() {
        Some("csv") | Some("tbl") => {
            // Try to detect CSV schema from filename
            let filename = input.file_stem()
                .and_then(|stem| stem.to_str())
                .map(|s| s.to_lowercase())
                .unwrap_or_default();
            
            let schema = if filename.contains("lineitem") {
                CsvSchema::Lineitem
            } else if filename.contains("yellow") && filename.contains("trip") {
                CsvSchema::YellowTrip
            } else {
                // Try to auto-detect schema by reading first few lines
                match detect_csv_schema(input, delimiter) {
                    Ok(detected_schema) => detected_schema,
                    Err(e) => {
                        return Err(format!("Could not detect CSV schema: {}. Please use 'csv' subcommand with explicit --schema", e).into());
                    }
                }
            };
            
            println!("Detected CSV file with schema: {:?}", schema);
            sort_csv(
                input,
                schema,
                run_gen_threads,
                merge_threads,
                memory_mb,
                delimiter,
                temp_dir,
                false,  // benchmark
                3,      // benchmark_runs
                None,   // thread_counts
                None,   // memory_sizes
            )
        }
        Some("gensort") | Some("dat") => {
            // Try to verify it's a GenSort file by checking if file size is multiple of 100
            match std::fs::metadata(input) {
                Ok(metadata) => {
                    if metadata.len() % 100 == 0 {
                        println!("Detected GenSort binary file");
                        sort_gensort(
                            input,
                            run_gen_threads,
                            merge_threads,
                            memory_mb,
                            temp_dir,
                            false,  // verify
                        )
                    } else {
                        Err(format!("File has .dat extension but size {} is not a multiple of 100 (GenSort record size)", metadata.len()).into())
                    }
                }
                Err(e) => Err(format!("Could not read file metadata: {}", e).into()),
            }
        }
        _ => {
            // Try to detect by content
            match detect_file_format(input) {
                Ok(FileFormat::Csv) => {
                    match detect_csv_schema(input, delimiter) {
                        Ok(schema) => {
                            println!("Detected CSV file with schema: {:?}", schema);
                            sort_csv(
                                input,
                                schema,
                                run_gen_threads,
                                merge_threads,
                                memory_mb,
                                delimiter,
                                temp_dir,
                                false,  // benchmark
                                3,      // benchmark_runs
                                None,   // thread_counts
                                None,   // memory_sizes
                            )
                        }
                        Err(e) => Err(format!("Could not detect CSV schema: {}", e).into()),
                    }
                }
                Ok(FileFormat::GenSort) => {
                    println!("Detected GenSort binary file");
                    sort_gensort(
                        input,
                        run_gen_threads,
                        merge_threads,
                        memory_mb,
                        temp_dir,
                        false,  // verify
                    )
                }
                Err(e) => Err(format!("Could not detect file format: {}", e).into()),
            }
        }
    }
}

enum FileFormat {
    Csv,
    GenSort,
}

fn detect_file_format(path: &Path) -> Result<FileFormat, String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    // Check if file size is multiple of 100 (GenSort record size)
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("Could not read file metadata: {}", e))?;
    
    if metadata.len() % 100 == 0 && metadata.len() > 0 {
        // Could be GenSort, let's verify by reading a sample
        let mut file = File::open(path)
            .map_err(|e| format!("Could not open file: {}", e))?;
        
        let mut buffer = vec![0u8; 100.min(metadata.len() as usize)];
        use std::io::Read;
        file.read_exact(&mut buffer)
            .map_err(|e| format!("Could not read file: {}", e))?;
        
        // GenSort files have 10-byte keys followed by 90-byte payloads
        // Keys are typically ASCII or binary but structured
        // If we see common CSV patterns, it's likely CSV
        let as_string = String::from_utf8_lossy(&buffer);
        if as_string.contains(',') || as_string.contains('|') || as_string.contains('\t') {
            return Ok(FileFormat::Csv);
        }
        
        // Likely GenSort
        return Ok(FileFormat::GenSort);
    }
    
    // Not a multiple of 100, check if it's CSV
    let file = File::open(path)
        .map_err(|e| format!("Could not open file: {}", e))?;
    let reader = BufReader::new(file);
    
    // Read first line
    let mut lines = reader.lines();
    if let Some(Ok(first_line)) = lines.next() {
        // Check for common delimiters
        if first_line.contains(',') || first_line.contains('|') || first_line.contains('\t') {
            return Ok(FileFormat::Csv);
        }
    }
    
    Err("Could not determine file format".to_string())
}

fn detect_csv_schema(path: &Path, delimiter: char) -> Result<CsvSchema, String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    let file = File::open(path)
        .map_err(|e| format!("Could not open file: {}", e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // Read first line (potential header)
    let first_line = lines.next()
        .ok_or("File is empty")?
        .map_err(|e| format!("Could not read first line: {}", e))?;
    
    let fields: Vec<&str> = first_line.split(delimiter).collect();
    
    // Check for lineitem columns
    let lineitem_cols = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", 
                         "l_quantity", "l_extendedprice", "l_discount", "l_tax"];
    let yellow_trip_cols = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
                            "passenger_count", "trip_distance", "RatecodeID"];
    
    // Count matches
    let lineitem_matches = fields.iter()
        .filter(|f| lineitem_cols.iter().any(|col| f.contains(col)))
        .count();
    let yellow_trip_matches = fields.iter()
        .filter(|f| yellow_trip_cols.iter().any(|col| f.contains(col)))
        .count();
    
    if lineitem_matches > yellow_trip_matches && lineitem_matches > 0 {
        return Ok(CsvSchema::Lineitem);
    } else if yellow_trip_matches > 0 {
        return Ok(CsvSchema::YellowTrip);
    }
    
    // Try to detect by number of fields
    match fields.len() {
        16 => Ok(CsvSchema::Lineitem),    // Lineitem has 16 columns
        18 | 19 => Ok(CsvSchema::YellowTrip),  // Yellow trip varies by year
        _ => Err(format!("Could not determine schema from {} fields", fields.len())),
    }
}