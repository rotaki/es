//! Refactored lineitem benchmark matching gen_sort_cli style
//! Supports CSV format with policy-based optimization

use arrow::datatypes::{DataType, Field, Schema};
use clap::Parser;
use es::sort_policy::{SortConfig, get_all_policies};
use es::{
    CsvDirectConfig, CsvInputDirect, ExternalSorter, RunsOutput, SortInput, SortOutput, SortStats,
    order_preserving_encoding::decode_bytes,
};
use es::{ExternalSorterWithOVC, RunsOutputWithOVC};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// TPC-H lineitem column cardinality reference table
// Column Index | Column Name         | DataType      | Cardinality | Distinct Values    | Category
// -------------|-------------------- |---------------|-------------|--------------------|-----------
// 0            | l_orderkey          | Int64         | High        | ~1.5M * SF         | High (foreign key to Orders)
// 1            | l_partkey           | Int64         | High        | 200,000 * SF       | High (foreign key to Part)
// 2            | l_suppkey           | Int64         | Medium      | 10,000 * SF        | Medium (foreign key to Supplier)
// 3            | l_linenumber        | Int32         | Ultra-Low   | 7                  | Ultra-Low (values: 1-7)
// 4            | l_quantity          | Float64       | Low         | 50                 | Low (integer values: 1-50)
// 5            | l_extendedprice     | Float64       | High        | Continuous         | High (calculated: quantity * price)
// 6            | l_discount          | Float64       | Ultra-Low   | 11                 | Ultra-Low (values: 0.00-0.10, step 0.01)
// 7            | l_tax               | Float64       | Ultra-Low   | 9                  | Ultra-Low (values: 0.00-0.08, step 0.01)
// 8            | l_returnflag        | Utf8          | Ultra-Low   | 3                  | Ultra-Low (values: 'R', 'A', 'N')
// 9            | l_linestatus        | Utf8          | Binary      | 2                  | Binary (values: 'O', 'F')
// 10           | l_shipdate          | Date32        | Medium      | ~2,526             | Medium (range: 1992-01-02 to 1998-12-01)
// 11           | l_commitdate        | Date32        | Medium      | ~2,466             | Medium (similar range to shipdate)
// 12           | l_receiptdate       | Date32        | Medium      | ~2,554             | Medium (typically after shipdate)
// 13           | l_shipinstruct      | Utf8          | Ultra-Low   | 4                  | Ultra-Low (values: 'DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN')
// 14           | l_shipmode          | Utf8          | Ultra-Low   | 7                  | Ultra-Low (values: 'REG AIR', 'AIR', 'RAIL', 'SHIP', 'TRUCK', 'MAIL', 'FOB')
// 15           | l_comment           | Utf8          | High        | Nearly unique      | High (random text comments)

// Cardinality categories for sorting algorithm selection:
// - Binary (2 values):      Use simple partition - Column 9
// - Ultra-Low (≤11 values): Use counting sort - Columns 3, 6, 7, 8, 9, 13, 14
// - Low (12-100 values):    Use counting sort - Column 4
// - Medium (100-10K):       Use radix or quicksort - Columns 2, 10, 11, 12
// - High (>10K):           Use quicksort or parallel sort - Columns 0, 1, 5, 15

// Sorting-friendly columns (counting sort applicable): 3, 4, 6, 7, 8, 9, 13, 14
// String columns requiring expensive comparisons: 8, 9, 13, 14, 15
// Continuous numeric requiring comparison-based sort: 0, 1, 2, 5
// Date columns with temporal clustering: 10, 11, 12

// Scale Factor (SF) impact on row counts:
// SF=1:    ~6 million rows
// SF=10:   ~60 million rows
// SF=100:  ~600 million rows
// SF=1000: ~6 billion rows

// TPC-H lineitem schema definition - made global for access by estimation functions
fn get_lineitem_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
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
    ]))
}

#[derive(Parser)]
#[command(name = "lineitem_benchmark")]
#[command(about = "TPC-H Lineitem CSV Sort Benchmark with policy-based optimization")]
struct Args {
    /// Input CSV file path
    #[arg(short, long)]
    input: PathBuf,

    /// Key column indices (comma-separated)
    #[arg(short = 'k', long, default_value = "8,9,13,14,15")]
    key_columns: String,

    /// Value column indices (comma-separated)  
    #[arg(short = 'v', long, default_value = "0,3,11")]
    value_columns: String,

    /// Max number of threads
    #[arg(short, long, default_value = "4")]
    threads: usize,

    /// Maximum total memory in MB
    #[arg(short, long, default_value = "1024")]
    memory_mb: usize,

    /// Use OVC encoding for keys
    #[arg(long, default_value = "false")]
    ovc: bool,

    /// Directory for temporary files
    #[arg(short, long, default_value = ".")]
    dir: PathBuf,

    /// CSV delimiter character
    #[arg(long, default_value = ",")]
    delimiter: char,

    /// CSV has headers
    #[arg(long, default_value = "true")]
    headers: bool,

    /// Verify sorted output
    #[arg(long, default_value = "false")]
    verify: bool,

    /// Number of benchmark runs per configuration
    #[arg(long, default_value = "1")]
    benchmark_runs: usize,

    /// Number of warmup runs before benchmarking
    #[arg(long, default_value = "0")]
    warmup_runs: usize,
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

// Statistics accumulator for multiple runs
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
    imbalance_sum: f64,
    imbalance_count: usize,
}

#[derive(Clone)]
struct BenchmarkResult {
    policy_name: String,
    threads: usize,
    memory_mb: usize,
    memory_str: String,
    run_size_mb: f64,
    run_gen_threads: usize,
    merge_threads: usize,
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
    imbalance_factor: f64,
    read_amplification: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Parse column indices
    let key_columns: Vec<usize> = args
        .key_columns
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let value_columns: Vec<usize> = args
        .value_columns
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    sort_lineitem(
        &args.input,
        args.threads,
        args.memory_mb,
        args.ovc,
        &args.dir,
        key_columns,
        value_columns,
        args.delimiter,
        args.headers,
        args.verify,
        args.benchmark_runs,
        args.warmup_runs,
    )?;

    Ok(())
}

fn sort_lineitem(
    input_path: &Path,
    threads: usize,
    memory_mb: usize,
    ovc: bool,
    dir: &Path,
    key_columns: Vec<usize>,
    value_columns: Vec<usize>,
    delimiter: char,
    has_headers: bool,
    verify: bool,
    benchmark_runs: usize,
    warmup_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Verify file is CSV
    if !input_path
        .to_string_lossy()
        .to_lowercase()
        .ends_with(".csv")
    {
        return Err("File must have .csv extension".into());
    }

    let file_metadata = std::fs::metadata(input_path)?;
    let file_size = file_metadata.len();

    // Get record count and create initial input for analysis
    let total_entries = 0usize; // Will be determined during first run

    // Estimate actual sorting data size (not entire file size)
    let sorting_data_mb = estimate_sorting_data_size(
        input_path,
        file_size,
        &key_columns,
        &value_columns,
        delimiter,
        has_headers,
    )?;

    // Get all policies based on actual sorting data size
    let policies = get_all_policies(SortConfig {
        memory_mb: memory_mb as f64,
        dataset_mb: sorting_data_mb,
        page_size_kb: 64.0,
        max_threads: threads as f64,
    });

    let mut all_results = Vec::new();

    println!("\n=== LINEITEM CSV BENCHMARK MODE ===");
    println!("Input file: {:?}", input_path);
    println!("Format: CSV");
    println!("File size: {}", bytes_to_human_readable(file_size as usize));
    println!(
        "Sorting data size: {} (estimated from selected columns)",
        bytes_to_human_readable((sorting_data_mb * 1024.0 * 1024.0) as usize)
    );
    println!(
        "  Selected {} of 16 columns = {:.1}% of data",
        key_columns.len() + value_columns.len(),
        ((key_columns.len() + value_columns.len()) as f64 / 16.0 * 100.0)
    );
    if total_entries > 0 {
        println!("Total entries: {}", total_entries);
    }
    println!(
        "Key columns: {:?} ({})",
        key_columns,
        format_column_names(&key_columns)
    );
    println!(
        "Value columns: {:?} ({})",
        value_columns,
        format_column_names(&value_columns)
    );
    println!("CSV delimiter: '{}'", delimiter);
    println!("CSV has headers: {}", has_headers);
    println!("Threads: {}", threads);
    println!("Memory limit: {} MB", memory_mb);
    println!("OVC enabled: {}", ovc);
    println!("Temporary directory: {:?}", dir);
    println!("Warmup runs: {}", warmup_runs);
    println!("Runs per configuration: {}", benchmark_runs);
    println!("Verify output: {}", verify);
    println!();

    // Run benchmarks for each policy
    for (policy, params) in policies {
        println!("Running benchmark for policy: {}", policy.name());
        println!("Parameters: {}", params);
        println!("{}", "=".repeat(80));

        // Perform warmup runs
        if warmup_runs > 0 {
            println!("  Performing {} warmup run(s)...", warmup_runs);
            for warmup in 1..=warmup_runs {
                print!("    Warmup {}/{}: ", warmup, warmup_runs);

                let temp_dir = dir.join(format!(
                    "warmup_{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_secs()
                ));
                std::fs::create_dir_all(&temp_dir)?;

                let input = create_input(
                    input_path,
                    &key_columns,
                    &value_columns,
                    delimiter,
                    has_headers,
                )?;

                // Ensure minimum run size of 1MB to avoid buffer issues
                let run_size_bytes =
                    ((params.run_size_mb * 1024.0 * 1024.0).max(1024.0 * 1024.0)) as usize;

                let (run_gen_stats, merge_stats) = if ovc {
                    let (runs, sketch, run_gen_stats) = ExternalSorterWithOVC::run_generation(
                        input,
                        params.run_gen_threads as usize,
                        run_size_bytes,
                        &temp_dir,
                    )?;

                    let (_merged_runs, merge_stats) = ExternalSorterWithOVC::merge(
                        runs,
                        params.merge_threads as usize,
                        sketch,
                        &temp_dir,
                    )?;
                    drop(_merged_runs); // Drop to avoid unused variable warning

                    (run_gen_stats, merge_stats)
                } else {
                    let (runs, sketch, run_gen_stats) = ExternalSorter::run_generation(
                        input,
                        params.run_gen_threads as usize,
                        run_size_bytes,
                        &temp_dir,
                    )?;

                    let (_merged_runs, merge_stats) = ExternalSorter::merge(
                        runs,
                        params.merge_threads as usize,
                        sketch,
                        &temp_dir,
                    )?;
                    drop(_merged_runs); // Drop to avoid unused variable warning

                    (run_gen_stats, merge_stats)
                };

                println!(
                    "{:.2}s",
                    run_gen_stats.time_ms as f64 / 1000.0 + merge_stats.time_ms as f64 / 1000.0
                );

                // Clean up
                std::fs::remove_dir_all(&temp_dir)?;
                unsafe {
                    let dir_fd = File::open(dir)?;
                    libc::syncfs(dir_fd.as_raw_fd());
                }
            }
            println!("  Warmup complete.\n");
        }

        // Actual benchmark runs
        let mut accumulated_stats = RunStats::default();
        let mut valid_runs = 0;
        let mut actual_entries = total_entries; // Will be updated from first run

        for run in 1..=benchmark_runs {
            print!("  Run {}/{}: ", run, benchmark_runs);

            let temp_dir = dir.join(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs()
                    .to_string(),
            );
            std::fs::create_dir_all(&temp_dir)?;

            let input = create_input(
                input_path,
                &key_columns,
                &value_columns,
                delimiter,
                has_headers,
            )?;

            // For CSV, entry count will be determined from merge stats

            // Ensure minimum run size of 1MB to avoid buffer issues
            let run_size_bytes =
                ((params.run_size_mb * 1024.0 * 1024.0).max(1024.0 * 1024.0)) as usize;

            let output = if ovc {
                let (runs, sketch, run_gen_stats) = ExternalSorterWithOVC::run_generation(
                    input,
                    params.run_gen_threads as usize,
                    run_size_bytes,
                    &temp_dir,
                )?;

                let (merged_runs, merge_stats) = ExternalSorterWithOVC::merge(
                    runs,
                    params.merge_threads as usize,
                    sketch,
                    &temp_dir,
                )?;

                let stats = SortStats {
                    num_runs: run_gen_stats.num_runs,
                    runs_info: run_gen_stats.runs_info,
                    run_generation_time_ms: Some(run_gen_stats.time_ms),
                    merge_entry_num: merge_stats.merge_entry_num,
                    merge_time_ms: Some(merge_stats.time_ms),
                    run_generation_io_stats: run_gen_stats.io_stats,
                    merge_io_stats: merge_stats.io_stats,
                };

                let output = Box::new(RunsOutputWithOVC {
                    runs: merged_runs,
                    stats: stats.clone(),
                }) as Box<dyn SortOutput>;

                output
            } else {
                let (runs, sketch, run_gen_stats) = ExternalSorter::run_generation(
                    input,
                    params.run_gen_threads as usize,
                    run_size_bytes,
                    &temp_dir,
                )?;

                let (merged_runs, merge_stats) =
                    ExternalSorter::merge(runs, params.merge_threads as usize, sketch, &temp_dir)?;

                let stats = SortStats {
                    num_runs: run_gen_stats.num_runs,
                    runs_info: run_gen_stats.runs_info,
                    run_generation_time_ms: Some(run_gen_stats.time_ms),
                    merge_entry_num: merge_stats.merge_entry_num,
                    merge_time_ms: Some(merge_stats.time_ms),
                    run_generation_io_stats: run_gen_stats.io_stats,
                    merge_io_stats: merge_stats.io_stats,
                };

                let output = Box::new(RunsOutput {
                    runs: merged_runs,
                    stats: stats.clone(),
                }) as Box<dyn SortOutput>;

                output
            };

            // Capture values before moving into SortStats
            let run_gen_time_ms = output.stats().run_generation_time_ms.unwrap();
            let merge_time_ms = output.stats().merge_time_ms.unwrap();
            let merge_entry_sum = output.stats().merge_entry_num.iter().sum::<u64>() as usize;

            println!("{}", output.stats());

            // Accumulate statistics
            accumulated_stats.total_time +=
                run_gen_time_ms as f64 / 1000.0 + merge_time_ms as f64 / 1000.0;
            accumulated_stats.runs_count += output.stats().num_runs;

            if let Some(rg_time) = output.stats().run_generation_time_ms {
                accumulated_stats.run_gen_time += rg_time as f64 / 1000.0;
            }
            if let Some(m_time) = output.stats().merge_time_ms {
                accumulated_stats.merge_time += m_time as f64 / 1000.0;
            }

            // Accumulate I/O stats
            if let Some(ref io) = output.stats().run_generation_io_stats {
                accumulated_stats.run_gen_read_ops += io.read_ops;
                accumulated_stats.run_gen_read_mb += io.read_bytes as f64 / 1_000_000.0;
                accumulated_stats.run_gen_write_ops += io.write_ops;
                accumulated_stats.run_gen_write_mb += io.write_bytes as f64 / 1_000_000.0;
            }
            if let Some(ref io) = output.stats().merge_io_stats {
                accumulated_stats.merge_read_ops += io.read_ops;
                accumulated_stats.merge_read_mb += io.read_bytes as f64 / 1_000_000.0;
                accumulated_stats.merge_write_ops += io.write_ops;
                accumulated_stats.merge_write_mb += io.write_bytes as f64 / 1_000_000.0;
            }

            // Calculate imbalance factor
            if output.stats().merge_entry_num.len() > 1 {
                let min_entries = *output.stats().merge_entry_num.iter().min().unwrap_or(&0);
                let max_entries = *output.stats().merge_entry_num.iter().max().unwrap_or(&0);
                if min_entries > 0 {
                    let imbalance = max_entries as f64 / min_entries as f64;
                    accumulated_stats.imbalance_sum += imbalance;
                    accumulated_stats.imbalance_count += 1;
                }
            } else if output.stats().merge_entry_num.len() == 1 {
                accumulated_stats.imbalance_sum += 1.0;
                accumulated_stats.imbalance_count += 1;
            }

            valid_runs += 1;
            println!(
                "{:.2}s",
                run_gen_time_ms as f64 / 1000.0 + merge_time_ms as f64 / 1000.0
            );

            // Update actual entries from merge stats if not known
            if actual_entries == 0 {
                actual_entries = merge_entry_sum;
            }

            // Verify if requested (only on first run)
            if verify && run == 1 {
                verify_sorted_output(&output, &key_columns)?;
                println!("    Verification passed!");
            } else {
                drop(output); // Drop to avoid unused variable warning
            }
            println!();

            // Clean up
            std::fs::remove_dir_all(&temp_dir)?;
            unsafe {
                let dir_fd = File::open(dir)?;
                libc::syncfs(dir_fd.as_raw_fd());
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

        // Calculate average imbalance factor
        let avg_imbalance_factor = if accumulated_stats.imbalance_count > 0 {
            accumulated_stats.imbalance_sum / accumulated_stats.imbalance_count as f64
        } else {
            1.0
        };

        // Calculate read amplification
        let read_amplification = if rg_write_mb > 0.0 {
            m_read_mb / rg_write_mb
        } else {
            1.0
        };

        all_results.push(BenchmarkResult {
            policy_name: policy.name(),
            threads,
            memory_mb,
            memory_str: bytes_to_human_readable(memory_mb * 1024 * 1024),
            run_size_mb: params.run_size_mb,
            run_gen_threads: params.run_gen_threads as usize,
            merge_threads: params.merge_threads as usize,
            runs: avg_runs,
            total_time: avg_total,
            run_gen_time: avg_run_gen,
            merge_time: avg_merge,
            entries: actual_entries,
            throughput: actual_entries as f64 / avg_total / 1_000_000.0,
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
            imbalance_factor: avg_imbalance_factor,
            read_amplification,
        });
    }

    // Print summary tables (matching gen_sort_cli format exactly)
    print_benchmark_summary(&all_results);

    Ok(())
}

fn estimate_sorting_data_size(
    file_path: &Path,
    file_size: u64,
    key_columns: &[usize],
    value_columns: &[usize],
    delimiter: char,
    has_headers: bool,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Sample the first 1000 lines to estimate column sizes
    const SAMPLE_SIZE: usize = 1000;
    let schema = get_lineitem_schema();
    let all_columns = [key_columns, value_columns].concat();

    let mut line_count = 0;
    let mut total_selected_bytes = 0;
    let mut total_line_bytes = 0;

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    for (i, line) in reader.lines().enumerate() {
        let line = line?;

        if has_headers && i == 0 {
            continue; // Skip header
        }

        if line_count >= SAMPLE_SIZE {
            break;
        }

        total_line_bytes += line.len() + 1; // +1 for newline

        let fields: Vec<&str> = line.split(delimiter).collect();

        // Calculate size of selected columns
        for &col_idx in &all_columns {
            if col_idx < fields.len() {
                // Get encoded size for this field
                let size = if col_idx < schema.fields().len() {
                    let field_type = schema.field(col_idx).data_type();
                    match field_type {
                        DataType::Utf8 => fields[col_idx].len() + 2, // Actual length + overhead
                        _ => get_datatype_size(field_type, schema.field(col_idx).name()) + 1,
                    }
                } else {
                    fields[col_idx].len() + 2 // Unknown column
                };
                total_selected_bytes += size;
            }
        }

        line_count += 1;
    }

    if line_count > 0 {
        // Estimate total rows based on average line size
        let avg_line_size = total_line_bytes as f64 / line_count as f64;
        let estimated_total_rows = (file_size as f64 / avg_line_size) as usize;

        // Average selected data per row
        let avg_selected_bytes_per_row = total_selected_bytes as f64 / line_count as f64;

        // Total estimated sorting data size
        let estimated_mb =
            (estimated_total_rows as f64 * avg_selected_bytes_per_row) / (1024.0 * 1024.0);

        Ok(estimated_mb)
    } else {
        // Fallback: use schema-based estimates
        let bytes_per_row: usize = all_columns
            .iter()
            .map(|&col_idx| {
                if col_idx < schema.fields().len() {
                    let field = schema.field(col_idx);
                    get_datatype_size(field.data_type(), field.name())
                } else {
                    8 // Default for unknown columns
                }
            })
            .sum::<usize>()
            + 4; // Add encoding overhead

        // Rough estimate: assume 6M rows per GB of CSV for lineitem
        let estimated_rows = (file_size as f64 / (1024.0 * 1024.0 * 1024.0)) * 6_000_000.0;
        Ok((estimated_rows * bytes_per_row as f64) / (1024.0 * 1024.0))
    }
}

// Unified function to get size for a data type
fn get_datatype_size(data_type: &DataType, field_name: &str) -> usize {
    match data_type {
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _) => 8,
        DataType::Int32
        | DataType::UInt32
        | DataType::Float32
        | DataType::Date32
        | DataType::Time32(_) => 4,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int8 | DataType::UInt8 | DataType::Boolean => 1,
        DataType::Utf8 | DataType::LargeUtf8 => {
            // Estimate string size based on field name for TPC-H lineitem
            match field_name {
                "l_returnflag" | "l_linestatus" => 1, // Single character flags
                "l_shipmode" => 7,                    // Short strings like "TRUCK", "RAIL"
                "l_shipinstruct" => 25,               // Medium strings
                "l_comment" => 44,                    // Longer comment field
                _ => 20,                              // Default for unknown string fields
            }
        }
        DataType::Binary | DataType::LargeBinary => 32, // Default binary size
        _ => 8,                                         // Default for other types
    }
}

fn create_input(
    path: &Path,
    key_columns: &[usize],
    value_columns: &[usize],
    delimiter: char,
    has_headers: bool,
) -> Result<Box<dyn SortInput>, Box<dyn std::error::Error>> {
    // Use the shared schema definition
    let schema = get_lineitem_schema();

    let mut config = CsvDirectConfig::new(schema);
    config.delimiter = delimiter as u8;
    config.key_columns = key_columns.to_vec();
    config.value_columns = value_columns.to_vec();
    config.has_headers = has_headers;

    Ok(Box::new(CsvInputDirect::new(
        path.to_str().unwrap(),
        config,
    )?))
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

fn print_benchmark_summary(results: &[BenchmarkResult]) {
    // Main summary table - EXACT format from gen_sort_cli
    println!("\n{}", "=".repeat(200));
    println!("Benchmark Results Summary");
    println!("{}", "=".repeat(200));
    println!(
        "{:<50} {:<8} {:<12} {:<10} {:<10} {:<10} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10} {:<12}",
        "Policy",
        "Threads",
        "Memory",
        "Run Size",
        "Gen Thr",
        "Merge Thr",
        "Runs",
        "Total (s)",
        "RunGen (s)",
        "Merge (s)",
        "Entries",
        "Throughput",
        "Read MB",
        "Write MB",
        "Imbalance"
    );
    println!(
        "{:<50} {:<8} {:<12} {:<10} {:<10} {:<10} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10} {:<12}",
        "", "", "", "(MB)", "", "", "", "", "", "", "", "(M entries/s)", "", "", "Factor"
    );
    println!("{}", "-".repeat(200));

    for result in results {
        println!(
            "{:<50} {:<8} {:<12} {:<10.1} {:<10} {:<10} {:<6} {:<10.2} {:<12.2} {:<10.2} {:<12} {:<16.2} {:<10.1} {:<10.1} {:<12}",
            result.policy_name,
            result.threads,
            result.memory_str,
            result.run_size_mb,
            result.run_gen_threads,
            result.merge_threads,
            result.runs,
            result.total_time,
            result.run_gen_time,
            result.merge_time,
            result.entries,
            result.throughput,
            result.read_mb,
            result.write_mb,
            if result.imbalance_factor > 1.0 {
                format!("{:.5}x", result.imbalance_factor)
            } else {
                "N/A".to_string()
            },
        );
    }
    println!("{}", "=".repeat(200));

    // Detailed I/O Statistics - EXACT format from gen_sort_cli
    println!("\nDetailed I/O Statistics Summary:");
    println!("{}", "-".repeat(180));
    println!(
        "{:<50} {:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
        "Policy",
        "Threads",
        "Memory",
        "Run Gen Reads",
        "Run Gen Writes",
        "Merge Reads",
        "Merge Writes",
        "Read Amp."
    );
    println!(
        "{:<50} {:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
        "", "", "", "(ops / MB)", "(ops / MB)", "(ops / MB)", "(ops / MB)", "Factor"
    );
    println!("{}", "-".repeat(180));

    for result in results {
        println!(
            "{:<50} {:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
            result.policy_name,
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
            format!("{:.3}x", result.read_amplification),
        );
    }
    println!("{}", "-".repeat(180));
}

fn verify_sorted_output(
    output: &Box<dyn SortOutput>,
    key_columns: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut prev_key: Option<Vec<u8>> = None;
    let mut count: usize = 0;

    // Store first and last 10 records with their sizes
    struct RecordInfo {
        decoded: String,
        key_size: usize,
        value_size: usize,
    }

    let mut first_records = Vec::new();
    let mut last_records = Vec::new();

    // Track total sizes for statistics
    let mut total_key_size = 0usize;
    let mut total_value_size = 0usize;

    for (key, value) in output.iter() {
        let key_size = key.len();
        let value_size = value.len();

        // Accumulate sizes
        total_key_size += key_size;
        total_value_size += value_size;

        // Store first 10 records
        if first_records.len() < 10 {
            first_records.push(RecordInfo {
                decoded: decode_key(&key, key_columns),
                key_size,
                value_size,
            });
        }

        // Keep last 10 records in a sliding window
        last_records.push(RecordInfo {
            decoded: decode_key(&key, key_columns),
            key_size,
            value_size,
        });
        if last_records.len() > 10 {
            last_records.remove(0);
        }

        // Check sort order
        if let Some(ref prev) = prev_key {
            if key < *prev {
                eprintln!("ERROR: Sort order violation at record {}", count);
                eprintln!("  Previous key: {:?}", decode_key(prev, key_columns));
                eprintln!("  Current key: {:?}", decode_key(&key, key_columns));
                return Err("Sort order violation".into());
            }
        }
        prev_key = Some(key);
        count += 1;
    }

    println!("    Verified {} records - all correctly sorted!", count);

    // Print size statistics
    println!("\n    === SIZE STATISTICS ===");
    println!("    Total records: {}", count);
    println!(
        "    Average key size: {:.1} bytes",
        total_key_size as f64 / count as f64
    );
    println!(
        "    Average value size: {:.1} bytes",
        total_value_size as f64 / count as f64
    );
    println!(
        "    Total key bytes: {} ({:.2} MB)",
        total_key_size,
        total_key_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    Total value bytes: {} ({:.2} MB)",
        total_value_size,
        total_value_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    Total sort output: {} bytes ({:.2} MB)",
        total_key_size + total_value_size,
        (total_key_size + total_value_size) as f64 / (1024.0 * 1024.0)
    );

    // Print first 10 records with sizes
    println!(
        "\n    First {} records (with sizes):",
        first_records.len().min(10)
    );
    println!(
        "    {:>5} {:>10} {:>10}  {}",
        "#", "Key Size", "Val Size", "Key Values"
    );
    println!("    {}", "-".repeat(80));
    for (i, record) in first_records.iter().enumerate() {
        println!(
            "    {:>5} {:>10} {:>10}  {}",
            i, record.key_size, record.value_size, record.decoded
        );
    }

    // Print last 10 records with sizes
    println!(
        "\n    Last {} records (with sizes):",
        last_records.len().min(10)
    );
    println!(
        "    {:>5} {:>10} {:>10}  {}",
        "#", "Key Size", "Val Size", "Key Values"
    );
    println!("    {}", "-".repeat(80));
    let start_idx = count.saturating_sub(10);
    for (i, record) in last_records.iter().enumerate() {
        println!(
            "    {:>5} {:>10} {:>10}  {}",
            start_idx + i,
            record.key_size,
            record.value_size,
            record.decoded
        );
    }

    Ok(())
}

fn decode_key(key: &[u8], key_columns: &[usize]) -> String {
    let schema = get_lineitem_schema();
    let mut parts = Vec::new();
    let mut offset = 0;

    for &col_idx in key_columns {
        if offset >= key.len() {
            break;
        }

        let (field_type, expected_size) = if col_idx < schema.fields().len() {
            let field = schema.field(col_idx);
            let type_str = match field.data_type() {
                DataType::Int64 => "Int64",
                DataType::Int32 => "Int32",
                DataType::Float64 => "Float64",
                DataType::Date32 => "Date32",
                DataType::Utf8 => "Utf8",
                _ => "Unknown",
            };
            let size = match field.data_type() {
                DataType::Int64 => 8,
                DataType::Int32 => 4,
                DataType::Float64 => 8,
                DataType::Date32 => 4,
                DataType::Utf8 => 0, // Variable length
                _ => 0,
            };
            (type_str, size)
        } else {
            ("Unknown", 0)
        };

        if expected_size > 0 {
            let field_end = offset + expected_size;
            if field_end > key.len() {
                break;
            }

            let field_bytes = &key[offset..field_end];
            let decoded = decode_bytes(field_bytes, field_type)
                .unwrap_or_else(|_| format!("{:?}", field_bytes));

            if col_idx < COLUMN_NAMES.len() {
                parts.push(format!("{}={}", COLUMN_NAMES[col_idx], decoded));
            } else {
                parts.push(format!("col{}={}", col_idx, decoded));
            }

            offset = field_end;
            if offset < key.len() && key[offset] == 0 {
                offset += 1;
            }
        } else {
            // Variable length string
            let field_end = key[offset..]
                .iter()
                .position(|&b| b == 0)
                .map(|pos| offset + pos)
                .unwrap_or(key.len());

            let field_bytes = &key[offset..field_end];
            let decoded = String::from_utf8_lossy(field_bytes);

            if col_idx < COLUMN_NAMES.len() {
                parts.push(format!("{}={}", COLUMN_NAMES[col_idx], decoded));
            } else {
                parts.push(format!("col{}={}", col_idx, decoded));
            }

            offset = field_end;
            if offset < key.len() && key[offset] == 0 {
                offset += 1;
            }
        }
    }

    parts.join(", ")
}
