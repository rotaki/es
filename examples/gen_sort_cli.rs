use clap::Parser;
use es::sort_policy::{SortConfig, get_all_policies};
use es::{ExternalSorter, GenSortInputDirect};
use es::{RunsOutput, SortOutput, SortStats};
use std::path::{Path, PathBuf};

#[derive(Parser)]
struct SortArgs {
    /// Input GenSort file path
    #[arg(short, long)]
    input: PathBuf,

    /// Max number of threads
    #[arg(short, long, default_value = "4")]
    threads: usize,

    /// Maximum total memory
    #[arg(short, long, default_value = "1024")]
    memory_mb: usize,

    /// Directory for temporary files
    #[arg(short, long, default_value = ".")]
    dir: PathBuf,

    /// Verify sorted output
    #[arg(short, long)]
    verify: bool,

    /// Number of benchmark runs per configuration
    #[arg(long, default_value = "1")]
    benchmark_runs: usize,

    /// Number of warmup runs before benchmarking (not included in results)
    #[arg(long, default_value = "0")]
    warmup_runs: usize,
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
    imbalance_sum: f64,
    imbalance_count: usize,
}

#[derive(Clone)]
struct BenchmarkResult {
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
    let args = SortArgs::parse();
    sort_gensort(
        &args.input,
        args.threads,
        args.memory_mb,
        &args.dir,
        args.verify,
        args.benchmark_runs,
        args.warmup_runs,
    )?;

    Ok(())
}

fn sort_gensort(
    input: &Path,
    threads: usize,
    memory_mb: usize,
    dir: &Path,
    verify: bool,
    benchmark_runs: usize,
    warmup_runs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get record count
    let gensort_input = GenSortInputDirect::new(input)?;
    let file_size = gensort_input.file_size()?;
    let total_entries = gensort_input.len();

    let policies = get_all_policies(SortConfig {
        memory_mb: memory_mb as f64,
        dataset_mb: file_size as f64 / (1024.0 * 1024.0),
        page_size_kb: 64.0,
        max_threads: threads as f64,
    });

    let mut all_results = Vec::new();

    println!("\n=== GENSORT BENCHMARK MODE ===");
    println!("Input file: {:?}", input);
    println!("File size: {}", bytes_to_human_readable(file_size as usize));
    println!("Total entries: {}", total_entries);
    println!("Warmup runs: {}", warmup_runs);
    println!("Runs per configuration: {}", benchmark_runs);
    println!("Verify output: {}", verify);
    println!();

    // Run benchmarks for each configuration
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

                let gensort_input = Box::new(GenSortInputDirect::new(input)?);

                let (runs, run_gen_stats) = ExternalSorter::run_generation(
                    gensort_input.clone(),
                    params.run_gen_threads as usize,
                    (params.run_size_mb * 1024.0 * 1024.0) as usize,
                    &temp_dir,
                )?;

                let (_merged_runs, merge_stats) =
                    ExternalSorter::merge(runs, params.merge_threads as usize, &temp_dir)?;

                println!(
                    "{:.2}s",
                    run_gen_stats.time_ms as f64 / 1000.0 + merge_stats.time_ms as f64 / 1000.0
                );

                // Remove temporary directory
                std::fs::remove_dir_all(&temp_dir)?;
            }
            println!("  Warmup complete.\n");
        }

        let mut accumulated_stats = RunStats::default();
        let mut valid_runs = 0;

        for run in 1..=benchmark_runs {
            print!("  Run {}/{}: ", run, benchmark_runs);

            let temp_dir = dir.join(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs()
                    .to_string(),
            );
            std::fs::create_dir_all(&temp_dir)?;

            let gensort_input = Box::new(GenSortInputDirect::new(input)?);

            let (runs, run_gen_stats) = ExternalSorter::run_generation(
                gensort_input.clone(),
                params.run_gen_threads as usize,
                (params.run_size_mb * 1024.0 * 1024.0) as usize,
                &temp_dir,
            )?;

            let (merged_runs, merge_stats) =
                ExternalSorter::merge(runs, params.merge_threads as usize, &temp_dir)?;

            let stats = SortStats {
                num_runs: run_gen_stats.num_runs,
                runs_info: run_gen_stats.runs_info,
                run_generation_time_ms: Some(run_gen_stats.time_ms),
                merge_entry_num: merge_stats.merge_entry_num,
                merge_time_ms: Some(merge_stats.time_ms),
                run_generation_io_stats: run_gen_stats.io_stats,
                merge_io_stats: merge_stats.io_stats,
            };

            println!("{}", stats);

            // Accumulate stats
            accumulated_stats.total_time +=
                run_gen_stats.time_ms as f64 / 1000.0 + merge_stats.time_ms as f64 / 1000.0;
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

            // Calculate imbalance factor
            if stats.merge_entry_num.len() > 1 {
                // Multiple partitions - calculate actual imbalance
                let min_entries = *stats.merge_entry_num.iter().min().unwrap_or(&0);
                let max_entries = *stats.merge_entry_num.iter().max().unwrap_or(&0);
                if min_entries > 0 {
                    let imbalance = max_entries as f64 / min_entries as f64;
                    accumulated_stats.imbalance_sum += imbalance;
                    accumulated_stats.imbalance_count += 1;
                }
            } else if stats.merge_entry_num.len() == 1 {
                // Single partition - perfect balance
                accumulated_stats.imbalance_sum += 1.0;
                accumulated_stats.imbalance_count += 1;
            }

            valid_runs += 1;
            println!(
                "{:.2}s",
                run_gen_stats.time_ms as f64 / 1000.0 + merge_stats.time_ms as f64 / 1000.0
            );

            // Verify if requested (only on first run to save time)
            if verify && run == 1 {
                println!("    Verifying sorted output...");
                let output = Box::new(RunsOutput {
                    runs: merged_runs,
                    stats: stats.clone(),
                }) as Box<dyn SortOutput>;
                verify_sorted_output(true, &output)?;
                println!("    Verification passed!");
            }
            println!();

            // Remove temporary directory
            std::fs::remove_dir_all(&temp_dir)?;
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
        println!("Imbalance sum: {}", accumulated_stats.imbalance_sum);
        println!("Imbalance count: {}", accumulated_stats.imbalance_count);
        let avg_imbalance_factor = if accumulated_stats.imbalance_count > 0 {
            accumulated_stats.imbalance_sum / accumulated_stats.imbalance_count as f64
        } else {
            1.0 // Perfect balance or single partition
        };
        println!("Average imbalance factor: {:.2}x", avg_imbalance_factor);

        // Calculate read amplification
        let read_amplification = if rg_write_mb > 0.0 {
            m_read_mb / rg_write_mb
        } else {
            1.0
        };

        all_results.push(BenchmarkResult {
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
            imbalance_factor: avg_imbalance_factor,
            read_amplification,
        });
    }

    // Print summary table
    print_benchmark_summary(&all_results);

    Ok(())
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

fn print_benchmark_summary(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(160));
    println!("Benchmark Results Summary");
    println!("{}", "=".repeat(160));
    println!(
        "{:<8} {:<12} {:<10} {:<10} {:<10} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10} {:<12}",
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
        "{:<8} {:<12} {:<10} {:<10} {:<10} {:<6} {:<10} {:<12} {:<10} {:<12} {:<16} {:<10} {:<10} {:<12}",
        "", "", "(MB)", "", "", "", "", "", "", "", "(M entries/s)", "", "", "Factor"
    );
    println!("{}", "-".repeat(160));

    for result in results {
        println!(
            "{:<8} {:<12} {:<10.1} {:<10} {:<10} {:<6} {:<10.2} {:<12.2} {:<10.2} {:<12} {:<16.2} {:<10.1} {:<10.1} {:<12.5}",
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
    println!("{}", "=".repeat(152));

    // Print detailed I/O statistics
    println!("\nDetailed I/O Statistics Summary:");
    println!("{}", "-".repeat(140));
    println!(
        "{:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
        "Threads",
        "Memory",
        "Run Gen Reads",
        "Run Gen Writes",
        "Merge Reads",
        "Merge Writes",
        "Read Amplif."
    );
    println!(
        "{:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
        "", "", "(ops / MB)", "(ops / MB)", "(ops / MB)", "(ops / MB)", "Factor"
    );
    println!("{}", "-".repeat(140));

    for result in results {
        println!(
            "{:<8} {:<12} {:<20} {:<20} {:<20} {:<20} {:<15}",
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
    println!("{}", "-".repeat(140));
}

fn verify_sorted_output(
    print_few_entries: bool,
    output: &Box<dyn es::SortOutput>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut prev_key: Option<Vec<u8>> = None;
    let mut count = 0;

    let mut first_records = Vec::new();
    let mut last_records = Vec::new();

    for (key, _value) in output.iter() {
        if print_few_entries {
            if first_records.len() < 5 {
                first_records.push(key.clone());
            }
            last_records.push(key.clone());
            if last_records.len() > 5 {
                last_records.remove(0); // Keep only last 5
            }
        }

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
    if print_few_entries {
        println!("\nFirst 5 records:");
        for (i, key) in first_records.iter().enumerate() {
            println!("  Record {}: Key = {:?}", i, key);
        }

        println!("\nLast 5 records:");
        for (i, key) in last_records.iter().enumerate() {
            println!("  Record {}: Key = {:?}", count - 5 + i, key);
        }
    }
    Ok(())
}
