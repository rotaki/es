//! Standalone CLI for the Myung baseline. Mirrors `examples/gen_sort_cli.rs`.
//!
//! Input variant is selected by `--format`:
//!   gensort : 10 B key + 90 B payload fixed — uses es::GenSortInputDirect
//! Record parsing is pluggable via the `SortInput` trait, so adding CSV or
//! KVBin later is a small CLI change — the sort path already supports
//! variable-length records through the length-prefixed page buffer.

use clap::{Parser, ValueEnum};
use es::input_reader::kvbin_input_direct::KvBinInputDirect;
use es::{GenSortInputDirect, SortInput};
use myung::arbitration::DeviceArbiter;
use myung::merge::{MergeConfig, merge};
use myung::planner::MyungPlan;
use myung::run_formation::{RunFormationConfig, run_formation};
use myung::sampler::{derive_range_boundaries, derive_range_boundaries_kvbin};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const GENSORT_KEY_LEN: usize = 10;
const GENSORT_RECORD_SIZE: usize = 100;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum InputFormat {
    Gensort,
    Kvbin,
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Myung et al. (IEEE TC 2021) external sort baseline"
)]
struct Args {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    dir: PathBuf,

    #[arg(long, default_value = "/tmp/myung_output.bin")]
    output: PathBuf,

    #[arg(long, default_value_t = 16)]
    threads: usize,

    #[arg(long, default_value_t = 2048.0)]
    memory_mb: f64,

    #[arg(long)]
    num_ranges: Option<usize>,

    #[arg(long, default_value_t = 64 * 1024)]
    interleave_floor_bytes: u64,

    #[arg(long, default_value_t = 0.001)]
    sample_ratio: f64,

    #[arg(long, default_value_t = false)]
    no_arbitration: bool,

    #[arg(long, default_value_t = false)]
    discard_final_output: bool,

    #[arg(long, default_value = "gensort")]
    format: InputFormat,

    /// KVBin sidecar index path (required when --format kvbin).
    #[arg(long)]
    idx: Option<PathBuf>,

    /// For KVBin sampler: number of records to read at each sampled anchor.
    #[arg(long, default_value_t = 8)]
    sample_records_per_anchor: usize,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let file_size = std::fs::metadata(&args.input)
        .map_err(|e| format!("stat input: {}", e))?
        .len();
    let memory_bytes = (args.memory_mb * (1024.0 * 1024.0)) as u64;
    // Sample reserve = sample_ratio × file_size. Bounds the upfront sample
    // buffer so per-thread budgets account for it (Vec<Vec<u8>> overhead is
    // ignored — a 0.1% sample of 200 GiB is ~200 MiB, dwarfs key vec overhead).
    let sample_reserve_bytes = ((file_size as f64) * args.sample_ratio).ceil() as u64;
    let plan = MyungPlan::new(
        file_size,
        memory_bytes,
        args.threads,
        args.interleave_floor_bytes,
        sample_reserve_bytes,
    );
    plan.log_summary();
    let num_ranges = args.num_ranges.unwrap_or(plan.threads).max(1);

    // Build the SortInput implementation for the chosen format.
    let input: Arc<dyn SortInput + Send + Sync> = match args.format {
        InputFormat::Gensort => Arc::new(GenSortInputDirect::new(&args.input)?),
        InputFormat::Kvbin => {
            let idx = args
                .idx
                .as_ref()
                .ok_or_else(|| "--idx is required for --format kvbin".to_string())?;
            Arc::new(KvBinInputDirect::new(&args.input, idx)?)
        }
    };

    // Boundary sampling.
    eprintln!(
        "[myung] sampling {:.3}% of input...",
        args.sample_ratio * 100.0
    );
    let t_sample = Instant::now();
    let boundary_keys = match args.format {
        InputFormat::Gensort => derive_range_boundaries(
            &args.input,
            GENSORT_RECORD_SIZE,
            GENSORT_KEY_LEN,
            num_ranges,
            args.sample_ratio,
        )?,
        InputFormat::Kvbin => {
            let idx = args
                .idx
                .as_ref()
                .ok_or_else(|| "--idx is required for --format kvbin".to_string())?;
            derive_range_boundaries_kvbin(
                &args.input,
                idx,
                num_ranges,
                args.sample_ratio,
                args.sample_records_per_anchor,
            )?
        }
    };
    eprintln!(
        "[myung] sampled {} boundaries in {} ms",
        boundary_keys.len(),
        t_sample.elapsed().as_millis()
    );

    let arbiter = DeviceArbiter::new(!args.no_arbitration);
    let rf = run_formation(
        input.clone(),
        RunFormationConfig {
            temp_dir: args.dir.clone(),
            threads: plan.threads,
            memory_bytes: plan.usable_memory_bytes(),
            boundary_keys: boundary_keys.clone(),
            arbiter: arbiter.clone(),
        },
    )?;
    eprintln!(
        "[myung] run formation: {} ms, {} runs, arbiter={}",
        rf.elapsed_ms,
        rf.metadata.num_runs,
        arbiter.enabled(),
    );

    // Eq. 2 per-run read block: (M / T_merge) / R, rounded up to 512 B and
    // floored at C. Using the actual number of runs produced (rf.metadata.num_runs)
    // rather than the planner's pre-run estimate.
    let t_merge = plan.threads as u64;
    let runs = rf.metadata.num_runs.max(1) as u64;
    let raw = ((plan.usable_memory_bytes() / t_merge) / runs) as usize;
    let floor = args.interleave_floor_bytes as usize;
    let read_block_bytes = raw.max(floor);
    eprintln!(
        "[myung] merge read block = {} bytes (Eq. 2 (M/T)/R = {}, C floor = {})",
        read_block_bytes, raw, floor
    );

    let mg = merge(MergeConfig {
        metadata: Arc::new(rf.metadata.clone()),
        threads: plan.threads,
        output_path: args.output.clone(),
        write_output: !args.discard_final_output,
        read_block_bytes,
    })?;
    eprintln!(
        "[myung] merge: {} ms, {} records emitted",
        mg.elapsed_ms, mg.records_emitted
    );

    let total_ms = rf.elapsed_ms + mg.elapsed_ms;
    println!(
        "myung_sort: D={} bytes, M={} MiB, T(user)={}, T(eff)={}, runs={}, ranges={}, total={} ms (rf={}, merge={})",
        file_size,
        args.memory_mb as u64,
        plan.user_threads,
        plan.threads,
        rf.metadata.num_runs,
        num_ranges,
        total_ms,
        rf.elapsed_ms,
        mg.elapsed_ms,
    );
    Ok(())
}
