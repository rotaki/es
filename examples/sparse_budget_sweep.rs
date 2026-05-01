//! Offline sweep: how does the sparse-index budget percentage affect partition
//! quality under skewed workloads?
//!
//! Build & run:
//!
//! ```bash
//! cargo run --release --features internal-bench --example sparse_budget_sweep -- \
//!     --data-mb 256 --num-runs 8 --num-partitions 14 --trials 3 \
//!     --workloads freq_key,heavy_key,heavy_range \
//!     --partition-types count,size \
//!     --budget-pcts 0.5,1,2,3,4,5 \
//!     > sweep.csv
//! ```
//!
//! What this does
//! --------------
//!
//! - Synthesizes K sorted runs of skewed key/value records (no I/O).
//! - Builds real `SparseIndex`es by driving the production `should_sample` /
//!   `record_sample` API at a configured byte-stride (size mode) or
//!   record-stride (count mode) budget.
//! - Calls the production `select_boundary_by_count` / `select_boundary_by_size`
//!   on the union of those sparse indexes via `MultiSparseIndexes`.
//! - Walks the in-memory record stream with the resulting boundaries to compute
//!   exact per-thread byte and record counts for each partition.
//! - Emits CSV: imbalance metrics, sample density, observed gap.
//!
//! Why the offline harness
//! -----------------------
//!
//! This bypasses run generation, replacement selection, and disk I/O entirely
//! while still exercising the actual production boundary-search and
//! sparse-index-sampling code paths. A complete sweep across
//! `(workload x partition_type x budget_pct x trial)` runs in seconds, so we
//! can measure many points cheaply.
//!
//! Workload defaults match `docs/kvbin_generators.md` shapes:
//! - `freq_key`:    50% of rows have `key=heavy_key`, all 512 B values.
//! - `heavy_key`:   10% of rows have `key=heavy_key` with 32 KiB values, the
//!                  rest get 128 B values.
//! - `heavy_range`: 20% of rows are uniform in `[0, 65535)` with 32 KiB values,
//!                  the rest are uniform in `[65535, u64::MAX)` with 128 B.
//!
//! Memory ratio
//! ------------
//!
//! `--memory-mb` defaults to `--data-mb / 25.0`, matching the production 50 GiB
//! data : 2 GiB memory ratio. With this default, a 1% budget here gives the
//! same `S / total_records` ratio as 1% in production.

use std::collections::BTreeMap;
use std::time::Instant;

use clap::Parser;
use es::internal_bench::{
    IndexingInterval, KeyRunIdOffsetBound, MultiSparseIndexes, SparseIndex,
    build_multi_sparse_indexes, select_boundary_by_count, select_boundary_by_size,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Parser, Debug)]
#[command(
    name = "sparse_budget_sweep",
    about = "Offline sweep of sparse-index budget % vs partition quality"
)]
struct Args {
    /// Total simulated data per (workload, trial), in MiB.
    #[arg(long, default_value_t = 256)]
    data_mb: usize,

    /// Simulated total memory budget that "%" applies to (MiB).
    /// Default: data_mb / 25.0 (production-like 50 GiB / 2 GiB ratio).
    #[arg(long)]
    memory_mb: Option<f64>,

    /// Comma-separated budget percentages to sweep.
    #[arg(long, default_value = "0.5,1,2,3,4,5")]
    budget_pcts: String,

    /// Comma-separated workloads to run. Choices: freq_key, heavy_key, heavy_range.
    #[arg(long, default_value = "freq_key,heavy_key,heavy_range")]
    workloads: String,

    /// Comma-separated partition types. Choices: count, size.
    #[arg(long, default_value = "count,size")]
    partition_types: String,

    /// Number of simulated input runs per trial.
    #[arg(long, default_value_t = 8)]
    num_runs: usize,

    /// Number of merge partitions (= merge threads).
    #[arg(long, default_value_t = 14)]
    num_partitions: usize,

    /// Trials per (workload, partition_type, budget_pct) cell.
    #[arg(long, default_value_t = 3)]
    trials: usize,

    /// Base RNG seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Clone, Copy, Debug)]
enum WorkloadKind {
    FreqKey,
    HeavyKey,
    HeavyRange,
}

impl WorkloadKind {
    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "freq_key" => Some(Self::FreqKey),
            "heavy_key" => Some(Self::HeavyKey),
            "heavy_range" => Some(Self::HeavyRange),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::FreqKey => "freq_key",
            Self::HeavyKey => "heavy_key",
            Self::HeavyRange => "heavy_range",
        }
    }

    /// Approximate average record byte size (key+value+headers), used both for
    /// sizing the run and for the dynamic-stride bootstrap.
    fn avg_record_bytes(self) -> f64 {
        match self {
            // key=8 + value=512 + 4+4 headers
            Self::FreqKey => (4 + 4 + 8 + 512) as f64,
            // 0.10 * (4+4+8+32768) + 0.90 * (4+4+8+128)
            Self::HeavyKey => 0.10 * (16.0 + 32768.0) + 0.90 * (16.0 + 128.0),
            // 0.20 * (4+4+8+32768) + 0.80 * (4+4+8+128)
            Self::HeavyRange => 0.20 * (16.0 + 32768.0) + 0.80 * (16.0 + 128.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PartType {
    Count,
    Size,
}

impl PartType {
    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "count" => Some(Self::Count),
            "size" => Some(Self::Size),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Size => "size",
        }
    }
}

fn parse_csv_list<T>(s: &str, parse: impl Fn(&str) -> Option<T>) -> Vec<T> {
    s.split(',')
        .filter_map(|tok| {
            let t = tok.trim();
            if t.is_empty() { None } else { parse(t) }
        })
        .collect()
}

/// Generate a sorted run of `num` records for the given workload. Records are
/// `(key_bytes, value_len)` pairs; the actual value bytes are not materialized
/// (we only need lengths for sparse-index sampling).
fn gen_records(workload: WorkloadKind, num: usize, rng: &mut StdRng) -> Vec<(Vec<u8>, usize)> {
    let mut recs: Vec<(Vec<u8>, usize)> = Vec::with_capacity(num);
    match workload {
        WorkloadKind::FreqKey => {
            let heavy = num / 2;
            for _ in 0..heavy {
                recs.push((0u64.to_be_bytes().to_vec(), 512));
            }
            for _ in heavy..num {
                let k: u64 = loop {
                    let v: u64 = rng.random();
                    if v != 0 {
                        break v;
                    }
                };
                recs.push((k.to_be_bytes().to_vec(), 512));
            }
        }
        WorkloadKind::HeavyKey => {
            let heavy = num / 10;
            for _ in 0..heavy {
                recs.push((0u64.to_be_bytes().to_vec(), 32 * 1024));
            }
            for _ in heavy..num {
                let k: u64 = loop {
                    let v: u64 = rng.random();
                    if v != 0 {
                        break v;
                    }
                };
                recs.push((k.to_be_bytes().to_vec(), 128));
            }
        }
        WorkloadKind::HeavyRange => {
            let hot = num / 5;
            for _ in 0..hot {
                let k: u64 = rng.random_range(0..65536);
                recs.push((k.to_be_bytes().to_vec(), 32 * 1024));
            }
            for _ in hot..num {
                let k: u64 = rng.random_range(65536..u64::MAX);
                recs.push((k.to_be_bytes().to_vec(), 128));
            }
        }
    }
    recs.sort_by(|a, b| a.0.cmp(&b.0));
    recs
}

/// Drive `should_sample` / `record_sample` on a sorted record stream exactly
/// the way `Run::append` does in production.
struct PerRunState {
    sparse_index: SparseIndex,
    total_entries: usize,
    total_bytes: usize,
    /// Largest gap between consecutive sample file_offsets. (For size mode this
    /// is the worst-case per-boundary byte error contribution.)
    worst_byte_gap: usize,
    /// Largest gap between consecutive sample record indices. (For count mode
    /// this is the worst-case per-boundary record error contribution.)
    worst_record_gap: usize,
}

fn build_run_index(
    records: &[(Vec<u8>, usize)],
    part_type: PartType,
    budget_bytes: usize,
    estimated_run_bytes: usize,
    avg_record_bytes_hint: f64,
) -> PerRunState {
    let avg_key_bytes_hint = 8.0; // every workload here uses 8-byte keys

    // The fallback stride is only used if `dynamic_stride()` cannot derive a
    // value (e.g., no budget set). We pre-bootstrap below so dynamic_stride
    // works from the very first sample, but we still pick a sane fallback.
    let initial_stride = match part_type {
        PartType::Size => 1024,
        PartType::Count => 1,
    };
    let interval = match part_type {
        PartType::Size => IndexingInterval::bytes(initial_stride),
        PartType::Count => IndexingInterval::records(initial_stride),
    }
    .with_budget_bytes(budget_bytes)
    .with_estimated_total_data_bytes(estimated_run_bytes);

    let mut sparse_index = SparseIndex::new(interval);
    // Match production's RS-bootstrap behavior: feed avg key/record sizes so
    // the first dynamic stride is computed correctly.
    sparse_index.set_bootstrap(
        avg_key_bytes_hint,
        avg_record_bytes_hint,
        records.len().max(1),
    );

    let mut total_entries: usize = 0;
    let mut total_bytes: usize = 0;
    let mut last_sample_offset: Option<usize> = None;
    let mut last_sample_index: Option<usize> = None;
    let mut worst_byte_gap: usize = 0;
    let mut worst_record_gap: usize = 0;

    // Mirror Run::append exactly:
    //   1. Save should_sample, record_index, record_offset (pre-write state).
    //   2. (Skip actual write — no I/O.)
    //   3. Bump totals.
    //   4. observe_record (post-write).
    //   5. record_sample (post-write, post-observe).
    for (key, value_len) in records.iter() {
        // 4 bytes key_len + 4 bytes value_len + key + value
        let record_bytes = 4 + 4 + key.len() + *value_len;
        let record_offset = total_bytes;
        let record_index = total_entries;
        let should_sample = sparse_index.should_sample(total_entries, total_bytes);

        total_entries += 1;
        total_bytes += record_bytes;

        sparse_index.observe_record(key.len(), record_bytes);
        if should_sample {
            sparse_index.record_sample(key, record_offset, record_index, record_offset);
            if let Some(prev) = last_sample_offset {
                let gap = record_offset.saturating_sub(prev);
                if gap > worst_byte_gap {
                    worst_byte_gap = gap;
                }
            }
            if let Some(prev) = last_sample_index {
                let gap = record_index.saturating_sub(prev);
                if gap > worst_record_gap {
                    worst_record_gap = gap;
                }
            }
            last_sample_offset = Some(record_offset);
            last_sample_index = Some(record_index);
        }
    }

    PerRunState {
        sparse_index,
        total_entries,
        total_bytes,
        worst_byte_gap,
        worst_record_gap,
    }
}

/// Number of internal bounds the record `(key, run_id, offset)` is `>= ` to.
/// This equals the partition index in `[0, num_partitions)` because internal
/// bounds are sorted ascending.
fn assign_partition(
    bounds: &[KeyRunIdOffsetBound],
    key: &[u8],
    run_id: u32,
    offset: usize,
) -> usize {
    bounds.iter().filter(|b| b.ge(key, run_id, offset)).count()
}

#[derive(Default, Debug, Clone, Copy)]
struct ImbalanceStats {
    min: u64,
    max: u64,
    target: u64,
    /// max / target (1.0 = perfect)
    ratio: f64,
    /// max - target, signed-ish; reported as a fraction of target
    max_dev_pct: f64,
}

fn imbalance(values: &[u64]) -> ImbalanceStats {
    if values.is_empty() {
        return ImbalanceStats::default();
    }
    let sum: u128 = values.iter().map(|v| *v as u128).sum();
    let target = (sum / values.len() as u128) as u64;
    let min = *values.iter().min().unwrap();
    let max = *values.iter().max().unwrap();
    let ratio = if target == 0 {
        0.0
    } else {
        max as f64 / target as f64
    };
    let max_dev_pct = if target == 0 {
        0.0
    } else {
        ((max as f64) - (target as f64)) / (target as f64) * 100.0
    };
    ImbalanceStats {
        min,
        max,
        target,
        ratio,
        max_dev_pct,
    }
}

#[derive(Debug, Clone)]
struct TrialRecord {
    workload: &'static str,
    part_type: &'static str,
    budget_pct: f64,
    trial: usize,
    num_runs: usize,
    num_partitions: usize,
    total_entries: u64,
    total_bytes: u64,
    total_samples: u64,
    samples_per_run_min: u64,
    samples_per_run_max: u64,
    samples_per_run_avg: f64,
    expected_samples: f64,
    worst_byte_gap: u64,
    worst_record_gap: u64,
    expected_byte_stride: f64,
    expected_record_stride: f64,
    boundary_search_ms: f64,
    count_min: u64,
    count_max: u64,
    count_target: u64,
    count_imbalance: f64,
    count_max_dev_pct: f64,
    byte_min: u64,
    byte_max: u64,
    byte_target: u64,
    byte_imbalance: f64,
    byte_max_dev_pct: f64,
}

fn run_one_trial(
    workload: WorkloadKind,
    part_type: PartType,
    num_runs: usize,
    num_partitions: usize,
    rows_per_run: usize,
    budget_bytes_per_run: usize,
    estimated_data_bytes_per_run: usize,
    seed: u64,
) -> TrialRecord {
    let avg_rec = workload.avg_record_bytes();

    // Generate runs (sorted records) and build their sparse indexes.
    let mut record_runs: Vec<Vec<(Vec<u8>, usize)>> = Vec::with_capacity(num_runs);
    let mut run_states: Vec<PerRunState> = Vec::with_capacity(num_runs);
    for r in 0..num_runs {
        let run_seed = seed
            .wrapping_mul(1009)
            .wrapping_add(r as u64)
            .wrapping_add(workload as u64 * 7919)
            .wrapping_add(part_type as u64 * 31);
        let mut rng = StdRng::seed_from_u64(run_seed);
        let records = gen_records(workload, rows_per_run, &mut rng);
        let state = build_run_index(
            &records,
            part_type,
            budget_bytes_per_run,
            estimated_data_bytes_per_run,
            avg_rec,
        );
        record_runs.push(records);
        run_states.push(state);
    }

    let total_bytes: u64 = run_states.iter().map(|s| s.total_bytes as u64).sum();
    let total_entries: u64 = run_states.iter().map(|s| s.total_entries as u64).sum();
    let total_samples: u64 = run_states.iter().map(|s| s.sparse_index.len() as u64).sum();
    let samples_per_run_min = run_states
        .iter()
        .map(|s| s.sparse_index.len() as u64)
        .min()
        .unwrap_or(0);
    let samples_per_run_max = run_states
        .iter()
        .map(|s| s.sparse_index.len() as u64)
        .max()
        .unwrap_or(0);
    let samples_per_run_avg = if num_runs > 0 {
        total_samples as f64 / num_runs as f64
    } else {
        0.0
    };
    let worst_byte_gap = run_states
        .iter()
        .map(|s| s.worst_byte_gap as u64)
        .max()
        .unwrap_or(0);
    let worst_record_gap = run_states
        .iter()
        .map(|s| s.worst_record_gap as u64)
        .max()
        .unwrap_or(0);

    // Build one MultiSparseIndexes per input run, mirroring how
    // `merge_once_with_hooks` builds `Vec<MultiSparseIndexes>` from
    // `runs.iter().map(|run| sparse_indexes(run))`.
    let mut multi_indexes: Vec<MultiSparseIndexes<'_>> = Vec::with_capacity(num_runs);
    for (run_id, state) in run_states.iter().enumerate() {
        let one = [(run_id as u32, &state.sparse_index, state.total_bytes)];
        multi_indexes.push(build_multi_sparse_indexes(&one));
    }

    // Compute (N-1) internal boundaries at i/N of the chosen objective,
    // matching the production targets at imbalance_factor=1.0.
    let bs_start = Instant::now();
    let mut bounds: Vec<KeyRunIdOffsetBound> = Vec::new();
    for i in 1..num_partitions {
        let bound = match part_type {
            PartType::Size => {
                let target_bytes =
                    ((total_bytes as u128) * (i as u128) / (num_partitions as u128)) as usize;
                select_boundary_by_size(&multi_indexes, target_bytes)
            }
            PartType::Count => {
                // Count mode targets sparse-entry count, not record count
                // (matches engine.rs `partition_entry_target_at`).
                let target_entries =
                    ((total_samples as u128) * (i as u128) / (num_partitions as u128)) as usize;
                select_boundary_by_count(&multi_indexes, target_entries)
            }
        };
        if let Some(b) = bound {
            bounds.push(b);
        }
    }
    let boundary_search_ms = bs_start.elapsed().as_secs_f64() * 1000.0;

    // Walk the actual record stream and assign each to a partition.
    let mut counts = vec![0u64; num_partitions];
    let mut bytes = vec![0u64; num_partitions];
    for (run_id, records) in record_runs.iter().enumerate() {
        let mut offset: usize = 0;
        for (key, value_len) in records.iter() {
            let rec_bytes = 4 + 4 + key.len() + *value_len;
            let part = assign_partition(&bounds, key, run_id as u32, offset);
            // Defensive clamp: if for any reason `part >= num_partitions`,
            // pin to the last partition instead of panicking.
            let part = part.min(num_partitions - 1);
            counts[part] += 1;
            bytes[part] += rec_bytes as u64;
            offset += rec_bytes;
        }
    }

    let cnt_stats = imbalance(&counts);
    let byte_stats = imbalance(&bytes);

    let expected_samples = if budget_bytes_per_run > 0 {
        // (budget per run / avg entry bytes) * num_runs
        let entry_bytes = 14.0 + 8.0; // header (file_offset 8 + key_len 2 + pos handle 4) + 8B key
        (budget_bytes_per_run as f64 / entry_bytes) * num_runs as f64
    } else {
        0.0
    };
    let expected_byte_stride = if total_samples > 0 {
        total_bytes as f64 / total_samples as f64
    } else {
        f64::INFINITY
    };
    let expected_record_stride = if total_samples > 0 {
        total_entries as f64 / total_samples as f64
    } else {
        f64::INFINITY
    };

    TrialRecord {
        workload: workload.name(),
        part_type: part_type.name(),
        budget_pct: 0.0, // set by caller
        trial: 0,        // set by caller
        num_runs,
        num_partitions,
        total_entries,
        total_bytes,
        total_samples,
        samples_per_run_min,
        samples_per_run_max,
        samples_per_run_avg,
        expected_samples,
        worst_byte_gap,
        worst_record_gap,
        expected_byte_stride,
        expected_record_stride,
        boundary_search_ms,
        count_min: cnt_stats.min,
        count_max: cnt_stats.max,
        count_target: cnt_stats.target,
        count_imbalance: cnt_stats.ratio,
        count_max_dev_pct: cnt_stats.max_dev_pct,
        byte_min: byte_stats.min,
        byte_max: byte_stats.max,
        byte_target: byte_stats.target,
        byte_imbalance: byte_stats.ratio,
        byte_max_dev_pct: byte_stats.max_dev_pct,
    }
}

fn print_csv_header() {
    println!(
        "workload,partition_type,budget_pct,trial,num_runs,num_partitions,\
total_entries,total_bytes,total_samples,\
samples_per_run_min,samples_per_run_max,samples_per_run_avg,expected_samples,\
worst_byte_gap,worst_record_gap,expected_byte_stride,expected_record_stride,\
boundary_search_ms,\
count_min,count_max,count_target,count_imbalance,count_max_dev_pct,\
byte_min,byte_max,byte_target,byte_imbalance,byte_max_dev_pct"
    );
}

fn print_csv_row(t: &TrialRecord) {
    println!(
        "{},{},{:.4},{},{},{},{},{},{},{},{},{:.2},{:.2},{},{},{:.2},{:.2},{:.3},{},{},{},{:.6},{:.6},{},{},{},{:.6},{:.6}",
        t.workload,
        t.part_type,
        t.budget_pct,
        t.trial,
        t.num_runs,
        t.num_partitions,
        t.total_entries,
        t.total_bytes,
        t.total_samples,
        t.samples_per_run_min,
        t.samples_per_run_max,
        t.samples_per_run_avg,
        t.expected_samples,
        t.worst_byte_gap,
        t.worst_record_gap,
        t.expected_byte_stride,
        t.expected_record_stride,
        t.boundary_search_ms,
        t.count_min,
        t.count_max,
        t.count_target,
        t.count_imbalance,
        t.count_max_dev_pct,
        t.byte_min,
        t.byte_max,
        t.byte_target,
        t.byte_imbalance,
        t.byte_max_dev_pct,
    );
}

fn main() {
    let args = Args::parse();

    let workloads = parse_csv_list(&args.workloads, |s| WorkloadKind::parse(s));
    if workloads.is_empty() {
        eprintln!(
            "ERROR: no valid workloads in --workloads (got {:?})",
            args.workloads
        );
        std::process::exit(2);
    }
    let part_types = parse_csv_list(&args.partition_types, |s| PartType::parse(s));
    if part_types.is_empty() {
        eprintln!(
            "ERROR: no valid partition types in --partition-types (got {:?})",
            args.partition_types
        );
        std::process::exit(2);
    }
    let budget_pcts: Vec<f64> = parse_csv_list(&args.budget_pcts, |s| s.parse::<f64>().ok());
    if budget_pcts.is_empty() {
        eprintln!(
            "ERROR: no valid percentages in --budget-pcts (got {:?})",
            args.budget_pcts
        );
        std::process::exit(2);
    }

    let data_bytes = (args.data_mb as u128) * 1024 * 1024;
    let memory_mb = args.memory_mb.unwrap_or(args.data_mb as f64 / 25.0);
    let memory_bytes = (memory_mb * 1024.0 * 1024.0) as u128;

    eprintln!(
        "[sweep] data={} MiB, memory={:.2} MiB ({:.1}:1 ratio), runs={}, partitions={}, trials={}",
        args.data_mb,
        memory_mb,
        args.data_mb as f64 / memory_mb.max(f64::EPSILON),
        args.num_runs,
        args.num_partitions,
        args.trials,
    );

    print_csv_header();

    // Cache row counts per workload (since data_mb is shared across cells, the
    // record count for each workload is fixed; the random keys vary by trial).
    let mut rows_per_workload: BTreeMap<&'static str, usize> = BTreeMap::new();
    for w in &workloads {
        let avg_rec = w.avg_record_bytes();
        let total_rows = (data_bytes as f64 / avg_rec) as usize;
        let rows_per_run = (total_rows / args.num_runs).max(1);
        rows_per_workload.insert(w.name(), rows_per_run);
        eprintln!(
            "[sweep] {} rows/run={} (total ~{} rows, avg_rec={:.0} B)",
            w.name(),
            rows_per_run,
            rows_per_run * args.num_runs,
            avg_rec
        );
    }

    let total_cells = workloads.len() * part_types.len() * budget_pcts.len() * args.trials;
    let mut cell_idx = 0usize;

    for &workload in &workloads {
        let rows_per_run = *rows_per_workload.get(workload.name()).unwrap();
        let estimated_run_bytes = (rows_per_run as f64 * workload.avg_record_bytes()) as usize;
        for &part_type in &part_types {
            for &pct in &budget_pcts {
                let total_budget_bytes = (memory_bytes as f64 * pct / 100.0) as usize;
                let budget_per_run = (total_budget_bytes / args.num_runs).max(1);
                for trial in 0..args.trials {
                    cell_idx += 1;
                    eprintln!(
                        "[sweep] [{}/{}] {} {} pct={} trial={} budget/run={} B est_run_bytes={} B",
                        cell_idx,
                        total_cells,
                        workload.name(),
                        part_type.name(),
                        pct,
                        trial,
                        budget_per_run,
                        estimated_run_bytes,
                    );
                    let trial_seed = args.seed.wrapping_add((cell_idx as u64).wrapping_mul(13));
                    let mut rec = run_one_trial(
                        workload,
                        part_type,
                        args.num_runs,
                        args.num_partitions,
                        rows_per_run,
                        budget_per_run,
                        estimated_run_bytes,
                        trial_seed,
                    );
                    rec.budget_pct = pct;
                    rec.trial = trial;
                    print_csv_row(&rec);
                }
            }
        }
    }

    eprintln!("[sweep] done.");
}
