//! Myung et al. (IEEE TC 2021) NVMe external sort — faithful reimplementation
//! used as a baseline for CrocSort evaluation.
//!
//! This crate deliberately avoids importing CrocSort's algorithmic wins
//! (replacement selection, sparse indexes, OVC, tree-of-losers). Only
//! device-level primitives (Direct I/O, GenSort record parsing, I/O stats)
//! are shared through the parent `es` crate.

pub mod arbitration;
pub mod merge;
pub mod page_buffer;
pub mod planner;
pub mod run_formation;
pub mod sampler;
pub mod winner_tree;

/// Sort record (key, payload) pair. Fixed-width for this baseline to match
/// Myung §4.1's 128-byte GenSort format (16B key + 112B value) — variable-
/// length support would require a slab allocator the paper does not describe.
pub type Record = (Vec<u8>, Vec<u8>);

/// Metadata persisted between run formation and merge so merge threads know
/// where each range-partition of each run begins on disk.
#[derive(Clone, Debug)]
pub struct RunRangeEntry {
    pub run_id: u32,
    pub range_id: u32,
    /// Path to the per-range run file.
    pub path: std::path::PathBuf,
    /// Number of records.
    pub num_records: u64,
    /// Bytes (for fixed-width records this is num_records * record_size).
    pub num_bytes: u64,
}

/// Global metadata handed from run formation to merge.
#[derive(Clone, Debug)]
pub struct MyungMetadata {
    pub num_runs: u32,
    pub num_ranges: u32,
    pub record_size: usize,
    /// Range boundary keys: ranges[i] = [boundary_keys[i-1], boundary_keys[i]).
    /// Length is num_ranges - 1. Range 0 is (-inf, boundary_keys[0]),
    /// last range is [boundary_keys.last(), +inf).
    pub boundary_keys: Vec<Vec<u8>>,
    /// Per-(run, range) file info.
    pub run_ranges: Vec<RunRangeEntry>,
}
