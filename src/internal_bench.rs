//! Visibility-only re-exports for offline benchmarking of internal
//! sparse-index and partitioning machinery.
//!
//! This module is **only** compiled when the `internal-bench` feature is
//! enabled. It contains no production logic; it merely makes a small set of
//! types and functions reachable from out-of-crate consumers (specifically
//! `examples/sparse_budget_sweep.rs`).
//!
//! Design notes:
//! - All re-exports forward to existing items. No behavior is altered.
//! - The boundary-selection routines (`select_boundary_by_count`,
//!   `select_boundary_by_size`) are wrapped via `engine::bench_api`; the
//!   originals remain private to `engine.rs`.
//! - `MultiSparseIndexes` cannot be constructed externally because
//!   `SparseIndexSegment` is private. `build_multi_sparse_indexes` provides a
//!   feature-gated constructor that takes plain
//!   `(run_id, &SparseIndex, total_bytes)` tuples.

pub use crate::sort::core::engine::bench_api::{select_boundary_by_count, select_boundary_by_size};
pub use crate::sort::core::run_format::bench_api::build_multi_sparse_indexes;
pub use crate::sort::core::run_format::{
    IndexingInterval, KeyRunIdOffsetBound, MultiSparseIndexes, SparseIndexRef, cmp_key_run_offset,
};
pub use crate::sort::core::sparse_index::SparseIndex;
