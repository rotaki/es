// Simple External Sort Library

#![allow(clippy::needless_range_loop)]

// Core traits from sorter.rs
pub trait Sorter {
    fn sort(&mut self, sort_input: Box<dyn SortInput>) -> Result<Box<dyn SortOutput>, String>;
}

pub trait SortBuffer {
    fn is_empty(&self) -> bool;
    fn append(&mut self, key: &[u8], value: &[u8]) -> bool;
    fn sort(&mut self);
    fn drain(&mut self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>;
    fn reset(&mut self);
}

pub trait Run {
    fn samples(&self) -> Vec<(Vec<u8>, usize, usize)>; // Key, file offset, entry number
    fn append(&mut self, key: Vec<u8>, value: Vec<u8>);
    fn scan_range(
        &self,
        lower_inc: &[u8],
        upper_exc: &[u8],
    ) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>;
    fn scan_range_with_io_tracker(
        &self,
        lower_inc: &[u8],
        upper_exc: &[u8],
        io_tracker: Option<IoStatsTracker>,
    ) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>;
}

// Input/Output traits
pub trait SortInput {
    fn create_parallel_scanners(
        &self,
        num_scanners: usize,
        io_tracker: Option<IoStatsTracker>,
    ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>>;
}

pub trait SortOutput {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>;

    /// Get statistics about the sort operation
    fn stats(&self) -> SortStats {
        // Default implementation returns unknown stats
        SortStats {
            num_runs: 0,
            runs_info: vec![],
            run_generation_time_ms: None,
            merge_entry_num: vec![],
            merge_time_ms: None,
            run_generation_io_stats: None,
            merge_io_stats: None,
        }
    }
}

/// Statistics about a sort operation
#[derive(Clone, Debug)]
pub struct SortStats {
    pub num_runs: usize,
    pub runs_info: Vec<RunInfo>,
    pub run_generation_time_ms: Option<u128>,
    pub merge_entry_num: Vec<u64>,
    pub merge_time_ms: Option<u128>,
    pub run_generation_io_stats: Option<IoStats>,
    pub merge_io_stats: Option<IoStats>,
}

/// Information about a single run
#[derive(Clone, Debug)]
pub struct RunInfo {
    pub entries: usize,
    pub file_size: u64,
}

// Implementations
pub mod aligned_buffer;
pub mod aligned_reader;
pub mod aligned_writer;
pub mod constants;
pub mod csv_input_direct;
pub mod global_file_manager;
pub mod io_stats;
pub mod merge;
pub mod order_preserving_encoding;
pub mod parquet_input_direct;
pub mod run;
pub mod sort_buffer;
pub mod sorter;

// Export the main types
pub use aligned_reader::{AlignedChunkReader, AlignedReader};
pub use aligned_writer::AlignedWriter;
pub use csv_input_direct::{CsvDirectConfig, CsvInputDirect};
pub use global_file_manager::{file_size_fd, pread_fd, pwrite_fd, GlobalFileManager};
pub use io_stats::{IoStats, IoStatsTracker};
pub use parquet_input_direct::{ParquetDirectConfig, ParquetInputDirect};
pub use sorter::{ExternalSorter, Input, Output};
