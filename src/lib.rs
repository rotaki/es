// Simple External Sort Library

#![allow(clippy::needless_range_loop)]

// Core traits from sorter.rs
pub trait Sorter {
    fn sort(&mut self, sort_input: Box<dyn SortInput>) -> Result<Box<dyn SortOutput>, String>;
}

pub trait SortBuffer {
    fn is_empty(&self) -> bool;
    fn has_space(&self, key: &[u8], value: &[u8]) -> bool;
    fn append(&mut self, key: Vec<u8>, value: Vec<u8>) -> bool;
    fn sort(&mut self);
    fn drain(&mut self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>;
    fn reset(&mut self);
}

pub trait Run {
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

impl std::fmt::Display for SortStats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Clean display implementation
        writeln!(f, "SortStats:")?;
        writeln!(f, "  Number of runs: {}", self.num_runs)?;
        writeln!(
            f,
            "  (R) time: {} ms",
            self.run_generation_time_ms.unwrap_or(0)
        )?;
        writeln!(f, "  (M) time: {} ms", self.merge_time_ms.unwrap_or(0))?;
        if let Some(io_stats) = &self.run_generation_io_stats {
            writeln!(f, "  (R) I/O stats: {}", io_stats)?;
        }
        if let Some(io_stats) = &self.merge_io_stats {
            writeln!(f, "  (M) I/O stats: {}", io_stats)?;
        }

        // Display read amplification (sparse indexing effectiveness)
        if let (Some(run_gen_io), Some(merge_io)) =
            (&self.run_generation_io_stats, &self.merge_io_stats)
        {
            if run_gen_io.write_bytes > 0 {
                let read_amplification = merge_io.read_bytes as f64 / run_gen_io.write_bytes as f64;
                let excess_read_pct = (read_amplification - 1.0) * 100.0;

                writeln!(f, "  Read amplification:")?;
                writeln!(
                    f,
                    "    Run generation writes: {} bytes ({:.2} GB)",
                    run_gen_io.write_bytes,
                    run_gen_io.write_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )?;
                writeln!(
                    f,
                    "    Merge phase reads: {} bytes ({:.2} GB)",
                    merge_io.read_bytes,
                    merge_io.read_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )?;
                writeln!(
                    f,
                    "    Read amplification factor: {:.2}x",
                    read_amplification
                )?;
                if excess_read_pct > 0.0 {
                    writeln!(
                        f,
                        "    Excess reads: {:.1}% (sparse indexing overhead)",
                        excess_read_pct
                    )?;
                } else if excess_read_pct < 0.0 {
                    writeln!(
                        f,
                        "    Read reduction: {:.1}% (possible data compression or skipping)",
                        -excess_read_pct
                    )?;
                } else {
                    writeln!(f, "    Perfect read efficiency (1.0x)")?;
                }
            }
        }

        // Display partition imbalance if merge was parallelized
        if self.merge_entry_num.len() > 1 {
            writeln!(f, "  Partition imbalance:")?;

            let total_entries: u64 = self.merge_entry_num.iter().sum();
            let avg_entries = total_entries as f64 / self.merge_entry_num.len() as f64;
            let min_entries = *self.merge_entry_num.iter().min().unwrap_or(&0);
            let max_entries = *self.merge_entry_num.iter().max().unwrap_or(&0);

            // Calculate standard deviation
            let variance = self
                .merge_entry_num
                .iter()
                .map(|&x| {
                    let diff = x as f64 - avg_entries;
                    diff * diff
                })
                .sum::<f64>()
                / self.merge_entry_num.len() as f64;
            let std_dev = variance.sqrt();

            // Calculate coefficient of variation (CV) as a percentage
            let cv = if avg_entries > 0.0 {
                (std_dev / avg_entries) * 100.0
            } else {
                0.0
            };

            // Calculate imbalance factor (max/min)
            let imbalance_factor = if min_entries > 0 {
                max_entries as f64 / min_entries as f64
            } else {
                0.0
            };

            writeln!(f, "    Partitions: {}", self.merge_entry_num.len())?;
            writeln!(f, "    Total entries: {}", total_entries)?;
            writeln!(f, "    Avg per partition: {:.0}", avg_entries)?;
            writeln!(
                f,
                "    Min entries: {} ({:.1}% of avg)",
                min_entries,
                if avg_entries > 0.0 {
                    (min_entries as f64 / avg_entries) * 100.0
                } else {
                    0.0
                }
            )?;
            writeln!(
                f,
                "    Max entries: {} ({:.1}% of avg)",
                max_entries,
                if avg_entries > 0.0 {
                    (max_entries as f64 / avg_entries) * 100.0
                } else {
                    0.0
                }
            )?;
            writeln!(f, "    Std deviation: {:.0}", std_dev)?;
            writeln!(f, "    Coefficient of variation: {:.1}%", cv)?;
            writeln!(
                f,
                "    Imbalance factor (max/min): {:.2}x",
                imbalance_factor
            )?;

            // Show distribution if not too many partitions
            if self.merge_entry_num.len() <= 32 {
                writeln!(f, "    Distribution: {:?}", self.merge_entry_num)?;
            }
        }

        Ok(())
    }
}

/// Information about a single run
#[derive(Clone, Debug)]
pub struct RunInfo {
    pub entries: usize,
    pub file_size: u64,
}

impl std::fmt::Display for RunInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "entries={}, file_size={}", self.entries, self.file_size)
    }
}

// Implementations
pub mod aligned_buffer;
pub mod aligned_reader;
pub mod aligned_writer;
pub mod constants;
pub mod csv_input_direct;
pub mod file;
pub mod gensort_input_direct;
pub mod io_stats;
pub mod kll;
pub mod merge;
pub mod order_preserving_encoding;
pub mod ovc;
pub mod parquet_input_direct;
pub mod rand;
pub mod run;
pub mod sort_buffer;
pub mod sort_policy;
pub mod sorter;

// Export the main types
pub use aligned_reader::{AlignedChunkReader, AlignedReader};
pub use aligned_writer::AlignedWriter;
pub use csv_input_direct::{CsvDirectConfig, CsvInputDirect};
pub use file::{file_size_fd, pread_fd, pwrite_fd}; // , GlobalFileManager};
pub use gensort_input_direct::GenSortInputDirect;
pub use io_stats::{IoStats, IoStatsTracker};
pub use parquet_input_direct::{ParquetDirectConfig, ParquetInputDirect};
pub use run::RunImpl;
pub use sorter::{
    ExternalSorter, Input, MergeStats, Output, RunGenerationStats, RunsOutput, TempDirInfo,
};
