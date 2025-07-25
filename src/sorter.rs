use std::os::fd::IntoRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::aligned_writer::AlignedWriter;
use crate::constants::open_file_with_direct_io;
use crate::merge::MergeIterator;
use crate::run::RunImpl;
use crate::sort_buffer::SortBufferImpl;
use crate::{
    IoStats, IoStatsTracker, Run, RunInfo, SortBuffer, SortInput, SortOutput, SortStats, Sorter,
};

/// Statistics from the run generation phase
#[derive(Clone, Debug)]
pub struct RunGenerationStats {
    pub num_runs: usize,
    pub runs_info: Vec<RunInfo>,
    pub time_ms: u128,
    pub io_stats: Option<IoStats>,
}

/// Statistics from the merge phase
#[derive(Clone, Debug)]
pub struct MergeStats {
    pub output_runs: usize,
    pub merge_entry_num: Vec<u64>,
    pub time_ms: u128,
    pub io_stats: Option<IoStats>,
}

/// Compute partition points for parallel merging
fn compute_partition_points(runs: &[RunImpl], num_partitions: usize) -> Vec<Vec<u8>> {
    if num_partitions <= 1 || runs.is_empty() {
        return vec![];
    }

    // Sample from all runs
    let mut all_samples = Vec::new();

    for run in runs {
        all_samples.extend(
            run.samples()
                .into_iter()
                .map(|(key, _file_offset, _entry_number)| key),
        );
    }

    // Remove empty keys from samples as they don't make good partition points
    all_samples.retain(|k| !k.is_empty());

    // Sort all samples
    all_samples.sort_unstable();

    // Pick partition points
    let mut partition_points = Vec::with_capacity(num_partitions - 1);

    if !all_samples.is_empty() {
        let step = all_samples.len() as f64 / num_partitions as f64;
        for i in 1..num_partitions {
            let idx = (i as f64 * step).round() as usize;
            if idx < all_samples.len() {
                partition_points.push(all_samples[idx].clone());
            }
        }
    }

    partition_points
}

// Input implementation
pub struct Input {
    pub data: Vec<(Vec<u8>, Vec<u8>)>,
}

impl SortInput for Input {
    fn create_parallel_scanners(
        &self,
        num_scanners: usize,
        _io_tracker: Option<IoStatsTracker>,
    ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>> {
        if self.data.is_empty() {
            return vec![];
        }

        let chunk_size = self.data.len().div_ceil(num_scanners);
        let mut scanners = Vec::new();

        // Clone the data for each scanner since we can't move out of &self
        let data = self.data.clone();
        for chunk in data.into_iter().collect::<Vec<_>>().chunks(chunk_size) {
            let chunk_vec = chunk.to_vec();
            scanners.push(Box::new(chunk_vec.into_iter())
                as Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>);
        }

        scanners
    }
}

// Output implementation that materializes all data
pub struct Output {
    pub data: Vec<(Vec<u8>, Vec<u8>)>,
    pub stats: Option<SortStats>,
}

impl SortOutput for Output {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        Box::new(self.data.clone().into_iter())
    }

    fn stats(&self) -> SortStats {
        self.stats.clone().unwrap_or_else(|| {
            // Default stats for in-memory output
            SortStats {
                num_runs: 1,
                runs_info: vec![RunInfo {
                    entries: self.data.len(),
                    file_size: 0, // No file for in-memory data
                }],
                run_generation_time_ms: None,
                merge_entry_num: vec![],
                merge_time_ms: None,
                run_generation_io_stats: None,
                merge_io_stats: None,
            }
        })
    }
}

// Output implementation that chains run iterators without materializing
pub struct RunsOutput {
    pub runs: Vec<RunImpl>,
    pub stats: SortStats,
}

impl SortOutput for RunsOutput {
    fn iter(&self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        // Create a chained iterator from all runs
        let mut iterators: Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>> = Vec::new();

        for run in &self.runs {
            iterators.push(run.scan_range(&[], &[]));
        }

        // Chain all iterators together
        Box::new(ChainedIterator {
            iterators,
            current: 0,
        })
    }

    fn stats(&self) -> SortStats {
        self.stats.clone()
    }
}

// Helper iterator that chains multiple iterators
struct ChainedIterator {
    iterators: Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>>,
    current: usize,
}

impl Iterator for ChainedIterator {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < self.iterators.len() {
            if let Some(item) = self.iterators[self.current].next() {
                return Some(item);
            }
            self.current += 1;
        }
        None
    }
}

// Shared directory info for cleanup coordination
pub struct TempDirInfo {
    path: PathBuf,
    should_delete: bool,
}

impl TempDirInfo {
    /// Create a new TempDirInfo with the specified path
    pub fn new(path: impl AsRef<Path>, should_delete: bool) -> Self {
        // Create the directory if it doesn't exist
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            std::fs::create_dir_all(&path).expect("Failed to create temp directory");
        }
        Self {
            path,
            should_delete,
        }
    }
}

impl Drop for TempDirInfo {
    fn drop(&mut self) {
        if self.should_delete {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

impl AsRef<Path> for TempDirInfo {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

// External sorter following sorter.rs pattern
pub struct ExternalSorter {
    run_gen_threads: usize,
    merge_threads: usize,
    max_memory: usize,
    temp_dir_info: Arc<TempDirInfo>,
}

impl ExternalSorter {
    /// Create a new ExternalSorter with temporary files in the current directory
    pub fn new(num_threads: usize, max_memory: usize) -> Self {
        Self::new_with_threads_and_dir(num_threads, num_threads, max_memory, ".")
    }

    /// Create a new ExternalSorter with temporary files in the specified directory
    /// Uses same thread count for both run generation and merge phases
    pub fn new_with_dir(
        num_threads: usize,
        max_memory: usize,
        base_dir: impl AsRef<std::path::Path>,
    ) -> Self {
        Self::new_with_threads_and_dir(num_threads, num_threads, max_memory, base_dir)
    }

    /// Create a new ExternalSorter with separate thread counts for run generation and merge
    /// The temporary directory will be automatically cleaned up when the sorter is dropped
    pub fn new_with_threads_and_dir(
        run_gen_threads: usize,
        merge_threads: usize,
        max_memory: usize,
        base_dir: impl AsRef<std::path::Path>,
    ) -> Self {
        // Use a unique temp directory for each instance to avoid conflicts
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir = base_dir.as_ref().join(format!(
            "external_sort_{}_{}",
            std::process::id(),
            timestamp
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        Self {
            run_gen_threads,
            merge_threads,
            max_memory,
            temp_dir_info: Arc::new(TempDirInfo {
                path: temp_dir,
                should_delete: true,
            }),
        }
    }

    pub fn run_generation(
        sort_input: Box<dyn SortInput>,
        num_threads: usize,
        per_thread_mem: usize,
        dir: impl AsRef<Path>,
    ) -> Result<(Vec<RunImpl>, RunGenerationStats), String> {
        // Start timing for run generation
        let run_generation_start = Instant::now();

        // Create IO tracker for run generation phase
        let run_generation_io_tracker = Arc::new(IoStatsTracker::new());

        // Create parallel scanners for input data with IO tracking
        let scanners = sort_input
            .create_parallel_scanners(num_threads, Some((*run_generation_io_tracker).clone()));

        println!(
            "Starting sort with {} parallel scanners for run generation",
            scanners.len()
        );

        if scanners.is_empty() {
            return Ok((
                vec![],
                RunGenerationStats {
                    num_runs: 0,
                    runs_info: vec![],
                    time_ms: 0,
                    io_stats: None,
                },
            ));
        }

        // Run Generation Phase (following sorter.rs pattern)
        let mut handles = vec![];

        for (thread_id, scanner) in scanners.into_iter().enumerate() {
            let io_tracker = Arc::clone(&run_generation_io_tracker);
            let dir = dir.as_ref().to_path_buf();

            let handle = thread::spawn(move || {
                let mut local_runs = Vec::new();
                let mut sort_buffer = SortBufferImpl::new(per_thread_mem);

                let run_path = dir.join(format!("intermediate_{}.dat", thread_id));
                let fd = open_file_with_direct_io(&run_path)
                    .expect("Failed to open run file with direct IO")
                    .into_raw_fd();
                let mut run_writer = Option::Some(
                    AlignedWriter::from_raw_fd_with_tracker(fd, Some((*io_tracker).clone()))
                        .expect("Failed to create run writer"),
                );

                for (key, value) in scanner {
                    if !sort_buffer.append(&key, &value) {
                        // Buffer is full, sort and flush
                        sort_buffer.sort();

                        let mut output_run = RunImpl::from_writer(run_writer.take().unwrap())
                            .expect("Failed to create run");

                        for (k, v) in sort_buffer.drain() {
                            output_run.append(k, v);
                        }

                        // Finalize the run and get writer back
                        run_writer = Some(output_run.finalize_write());

                        // Create a read-only run with proper metadata
                        local_runs.push(output_run);

                        sort_buffer.reset();
                        if !sort_buffer.append(&key, &value) {
                            panic!("Single entry too large for buffer");
                        }
                    }
                }

                // Final buffer
                if !sort_buffer.is_empty() {
                    sort_buffer.sort();

                    let mut output_run = RunImpl::from_writer(run_writer.take().unwrap())
                        .expect("Failed to create run");

                    for (k, v) in sort_buffer.drain() {
                        output_run.append(k, v);
                    }

                    // Finalize the run and get writer back
                    run_writer = Some(output_run.finalize_write());

                    // Create a read-only run with proper metadata
                    local_runs.push(output_run);
                }

                drop(run_writer); // Ensure the writer is closed

                local_runs
            });

            handles.push(handle);
        }

        // Collect runs from all threads
        let mut output_runs = Vec::new();
        for handle in handles {
            let runs = handle.join().unwrap();
            output_runs.extend(runs);
        }

        let initial_runs_count = output_runs.len();

        // Capture run generation time
        let run_generation_time = run_generation_start.elapsed();
        let run_generation_time_ms = run_generation_time.as_millis();
        println!(
            "Generated {} runs in {} ms",
            initial_runs_count, run_generation_time_ms
        );

        // Capture initial run info (before any merging)
        let initial_runs_info: Vec<RunInfo> = output_runs
            .iter()
            .map(|run| RunInfo {
                entries: run.total_entries(),
                file_size: run.total_bytes() as u64,
            })
            .collect();

        // Capture run generation IO stats
        let run_generation_io_stats = Some(run_generation_io_tracker.get_detailed_stats());

        Ok((
            output_runs,
            RunGenerationStats {
                num_runs: initial_runs_count,
                runs_info: initial_runs_info,
                time_ms: run_generation_time_ms,
                io_stats: run_generation_io_stats,
            },
        ))
    }

    pub fn merge(
        output_runs: Vec<RunImpl>,
        num_threads: usize,
        dir: impl AsRef<Path>,
    ) -> Result<(Vec<RunImpl>, MergeStats), String> {
        // If no runs or single run, return early
        if output_runs.is_empty() {
            let merge_stats = MergeStats {
                output_runs: 0,
                merge_entry_num: vec![],
                time_ms: 0,
                io_stats: None,
            };
            return Ok((vec![], merge_stats));
        }

        if output_runs.len() == 1 {
            let merge_stats = MergeStats {
                output_runs: 1,
                merge_entry_num: vec![output_runs[0].total_entries() as u64],
                time_ms: 0,
                io_stats: None,
            };
            return Ok((output_runs, merge_stats));
        }

        // Start timing for merge phase
        let merge_start = Instant::now();

        // Create IO tracker for merge phase
        let merge_io_tracker = Arc::new(IoStatsTracker::new());

        // Parallel Merge Phase for many runs
        let desired_threads = num_threads;

        // Compute partition points
        let partition_points = compute_partition_points(&output_runs, desired_threads);

        // Actual number of threads is based on partition points
        let merge_threads = if partition_points.is_empty() {
            1 // Fall back to single thread if no partition points
        } else {
            partition_points.len() + 1
        };

        println!(
            "Merging {} runs using {} threads",
            output_runs.len(),
            merge_threads
        );

        // Share runs across threads using Arc
        let runs_arc = Arc::new(output_runs);

        // Create merge tasks
        let mut merge_handles = vec![];

        for thread_id in 0..merge_threads {
            let runs = Arc::clone(&runs_arc);
            let dir = dir.as_ref().to_path_buf();
            let io_tracker = Arc::clone(&merge_io_tracker);
            // Determine the key range for this thread
            let lower_bound = if thread_id == 0 {
                vec![]
            } else {
                partition_points[thread_id - 1].clone()
            };

            let upper_bound = if thread_id < partition_points.len() {
                partition_points[thread_id].clone()
            } else {
                vec![]
            };

            let handle = thread::spawn(move || {
                // Create iterators for this key range from all runs
                let iterators: Vec<_> = runs
                    .iter()
                    .map(|run| {
                        run.scan_range_with_io_tracker(
                            &lower_bound,
                            &upper_bound,
                            Some((*io_tracker).clone()),
                        )
                    })
                    .collect();

                // Create output run for this thread
                let run_path = dir.join(format!("merge_output_{}.dat", thread_id));
                let fd = open_file_with_direct_io(&run_path)
                    .expect("Failed to open run file with direct IO")
                    .into_raw_fd();
                let writer =
                    AlignedWriter::from_raw_fd_with_tracker(fd, Some((*io_tracker).clone()))
                        .expect("Failed to create run writer");
                let mut output_run =
                    RunImpl::from_writer(writer).expect("Failed to create merge output run");

                // Merge this range directly into the output run
                let merge_iter = MergeIterator::new(iterators);

                for (key, value) in merge_iter {
                    output_run.append(key, value);
                }

                // Finalize to flush
                let writer = output_run.finalize_write();
                drop(writer);

                // Return a read-only run
                output_run
            });

            merge_handles.push(handle);
        }

        // Collect merge results as runs
        let mut output_runs = Vec::new();

        for handle in merge_handles {
            output_runs.push(handle.join().unwrap());
        }

        // Capture merge time
        let merge_time = merge_start.elapsed();
        let merge_time_ms = merge_time.as_millis();

        // Capture merge IO stats
        let merge_io_stats = Some(merge_io_tracker.get_detailed_stats());

        println!("Merge phase took {} ms", merge_time_ms);

        let merge_entry_num: Vec<u64> = output_runs
            .iter()
            .map(|run| run.total_entries() as u64)
            .collect();

        let merge_stats = MergeStats {
            output_runs: output_runs.len(),
            merge_entry_num,
            time_ms: merge_time_ms,
            io_stats: merge_io_stats,
        };

        Ok((output_runs, merge_stats))
    }
}

impl Sorter for ExternalSorter {
    /// Sort method that runs generation and merging phases
    fn sort(&mut self, sort_input: Box<dyn SortInput>) -> Result<Box<dyn SortOutput>, String> {
        // Run generation phase
        let (runs, run_gen_stats) = ExternalSorter::run_generation(
            sort_input,
            self.run_gen_threads,
            self.max_memory / self.run_gen_threads,
            self.temp_dir_info.as_ref(),
        )?;

        // Merge phase
        let (merged_runs, merge_stats) =
            ExternalSorter::merge(runs, self.merge_threads, self.temp_dir_info.as_ref())?;

        // Combine stats
        let sort_stats = SortStats {
            num_runs: run_gen_stats.num_runs,
            runs_info: run_gen_stats.runs_info,
            run_generation_time_ms: Some(run_gen_stats.time_ms),
            merge_entry_num: merge_stats.merge_entry_num,
            merge_time_ms: Some(merge_stats.time_ms),
            run_generation_io_stats: run_gen_stats.io_stats,
            merge_io_stats: merge_stats.io_stats,
        };

        // Create output
        Ok(Box::new(RunsOutput {
            runs: merged_runs,
            stats: sort_stats,
        }))
    }
}
