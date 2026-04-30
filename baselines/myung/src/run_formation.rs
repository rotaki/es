//! Run formation (Myung §3.1, Algorithm 1) — redesigned around a page-based
//! record buffer with AlphaSort-style key/pointer sorting.
//!
//! Per-thread loop:
//!   1. READ step (shared lock): pull records from the input scanner into
//!      a `PageBuffer` until memory budget is reached.
//!   2. SORT step (no lock): sort the 8-byte pointer vector by key.
//!   3. WRITE step (exclusive lock): walk sorted pointers, classify each
//!      record into a range, and append its length-prefixed bytes to the
//!      corresponding per-(run, range) file.
//!
//! Each `try_push` failure triggers a flush (one run) and a fresh buffer —
//! so a single input segment typically produces multiple runs.
//!
//! Variable-length records are supported natively via length-prefixed
//! on-page encoding. Record boundaries are self-describing on disk too, so
//! the merge reader does not need a record_size parameter.

use crate::arbitration::DeviceArbiter;
use crate::page_buffer::PageBuffer;
use crate::{MyungMetadata, RunRangeEntry};
use es::diskio::aligned_writer::AlignedWriter;
use es::diskio::file::SharedFd;
use es::{IoStatsTracker, SortInput};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct RunFormationConfig {
    pub temp_dir: PathBuf,
    pub threads: usize,
    pub memory_bytes: u64,
    pub boundary_keys: Vec<Vec<u8>>,
    pub arbiter: DeviceArbiter,
}

pub struct RunFormationResult {
    pub metadata: MyungMetadata,
    pub elapsed_ms: u128,
    pub per_thread_elapsed_ms: Vec<u128>,
}

/// Per-run metadata carried through until the merge phase.
///
/// `input` must provide `cfg.threads` scanners — each thread consumes one.
pub fn run_formation(
    input: Arc<dyn SortInput + Send + Sync>,
    cfg: RunFormationConfig,
) -> Result<RunFormationResult, String> {
    let num_ranges = cfg.boundary_keys.len() + 1;
    assert!(num_ranges >= 1);
    assert!(cfg.threads >= 1);

    std::fs::create_dir_all(&cfg.temp_dir).map_err(|e| format!("mkdir temp_dir: {}", e))?;

    // Build scanners. SortInput requires &self, so we call through the Arc.
    let scanners = input.create_parallel_scanners(cfg.threads, None::<IoStatsTracker>);
    if scanners.is_empty() {
        // Empty input — return a plausible empty result.
        return Ok(RunFormationResult {
            metadata: MyungMetadata {
                num_runs: 0,
                num_ranges: num_ranges as u32,
                record_size: 0,
                boundary_keys: cfg.boundary_keys,
                run_ranges: Vec::new(),
            },
            elapsed_ms: 0,
            per_thread_elapsed_ms: Vec::new(),
        });
    }

    // Some SortInput implementations return fewer scanners than requested
    // (e.g. tiny input). Clamp threads accordingly.
    let effective_threads = scanners.len().min(cfg.threads);
    let per_thread_mem = cfg.memory_bytes / effective_threads as u64;

    let next_run_id = Arc::new(AtomicU32::new(0));
    let boundary_keys = Arc::new(cfg.boundary_keys);
    let temp_dir = Arc::new(cfg.temp_dir);
    let arbiter = cfg.arbiter;

    let t0 = Instant::now();

    // Worker pool: one thread per scanner. We hand ownership of each iterator
    // into its worker so the closure is Send-safe.
    let results: Arc<Mutex<Vec<(Vec<RunRangeEntry>, u128)>>> = Arc::new(Mutex::new(Vec::new()));
    std::thread::scope(|s| -> Result<(), String> {
        let mut handles = Vec::with_capacity(effective_threads);
        for (tid, scanner) in scanners.into_iter().take(effective_threads).enumerate() {
            let next_run_id = next_run_id.clone();
            let boundary_keys = boundary_keys.clone();
            let temp_dir = temp_dir.clone();
            let arbiter = arbiter.clone();
            let results = results.clone();
            let mem_budget = per_thread_mem;

            let h = s.spawn(move || {
                let out = run_one_thread(
                    tid,
                    scanner,
                    mem_budget,
                    &boundary_keys,
                    &temp_dir,
                    &next_run_id,
                    &arbiter,
                );
                results.lock().unwrap().push(match out {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[myung] thread {} error: {}", tid, e);
                        (Vec::new(), 0)
                    }
                });
            });
            handles.push(h);
        }
        for h in handles {
            h.join().map_err(|_| "worker panicked".to_string())?;
        }
        Ok(())
    })?;

    let mut run_ranges = Vec::new();
    let mut per_thread_elapsed_ms = Vec::new();
    for (mut entries, elapsed) in Arc::try_unwrap(results)
        .map_err(|_| "results arc still held".to_string())?
        .into_inner()
        .map_err(|_| "results mutex poisoned".to_string())?
    {
        run_ranges.append(&mut entries);
        per_thread_elapsed_ms.push(elapsed);
    }
    let elapsed_ms = t0.elapsed().as_millis();

    let metadata = MyungMetadata {
        num_runs: next_run_id.load(Ordering::SeqCst),
        num_ranges: num_ranges as u32,
        record_size: 0, // variable length
        boundary_keys: Arc::try_unwrap(boundary_keys).unwrap_or_else(|arc| (*arc).clone()),
        run_ranges,
    };
    Ok(RunFormationResult {
        metadata,
        elapsed_ms,
        per_thread_elapsed_ms,
    })
}

fn run_one_thread(
    tid: usize,
    mut scanner: Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>,
    memory_bytes: u64,
    boundary_keys: &[Vec<u8>],
    temp_dir: &Path,
    next_run_id: &AtomicU32,
    arbiter: &DeviceArbiter,
) -> Result<(Vec<RunRangeEntry>, u128), String> {
    let t0 = Instant::now();
    let mut out = Vec::new();
    let mut buffer = PageBuffer::new(memory_bytes);

    // Peek-one lookahead so we can decide whether to flush before pushing.
    let mut pending: Option<(Vec<u8>, Vec<u8>)> = None;

    loop {
        // ---- READ STEP (shared lock) ----
        // We hold the read lock for the duration of a read burst: many
        // records from the scanner get loaded into the page buffer while
        // arbitration allows all readers concurrent access.
        let filled = {
            let _g = arbiter.read();
            fill_buffer(&mut buffer, &mut scanner, &mut pending)?
        };

        if buffer.is_empty() && !filled.had_records {
            break; // no more input, buffer empty — done
        }

        // ---- SORT STEP (no lock) ----
        buffer.sort();

        // ---- WRITE STEP (exclusive lock) ----
        {
            let _g = arbiter.write();
            let run_id = next_run_id.fetch_add(1, Ordering::SeqCst);
            flush_run_to_disk(tid, run_id, &buffer, boundary_keys, temp_dir, &mut out)?;
        }

        // Reset the buffer for the next run. Pages + pointer vec are freed.
        buffer.clear();

        if !filled.had_records && pending.is_none() {
            break;
        }
    }

    Ok((out, t0.elapsed().as_millis()))
}

struct FillOutcome {
    had_records: bool,
}

fn fill_buffer(
    buffer: &mut PageBuffer,
    scanner: &mut Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>,
    pending: &mut Option<(Vec<u8>, Vec<u8>)>,
) -> Result<FillOutcome, String> {
    let mut had_records = false;

    // Flush any record left from the previous iteration.
    if let Some((k, v)) = pending.take() {
        if buffer.try_push(&k, &v)? {
            had_records = true;
        } else {
            *pending = Some((k, v));
            return Ok(FillOutcome { had_records });
        }
    }

    while let Some((k, v)) = scanner.next() {
        had_records = true;
        if !buffer.try_push(&k, &v)? {
            // Buffer full — stash the record for the next iteration.
            *pending = Some((k, v));
            break;
        }
    }
    Ok(FillOutcome { had_records })
}

fn flush_run_to_disk(
    tid: usize,
    run_id: u32,
    buffer: &PageBuffer,
    boundary_keys: &[Vec<u8>],
    temp_dir: &Path,
    out: &mut Vec<RunRangeEntry>,
) -> Result<(), String> {
    let num_ranges = boundary_keys.len() + 1;

    // Per-range state. Writers are lazily opened on the first record that
    // falls into a range, so empty ranges never create files. `logical_bytes`
    // tracks the byte count of real record data written (excluding any 512 B
    // zero-padding that AlignedWriter adds on final flush). The merge reader
    // uses this to stop before the padding.
    let mut writers: Vec<Option<AlignedWriter>> = (0..num_ranges).map(|_| None).collect();
    let mut paths: Vec<PathBuf> = (0..num_ranges)
        .map(|range_id| temp_dir.join(format!("run_{:06}_range_{:03}.bin", run_id, range_id)))
        .collect();
    let mut logical_bytes: Vec<u64> = vec![0; num_ranges];
    let mut record_counts: Vec<u64> = vec![0; num_ranges];

    // Two-pointer merge between the sorted pointer vector and the sorted
    // boundary keys. Both buffer.refs() and boundary_keys are ascending, so
    // we advance `cur_range` only when the current key crosses the next
    // boundary — O(n + b) comparisons total instead of O(n log b) per-record
    // binary searches. Matches the paper's L7 description ("the records are
    // scanned in the thread local buffer and compared to the global
    // boundary key", §3.1.3).
    let mut cur_range: usize = 0;
    let num_boundaries = boundary_keys.len();
    for r in buffer.refs() {
        let k = buffer.key_at(r);
        // boundary_keys[i] is the exclusive lower bound for range i+1.
        while cur_range < num_boundaries && k >= boundary_keys[cur_range].as_slice() {
            cur_range += 1;
        }
        let bytes = buffer.raw_record_bytes(r);

        let writer = match &mut writers[cur_range] {
            Some(w) => w,
            None => {
                // SharedFd opens with O_DIRECT via `open_file_with_direct_io`.
                let fd = Arc::new(
                    SharedFd::new_from_path(&paths[cur_range], false)
                        .map_err(|e| format!("tid {} open {:?}: {}", tid, paths[cur_range], e))?,
                );
                let w = AlignedWriter::from_fd(fd)
                    .map_err(|e| format!("tid {} aligned writer: {}", tid, e))?;
                writers[cur_range] = Some(w);
                writers[cur_range].as_mut().unwrap()
            }
        };
        writer
            .write_all(bytes)
            .map_err(|e| format!("tid {} write: {}", tid, e))?;
        logical_bytes[cur_range] += bytes.len() as u64;
        record_counts[cur_range] += 1;
    }

    // Flush each open writer (Drop also flushes as a safety net). Dropping
    // releases the FD; the 512-B zero padding written at final flush is
    // harmless because the merge reader respects `num_bytes`.
    for (range_id, slot) in writers.iter_mut().enumerate() {
        if let Some(mut w) = slot.take() {
            w.flush()
                .map_err(|e| format!("tid {} final flush: {}", tid, e))?;
            drop(w);
            out.push(RunRangeEntry {
                run_id,
                range_id: range_id as u32,
                path: paths[range_id].clone(),
                num_records: record_counts[range_id],
                num_bytes: logical_bytes[range_id],
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::classify;

    /// The flush loop's linear advance through `boundary_keys` must produce
    /// the same range assignment as the per-record binary search `classify()`,
    /// because the records are sorted ascending by the time we reach flush.
    #[test]
    fn linear_walk_matches_classify_on_sorted_buffer() {
        // 1 MiB budget: comfortably fits one page + a dozen pointers.
        let mut buf = PageBuffer::new(1024 * 1024);
        // Insert records in random order so the sort path actually runs.
        let unsorted: &[&[u8]] = &[
            b"mango", b"apple", b"zebra", b"cherry", b"banana", b"date", b"fig", b"grape",
            b"lemon", b"orange", b"papaya", b"quince",
        ];
        for k in unsorted {
            assert!(buf.try_push(k, b"v").unwrap());
        }
        buf.sort();

        // Boundaries chosen to put records into ranges 0..3 (3 boundaries → 4 ranges).
        let boundary_keys: Vec<Vec<u8>> = vec![
            b"cherry".to_vec(), // exclusive lower bound for range 1
            b"lemon".to_vec(),  //                          range 2
            b"papaya".to_vec(), //                          range 3
        ];
        let num_boundaries = boundary_keys.len();

        // Walk the sorted refs the same way flush_run_to_disk does.
        let mut cur_range: usize = 0;
        let mut got: Vec<usize> = Vec::new();
        for r in buf.refs() {
            let k = buf.key_at(r);
            while cur_range < num_boundaries && k >= boundary_keys[cur_range].as_slice() {
                cur_range += 1;
            }
            got.push(cur_range);
        }

        // Reference: per-record binary search classify().
        let want: Vec<usize> = buf
            .refs()
            .iter()
            .map(|r| classify(buf.key_at(r), &boundary_keys))
            .collect();

        assert_eq!(got, want, "linear walk diverged from classify()");
        // Sanity: at least one record in each of the 4 ranges.
        for expected_range in 0..=3 {
            assert!(
                got.contains(&expected_range),
                "no records in range {}",
                expected_range
            );
        }
    }

    #[test]
    fn linear_walk_handles_keys_equal_to_boundary() {
        // Boundary semantics: key == boundary[i] goes to range i+1
        // (boundary[i] is exclusive lower bound for range i+1).
        let mut buf = PageBuffer::new(1024 * 1024);
        for k in [b"a", b"b", b"c", b"d", b"e"] {
            assert!(buf.try_push(k, b"v").unwrap());
        }
        buf.sort();
        let boundary_keys = vec![b"b".to_vec(), b"d".to_vec()]; // 3 ranges
        let mut cur_range: usize = 0;
        let mut got = Vec::new();
        for r in buf.refs() {
            let k = buf.key_at(r);
            while cur_range < boundary_keys.len() && k >= boundary_keys[cur_range].as_slice() {
                cur_range += 1;
            }
            got.push(cur_range);
        }
        // a < b → 0; b == b → 1; c → 1; d == d → 2; e → 2
        assert_eq!(got, vec![0, 1, 1, 2, 2]);
    }
}
