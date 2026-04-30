//! Merge (Myung §3.2, Algorithm 2).
//!
//! Each merge thread handles one (or more) ranges and merges across all
//! per-range run files for that range. Implicit shuffling at run-formation
//! time already separated records into per-range files, so merge threads
//! never seek within a run file — just linear scans.
//!
//! Records on disk are length-prefixed: [u32 klen][u32 vlen][key][value].
//! The merge data structure is a winner tree (basic tournament tree),
//! matching Alg. 2's `initialize_tournament(run)` / `pq.pop()` / `pq.push()`.
//! Stability tie-break is achieved structurally: run files are opened in
//! ascending run_id order, and `WinnerTree::compare_pick` breaks ties to the
//! smaller leaf index — so a smaller run_id wins on key tie, matching the
//! paper's stable_priority_queue behaviour.
//!
//! Output buffer per thread: 2 MiB (Myung §3.2 MDTS).

use crate::MyungMetadata;
use crate::winner_tree::{NIL_LEAF, WinnerTree};
use es::diskio::aligned_reader::AlignedReader;
use es::diskio::aligned_writer::AlignedWriter;
use es::diskio::file::SharedFd;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Size of the merge output buffer (Myung §3.2: matches NVMe MDTS).
pub const MERGE_OUTPUT_BUF_BYTES: usize = 2 * 1024 * 1024;
/// Direct-I/O alignment (matches `es::diskio::constants::DIRECT_IO_ALIGNMENT`).
const DIO_ALIGN: usize = 512;

pub struct MergeConfig {
    pub metadata: Arc<MyungMetadata>,
    pub threads: usize,
    /// Output directory — one `merged_t###_r###.bin` file emitted per
    /// (thread, range). Concatenating those in (range, thread) order yields
    /// the globally sorted stream.
    pub output_path: PathBuf,
    pub write_output: bool,
    /// Per-run read block size in bytes. Myung Eq. 2 says this should be
    /// `(M / T_merge) / R` and must be ≥ `C` (the NVMe interleaving floor).
    /// Rounded up to a 512 B multiple by the caller.
    pub read_block_bytes: usize,
}

pub struct MergeResult {
    pub elapsed_ms: u128,
    pub per_thread_elapsed_ms: Vec<u128>,
    pub records_emitted: u64,
}

pub fn merge(cfg: MergeConfig) -> Result<MergeResult, String> {
    let num_ranges = cfg.metadata.num_ranges as usize;
    let threads = cfg.threads.min(num_ranges).max(1);

    let mut thread_ranges: Vec<Vec<u32>> = vec![Vec::new(); threads];
    for r in 0..num_ranges {
        thread_ranges[r % threads].push(r as u32);
    }

    let output_root = cfg.output_path.clone();
    if cfg.write_output {
        std::fs::create_dir_all(&output_root)
            .map_err(|e| format!("create output dir {:?}: {}", output_root, e))?;
    }

    let metadata = cfg.metadata.clone();
    let write_output = cfg.write_output;
    let read_block = align_up_to_dio(cfg.read_block_bytes.max(DIO_ALIGN));

    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(threads);
    for (tid, ranges) in thread_ranges.into_iter().enumerate() {
        let metadata = metadata.clone();
        let output_root = output_root.clone();
        let h = std::thread::spawn(move || -> Result<(u128, u64), String> {
            merge_one_thread(
                tid,
                &ranges,
                &metadata,
                &output_root,
                write_output,
                read_block,
            )
        });
        handles.push(h);
    }

    let mut per_thread_elapsed_ms = Vec::with_capacity(threads);
    let mut records_emitted = 0u64;
    for h in handles {
        let (elapsed, n) = h
            .join()
            .map_err(|_| "merge thread panicked".to_string())??;
        per_thread_elapsed_ms.push(elapsed);
        records_emitted += n;
    }

    Ok(MergeResult {
        elapsed_ms: t0.elapsed().as_millis(),
        per_thread_elapsed_ms,
        records_emitted,
    })
}

fn merge_one_thread(
    tid: usize,
    ranges: &[u32],
    metadata: &MyungMetadata,
    output_root: &std::path::Path,
    write_output: bool,
    read_block_bytes: usize,
) -> Result<(u128, u64), String> {
    let t0 = Instant::now();
    let mut total = 0u64;
    for &range_id in ranges {
        total += merge_one_range(
            tid,
            range_id,
            metadata,
            output_root,
            write_output,
            read_block_bytes,
        )?;
    }
    Ok((t0.elapsed().as_millis(), total))
}

fn align_up_to_dio(n: usize) -> usize {
    (n + DIO_ALIGN - 1) & !(DIO_ALIGN - 1)
}

/// Reader over a length-prefixed record stream, backed by Direct I/O via
/// `AlignedReader`. Bounded by `max_bytes` — the logical (unpadded) size of
/// the run file, as recorded in `RunRangeEntry::num_bytes`. Direct-I/O-
/// written intermediate files have up to 511 B of trailing zero padding;
/// stopping at the logical boundary avoids reading padding as records.
///
/// Per-run buffer size is derived from Myung Eq. 2: `(M / T_merge) / R`,
/// rounded up to a 512 B multiple, with a hard floor of C (the NVMe
/// interleaving threshold). This is the "read block size" §3.2 talks about.
struct RunReader {
    reader: AlignedReader,
    /// Last record bytes (8-byte header + key + value). Length grows on
    /// demand to fit each variable-length record.
    buf: Vec<u8>,
    /// Length of the current key inside `buf` (after the 8-byte header).
    key_len: u32,
    /// Whether `buf` currently holds a valid record.
    has_data: bool,
    /// Logical byte budget — stop when `bytes_read >= max_bytes`.
    max_bytes: u64,
    bytes_read: u64,
}

impl RunReader {
    fn open(path: &std::path::Path, max_bytes: u64, block_bytes: usize) -> Result<Self, String> {
        let fd = Arc::new(
            SharedFd::new_from_path(path, false).map_err(|e| format!("open {:?}: {}", path, e))?,
        );
        let reader = AlignedReader::from_fd_with_buffer_size(fd, block_bytes, 0, None)
            .map_err(|e| format!("aligned reader {:?}: {}", path, e))?;
        Ok(RunReader {
            reader,
            buf: Vec::new(),
            key_len: 0,
            has_data: false,
            max_bytes,
            bytes_read: 0,
        })
    }

    /// Advance to the next record. Returns `Ok(true)` if populated, `Ok(false)` on EOF.
    fn advance(&mut self) -> Result<bool, String> {
        if self.bytes_read >= self.max_bytes {
            self.has_data = false;
            return Ok(false);
        }
        let mut hdr = [0u8; 8];
        match self.reader.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.has_data = false;
                return Ok(false);
            }
            Err(e) => return Err(format!("read header: {}", e)),
        }
        let klen = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        let vlen = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        let total = 8usize + klen as usize + vlen as usize;
        self.buf.resize(total, 0);
        self.buf[..8].copy_from_slice(&hdr);
        self.reader
            .read_exact(&mut self.buf[8..total])
            .map_err(|e| format!("read body: {}", e))?;
        self.key_len = klen;
        self.has_data = true;
        self.bytes_read += total as u64;
        Ok(true)
    }

    fn current_key(&self) -> &[u8] {
        debug_assert!(self.has_data);
        &self.buf[8..8 + self.key_len as usize]
    }

    fn current_record_bytes(&self) -> &[u8] {
        debug_assert!(self.has_data);
        &self.buf
    }
}

fn merge_one_range(
    tid: usize,
    range_id: u32,
    metadata: &MyungMetadata,
    output_root: &std::path::Path,
    write_output: bool,
    read_block_bytes: usize,
) -> Result<u64, String> {
    // Open run files in ascending run_id order. Leaf index in the winner
    // tree therefore equals run-id rank, so the tree's tie-break-to-smaller-
    // leaf-index rule gives us the paper's stability semantics for free.
    let mut files: Vec<&crate::RunRangeEntry> = metadata
        .run_ranges
        .iter()
        .filter(|e| e.range_id == range_id)
        .collect();
    files.sort_by_key(|e| e.run_id);
    if files.is_empty() {
        return Ok(0);
    }

    let mut readers: Vec<RunReader> = Vec::with_capacity(files.len());
    for f in &files {
        let mut r = RunReader::open(&f.path, f.num_bytes, read_block_bytes)?;
        r.advance()?; // pre-load first record (or mark exhausted)
        readers.push(r);
    }

    let mut tree = WinnerTree::new(readers.len());
    tree.build(|leaf| {
        let r = &readers[leaf as usize];
        if r.has_data {
            Some(r.current_key())
        } else {
            None
        }
    });

    // Output: Direct I/O via AlignedWriter with a 2 MiB (MDTS-sized) internal
    // buffer — matches Myung §3.2's "2 MB for each merge buffer, which
    // corresponds to max data transfer size (MDTS) of the NVMe SSD".
    let output_path = output_root.join(format!("merged_t{:03}_r{:03}.bin", tid, range_id));
    let mut writer: Option<AlignedWriter> = if write_output {
        let fd = Arc::new(
            SharedFd::new_from_path(&output_path, false)
                .map_err(|e| format!("create {:?}: {}", output_path, e))?,
        );
        Some(
            AlignedWriter::from_fd_with_buffer_size(fd, MERGE_OUTPUT_BUF_BYTES)
                .map_err(|e| format!("aligned writer {:?}: {}", output_path, e))?,
        )
    } else {
        None
    };

    let mut emitted = 0u64;
    let mut output_logical_bytes = 0u64;
    loop {
        let w = tree.winner();
        if w == NIL_LEAF {
            break;
        }
        let idx = w as usize;

        // Emit the current record (still in readers[idx].buf), then advance.
        let rec_bytes = readers[idx].current_record_bytes();
        if let Some(out_w) = writer.as_mut() {
            out_w
                .write_all(rec_bytes)
                .map_err(|e| format!("write: {}", e))?;
        }
        output_logical_bytes += rec_bytes.len() as u64;
        emitted += 1;

        let still_has = readers[idx].advance()?;
        // Replay this leaf — closure captures `readers` immutably for the
        // O(log k) compares along the leaf-to-root path.
        tree.replay(idx, still_has, |leaf| {
            let r = &readers[leaf as usize];
            if r.has_data {
                Some(r.current_key())
            } else {
                None
            }
        });
    }

    if let Some(mut w) = writer {
        w.flush().map_err(|e| format!("flush: {}", e))?;
        // AlignedWriter pads to a 512 B boundary; ftruncate back to the
        // logical byte count so downstream consumers see exactly the sorted
        // records with no trailing zeros.
        let fd = w.get_fd();
        let rc = unsafe { libc::ftruncate(fd.as_raw_fd(), output_logical_bytes as libc::off_t) };
        if rc < 0 {
            return Err(format!(
                "ftruncate {:?}: {}",
                output_path,
                std::io::Error::last_os_error()
            ));
        }
    }
    Ok(emitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arbitration::DeviceArbiter;
    use crate::page_buffer::PAGE_SIZE;
    use crate::run_formation::{RunFormationConfig, run_formation};
    use es::{IoStatsTracker, SortInput};
    use tempfile::TempDir;

    /// In-memory `SortInput` that yields a fixed list of records on a single
    /// scanner. Sufficient for driving run_formation in tests.
    struct InMemoryInput {
        records: Vec<(Vec<u8>, Vec<u8>)>,
    }

    impl SortInput for InMemoryInput {
        fn create_parallel_scanners(
            &self,
            _num_scanners: usize,
            _io_tracker: Option<IoStatsTracker>,
        ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>> {
            let recs = self.records.clone();
            vec![Box::new(recs.into_iter())]
        }
    }

    /// Decode a length-prefixed record stream produced by `merge` (no
    /// trailing padding because we ftruncate after flush).
    fn read_lp_records(path: &std::path::Path) -> Vec<(Vec<u8>, Vec<u8>)> {
        let bytes = std::fs::read(path).expect("read merged file");
        let mut out = Vec::new();
        let mut i = 0usize;
        while i + 8 <= bytes.len() {
            let klen = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
            let vlen = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
            i += 8;
            let k = bytes[i..i + klen].to_vec();
            i += klen;
            let v = bytes[i..i + vlen].to_vec();
            i += vlen;
            out.push((k, v));
        }
        out
    }

    /// End-to-end: feed unsorted records → run_formation → merge → assert the
    /// merged per-range output, when concatenated in range order, equals the
    /// sorted input. Exercises the WinnerTree, RunReader length-prefix
    /// parsing, per-range output emission, and the multi-range parallelism
    /// path (one merge thread per range).
    #[test]
    fn end_to_end_run_formation_then_merge() {
        // 50 records fed in descending order so the input is unsorted.
        let records: Vec<(Vec<u8>, Vec<u8>)> = (0..50u32)
            .rev()
            .map(|i| {
                let k = format!("key{:07}", i).into_bytes();
                let v = format!("val{:07}", i).into_bytes();
                (k, v)
            })
            .collect();

        let temp = TempDir::new().expect("tempdir");
        let runs_dir = temp.path().join("runs");
        let out_dir = temp.path().join("out");

        // Budget = 1 page + 10 pointer slots. Each push past the 10th in a
        // page would need an additional pointer that busts the budget,
        // forcing a flush. With ~50 records this yields multiple runs.
        let memory_bytes = (PAGE_SIZE as u64) + 10 * 8;

        // Boundary "key0000025" splits records evenly:
        // range 0: 0..24 (25 records), range 1: 25..49 (25 records).
        let boundary_keys: Vec<Vec<u8>> = vec![b"key0000025".to_vec()];
        let num_ranges = boundary_keys.len() + 1;

        let input: Arc<dyn SortInput + Send + Sync> = Arc::new(InMemoryInput {
            records: records.clone(),
        });

        let rf = run_formation(
            input,
            RunFormationConfig {
                temp_dir: runs_dir.clone(),
                threads: 1,
                memory_bytes,
                boundary_keys: boundary_keys.clone(),
                arbiter: DeviceArbiter::new(false),
            },
        )
        .expect("run formation");
        assert!(
            rf.metadata.num_runs >= 2,
            "tight budget should force multiple runs, got {}",
            rf.metadata.num_runs
        );
        assert_eq!(rf.metadata.num_ranges, num_ranges as u32);

        let merge_result = merge(MergeConfig {
            metadata: Arc::new(rf.metadata.clone()),
            threads: num_ranges,
            output_path: out_dir.clone(),
            write_output: true,
            read_block_bytes: 4096,
        })
        .expect("merge");
        assert_eq!(merge_result.records_emitted, records.len() as u64);

        // Read each range's output. With threads == num_ranges, range r
        // is assigned to thread r (round-robin in merge.rs:57-60), so the
        // file name is merged_t{r}_r{r}.bin.
        let mut all: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for r in 0..num_ranges as u32 {
            let path = out_dir.join(format!("merged_t{:03}_r{:03}.bin", r, r));
            let recs = read_lp_records(&path);
            assert!(!recs.is_empty(), "range {} output is empty", r);
            for w in recs.windows(2) {
                assert!(w[0].0 <= w[1].0, "range {} not sorted", r);
            }
            all.extend(recs);
        }

        let mut expected = records;
        expected.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(all, expected, "merged output != sorted input");
    }

    /// Stable-tie-break sanity: when multiple runs contain the same key, the
    /// merge must emit records in run_id order (smaller run_id first). The
    /// winner tree's tie-break-to-smaller-leaf rule plus opening files in
    /// ascending run_id order at merge.rs:221 give us this for free.
    #[test]
    fn merge_breaks_ties_by_run_id() {
        // All records share the same key — only `value` distinguishes them,
        // and we tag value with the order they were inserted so we can
        // verify run_id ordering after the merge.
        let records: Vec<(Vec<u8>, Vec<u8>)> = (0..30u32)
            .map(|i| (b"same_key__".to_vec(), format!("v{:08}", i).into_bytes()))
            .collect();

        let temp = TempDir::new().expect("tempdir");
        let runs_dir = temp.path().join("runs");
        let out_dir = temp.path().join("out");

        let memory_bytes = (PAGE_SIZE as u64) + 10 * 8;
        let input: Arc<dyn SortInput + Send + Sync> = Arc::new(InMemoryInput {
            records: records.clone(),
        });

        let rf = run_formation(
            input,
            RunFormationConfig {
                temp_dir: runs_dir.clone(),
                threads: 1,
                memory_bytes,
                boundary_keys: Vec::new(), // single range
                arbiter: DeviceArbiter::new(false),
            },
        )
        .expect("run formation");
        assert!(
            rf.metadata.num_runs >= 2,
            "need multiple runs for tie-break test"
        );

        let _ = merge(MergeConfig {
            metadata: Arc::new(rf.metadata.clone()),
            threads: 1,
            output_path: out_dir.clone(),
            write_output: true,
            read_block_bytes: 4096,
        })
        .expect("merge");

        let merged = read_lp_records(&out_dir.join("merged_t000_r000.bin"));
        assert_eq!(merged.len(), records.len());
        // Within the same key, values must appear in original-insertion
        // order — i.e. v00000000, v00000001, ... — because run_id is assigned
        // sequentially to flushed runs and ties break to smaller run_id.
        for w in merged.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "tie not broken stably: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
    }
}
