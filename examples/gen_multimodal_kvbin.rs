/// Generate a synthetic KVBin dataset for **Experiment 2.3 — Multimodal**.
///
/// # Dataset characteristics
///
/// Keys are drawn from a mixture of K Gaussians. K and the per-component
/// (mean, stddev, weight) are configurable; the defaults below give 3
/// well-separated modes that probe whether the partition planner's sample
/// captures inter-mode boundaries (rather than over-sampling a single mode).
///
/// Default mixture (K = 3, equal weights):
///   N(  1.0e18 , 5.0e16 )
///   N(  9.2e18 , 5.0e16 )
///   N( 1.74e19 , 5.0e16 )
/// (means span the u64 range with ~3-σ separation between adjacent modes.)
///
/// Sampled f64 values are clamped to `[0, u64::MAX]` and cast to u64. Payloads
/// are uniform-sized (`--payload`, default 512 B), so this isolates the
/// *key-distribution sampling* effect from any size skew.
///
/// # Pairing with stride sweep
///
/// To run Experiment 2.3, sweep the per-record sampling stride over
/// {1/512, 1/1024, 1/2048} via the env-var `ES_FIXED_STRIDE_RECORDS=N`
/// (or `ES_FIXED_STRIDE_BYTES=...` for size-balanced / CrocSort, where the
/// sparse index is byte-strided). See `scripts/crocsort_skew_bench.sh`.
///
/// # KVBin record format
///
/// ```text
/// ┌──────────┬─────────────┬──────────┬─────────────────┐
/// │ klen: u32│ key (klen B)│ vlen: u32│ value (vlen B)  │
/// └──────────┴─────────────┴──────────┴─────────────────┘
/// ```
/// Integers are little-endian; key is always 8 bytes (a `u64`, big-endian
/// encoded so lexicographic byte order matches numeric order).
///
/// # Sparse index format (`.idx`)
///
/// A sequence of `u64` little-endian byte offsets, one per ~4 MiB block start.
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use clap::Parser;
use crossbeam::channel;

const INDEX_STRIDE_BYTES: usize = 4 * 1024 * 1024;
const CHUNK_ROWS: u64 = 50_000;

/// Default mixture: 3 equally-weighted Gaussians spanning the u64 range with
/// roughly 3-σ inter-mode separation.
const DEFAULT_MEANS: &str = "1.0e18,9.2e18,1.74e19";
const DEFAULT_STDDEVS: &str = "5.0e16,5.0e16,5.0e16";
const DEFAULT_WEIGHTS: &str = "1,1,1";

#[derive(Parser, Debug)]
struct Args {
    /// Output .kvbin path
    #[arg(long)]
    out: PathBuf,

    /// Output .idx path (u64 offsets, no keys)
    #[arg(long)]
    idx: PathBuf,

    /// Total rows to generate.
    /// Default targets ~200 GiB at the default 512 B payload:
    /// record = 4+8+4+512 = 528 B → 200 GiB / 528 ≈ 406_720_387 rows
    #[arg(long, default_value_t = 406_720_387)]
    rows: u64,

    /// Comma-separated component means (f64). Length determines K.
    #[arg(long, default_value = DEFAULT_MEANS)]
    means: String,

    /// Comma-separated component stddevs (f64). Length must match means.
    #[arg(long, default_value = DEFAULT_STDDEVS)]
    stddevs: String,

    /// Comma-separated mixture weights (need not sum to 1; will be normalised).
    /// Length must match means.
    #[arg(long, default_value = DEFAULT_WEIGHTS)]
    weights: String,

    /// Payload size in bytes for every row (uniform — no size skew)
    #[arg(long, default_value_t = 512)]
    payload: usize,

    /// Worker threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Seed for PRNG
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Clone)]
struct SplitMix64 {
    x: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { x: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.x = self.x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    /// Standard normal via Box–Muller (one of two values; the other is discarded
    /// for simplicity — half the throughput, plenty fast for offline gen).
    fn next_gaussian(&mut self) -> f64 {
        // Avoid log(0) by drawing u1 from (0, 1].
        let mut u1 = self.next_f64();
        if u1 == 0.0 {
            u1 = f64::EPSILON;
        }
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        r * theta.cos()
    }
}

fn fill_payload(buf: &mut [u8], rng: &mut SplitMix64) {
    let mut i = 0;
    while i + 8 <= buf.len() {
        let v = rng.next_u64().to_le_bytes();
        buf[i..i + 8].copy_from_slice(&v);
        i += 8;
    }
    if i < buf.len() {
        let v = rng.next_u64().to_le_bytes();
        let remain = buf.len() - i;
        buf[i..].copy_from_slice(&v[..remain]);
    }
}

#[derive(Clone)]
struct Mixture {
    means: Vec<f64>,
    stddevs: Vec<f64>,
    /// CDF of weights (length K), monotone non-decreasing, last entry == 1.0.
    weight_cdf: Vec<f64>,
}

impl Mixture {
    fn new(means: Vec<f64>, stddevs: Vec<f64>, weights: Vec<f64>) -> io::Result<Self> {
        if means.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mixture must have at least one component",
            ));
        }
        if means.len() != stddevs.len() || means.len() != weights.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "means/stddevs/weights length mismatch: {} / {} / {}",
                    means.len(),
                    stddevs.len(),
                    weights.len()
                ),
            ));
        }
        for (i, w) in weights.iter().enumerate() {
            if !w.is_finite() || *w < 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("weight[{i}]={w} must be finite and non-negative"),
                ));
            }
        }
        for (i, s) in stddevs.iter().enumerate() {
            if !s.is_finite() || *s <= 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("stddev[{i}]={s} must be finite and positive"),
                ));
            }
        }
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "weights must sum to a positive value",
            ));
        }
        let mut weight_cdf = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for w in &weights {
            acc += w / sum;
            weight_cdf.push(acc);
        }
        // Pin the last entry to exactly 1.0 to avoid round-off boundary issues.
        if let Some(last) = weight_cdf.last_mut() {
            *last = 1.0;
        }
        Ok(Self {
            means,
            stddevs,
            weight_cdf,
        })
    }

    fn sample_key(&self, rng: &mut SplitMix64) -> u64 {
        let u = rng.next_f64();
        let mut k = 0;
        while k + 1 < self.weight_cdf.len() && u >= self.weight_cdf[k] {
            k += 1;
        }
        let z = rng.next_gaussian();
        let v = self.means[k] + self.stddevs[k] * z;
        // Clamp into [0, u64::MAX]. Casting f64 -> u64 is saturating in Rust.
        if !v.is_finite() {
            return 0;
        }
        if v <= 0.0 {
            return 0;
        }
        if v >= u64::MAX as f64 {
            return u64::MAX;
        }
        v as u64
    }
}

fn parse_csv_f64(s: &str, name: &str) -> io::Result<Vec<f64>> {
    s.split(',')
        .map(|tok| {
            tok.trim().parse::<f64>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("--{name}: failed to parse '{tok}' as f64: {e}"),
                )
            })
        })
        .collect()
}

#[derive(Clone)]
struct WorkerConfig {
    mixture: Mixture,
    payload: usize,
    seed: u64,
}

struct Block {
    data: Vec<u8>,
    rows: u64,
}

fn resolve_threads(requested: usize) -> usize {
    if requested == 0 {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1)
    } else {
        requested.max(1)
    }
}

fn append_record(buf: &mut Vec<u8>, key: u64, payload_len: usize, rng: &mut SplitMix64) {
    let klen: u32 = 8;
    let vlen: u32 = payload_len as u32;

    buf.extend_from_slice(&klen.to_le_bytes());
    // Big-endian so lexicographic byte order matches numeric order — matches
    // sister generators that sort by the raw key bytes.
    buf.extend_from_slice(&key.to_be_bytes());
    buf.extend_from_slice(&vlen.to_le_bytes());

    let start = buf.len();
    buf.reserve(payload_len);
    // SAFETY: fill_payload unconditionally writes all bytes in the slice.
    unsafe { buf.set_len(start + payload_len) };
    fill_payload(&mut buf[start..], rng);
}

fn write_blocks(
    rx: channel::Receiver<Block>,
    out_path: PathBuf,
    idx_path: PathBuf,
) -> io::Result<u64> {
    let out_f = File::create(out_path)?;
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, out_f);

    let idx_f = File::create(idx_path)?;
    let mut idx = BufWriter::with_capacity(256 * 1024, idx_f);

    let mut bytes_written: u64 = 0;
    let mut rows_written: u64 = 0;

    for block in rx.iter() {
        idx.write_all(&bytes_written.to_le_bytes())?;
        out.write_all(&block.data)?;
        bytes_written += block.data.len() as u64;
        rows_written += block.rows;
    }

    out.flush()?;
    idx.flush()?;
    Ok(rows_written)
}

fn worker_loop(
    thread_id: usize,
    cfg: WorkerConfig,
    total_rows: u64,
    next_row: &AtomicU64,
    tx: channel::Sender<Block>,
) -> io::Result<()> {
    let mut rng = SplitMix64::new(
        cfg.seed
            .wrapping_add((thread_id as u64 + 1) * 0x9E3779B97F4A7C15),
    );
    let max_record = 4 + 8 + 4 + cfg.payload;
    let mut buf = Vec::with_capacity(INDEX_STRIDE_BYTES + max_record);
    let mut rows_in_buf: u64 = 0;

    loop {
        let start = next_row.fetch_add(CHUNK_ROWS, Ordering::Relaxed);
        if start >= total_rows {
            break;
        }
        let mut remaining = (total_rows - start).min(CHUNK_ROWS);

        while remaining > 0 {
            remaining -= 1;
            let key = cfg.mixture.sample_key(&mut rng);
            append_record(&mut buf, key, cfg.payload, &mut rng);
            rows_in_buf += 1;

            if buf.len() >= INDEX_STRIDE_BYTES {
                tx.send(Block {
                    data: std::mem::take(&mut buf),
                    rows: rows_in_buf,
                })
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "writer thread ended"))?;
                buf = Vec::with_capacity(INDEX_STRIDE_BYTES + max_record);
                rows_in_buf = 0;
            }
        }
    }

    if rows_in_buf > 0 {
        tx.send(Block {
            data: buf,
            rows: rows_in_buf,
        })
        .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "writer thread ended"))?;
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let threads = resolve_threads(args.threads);

    let means = parse_csv_f64(&args.means, "means")?;
    let stddevs = parse_csv_f64(&args.stddevs, "stddevs")?;
    let weights = parse_csv_f64(&args.weights, "weights")?;
    let mixture = Mixture::new(means, stddevs, weights)?;

    eprintln!(
        "multimodal: K={} means={:?} stddevs={:?} weight_cdf={:?}",
        mixture.means.len(),
        mixture.means,
        mixture.stddevs,
        mixture.weight_cdf,
    );

    let (tx, rx) = channel::bounded::<Block>(threads * 2);
    let writer_out = args.out.clone();
    let writer_idx = args.idx.clone();
    let writer_handle = thread::spawn(move || write_blocks(rx, writer_out, writer_idx));

    let next_row = AtomicU64::new(0);
    let cfg = WorkerConfig {
        mixture,
        payload: args.payload,
        seed: args.seed,
    };

    thread::scope(|s| -> io::Result<()> {
        let mut handles = Vec::with_capacity(threads);
        for thread_id in 0..threads {
            let tx = tx.clone();
            let next_row = &next_row;
            let cfg = cfg.clone();
            handles.push(s.spawn(move || worker_loop(thread_id, cfg, args.rows, next_row, tx)));
        }

        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "worker thread panicked",
                    ));
                }
            }
        }
        Ok(())
    })?;

    drop(tx);

    let rows_written = match writer_handle.join() {
        Ok(Ok(rows)) => rows,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "writer thread panicked",
            ));
        }
    };

    if rows_written != args.rows {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "row count mismatch: wrote {rows_written}, expected {}",
                args.rows
            ),
        ));
    }

    Ok(())
}
