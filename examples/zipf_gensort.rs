use es::rand::small_thread_rng;
use rand::RngCore;
use rand::{SeedableRng, rngs::SmallRng};

pub struct FastZipf {
    nr: usize,
    alpha: f64,
    zetan: f64,
    eta: f64,
    threshold: f64,
}

impl FastZipf {
    /// Constructs a new `FastZipf` distribution with parameter `theta` over `nr` items.
    ///
    /// # Panics
    /// Panics if `theta < 0.0` or `theta >= 1.0` or if `nr < 1`.
    pub fn new(theta: f64, nr: usize) -> Self {
        assert!(nr >= 1, "nr must be at least 1");
        assert!(
            (0.0..1.0).contains(&theta),
            "theta must be in [0,1) for FastZipf"
        );

        // Precompute zeta(nr, theta)
        let zetan = Self::zeta(nr, theta);

        // compute alpha, eta, threshold
        let alpha = 1.0 / (1.0 - theta);
        let eta = {
            let numerator = 1.0 - (2.0 / nr as f64).powf(1.0 - theta);
            let denominator = 1.0 - Self::zeta(2, theta) / zetan;
            numerator / denominator
        };
        let threshold = 1.0 + 0.5f64.powf(theta);

        FastZipf {
            nr,
            alpha,
            zetan,
            eta,
            threshold,
        }
    }

    /// Constructs a `FastZipf` if you already have a precomputed `zetan` (zeta(nr, theta)).
    #[allow(dead_code)]
    pub fn with_zeta(theta: f64, nr: usize, zetan: f64) -> Self {
        assert!(nr >= 1, "nr must be at least 1");
        assert!(
            (0.0..1.0).contains(&theta),
            "theta must be in [0,1) for FastZipf"
        );

        let alpha = 1.0 / (1.0 - theta);
        let eta = {
            let numerator = 1.0 - (2.0 / nr as f64).powf(1.0 - theta);
            let denominator = 1.0 - Self::zeta(2, theta) / zetan;
            numerator / denominator
        };
        let threshold = 1.0 + 0.5f64.powf(theta);

        FastZipf {
            nr,
            alpha,
            zetan,
            eta,
            threshold,
        }
    }

    /// Samples a value in `[0, nr)`.
    pub fn sample(&mut self) -> usize {
        // Generate u in [0,1).
        let u = self.rand_f64();
        let uz = u * self.zetan;

        if uz < 1.0 {
            return 0;
        }
        if uz < self.threshold {
            return 1;
        }
        // main formula
        let val = (self.nr as f64) * ((self.eta * u) - self.eta + 1.0).powf(self.alpha);
        val as usize
    }

    /// Returns a raw 64-bit random value.
    pub fn rand_u64(&mut self) -> u64 {
        small_thread_rng().next_u64()
    }

    /// Returns a random f64 in `[0, 1)`.
    fn rand_f64(&mut self) -> f64 {
        (small_thread_rng().next_u64() as f64) / (u64::MAX as f64)
    }

    /// Computes the zeta function for `nr` terms with exponent `theta`.
    ///
    /// \[
    ///   \zeta(nr, \theta) = \sum_{i=1}^{nr} \frac{1}{i^\theta}
    /// \]
    #[inline]
    pub fn zeta(nr: usize, theta: f64) -> f64 {
        let mut sum = 0.0;
        for i in 1..=nr {
            sum += (1.0 / (i as f64)).powf(theta);
        }
        sum
    }
}

use clap::Parser;
use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;
use std::thread;

#[derive(Parser)]
#[command(name = "zipf_gensort")]
#[command(about = "Generate gensort data with Zipf distribution")]
struct Args {
    /// Zipf theta parameter (0.0 to 1.0)
    #[arg(short, long)]
    theta: f64,

    /// Number of rows to generate
    #[arg(short, long)]
    rows: usize,

    /// Output file path
    #[arg(short, long)]
    output: String,

    /// Number of threads to use
    #[arg(short = 'j', long, default_value = "4")]
    threads: usize,
}

fn generate_gensort_record(key_value: usize) -> Vec<u8> {
    let mut record = Vec::with_capacity(100);

    // Generate 10-byte key from the zipf value
    // Format: zero-padded integer representation
    let key_str = format!("{:010}", key_value);
    record.extend_from_slice(key_str.as_bytes());

    // Generate 90 bytes of random payload
    let mut rng = SmallRng::from_os_rng();
    let mut payload = vec![0u8; 90];
    rng.fill_bytes(&mut payload);

    record.extend_from_slice(&payload);

    record
}

const RECORD_SIZE: usize = 100;
const BUFFER_RECORDS: usize = 1000;
const BUFFER_SIZE: usize = RECORD_SIZE * BUFFER_RECORDS;

fn pwrite_all(fd: i32, buf: &[u8], offset: u64) -> std::io::Result<()> {
    let mut written = 0;
    while written < buf.len() {
        let result = unsafe {
            libc::pwrite(
                fd,
                buf[written..].as_ptr() as *const libc::c_void,
                buf.len() - written,
                (offset + written as u64) as libc::off_t,
            )
        };

        if result < 0 {
            return Err(std::io::Error::last_os_error());
        }

        written += result as usize;
    }
    Ok(())
}

fn worker_thread(
    thread_id: usize,
    start_record: usize,
    end_record: usize,
    theta: f64,
    total_rows: usize,
    fd: i32,
) -> std::io::Result<()> {
    // Create FastZipf distribution
    let mut fast_zipf = FastZipf::new(theta, total_rows);

    // Buffer for batching writes
    let mut buffer = Vec::with_capacity(BUFFER_SIZE);
    let mut buffer_count = 0;
    let mut current_offset = start_record * RECORD_SIZE;

    let num_records = end_record - start_record;

    for i in 0..num_records {
        if i % 1_000_000 == 0 && i > 0 {
            println!("Thread {}: Generated {} records...", thread_id, i);
        }

        // Sample from Zipf distribution
        let key_value = fast_zipf.sample();

        // Generate record
        let record = generate_gensort_record(key_value);
        buffer.extend_from_slice(&record);
        buffer_count += 1;

        // Write buffer when full or at the end
        if buffer_count == BUFFER_RECORDS || i == num_records - 1 {
            pwrite_all(fd, &buffer, current_offset as u64)?;
            current_offset += buffer.len();
            buffer.clear();
            buffer_count = 0;
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Validate arguments
    if args.theta < 0.0 || args.theta >= 1.0 {
        return Err("Theta must be in range [0.0, 1.0)".into());
    }

    if args.rows == 0 {
        return Err("Number of rows must be greater than 0".into());
    }

    if args.threads == 0 {
        return Err("Number of threads must be greater than 0".into());
    }

    println!(
        "Generating {} records with Zipf theta={}",
        args.rows, args.theta
    );
    println!("Output file: {}", args.output);
    println!("Using {} threads", args.threads);

    // Pre-allocate file to correct size
    let total_size = args.rows * RECORD_SIZE;
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&args.output)?;
    file.set_len(total_size as u64)?;

    // Get raw file descriptor
    let fd = file.as_raw_fd();

    // Calculate records per thread
    let records_per_thread = args.rows / args.threads;
    let remainder = args.rows % args.threads;

    let start = std::time::Instant::now();

    // Spawn threads
    let mut handles = vec![];
    let mut current_start = 0;

    for i in 0..args.threads {
        let thread_records = if i < remainder {
            records_per_thread + 1
        } else {
            records_per_thread
        };

        let end_record = current_start + thread_records;
        let theta = args.theta;
        let total_rows = args.rows;

        let handle = thread::spawn(move || {
            worker_thread(i, current_start, end_record, theta, total_rows, fd)
        });

        handles.push(handle);
        current_start = end_record;
    }

    // Wait for all threads to complete
    for handle in handles {
        match handle.join() {
            Ok(result) => result?,
            Err(_) => return Err("Thread panicked".into()),
        }
    }

    // Finally sync the file to disk
    unsafe {
        libc::fsync(fd);
    }

    let duration = start.elapsed();
    println!(
        "\nGenerated {} records in {:.2} seconds",
        args.rows,
        duration.as_secs_f64()
    );
    println!(
        "Throughput: {:.2} MB/s",
        (args.rows as f64 * 100.0) / (1024.0 * 1024.0) / duration.as_secs_f64()
    );

    Ok(())
}
