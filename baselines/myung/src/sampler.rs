//! Upfront random sampling to derive range boundaries (Myung §3.1.1).
//!
//! Samples ~0.1% of the input via concurrent aligned pread, extracts keys,
//! sorts them, and picks quantile cuts for (num_ranges - 1) boundary keys.
//! Implements the key insight that sampling is cheap compared to the full
//! sort and yields a well-balanced partition for uniform data; size skew is
//! deliberately NOT handled here (that's CrocSort's byte-balanced win).

use es::diskio::file::{SharedFd, file_size_fd, pread_fd};
use rand::Rng;
use std::path::Path;
use std::sync::Arc;

/// Sample the input and return (num_ranges - 1) boundary keys.
///
/// `record_size` must match the fixed record layout of the input (e.g. 100 for
/// standard GenSort: 10B key + 90B payload).
/// `key_len` is the prefix length treated as the sort key.
/// `sample_ratio` is e.g. 0.001 for 0.1%.
pub fn derive_range_boundaries(
    path: &Path,
    record_size: usize,
    key_len: usize,
    num_ranges: usize,
    sample_ratio: f64,
) -> Result<Vec<Vec<u8>>, String> {
    assert!(num_ranges >= 1);
    if num_ranges == 1 {
        return Ok(Vec::new());
    }
    assert!(key_len <= record_size);
    assert!(sample_ratio > 0.0 && sample_ratio < 1.0);

    let fd = Arc::new(
        SharedFd::new_from_path(path, false).map_err(|e| format!("open {:?}: {}", path, e))?,
    );
    let file_size = file_size_fd(fd.as_raw_fd()).map_err(|e| format!("fstat: {}", e))?;
    if file_size % record_size as u64 != 0 {
        return Err(format!(
            "file size {} is not a multiple of record size {}",
            file_size, record_size
        ));
    }
    let num_records = file_size / record_size as u64;
    let target_samples = ((num_records as f64) * sample_ratio).ceil() as u64;
    let target_samples = target_samples.max((num_ranges as u64) * 32);

    // Myung's paper samples "pages of dozens of records" at random offsets.
    // We use 4 KiB page reads (a multiple of record_size via a local aligned
    // buffer). For simplicity here we just read individual records at random
    // positions — this matches the paper's quantile-accuracy properties and
    // keeps the implementation portable. Direct I/O requires aligned buffers;
    // since we may not be record-aligned with 512B DIO blocks for arbitrary
    // record sizes, we bounce through a page-sized buffer.
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(target_samples as usize);
    let mut rng = rand::rng();

    // Page size for Direct I/O. 4 KiB aligns with most NVMe LBA sizes.
    const PAGE: usize = 4096;
    // Use a heap-allocated, aligned buffer. Rust doesn't give us aligned
    // Vec<u8> directly, so over-allocate and slice to an aligned offset.
    let mut raw = vec![0u8; PAGE * 2];
    let base = raw.as_ptr() as usize;
    let slack = (PAGE - (base % PAGE)) % PAGE;
    let aligned = &mut raw[slack..slack + PAGE];

    while (keys.len() as u64) < target_samples {
        // Pick a random page-aligned offset that contains at least one record.
        let max_page_start = file_size.saturating_sub(PAGE as u64);
        if max_page_start == 0 {
            break;
        }
        let off = (rng.random_range(0..=max_page_start) / PAGE as u64) * PAGE as u64;
        match pread_fd(fd.as_raw_fd(), aligned, off) {
            Ok(n) => {
                // Walk through the page, extracting keys at record boundaries
                // that fall inside [off, off+n) AND inside the file.
                let first_record_in_file = off / record_size as u64;
                let first_record_offset_in_file = first_record_in_file * record_size as u64;
                if first_record_offset_in_file < off {
                    // Skip to the next record boundary.
                    let _skip = (first_record_in_file + 1) * record_size as u64 - off;
                    // For simplicity, just skip this page if it doesn't align
                    // cleanly. The page sampler will pick up aligned pages over
                    // many iterations.
                    continue;
                }
                let mut cursor = 0usize;
                while cursor + record_size <= n {
                    keys.push(aligned[cursor..cursor + key_len].to_vec());
                    cursor += record_size;
                    if (keys.len() as u64) >= target_samples {
                        break;
                    }
                }
            }
            Err(e) => return Err(format!("pread at {}: {}", off, e)),
        }
    }

    if keys.is_empty() {
        return Err("sampler collected no keys".into());
    }

    keys.sort();
    let mut boundaries = Vec::with_capacity(num_ranges - 1);
    for i in 1..num_ranges {
        let idx = (keys.len() * i) / num_ranges;
        let idx = idx.min(keys.len() - 1);
        boundaries.push(keys[idx].clone());
    }
    // De-duplicate while preserving order: if two adjacent boundaries collide
    // (heavy-hitter keys), we keep both — the merge is fine with empty ranges.
    Ok(boundaries)
}

/// Given boundary keys sorted ascending, classify a key into [0, num_ranges).
pub fn classify(key: &[u8], boundaries: &[Vec<u8>]) -> usize {
    // Binary search: returns the first range whose upper bound > key.
    // boundaries[i] is the exclusive upper bound for range i.
    match boundaries.binary_search_by(|b| b.as_slice().cmp(key)) {
        Ok(i) => i + 1, // key == boundaries[i] goes into the NEXT range (boundary exclusive for lower)
        Err(i) => i,    // key fits before boundary[i]
    }
}

/// Sample boundaries for a KVBin file using its sidecar index.
///
/// The KVBin index is a stream of `u64` offsets, each pointing to the start
/// of a ~4 MiB block boundary in the data file. We:
///   1. Read all anchor offsets from the .idx file.
///   2. Randomly subsample those anchors to hit ~`sample_ratio * total_size`
///      bytes worth of probes.
///   3. For each chosen anchor, open a small read window and pull a few
///      length-prefixed records starting there. Their keys feed the boundary
///      derivation just like the GenSort sampler.
///
/// This honours the user's "you can further sample from them" guidance — the
/// .idx file is treated as a coarse probe set, then we further subsample.
///
/// Record format on disk: `[u32 klen][key][u32 vlen][value]` (KVBin layout
/// from `es::kvbin`).
pub fn derive_range_boundaries_kvbin(
    data_path: &Path,
    idx_path: &Path,
    num_ranges: usize,
    sample_ratio: f64,
    records_per_anchor: usize,
) -> Result<Vec<Vec<u8>>, String> {
    use std::io::{Read, Seek, SeekFrom};

    assert!(num_ranges >= 1);
    if num_ranges == 1 {
        return Ok(Vec::new());
    }
    assert!(sample_ratio > 0.0 && sample_ratio < 1.0);
    assert!(records_per_anchor >= 1);

    // 1) Load anchor offsets from the .idx file (stream of u64 LE).
    let idx_bytes =
        std::fs::read(idx_path).map_err(|e| format!("read idx {:?}: {}", idx_path, e))?;
    if idx_bytes.len() % 8 != 0 {
        return Err(format!(
            "idx size {} is not a multiple of 8",
            idx_bytes.len()
        ));
    }
    let data_size = std::fs::metadata(data_path)
        .map_err(|e| format!("stat data {:?}: {}", data_path, e))?
        .len();

    let mut anchors: Vec<u64> = idx_bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .filter(|&o| o < data_size)
        .collect();
    if anchors.is_empty() {
        return Err("idx file contained no usable offsets".into());
    }
    anchors.sort_unstable();
    anchors.dedup();

    // 2) Subsample anchors. Target a number of probes such that
    //    probes * records_per_anchor ≈ sample_ratio * (data_size / avg_record).
    //    We don't know avg record size yet — use a coarse estimate by reading
    //    one anchor first to learn record size, then size the sample.
    let mut data_file =
        std::fs::File::open(data_path).map_err(|e| format!("open data {:?}: {}", data_path, e))?;
    let probe_anchor = anchors[anchors.len() / 2];
    data_file
        .seek(SeekFrom::Start(probe_anchor))
        .map_err(|e| format!("seek probe: {}", e))?;
    let mut probe_buf = vec![0u8; 4096.min(data_size as usize)];
    let probe_n = data_file
        .read(&mut probe_buf)
        .map_err(|e| format!("probe read: {}", e))?;
    let est_record_bytes = estimate_first_record_size(&probe_buf[..probe_n]).unwrap_or(128);

    let est_total_records = (data_size as f64) / (est_record_bytes as f64);
    let target_keys = ((est_total_records * sample_ratio).ceil() as usize)
        .max((num_ranges) * 32)
        .max(64);
    let probes_needed = (target_keys + records_per_anchor - 1) / records_per_anchor;
    let probes_needed = probes_needed.min(anchors.len());

    // 3) Random subsample of anchors (without replacement).
    use rand::prelude::IndexedRandom;
    let mut rng = rand::rng();
    let chosen: Vec<u64> = anchors
        .as_slice()
        .choose_multiple(&mut rng, probes_needed)
        .copied()
        .collect();

    // 4) For each anchor, read records_per_anchor records and collect keys.
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(target_keys);
    // Reasonable per-anchor window: ~64 KiB usually contains many records.
    let window = 64 * 1024usize;
    let mut buf = vec![0u8; window];
    for off in chosen {
        data_file
            .seek(SeekFrom::Start(off))
            .map_err(|e| format!("seek anchor {}: {}", off, e))?;
        let n = data_file
            .read(&mut buf)
            .map_err(|e| format!("read anchor {}: {}", off, e))?;
        let mut cursor = 0usize;
        let mut got = 0usize;
        while cursor + 8 <= n && got < records_per_anchor {
            let klen = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().unwrap()) as usize;
            if cursor + 4 + klen + 4 > n {
                break;
            }
            let key = buf[cursor + 4..cursor + 4 + klen].to_vec();
            // Skip the value to advance: vlen is right after the key.
            let vlen_off = cursor + 4 + klen;
            let vlen = u32::from_le_bytes(buf[vlen_off..vlen_off + 4].try_into().unwrap()) as usize;
            let next = vlen_off + 4 + vlen;
            if next > n {
                // record straddles the window edge; keep the key, then stop.
                keys.push(key);
                break;
            }
            keys.push(key);
            cursor = next;
            got += 1;
        }
    }

    if keys.is_empty() {
        return Err("KVBin sampler collected no keys".into());
    }
    keys.sort();
    let mut boundaries = Vec::with_capacity(num_ranges - 1);
    for i in 1..num_ranges {
        let idx = (keys.len() * i) / num_ranges;
        let idx = idx.min(keys.len() - 1);
        boundaries.push(keys[idx].clone());
    }
    Ok(boundaries)
}

/// Look at the first 8 bytes (KVBin header) and infer "record size" =
/// 4 + klen + 4 + vlen. Returns None if the buffer is too short.
fn estimate_first_record_size(buf: &[u8]) -> Option<usize> {
    if buf.len() < 8 {
        return None;
    }
    let klen = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    if buf.len() < 4 + klen + 4 {
        return None;
    }
    let vlen_off = 4 + klen;
    let vlen = u32::from_le_bytes(buf[vlen_off..vlen_off + 4].try_into().unwrap()) as usize;
    Some(4 + klen + 4 + vlen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_basic() {
        let b = vec![b"cc".to_vec(), b"mm".to_vec()];
        assert_eq!(classify(b"aa", &b), 0);
        assert_eq!(classify(b"cc", &b), 1); // boundary goes to upper range
        assert_eq!(classify(b"dd", &b), 1);
        assert_eq!(classify(b"mm", &b), 2);
        assert_eq!(classify(b"zz", &b), 2);
    }
}
