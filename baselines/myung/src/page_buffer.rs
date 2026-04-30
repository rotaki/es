//! Page-based record buffer for Myung's run-formation sort buffer.
//!
//! Matches the AlphaSort-style key/pointer layout that the paper cites [8]:
//!   - Records are packed into fixed-size pages (64 KiB).
//!   - A separate pointer vector holds (page_idx, offset_in_page) tuples —
//!     sorting moves the 8-byte pointers, never the records themselves.
//!
//! Record encoding on-page:
//!     [u32 key_len][u32 value_len][key bytes][value bytes]
//! This length-prefixed format natively supports variable-length keys and
//! values (needed for TPC-H lineitem, HeavyKey). The same format is emitted
//! to per-range run files, so the merge reader can parse without knowing a
//! record_size upfront.
//!
//! Memory accounting:
//!   allocated_pages * PAGE_SIZE  +  num_pointers * sizeof(RecordRef)  <=  budget
//! Once a push would exceed `budget`, the caller must flush (sort + write)
//! and reset the buffer before pushing more records.

pub const PAGE_SIZE: usize = 64 * 1024;
pub const RECORD_HEADER_SIZE: usize = 8; // [u32 key_len][u32 value_len]
pub const POINTER_SIZE: usize = std::mem::size_of::<RecordRef>();

/// Pointer into the page buffer. Stays 8 bytes to minimise sort-vector size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct RecordRef {
    pub page_idx: u32,
    pub offset_in_page: u32,
}

/// A single 64 KiB page holding length-prefixed records.
pub struct Page {
    data: Box<[u8; PAGE_SIZE]>,
    used: u32,
}

impl Page {
    fn new() -> Self {
        Page {
            data: Box::new([0u8; PAGE_SIZE]),
            used: 0,
        }
    }

    fn remaining(&self) -> usize {
        PAGE_SIZE - self.used as usize
    }

    /// Try to append a record. Returns `Some(offset)` on success, `None` if
    /// this page doesn't have room.
    fn try_append(&mut self, key: &[u8], value: &[u8]) -> Option<u32> {
        let need = RECORD_HEADER_SIZE + key.len() + value.len();
        if need > self.remaining() {
            return None;
        }
        let off = self.used as usize;
        let klen = u32::try_from(key.len()).expect("key too long");
        let vlen = u32::try_from(value.len()).expect("value too long");
        self.data[off..off + 4].copy_from_slice(&klen.to_le_bytes());
        self.data[off + 4..off + 8].copy_from_slice(&vlen.to_le_bytes());
        let data_start = off + RECORD_HEADER_SIZE;
        self.data[data_start..data_start + key.len()].copy_from_slice(key);
        self.data[data_start + key.len()..data_start + key.len() + value.len()]
            .copy_from_slice(value);
        self.used = (off + need) as u32;
        Some(off as u32)
    }

    /// Decode a record stored at `offset` into (key, value) byte slices.
    /// Only valid while `self` is not mutated.
    fn decode(&self, offset: u32) -> (&[u8], &[u8]) {
        let off = offset as usize;
        let klen = u32::from_le_bytes(self.data[off..off + 4].try_into().unwrap()) as usize;
        let vlen = u32::from_le_bytes(self.data[off + 4..off + 8].try_into().unwrap()) as usize;
        let data_start = off + RECORD_HEADER_SIZE;
        (
            &self.data[data_start..data_start + klen],
            &self.data[data_start + klen..data_start + klen + vlen],
        )
    }

    /// Decode just the key. Skips reading `value_len` and avoids slicing the
    /// value range — the hot path for sort comparisons.
    fn decode_key(&self, offset: u32) -> &[u8] {
        let off = offset as usize;
        let klen = u32::from_le_bytes(self.data[off..off + 4].try_into().unwrap()) as usize;
        let data_start = off + RECORD_HEADER_SIZE;
        &self.data[data_start..data_start + klen]
    }

    /// Raw bytes of the record (header + key + value), as written on disk.
    fn raw_record_bytes(&self, offset: u32) -> &[u8] {
        let off = offset as usize;
        let klen = u32::from_le_bytes(self.data[off..off + 4].try_into().unwrap()) as usize;
        let vlen = u32::from_le_bytes(self.data[off + 4..off + 8].try_into().unwrap()) as usize;
        let total = RECORD_HEADER_SIZE + klen + vlen;
        &self.data[off..off + total]
    }
}

/// Page-based record buffer with a separate sortable pointer vector.
pub struct PageBuffer {
    pages: Vec<Page>,
    refs: Vec<RecordRef>,
    /// Memory budget in bytes for pages + pointer vector.
    budget_bytes: u64,
    /// Records larger than PAGE_SIZE - RECORD_HEADER_SIZE cannot fit.
    largest_record_bytes: u32,
}

impl PageBuffer {
    pub fn new(budget_bytes: u64) -> Self {
        PageBuffer {
            pages: Vec::new(),
            refs: Vec::new(),
            budget_bytes,
            largest_record_bytes: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.refs.len()
    }

    /// Bytes currently committed (allocated pages + pointer vector). Used
    /// pages are counted at full PAGE_SIZE because the page stays allocated
    /// for the life of the run.
    pub fn committed_bytes(&self) -> u64 {
        (self.pages.len() as u64) * (PAGE_SIZE as u64)
            + (self.refs.len() as u64) * (POINTER_SIZE as u64)
    }

    /// Try to append a record. Returns `true` if inserted, `false` if doing so
    /// would exceed the budget — in which case the caller should flush.
    pub fn try_push(&mut self, key: &[u8], value: &[u8]) -> Result<bool, String> {
        let record_bytes = RECORD_HEADER_SIZE + key.len() + value.len();
        if record_bytes > PAGE_SIZE {
            return Err(format!(
                "record of {} bytes does not fit in a {} byte page",
                record_bytes, PAGE_SIZE
            ));
        }

        // Pointer vector cost is constant per record.
        let pointer_cost = POINTER_SIZE as u64;

        // Determine whether we can fit it in the last page, or need a new one.
        let need_new_page = self
            .pages
            .last()
            .map(|p| p.remaining() < record_bytes)
            .unwrap_or(true);

        let incremental_bytes = if need_new_page {
            PAGE_SIZE as u64 + pointer_cost
        } else {
            pointer_cost
        };

        if self.committed_bytes() + incremental_bytes > self.budget_bytes {
            return Ok(false);
        }

        if need_new_page {
            self.pages.push(Page::new());
        }
        let page_idx = (self.pages.len() - 1) as u32;
        let offset = self.pages[page_idx as usize]
            .try_append(key, value)
            .expect("checked capacity");
        self.refs.push(RecordRef {
            page_idx,
            offset_in_page: offset,
        });
        if record_bytes as u32 > self.largest_record_bytes {
            self.largest_record_bytes = record_bytes as u32;
        }
        Ok(true)
    }

    /// Decode the key of a pointer (no copy). Skips value-length parsing.
    pub fn key_at(&self, r: &RecordRef) -> &[u8] {
        self.pages[r.page_idx as usize].decode_key(r.offset_in_page)
    }

    /// Decode (key, value) bytes at a pointer.
    pub fn kv_at(&self, r: &RecordRef) -> (&[u8], &[u8]) {
        self.pages[r.page_idx as usize].decode(r.offset_in_page)
    }

    /// Raw on-disk bytes for the record at `r` (8-byte header + key + value).
    pub fn raw_record_bytes(&self, r: &RecordRef) -> &[u8] {
        self.pages[r.page_idx as usize].raw_record_bytes(r.offset_in_page)
    }

    /// Sort the pointer vector by key ascending (pdqsort via sort_unstable).
    /// Stability within a run is NOT guaranteed — consistent with Myung §3.2
    /// ("The merging phase presented in this paper produces an unstable sort
    /// because it does not maintain the sequence of objects with equal keys").
    pub fn sort(&mut self) {
        // Borrow the page pool immutably inside the compare closure by taking
        // a raw-like view: we split refs from pages via indexing.
        let pages = &self.pages;
        self.refs.sort_unstable_by(|a, b| {
            let ka = pages[a.page_idx as usize].decode_key(a.offset_in_page);
            let kb = pages[b.page_idx as usize].decode_key(b.offset_in_page);
            ka.cmp(kb)
        });
    }

    /// Reset for a new run. Drops pages and pointers.
    pub fn clear(&mut self) {
        self.pages.clear();
        self.refs.clear();
        self.largest_record_bytes = 0;
    }

    /// Iterator over pointers in current (possibly sorted) order.
    pub fn refs(&self) -> &[RecordRef] {
        &self.refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_sort_roundtrip_fixed_records() {
        // 100-byte records (10 key + 90 value) like GenSort.
        let mut buf = PageBuffer::new(1 * 1024 * 1024); // 1 MiB
        let keys = [b"cccccccccc", b"aaaaaaaaaa", b"bbbbbbbbbb"];
        for k in keys {
            let v = [0xAB; 90];
            assert!(buf.try_push(k, &v).unwrap());
        }
        buf.sort();
        let sorted_keys: Vec<Vec<u8>> = buf.refs().iter().map(|r| buf.key_at(r).to_vec()).collect();
        assert_eq!(sorted_keys[0], b"aaaaaaaaaa");
        assert_eq!(sorted_keys[1], b"bbbbbbbbbb");
        assert_eq!(sorted_keys[2], b"cccccccccc");
    }

    #[test]
    fn variable_length_records() {
        let mut buf = PageBuffer::new(1 * 1024 * 1024);
        assert!(buf.try_push(b"short_key", b"v").unwrap());
        assert!(
            buf.try_push(
                b"a_much_longer_key_with_many_bytes",
                b"a longer value payload"
            )
            .unwrap()
        );
        assert!(buf.try_push(b"mid", b"mm").unwrap());
        buf.sort();
        let sk: Vec<Vec<u8>> = buf.refs().iter().map(|r| buf.key_at(r).to_vec()).collect();
        assert_eq!(sk[0], b"a_much_longer_key_with_many_bytes");
        assert_eq!(sk[1], b"mid");
        assert_eq!(sk[2], b"short_key");

        // kv_at matches what we inserted
        let (k, v) = buf.kv_at(&buf.refs()[0]);
        assert_eq!(k, b"a_much_longer_key_with_many_bytes");
        assert_eq!(v, b"a longer value payload");
    }

    #[test]
    fn budget_enforced() {
        // Budget: one page + a couple of pointers. After the first page fills
        // we need a *second* page; the incremental cost (PAGE_SIZE) must push
        // us over, so try_push returns false.
        let budget = PAGE_SIZE as u64 + 128;
        let mut buf = PageBuffer::new(budget);
        // Fill first page.
        let k = vec![0u8; 256];
        let v = vec![0u8; 256];
        let mut pushed = 0;
        loop {
            match buf.try_push(&k, &v).unwrap() {
                true => pushed += 1,
                false => break,
            }
        }
        assert!(pushed > 0);
        // Adding one more must fail (we'd need a new page and bust the budget).
        assert!(!buf.try_push(&k, &v).unwrap());
        assert!(buf.committed_bytes() <= budget);
    }
}
