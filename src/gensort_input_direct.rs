use crate::aligned_reader::AlignedReader;
use crate::file::SharedFd;
use crate::{IoStatsTracker, SortInput, file_size_fd};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

// GenSort format constants
const KEY_SIZE: usize = 10;
const PAYLOAD_SIZE: usize = 90;
const RECORD_SIZE: usize = KEY_SIZE + PAYLOAD_SIZE;

/// Direct I/O reader for GenSort binary format
#[derive(Clone)]
pub struct GenSortInputDirect {
    fd: Arc<SharedFd>,
    num_records: usize,
}

impl GenSortInputDirect {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(format!("File does not exist: {:?}", path));
        }

        let fd = Arc::new(SharedFd::new_from_path(&path).map_err(|e| {
            format!(
                "Failed to open file with Direct I/O: {}: {}",
                path.display(),
                e
            )
        })?);

        let file_size = file_size_fd(fd.as_raw_fd())
            .map_err(|e| format!("Failed to get file size for {:?}: {}", path, e))?;

        // Verify file size is a multiple of record size
        if file_size % RECORD_SIZE as u64 != 0 {
            return Err(format!(
                "File size {} is not a multiple of record size {}",
                file_size, RECORD_SIZE
            ));
        }

        let num_records = (file_size / RECORD_SIZE as u64) as usize;

        Ok(Self { fd, num_records })
    }

    pub fn len(&self) -> usize {
        self.num_records
    }

    pub fn file_size(&self) -> Result<u64, String> {
        file_size_fd(self.fd.as_raw_fd()).map_err(|e| format!("Failed to get file size: {}", e))
    }

    pub fn is_empty(&self) -> bool {
        self.num_records == 0
    }
}

impl SortInput for GenSortInputDirect {
    fn create_parallel_scanners(
        &self,
        num_scanners: usize,
        io_tracker: Option<IoStatsTracker>,
    ) -> Vec<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>> {
        if self.num_records == 0 {
            return vec![];
        }

        let records_per_scanner = self.num_records.div_ceil(num_scanners);
        let mut scanners = Vec::new();

        for scanner_id in 0..num_scanners {
            let start_record = scanner_id * records_per_scanner;
            if start_record >= self.num_records {
                break;
            }

            let end_record = ((scanner_id + 1) * records_per_scanner).min(self.num_records);
            let scanner = GenSortScanner::new(
                self.fd.clone(),
                start_record,
                end_record,
                io_tracker.clone(),
            );

            match scanner {
                Ok(s) => scanners
                    .push(Box::new(s) as Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + Send>),
                Err(e) => {
                    eprintln!("Failed to create scanner {}: {}", scanner_id, e);
                    // Continue with other scanners
                }
            }
        }

        scanners
    }
}

/// Scanner for reading a range of records from a GenSort file
struct GenSortScanner {
    reader: AlignedReader,
    current_record: usize,
    end_record: usize,
}

impl GenSortScanner {
    fn new(
        fd: Arc<SharedFd>,
        start_record: usize,
        end_record: usize,
        io_tracker: Option<IoStatsTracker>,
    ) -> Result<Self, String> {
        // Open file with Direct I/O
        // Create aligned reader with optional IO tracking
        let mut reader = if let Some(tracker) = io_tracker {
            AlignedReader::from_fd_with_tracer(fd, Some(tracker))
                .map_err(|e| format!("Failed to create aligned reader: {}", e))?
        } else {
            AlignedReader::from_fd(fd)
                .map_err(|e| format!("Failed to create aligned reader: {}", e))?
        };

        // Seek to start position (must be aligned)
        reader
            .seek((start_record * RECORD_SIZE) as u64)
            .map_err(|e| format!("Failed to seek: {}", e))?;

        Ok(Self {
            reader,
            current_record: start_record,
            end_record,
        })
    }

    fn read_record(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        if self.current_record >= self.end_record {
            return None;
        }

        let mut key = vec![0u8; KEY_SIZE];
        let mut payload = vec![0u8; PAYLOAD_SIZE];

        // Read the next record into the buffer if needed

        self.reader
            .read_exact(&mut key)
            .map_err(|e| eprintln!("Failed to read key: {}", e))
            .ok()?;
        self.reader
            .read_exact(&mut payload)
            .map_err(|e| eprintln!("Failed to read payload: {}", e))
            .ok()?;

        self.current_record += 1;

        Some((key, payload))
    }
}

impl Iterator for GenSortScanner {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        self.read_record()
    }
}
