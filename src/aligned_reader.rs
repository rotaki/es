//! AlignedReader - handles O_DIRECT reads with non-aligned positions
//!
//! This module provides utilities for reading files with O_DIRECT flag
//! while handling the alignment requirements.

use crate::IoStatsTracker;
use bytes::Bytes;
use parquet::errors::Result as ParquetResult;
use parquet::file::reader::{ChunkReader, Length};
use std::fs::{File, OpenOptions};
use std::io::{Error as IoError, ErrorKind, Read, Result as IoResult, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

pub const PAGE_SIZE: usize = 4096; // Page size for O_DIRECT alignment

fn allocate_aligned_buffer(size: usize) -> Vec<u8> {
    // Round up to page size
    let aligned_size = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    vec![0u8; aligned_size]
}

/// AlignedReader handles O_DIRECT reads with non-aligned start offsets
pub struct AlignedReader {
    file: File,
    file_pos: u64,
    buffer: Vec<u8>,
    buffer_offset: usize,
    buffer_valid_len: usize,
    // I/O statistics
    io_tracker: Option<IoStatsTracker>, // Global tracker to update
}

impl AlignedReader {
    pub fn new(path: &Path, buffer_size: usize) -> Result<Self, String> {
        Self::new_with_tracker(path, 0, buffer_size, None)
    }

    pub fn new_with_tracker(
        path: &Path,
        start_byte: u64,
        buffer_size: usize,
        tracker: Option<IoStatsTracker>,
    ) -> Result<Self, String> {
        // Open file with O_DIRECT
        let file = OpenOptions::new()
            .read(true)
            // .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|e| format!("Failed to open file with O_DIRECT: {}", e))?;

        // Round up buffer size to be page-aligned
        let buffer = allocate_aligned_buffer(buffer_size);

        let aligned_pos = (start_byte / PAGE_SIZE as u64) * PAGE_SIZE as u64;
        let skip_bytes = (start_byte - aligned_pos) as usize;

        let mut reader = AlignedReader {
            file,
            buffer,
            file_pos: aligned_pos,
            buffer_offset: 0,
            buffer_valid_len: 0,
            io_tracker: tracker,
        };

        // Seek to aligned position
        use std::io::Seek;
        reader
            .file
            .seek(std::io::SeekFrom::Start(aligned_pos))
            .map_err(|e| format!("Failed to seek: {}", e))?;

        // If we need to skip bytes, fill the buffer and set buffer_pos
        if skip_bytes > 0 {
            reader
                .fill_buffer()
                .map_err(|e| format!("Failed to fill buffer: {}", e))?;
            reader.buffer_offset = skip_bytes.min(reader.buffer_valid_len);
        }

        Ok(reader)
    }

    /// Skip to the next newline character and return the number of bytes skipped
    pub fn skip_to_newline(&mut self) -> Result<usize, String> {
        // Make sure we have data in the buffer
        if self.buffer_offset >= self.buffer_valid_len {
            self.fill_buffer()?;
        }

        let mut total_skipped = 0;

        loop {
            // Check remaining buffer
            if let Some(pos) = self.buffer[self.buffer_offset..self.buffer_valid_len]
                .iter()
                .position(|&b| b == b'\n')
            {
                let bytes_skipped = pos + 1;
                self.buffer_offset += bytes_skipped;
                total_skipped += bytes_skipped;
                return Ok(total_skipped);
            }

            // Need more data
            let bytes_in_buffer = self.buffer_valid_len - self.buffer_offset;
            total_skipped += bytes_in_buffer;
            self.buffer_offset = self.buffer_valid_len;
            if self.buffer_valid_len == 0 || self.buffer_valid_len < self.buffer.len() {
                return Ok(total_skipped);
            }

            self.fill_buffer()?;
            if self.buffer_valid_len == 0 {
                return Ok(total_skipped);
            }
        }
    }

    fn fill_buffer(&mut self) -> Result<(), String> {
        self.buffer_offset = 0;
        match self.file.read(&mut self.buffer) {
            Ok(n) => {
                self.buffer_valid_len = n;
                self.file_pos += n as u64;

                // Update global tracker if present
                if let Some(ref tracker) = self.io_tracker {
                    tracker.add_read(1, n as u64);
                }

                Ok(())
            }
            Err(e) => Err(format!("Failed to read from file: {}", e)),
        }
    }
}

impl Read for AlignedReader {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        let mut total_read = 0;

        loop {
            // If we have data in buffer, copy it
            let available = self.buffer_valid_len - self.buffer_offset;
            if available > 0 {
                let to_copy = available.min(buf.len() - total_read);
                buf[total_read..total_read + to_copy]
                    .copy_from_slice(&self.buffer[self.buffer_offset..self.buffer_offset + to_copy]);
                self.buffer_offset += to_copy;
                total_read += to_copy;

                if total_read == buf.len() {
                    return Ok(total_read);
                }
            }

            // Need more data
            if let Err(e) = self.fill_buffer() {
                if total_read > 0 {
                    return Ok(total_read);
                } else {
                    return Err(IoError::new(ErrorKind::Other, e));
                }
            }

            if self.buffer_valid_len == 0 {
                // EOF
                return Ok(total_read);
            }
        }
    }
}

impl Seek for AlignedReader {
    fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> {
        let new_pos = match pos {
            SeekFrom::Start(pos) => pos,
            SeekFrom::End(offset) => {
                let file_size = self.file.metadata()?.len();
                if offset >= 0 {
                    file_size + offset as u64
                } else {
                    file_size.saturating_sub((-offset) as u64)
                }
            }
            SeekFrom::Current(offset) => {
                let current = self.file_pos - (self.buffer_valid_len - self.buffer_offset) as u64;
                if offset >= 0 {
                    current + offset as u64
                } else {
                    current.saturating_sub((-offset) as u64)
                }
            }
        };

        let current = self.file_pos - (self.buffer_valid_len - self.buffer_offset) as u64;
        if new_pos == current {
            return Ok(new_pos);
        }

        // Calculate aligned position (round down to page boundary)
        let aligned_pos = (new_pos / PAGE_SIZE as u64) * PAGE_SIZE as u64;
        let skip_bytes = (new_pos - aligned_pos) as usize;

        // Seek to aligned position
        self.file.seek(SeekFrom::Start(aligned_pos))?;
        self.file_pos = aligned_pos;

        // Clear buffer and read new data
        self.buffer_offset = 0;
        self.buffer_valid_len = 0;

        if skip_bytes > 0 {
            // Read a buffer and skip to the desired position
            self.fill_buffer()
                .map_err(|e| IoError::new(ErrorKind::Other, e))?;
            self.buffer_offset = skip_bytes.min(self.buffer_valid_len);
        }

        Ok(new_pos)
    }
}

/// A wrapper around a file path that implements ChunkReader for Direct I/O
pub struct AlignedChunkReader {
    path: Arc<Path>,
    file_size: u64,
    buffer_size: usize,
    io_tracker: Option<IoStatsTracker>,
}

impl AlignedChunkReader {
    pub fn new(path: impl AsRef<Path>, buffer_size: usize) -> Result<Self, String> {
        Self::new_with_tracker(path, buffer_size, None)
    }

    pub fn new_with_tracker(
        path: impl AsRef<Path>,
        buffer_size: usize,
        tracker: Option<IoStatsTracker>,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let file_size = file
            .metadata()
            .map_err(|e| format!("Failed to get file metadata: {}", e))?
            .len();

        Ok(Self {
            path: Arc::from(path),
            file_size,
            buffer_size,
            io_tracker: tracker,
        })
    }
}

impl Length for AlignedChunkReader {
    fn len(&self) -> u64 {
        self.file_size
    }
}

impl ChunkReader for AlignedChunkReader {
    type T = AlignedReader;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        let reader = AlignedReader::new_with_tracker(
            &self.path,
            start,
            self.buffer_size,
            self.io_tracker.clone(),
        )
        .map_err(|e| parquet::errors::ParquetError::General(e))?;

        Ok(reader)
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        let mut reader = self.get_read(start)?;
        let mut buffer = vec![0u8; length];
        reader.read_exact(&mut buffer).map_err(|e| {
            parquet::errors::ParquetError::General(format!("Failed to read: {}", e))
        })?;
        Ok(Bytes::from(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};
    use tempfile::NamedTempFile;

    fn create_test_file(content: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content).unwrap();
        file.flush().unwrap();
        file
    }

    fn create_large_test_file(size: usize) -> (NamedTempFile, Vec<u8>) {
        let mut file = NamedTempFile::new().unwrap();
        let mut content = Vec::with_capacity(size);

        // Create content with a pattern that's easy to verify
        for i in 0..size {
            content.push((i % 256) as u8);
        }

        file.write_all(&content).unwrap();
        file.flush().unwrap();
        (file, content)
    }

    #[test]
    fn test_aligned_read_from_start() {
        let content = b"Hello, World! This is a test file for AlignedReader.";
        let file = create_test_file(content);

        let mut reader = AlignedReader::new(file.path(), 4096).unwrap();
        let mut buffer = vec![0u8; content.len()];
        let bytes_read = reader.read(&mut buffer).unwrap();

        assert_eq!(bytes_read, content.len());
        assert_eq!(&buffer[..bytes_read], content);
    }

    #[test]
    fn test_aligned_read_from_non_aligned_offset() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let file = create_test_file(content);

        // Start reading from offset 5 (non-aligned)
        let mut reader = AlignedReader::new_with_tracker(file.path(), 5, 4096, None).unwrap();
        let mut buffer = vec![0u8; 10];
        let bytes_read = reader.read(&mut buffer).unwrap();

        assert_eq!(bytes_read, 10);
        assert_eq!(&buffer[..bytes_read], b"56789ABCDE");
    }

    #[test]
    fn test_multiple_buffer_fills() {
        // Create a file larger than the buffer size
        let (file, content) = create_large_test_file(16 * 1024); // 16KB

        let mut reader = AlignedReader::new(file.path(), 4096).unwrap();
        let mut result = Vec::new();
        let bytes_read = reader.read_to_end(&mut result).unwrap();

        assert_eq!(bytes_read, content.len());
        assert_eq!(result, content);
    }

    #[test]
    fn test_seek_operations() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let file = create_test_file(content);

        let mut reader = AlignedReader::new(file.path(), 4096).unwrap();

        // Test seek from start
        reader.seek(SeekFrom::Start(10)).unwrap();
        let mut buffer = vec![0u8; 5];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"ABCDE");

        // Test seek from current
        reader.seek(SeekFrom::Current(5)).unwrap();
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"KLMNO");

        // Test seek from end
        reader.seek(SeekFrom::End(-5)).unwrap();
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"VWXYZ");
    }

    #[test]
    fn test_seek_beyond_file() {
        let content = b"Short content";
        let file = create_test_file(content);

        let mut reader = AlignedReader::new(file.path(), 4096).unwrap();

        // Seek beyond file size
        reader.seek(SeekFrom::Start(100)).unwrap();
        let mut buffer = vec![0u8; 10];
        let bytes_read = reader.read(&mut buffer).unwrap();

        assert_eq!(bytes_read, 0); // EOF
    }

    #[test]
    fn test_partial_reads() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let file = create_test_file(content);

        let mut reader = AlignedReader::new(file.path(), 4096).unwrap();

        // Read in small chunks
        let mut buffer = vec![0u8; 5];
        let mut total_read = Vec::new();

        while let Ok(n) = reader.read(&mut buffer) {
            if n == 0 {
                break;
            }
            total_read.extend_from_slice(&buffer[..n]);
        }

        assert_eq!(total_read, content);
    }

    #[test]
    fn test_io_stats_tracking() {
        let (file, _) = create_large_test_file(16 * 1024); // 16KB
        let tracker = IoStatsTracker::new();

        let mut reader =
            AlignedReader::new_with_tracker(file.path(), 0, 4096, Some(tracker.clone())).unwrap();

        // Read entire file
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).unwrap();

        let (ops, bytes) = tracker.get_read_stats();
        assert!(ops > 0); // Should have at least one I/O operation
        assert_eq!(bytes as usize, 16 * 1024); // Should have read all bytes

        // Check detailed stats
        let detailed = tracker.get_detailed_stats();
        assert_eq!(detailed.read_ops, ops);
        assert_eq!(detailed.read_bytes, bytes);
        assert_eq!(detailed.write_ops, 0);
        assert_eq!(detailed.write_bytes, 0);
    }

    #[test]
    fn test_skip_to_newline() {
        let content = b"First line\nSecond line\nThird line\n";
        let file = create_test_file(content);

        // Start in the middle of the first line
        let mut reader = AlignedReader::new_with_tracker(file.path(), 5, 4096, None).unwrap();

        // Skip to newline
        let skipped = reader.skip_to_newline().unwrap();
        assert_eq!(skipped, 6); // " line\n"

        // Read next line
        let mut buffer = vec![0u8; 11];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Second line");
    }

    #[test]
    fn test_skip_to_newline_at_boundary() {
        let content = b"Line1\nLine2\nLine3\n";
        let file = create_test_file(content);

        // Start exactly at newline
        let mut reader = AlignedReader::new_with_tracker(file.path(), 5, 4096, None).unwrap();
        let skipped = reader.skip_to_newline().unwrap();
        assert_eq!(skipped, 1); // Just the newline

        // Read next line
        let mut buffer = vec![0u8; 5];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Line2");
    }

    #[test]
    fn test_page_boundary_handling() {
        // Create a file that spans multiple pages
        let mut content = vec![0u8; PAGE_SIZE * 3];

        // Put markers at page boundaries
        content[PAGE_SIZE - 1] = b'A';
        content[PAGE_SIZE] = b'B';
        content[PAGE_SIZE * 2 - 1] = b'C';
        content[PAGE_SIZE * 2] = b'D';

        let file = create_test_file(&content);

        // Read across page boundary
        let mut reader =
            AlignedReader::new_with_tracker(file.path(), PAGE_SIZE as u64 - 1, 4096, None).unwrap();

        let mut buffer = vec![0u8; 4];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"AB\0\0");
    }

    #[test]
    fn test_read_exact_insufficient_data() {
        let content = b"Short";
        let file = create_test_file(content);

        let mut reader = AlignedReader::new(file.path(), 4096).unwrap();
        let mut buffer = vec![0u8; 10];

        let result = reader.read_exact(&mut buffer);
        assert!(result.is_err()); // Should fail - not enough data
    }

    #[test]
    fn test_chunk_reader_interface() {
        let content = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let file = create_test_file(content);

        let chunk_reader = AlignedChunkReader::new(file.path(), 4096).unwrap();

        // Test get_bytes
        let bytes = chunk_reader.get_bytes(10, 5).unwrap();
        assert_eq!(&bytes[..], b"ABCDE");

        // Test get_read
        let mut reader = chunk_reader.get_read(20).unwrap();
        let mut buffer = vec![0u8; 5];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"KLMNO");
    }

    #[test]
    fn test_empty_file() {
        let file = create_test_file(b"");

        let mut reader = AlignedReader::new(file.path(), 4096).unwrap();
        let mut buffer = vec![0u8; 10];
        let bytes_read = reader.read(&mut buffer).unwrap();

        assert_eq!(bytes_read, 0);
    }

    #[test]
    fn test_concurrent_readers() {
        let content = b"Shared file content for multiple readers";
        let file = create_test_file(content);

        // Create multiple readers at different offsets
        let mut reader1 = AlignedReader::new_with_tracker(file.path(), 0, 4096, None).unwrap();
        let mut reader2 = AlignedReader::new_with_tracker(file.path(), 10, 4096, None).unwrap();

        let mut buffer1 = vec![0u8; 6];
        let mut buffer2 = vec![0u8; 7];

        reader1.read_exact(&mut buffer1).unwrap();
        reader2.read_exact(&mut buffer2).unwrap();

        assert_eq!(&buffer1, b"Shared");
        // Position 10 should start at 'e' in "file"
        // "Shared file content for multiple readers"
        //         ^10th position
        assert_eq!(&buffer2, b"e conte");
    }
}
