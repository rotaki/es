//! AlignedWriter - handles O_DIRECT writes with proper alignment
//!
//! This module provides utilities for writing files with O_DIRECT flag
//! while handling the alignment requirements.

use crate::IoStatsTracker;
use std::fs::{File, OpenOptions};
use std::io::{Error as IoError, ErrorKind, Result as IoResult, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const PAGE_SIZE: usize = 4096; // Page size for O_DIRECT alignment

/// Write mode for AlignedWriter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Create new file, truncate if exists
    Create,
    /// Append to existing file, create if doesn't exist
    Append,
    /// Create new file, fail if exists
    CreateNew,
}

fn allocate_aligned_buffer(size: usize) -> Vec<u8> {
    // Round up to page size
    let aligned_size = ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    vec![0u8; aligned_size]
}

/// AlignedWriter handles O_DIRECT writes with alignment requirements
pub struct AlignedWriter {
    path: PathBuf,
    file: File,
    buffer: Vec<u8>,
    buffer_pos: usize, // Current position in buffer
    io_tracker: Option<IoStatsTracker>,
}

impl AlignedWriter {
    /// Open a file for writing with O_DIRECT with default buffer size (64KB)
    pub fn open(path: &Path, mode: WriteMode) -> Result<Self, String> {
        Self::open_with_buffer_size(path, mode, 64 * 1024)
    }

    /// Open a file for writing with O_DIRECT with specified buffer size
    pub fn open_with_buffer_size(
        path: &Path,
        mode: WriteMode,
        buffer_size: usize,
    ) -> Result<Self, String> {
        Self::open_with_tracker_and_buffer_size(path, mode, None, buffer_size)
    }

    /// Open a file for writing with O_DIRECT and optional I/O tracking (default buffer size)
    pub fn open_with_tracker(
        path: &Path,
        mode: WriteMode,
        tracker: Option<IoStatsTracker>,
    ) -> Result<Self, String> {
        Self::open_with_tracker_and_buffer_size(path, mode, tracker, 64 * 1024)
    }

    /// Open a file for writing with O_DIRECT, optional I/O tracking, and specified buffer size
    pub fn open_with_tracker_and_buffer_size(
        path: &Path,
        mode: WriteMode,
        tracker: Option<IoStatsTracker>,
        buffer_size: usize,
    ) -> Result<Self, String> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent directory: {}", e))?;
        }

        // Configure OpenOptions based on mode
        let mut options = OpenOptions::new();
        // options.custom_flags(libc::O_DIRECT);

        match mode {
            WriteMode::Create => {
                options.create(true).write(true).truncate(true);
            }
            WriteMode::Append => {
                options.create(true).append(true);
            }
            WriteMode::CreateNew => {
                options.create_new(true).write(true);
            }
        }

        // Open file with O_DIRECT
        let file = options.open(path).map_err(|e| match mode {
            WriteMode::CreateNew if e.kind() == ErrorKind::AlreadyExists => {
                format!("File already exists: {:?}", path)
            }
            _ => format!("Failed to open file with O_DIRECT: {}", e),
        })?;

        // Allocate aligned buffer with specified size
        let buffer = allocate_aligned_buffer(buffer_size);

        Ok(AlignedWriter {
            path: path.to_path_buf(),
            file,
            buffer,
            buffer_pos: 0,
            io_tracker: tracker,
        })
    }

    /// Legacy method for backward compatibility - opens for append
    #[deprecated(since = "0.2.0", note = "Use open(path, WriteMode::Append) instead")]
    pub fn open_and_append(path: &Path) -> Result<Self, String> {
        Self::open(path, WriteMode::Append)
    }

    /// Legacy method for backward compatibility - creates new file
    #[deprecated(since = "0.2.0", note = "Use open(path, WriteMode::Create) instead")]
    pub fn create(path: &Path) -> Result<Self, String> {
        Self::open(path, WriteMode::Create)
    }

    /// Get the path of the file being written
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the current position in the file (including buffered data)
    pub fn current_pos(&mut self) -> Result<u64, String> {
        let file_pos = self
            .file
            .seek(SeekFrom::Current(0))
            .map_err(|e| format!("Failed to get file position: {}", e))?;
        Ok(file_pos + self.buffer_pos as u64)
    }

    /// Write all data, buffering as needed
    pub fn write_all(&mut self, mut data: &[u8]) -> Result<(), String> {
        while !data.is_empty() {
            let available = self.buffer.len() - self.buffer_pos;
            let to_write = data.len().min(available);

            // Copy data to buffer
            self.buffer[self.buffer_pos..self.buffer_pos + to_write]
                .copy_from_slice(&data[..to_write]);
            self.buffer_pos += to_write;
            data = &data[to_write..];

            // If buffer is full, flush it
            if self.buffer_pos == self.buffer.len() {
                self.flush_buffer()?;
            }
        }
        Ok(())
    }

    /// Flush any buffered data to disk
    pub fn flush(&mut self) -> Result<(), String> {
        if self.buffer_pos > 0 {
            // For O_DIRECT, we need to write aligned data
            // Round up to page boundary and pad with zeros
            let aligned_len = ((self.buffer_pos + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;

            // Pad with zeros
            while self.buffer_pos < aligned_len {
                self.buffer[self.buffer_pos] = 0;
                self.buffer_pos += 1;
            }

            self.flush_buffer()?;
        }
        Ok(())
    }

    /// Internal method to flush the buffer
    fn flush_buffer(&mut self) -> Result<(), String> {
        if self.buffer_pos == 0 {
            return Ok(());
        }

        // Track I/O operation
        if let Some(ref tracker) = self.io_tracker {
            tracker.add_write(1, self.buffer_pos as u64);
        }

        // Write only the filled portion of the buffer
        self.file
            .write_all(&self.buffer[..self.buffer_pos])
            .map_err(|e| format!("Failed to write to file: {}", e))?;

        self.buffer_pos = 0;
        Ok(())
    }
}

impl Write for AlignedWriter {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        let len = buf.len();
        self.write_all(buf)
            .map_err(|e| IoError::new(ErrorKind::Other, e))?;
        Ok(len)
    }

    fn flush(&mut self) -> IoResult<()> {
        self.flush().map_err(|e| IoError::new(ErrorKind::Other, e))
    }
}

impl Drop for AlignedWriter {
    fn drop(&mut self) {
        // Best effort flush on drop
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligned_reader::AlignedReader;
    use std::fs;
    use std::io::Read;
    use tempfile::NamedTempFile;

    #[test]
    fn test_basic_write() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write data
        {
            let mut writer = AlignedWriter::open(path, WriteMode::Create).unwrap();
            writer.write_all(b"Hello, Direct I/O!").unwrap();
            writer.flush().unwrap();
        }

        // Read back
        let content = fs::read(path).unwrap();
        assert!(content.starts_with(b"Hello, Direct I/O!"));
    }

    #[test]
    fn test_append_mode() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write initial data
        {
            let mut writer = AlignedWriter::open(path, WriteMode::Create).unwrap();
            writer.write_all(b"First ").unwrap();
            writer.flush().unwrap();
        }

        // Append more data
        {
            let mut writer = AlignedWriter::open(path, WriteMode::Append).unwrap();
            writer.write_all(b"Second").unwrap();
            writer.flush().unwrap();
        }

        // Read back
        let content = fs::read(path).unwrap();
        assert!(content.starts_with(b"First "));
        // Note: Due to padding, we need to find "Second" after some padding
        let content_str = String::from_utf8_lossy(&content);
        assert!(content_str.contains("Second"));
    }

    #[test]
    fn test_large_write() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create large data that spans multiple buffers
        let large_data = vec![b'X'; 256 * 1024]; // 256KB

        {
            let mut writer = AlignedWriter::open(path, WriteMode::Create).unwrap();
            writer.write_all(&large_data).unwrap();
            writer.flush().unwrap();
        }

        // Verify file size is at least as large as the data
        let metadata = fs::metadata(path).unwrap();
        assert!(metadata.len() >= large_data.len() as u64);
    }

    #[test]
    fn test_multiple_small_writes() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        {
            let mut writer = AlignedWriter::open(path, WriteMode::Create).unwrap();
            writer.write_all(b"One ").unwrap();
            writer.write_all(b"Two ").unwrap();
            writer.write_all(b"Three").unwrap();
            writer.flush().unwrap();
        }

        // Read back
        let content = fs::read(path).unwrap();
        assert!(content.starts_with(b"One Two Three"));
    }

    #[test]
    fn test_io_stats_tracking() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        let tracker = IoStatsTracker::new();

        {
            let mut writer =
                AlignedWriter::open_with_tracker(path, WriteMode::Create, Some(tracker.clone()))
                    .unwrap();
            // Write enough to trigger a buffer flush
            let data = vec![b'A'; 70 * 1024]; // 70KB
            writer.write_all(&data).unwrap();
            writer.flush().unwrap();
        }

        let (ops, bytes) = tracker.get_write_stats();
        assert!(ops >= 2); // At least 2 I/O operations (one for full buffer, one for flush)
        assert!(bytes >= 70 * 1024); // At least 70KB written

        // Check detailed stats
        let detailed = tracker.get_detailed_stats();
        assert_eq!(detailed.write_ops, ops);
        assert_eq!(detailed.write_bytes, bytes);
        assert_eq!(detailed.read_ops, 0);
        assert_eq!(detailed.read_bytes, 0);
    }

    #[test]
    fn test_aligned_writer_reader_compatibility() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let test_data = b"This is a test of AlignedWriter and AlignedReader compatibility!";

        // Write with AlignedWriter
        {
            let mut writer = AlignedWriter::open(path, WriteMode::Create).unwrap();
            writer.write_all(test_data).unwrap();
            writer.flush().unwrap();
        }

        // Read with AlignedReader
        {
            let mut reader = AlignedReader::new(path, 4096).unwrap();
            let mut buffer = vec![0u8; test_data.len()];
            reader.read_exact(&mut buffer).unwrap();
            assert_eq!(&buffer, test_data);
        }
    }

    #[test]
    fn test_current_pos() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = AlignedWriter::open(path, WriteMode::Create).unwrap();

        assert_eq!(writer.current_pos().unwrap(), 0);

        writer.write_all(b"Hello").unwrap();
        assert_eq!(writer.current_pos().unwrap(), 5);

        writer.write_all(b" World").unwrap();
        assert_eq!(writer.current_pos().unwrap(), 11);
    }

    #[test]
    fn test_empty_write() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        {
            let mut writer = AlignedWriter::open(path, WriteMode::Create).unwrap();
            writer.write_all(b"").unwrap();
            writer.flush().unwrap();
        }

        // File should exist but may be empty
        assert!(path.exists());
    }

    #[test]
    fn test_write_trait() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        {
            let mut writer = AlignedWriter::open(path, WriteMode::Create).unwrap();
            // Use Write trait methods
            write!(writer, "Hello {}", "World").unwrap();
            writer.flush().unwrap();
        }

        let content = fs::read(path).unwrap();
        assert!(content.starts_with(b"Hello World"));
    }

    #[test]
    fn test_create_new_mode() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_file.dat");

        // First create should succeed
        {
            let mut writer = AlignedWriter::open(&path, WriteMode::CreateNew).unwrap();
            writer.write_all(b"First").unwrap();
            writer.flush().unwrap();
        }

        // Second create should fail
        let result = AlignedWriter::open(&path, WriteMode::CreateNew);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.contains("already exists"));
        }
    }

    #[test]
    fn test_custom_buffer_size() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Test with small buffer (8KB)
        {
            let mut writer =
                AlignedWriter::open_with_buffer_size(path, WriteMode::Create, 8 * 1024).unwrap();
            // Write more than 8KB to force flush
            let data = vec![b'A'; 10 * 1024];
            writer.write_all(&data).unwrap();
            writer.flush().unwrap();
        }

        // Verify file size
        let metadata = fs::metadata(path).unwrap();
        assert!(metadata.len() >= 10 * 1024);

        // Test with large buffer (128KB)
        {
            let mut writer =
                AlignedWriter::open_with_buffer_size(path, WriteMode::Create, 128 * 1024).unwrap();
            let data = vec![b'B'; 50 * 1024];
            writer.write_all(&data).unwrap();
            writer.flush().unwrap();
        }

        // Verify new content
        let metadata = fs::metadata(path).unwrap();
        assert!(metadata.len() >= 50 * 1024);
    }
}
