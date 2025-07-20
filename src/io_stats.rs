//! I/O statistics tracking module

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Tracks I/O statistics for debugging and performance analysis
#[derive(Debug, Clone)]
pub struct IoStatsTracker {
    // Read statistics
    pub read_ops: Arc<AtomicU64>,
    pub read_bytes: Arc<AtomicU64>,
    // Write statistics
    pub write_ops: Arc<AtomicU64>,
    pub write_bytes: Arc<AtomicU64>,
}

impl Default for IoStatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl IoStatsTracker {
    pub fn new() -> Self {
        Self {
            read_ops: Arc::new(AtomicU64::new(0)),
            read_bytes: Arc::new(AtomicU64::new(0)),
            write_ops: Arc::new(AtomicU64::new(0)),
            write_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Add read statistics
    pub fn add_read(&self, ops: u64, bytes: u64) {
        self.read_ops.fetch_add(ops, Ordering::Relaxed);
        self.read_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Add write statistics
    pub fn add_write(&self, ops: u64, bytes: u64) {
        self.write_ops.fetch_add(ops, Ordering::Relaxed);
        self.write_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get read statistics
    pub fn get_read_stats(&self) -> (u64, u64) {
        (
            self.read_ops.load(Ordering::Relaxed),
            self.read_bytes.load(Ordering::Relaxed),
        )
    }

    /// Get write statistics
    pub fn get_write_stats(&self) -> (u64, u64) {
        (
            self.write_ops.load(Ordering::Relaxed),
            self.write_bytes.load(Ordering::Relaxed),
        )
    }

    /// Get detailed statistics
    pub fn get_detailed_stats(&self) -> IoStats {
        IoStats {
            read_ops: self.read_ops.load(Ordering::Relaxed),
            read_bytes: self.read_bytes.load(Ordering::Relaxed),
            write_ops: self.write_ops.load(Ordering::Relaxed),
            write_bytes: self.write_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Detailed I/O statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoStats {
    pub read_ops: u64,
    pub read_bytes: u64,
    pub write_ops: u64,
    pub write_bytes: u64,
}

impl IoStats {
    /// Get total operations
    pub fn total_ops(&self) -> u64 {
        self.read_ops + self.write_ops
    }

    /// Get total bytes
    pub fn total_bytes(&self) -> u64 {
        self.read_bytes + self.write_bytes
    }
}
