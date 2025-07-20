use std::alloc::{alloc, dealloc, Layout};
use std::io;
use std::slice;

/// A buffer that is aligned to a specific boundary for Direct I/O operations
pub struct AlignedBuffer {
    ptr: *mut u8,
    capacity: usize,
    alignment: usize,
    layout: Layout,
}

impl AlignedBuffer {
    /// Create a new aligned buffer with the specified size and alignment
    pub fn new(size: usize, alignment: usize) -> io::Result<Self> {
        // Ensure alignment is a power of 2
        if !alignment.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Alignment must be a power of 2",
            ));
        }

        // Create layout with proper alignment
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        // Allocate aligned memory
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "Failed to allocate aligned memory",
            ));
        }

        // Initialize with zeros for safety and predictability
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }

        Ok(Self {
            ptr,
            capacity: size,
            alignment,
            layout,
        })
    }

    /// Get the buffer as a slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.capacity) }
    }

    /// Get the buffer as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.capacity) }
    }

    /// Get the raw pointer to the buffer
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get the mutable raw pointer to the buffer
    #[allow(dead_code)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the alignment of the buffer
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Check if the buffer is properly aligned
    pub fn is_aligned(&self) -> bool {
        self.ptr as usize % self.alignment == 0
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}

// Safety: AlignedBuffer owns its memory and ensures proper cleanup
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_creation() {
        let buffer = AlignedBuffer::new(4096, 512).unwrap();
        assert_eq!(buffer.capacity(), 4096);
        assert_eq!(buffer.alignment(), 512);
        assert!(buffer.is_aligned());
    }

    #[test]
    fn test_aligned_buffer_read_write() {
        let mut buffer = AlignedBuffer::new(1024, 512).unwrap();

        // Write data
        let data = b"Hello, aligned buffer!";
        buffer.as_mut_slice()[..data.len()].copy_from_slice(data);

        // Read data
        assert_eq!(&buffer.as_slice()[..data.len()], data);
    }

    #[test]
    fn test_alignment_validation() {
        // Non-power-of-2 alignment should fail
        assert!(AlignedBuffer::new(1024, 513).is_err());

        // Power-of-2 alignments should succeed
        assert!(AlignedBuffer::new(1024, 512).is_ok());
        assert!(AlignedBuffer::new(1024, 1024).is_ok());
        assert!(AlignedBuffer::new(1024, 4096).is_ok());
    }

    #[test]
    fn test_buffer_alignment() {
        let alignments = vec![512, 1024, 4096];

        for align in alignments {
            let buffer = AlignedBuffer::new(8192, align).unwrap();
            assert_eq!(buffer.as_ptr() as usize % align, 0);
        }
    }
}
