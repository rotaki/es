use libc::{c_void, fstat, off_t, pread, pwrite};
use std::os::unix::io::RawFd;
use std::path::Path;
use std::{io, os::fd::IntoRawFd};

use crate::diskio::constants::{DIRECT_IO_ALIGNMENT, open_file_with_direct_io};

pub struct SharedFd {
    fd: RawFd,
}

impl From<RawFd> for SharedFd {
    fn from(fd: RawFd) -> Self {
        Self { fd }
    }
}

impl SharedFd {
    /// Create a new SharedFd from a raw file descriptor
    pub fn new(fd: RawFd) -> Self {
        Self { fd }
    }

    pub fn new_from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = open_file_with_direct_io(path.as_ref())?;
        let fd = file.into_raw_fd();
        Ok(Self { fd })
    }

    /// Get the raw file descriptor
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl Drop for SharedFd {
    fn drop(&mut self) {
        // Close the file descriptor when SharedFd is dropped
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// Get the size of a file using its raw file descriptor
pub fn file_size_fd(fd: RawFd) -> io::Result<u64> {
    let mut stat_buf: libc::stat = unsafe { std::mem::zeroed() };

    let result = unsafe { fstat(fd, &mut stat_buf) };

    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(stat_buf.st_size as u64)
    }
}

/// Perform pread using raw file descriptor
///
/// This function reads data from a file at a specific offset without changing
/// the file position. It's thread-safe and doesn't require synchronization.
pub fn pread_fd(fd: RawFd, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let result = unsafe {
        pread(
            fd,
            buf.as_mut_ptr() as *mut c_void,
            buf.len(),
            offset as off_t,
        )
    };

    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result as usize)
    }
}

/// Perform pwrite using raw file descriptor
///
/// This function writes data to a file at a specific offset without changing
/// the file position. It's thread-safe and doesn't require synchronization.
pub fn pwrite_fd(fd: RawFd, buf: &[u8], offset: u64) -> io::Result<usize> {
    // Ensure buffer is aligned
    let buf_addr = buf.as_ptr() as usize;
    if buf_addr % DIRECT_IO_ALIGNMENT != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Buffer is not properly aligned for Direct I/O",
        ));
    }

    // Ensure offset is aligned
    if offset % DIRECT_IO_ALIGNMENT as u64 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Offset is not properly aligned for Direct I/O",
        ));
    }

    // Ensure length is aligned
    if buf.len() % DIRECT_IO_ALIGNMENT != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Buffer length is not properly aligned for Direct I/O",
        ));
    }

    let result = unsafe {
        pwrite(
            fd,
            buf.as_ptr() as *const c_void,
            buf.len(),
            offset as off_t,
        )
    };

    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result as usize)
    }
}
