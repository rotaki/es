use libc::{c_void, fstat, off_t, pread, pwrite};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::ops::Deref;
use std::os::fd::IntoRawFd;
use std::os::unix::io::RawFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::constants::{open_file_with_direct_io, DIRECT_IO_ALIGNMENT};

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
