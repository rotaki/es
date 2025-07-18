use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::io;
use libc::{c_void, off_t, pread, pwrite, fstat, O_DIRECT};

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

/// Global file manager that maintains shared file descriptors
/// and provides thread-safe pread/pwrite operations
pub struct GlobalFileManager {
    /// Maps file paths to open file descriptors
    open_files: Arc<RwLock<HashMap<PathBuf, RawFd>>>,
    /// Alignment requirement for Direct I/O
    alignment: usize,
    /// Working directory for all file operations
    work_dir: PathBuf,
    /// Whether to remove the work directory on drop
    is_temporary: bool,
}

impl GlobalFileManager {
    /// Create a new GlobalFileManager with specified alignment
    pub fn new(alignment: usize) -> Self {
        // For backward compatibility, use current directory and non-temporary
        Self::new_with_dir(alignment, ".", false).unwrap()
    }
    
    /// Create a new GlobalFileManager with specified alignment and working directory
    pub fn new_with_dir(alignment: usize, work_dir: impl AsRef<Path>, is_temporary: bool) -> io::Result<Self> {
        let work_dir = work_dir.as_ref().to_path_buf();
        
        // Create the directory if it doesn't exist
        if !work_dir.exists() {
            fs::create_dir_all(&work_dir)?;
        }
        
        Ok(Self {
            open_files: Arc::new(RwLock::new(HashMap::new())),
            alignment,
            work_dir,
            is_temporary,
        })
    }

    /// Get or open a file, returning its file descriptor
    pub fn get_or_open_file(&self, path: &Path, flags: i32) -> io::Result<RawFd> {
        let path_buf = path.to_path_buf();
        
        // First try to get existing fd with read lock
        {
            let files = self.open_files.read().unwrap();
            if let Some(&fd) = files.get(&path_buf) {
                return Ok(fd);
            }
        }

        // Need to open file - acquire write lock
        let mut files = self.open_files.write().unwrap();
        
        // Double-check in case another thread opened it
        if let Some(&fd) = files.get(&path_buf) {
            return Ok(fd);
        }
        
        // Open the file with Direct I/O
        let file = OpenOptions::new()
            .read(true)
            .write(flags & libc::O_WRONLY != 0 || flags & libc::O_RDWR != 0)
            .create(flags & libc::O_CREAT != 0)
            .truncate(flags & libc::O_TRUNC != 0)
            .custom_flags(flags | O_DIRECT)
            .open(&path_buf)?;

        let fd = file.as_raw_fd();
        
        // Important: prevent file from being closed when it goes out of scope
        std::mem::forget(file);
        
        files.insert(path_buf, fd);
        Ok(fd)
    }

    /// Get the raw file descriptor for a given path
    /// 
    /// This method returns the raw file descriptor if the file is already open,
    /// or opens it with read-only access if not.
    pub fn get_fd(&self, path: &Path) -> io::Result<RawFd> {
        self.get_or_open_file(path, libc::O_RDONLY)
    }

    /// Get the raw file descriptor for a given path with specific flags
    /// 
    /// This method returns the raw file descriptor if the file is already open,
    /// or opens it with the specified flags if not.
    pub fn get_fd_with_flags(&self, path: &Path, flags: i32) -> io::Result<RawFd> {
        self.get_or_open_file(path, flags)
    }

    /// Perform aligned pread
    pub fn pread_aligned(&self, path: &Path, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let fd = self.get_or_open_file(path, libc::O_RDONLY)?;
        
        // Ensure buffer is aligned
        let buf_addr = buf.as_ptr() as usize;
        if buf_addr % self.alignment != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer is not properly aligned for Direct I/O"
            ));
        }

        // Ensure offset is aligned
        if offset % self.alignment as u64 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Offset is not properly aligned for Direct I/O"
            ));
        }

        // For Direct I/O, the read size should ideally be aligned too.
        // However, when reading the last block of a file, we might not have
        // an aligned amount of data left. The kernel will handle this by
        // returning fewer bytes than requested.
        let read_size = buf.len();
        
        // Note: We allow non-aligned read sizes for the last block.
        // The kernel will return the actual number of bytes read.
        
        let result = unsafe {
            pread(
                fd,
                buf.as_mut_ptr() as *mut c_void,
                read_size,
                offset as off_t,
            )
        };

        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(result as usize)
        }
    }

    /// Perform aligned pwrite
    pub fn pwrite_aligned(&self, path: &Path, buf: &[u8], offset: u64) -> io::Result<usize> {
        let fd = self.get_or_open_file(path, libc::O_WRONLY | libc::O_CREAT)?;
        
        // Ensure buffer is aligned
        let buf_addr = buf.as_ptr() as usize;
        if buf_addr % self.alignment != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer is not properly aligned for Direct I/O"
            ));
        }

        // Ensure offset is aligned
        if offset % self.alignment as u64 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Offset is not properly aligned for Direct I/O"
            ));
        }

        // Ensure length is aligned
        if buf.len() % self.alignment != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer length is not properly aligned for Direct I/O"
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

    /// Close a specific file
    pub fn close_file(&self, path: &Path) -> io::Result<()> {
        let mut files = self.open_files.write().unwrap();
        if let Some(fd) = files.remove(path) {
            unsafe {
                if libc::close(fd) < 0 {
                    return Err(io::Error::last_os_error());
                }
            }
        }
        Ok(())
    }

    /// Close all open files
    pub fn close_all(&self) -> io::Result<()> {
        let mut files = self.open_files.write().unwrap();
        for (_, fd) in files.drain() {
            unsafe {
                if libc::close(fd) < 0 {
                    return Err(io::Error::last_os_error());
                }
            }
        }
        Ok(())
    }
}

impl Drop for GlobalFileManager {
    fn drop(&mut self) {
        // Best effort to close all files
        let _ = self.close_all();
        
        // Remove the working directory if it's temporary
        if self.is_temporary && self.work_dir.exists() {
            let _ = fs::remove_dir_all(&self.work_dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligned_buffer::AlignedBuffer;
    use tempfile::TempDir;

    fn create_test_manager() -> (GlobalFileManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let manager = GlobalFileManager::new_with_dir(512, temp_dir.path(), false).unwrap();
        (manager, temp_dir)
    }

    #[test]
    fn test_file_manager_basic_operations() {
        let (manager, temp_dir) = create_test_manager();
        let file_path = temp_dir.path().join("test.dat");

        // Create aligned buffer
        let mut write_buf = AlignedBuffer::new(4096, 512).unwrap();
        let data = b"Hello, Direct I/O!";
        write_buf.as_mut_slice()[..data.len()].copy_from_slice(data);

        // Write data
        let written = manager.pwrite_aligned(&file_path, write_buf.as_slice(), 0).unwrap();
        assert_eq!(written, 4096);

        // Read data back
        let mut read_buf = AlignedBuffer::new(4096, 512).unwrap();
        let read = manager.pread_aligned(&file_path, read_buf.as_mut_slice(), 0).unwrap();
        assert_eq!(read, 4096);
        assert_eq!(&read_buf.as_slice()[..data.len()], data);

        // Close file
        manager.close_file(&file_path).unwrap();
    }

    #[test]
    fn test_file_manager_shared_fd() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("shared.dat");
        let manager = Arc::new(GlobalFileManager::new_with_dir(512, temp_dir.path(), false).unwrap());

        // Create file with some data
        let mut buf = AlignedBuffer::new(1024, 512).unwrap();
        buf.as_mut_slice()[..10].copy_from_slice(b"1234567890");
        manager.pwrite_aligned(&file_path, buf.as_slice(), 0).unwrap();

        // Simulate multiple threads reading different offsets
        let file_path1 = file_path.clone();
        let manager1 = Arc::clone(&manager);
        let handle1 = std::thread::spawn(move || {
            let mut buf = AlignedBuffer::new(512, 512).unwrap();
            manager1.pread_aligned(&file_path1, buf.as_mut_slice(), 0).unwrap();
            buf.as_slice()[0]
        });

        let file_path2 = file_path.clone();
        let manager2 = Arc::clone(&manager);
        let handle2 = std::thread::spawn(move || {
            let mut buf = AlignedBuffer::new(512, 512).unwrap();
            manager2.pread_aligned(&file_path2, buf.as_mut_slice(), 512).unwrap();
            buf.as_slice()[0]
        });

        let result1 = handle1.join().unwrap();
        let result2 = handle2.join().unwrap();

        assert_eq!(result1, b'1');
        assert_eq!(result2, 0); // Second block is empty
    }

    #[test]
    fn test_alignment_validation() {
        let (manager, temp_dir) = create_test_manager();
        let file_path = temp_dir.path().join("align_test.dat");

        // Test unaligned buffer
        let mut unaligned = vec![0u8; 1024];
        let result = manager.pread_aligned(&file_path, &mut unaligned, 0);
        assert!(result.is_err());

        // Test unaligned offset
        let mut aligned = AlignedBuffer::new(512, 512).unwrap();
        let result = manager.pread_aligned(&file_path, aligned.as_mut_slice(), 100);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_temporary_directory_cleanup() {
        let temp_path = {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test_work_dir");
            
            // Create manager with is_temporary = true
            {
                let manager = GlobalFileManager::new_with_dir(512, &path, true).unwrap();
                assert!(path.exists());
                
                // Create a file to ensure directory is not empty
                let file_path = path.join("test.dat");
                let mut buf = AlignedBuffer::new(512, 512).unwrap();
                buf.as_mut_slice()[..5].copy_from_slice(b"test\0");
                manager.pwrite_aligned(&file_path, buf.as_slice(), 0).unwrap();
            }
            
            // Manager is dropped here
            path
        };
        
        // Directory should be removed
        assert!(!temp_path.exists());
    }

    #[test]
    fn test_raw_fd_operations() {
        let (manager, temp_dir) = create_test_manager();
        let file_path = temp_dir.path().join("test_fd.dat");

        // Create aligned buffer for initial write
        let mut write_buf = AlignedBuffer::new(512, 512).unwrap();
        let data = b"Test data for FD operations";
        write_buf.as_mut_slice()[..data.len()].copy_from_slice(data);

        // Write initial data
        manager.pwrite_aligned(&file_path, write_buf.as_slice(), 0).unwrap();

        // Get the raw FD
        let fd = manager.get_fd(&file_path).unwrap();

        // Test file_size_fd
        let size = file_size_fd(fd).unwrap();
        assert_eq!(size, 512);

        // Test pread_fd with aligned buffer
        let mut read_buf = AlignedBuffer::new(512, 512).unwrap();
        let bytes_read = pread_fd(fd, read_buf.as_mut_slice(), 0).unwrap();
        assert_eq!(bytes_read, 512);
        assert_eq!(&read_buf.as_slice()[..data.len()], data);

        // Get FD with write flags
        let write_fd = manager.get_fd_with_flags(&file_path, libc::O_RDWR).unwrap();
        
        // Test pwrite_fd (write at offset 512) - must use aligned buffer
        let more_data = b"More data";
        let mut write_buf2 = AlignedBuffer::new(512, 512).unwrap();
        write_buf2.as_mut_slice()[..more_data.len()].copy_from_slice(more_data);
        let bytes_written = pwrite_fd(write_fd, write_buf2.as_slice(), 512).unwrap();
        assert_eq!(bytes_written, 512);

        // Verify the write with pread_fd
        let mut verify_buf = AlignedBuffer::new(512, 512).unwrap();
        let bytes_read = pread_fd(fd, verify_buf.as_mut_slice(), 512).unwrap();
        assert_eq!(bytes_read, 512);
        assert_eq!(&verify_buf.as_slice()[..more_data.len()], more_data);

        // Verify file size increased
        let new_size = file_size_fd(fd).unwrap();
        assert_eq!(new_size, 1024);
    }

    #[test]
    fn test_fd_operations_multiple_files() {
        let (manager, temp_dir) = create_test_manager();
        let file1 = temp_dir.path().join("file1.dat");
        let file2 = temp_dir.path().join("file2.dat");

        // Create some test data
        let data1 = b"File 1 data";
        let data2 = b"File 2 data";

        // Write to both files using manager
        let mut buf1 = AlignedBuffer::new(512, 512).unwrap();
        buf1.as_mut_slice()[..data1.len()].copy_from_slice(data1);
        manager.pwrite_aligned(&file1, buf1.as_slice(), 0).unwrap();

        let mut buf2 = AlignedBuffer::new(512, 512).unwrap();
        buf2.as_mut_slice()[..data2.len()].copy_from_slice(data2);
        manager.pwrite_aligned(&file2, buf2.as_slice(), 0).unwrap();

        // Get FDs for both files
        let fd1 = manager.get_fd(&file1).unwrap();
        let fd2 = manager.get_fd(&file2).unwrap();

        // Read from both using raw FDs with aligned buffers
        let mut read1 = AlignedBuffer::new(512, 512).unwrap();
        let mut read2 = AlignedBuffer::new(512, 512).unwrap();
        
        pread_fd(fd1, read1.as_mut_slice(), 0).unwrap();
        pread_fd(fd2, read2.as_mut_slice(), 0).unwrap();

        assert_eq!(&read1.as_slice()[..data1.len()], data1);
        assert_eq!(&read2.as_slice()[..data2.len()], data2);

        // Verify sizes
        assert_eq!(file_size_fd(fd1).unwrap(), 512);
        assert_eq!(file_size_fd(fd2).unwrap(), 512);
    }
}