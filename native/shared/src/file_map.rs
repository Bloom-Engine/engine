//! A file's bytes for loading, memory-mapped where the platform allows.
//!
//! A model file is mostly texture levels, and with texture streaming on most
//! of those are never uploaded (only the small levels are, the rest is read
//! later and only when needed). Reading the whole file brought every byte in
//! from disk and held it in memory for the load. A read-only mapping brings
//! in only the pages the loader touches, and those are clean file pages:
//! iOS does not count them against the app's memory limit.
//!
//! The mapping lives as long as the `FileBytes` (the load), so a file must not
//! be truncated while it is being loaded (reading past its new end would
//! fault). Elsewhere (Windows, the web) the file is read whole as before.

use std::ops::Deref;
use std::path::Path;

/// The bytes of a file (see [`read`]). The mapping is private to this module.
pub struct FileBytes(Inner);

enum Inner {
    #[cfg(unix)]
    Mapped {
        ptr: *const u8,
        len: usize,
    },
    Owned(Vec<u8>),
}

impl Deref for FileBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match &self.0 {
            #[cfg(unix)]
            Inner::Mapped { ptr, len } => {
                // SAFETY: ptr/len describe a live read-only mapping owned by self.
                unsafe { std::slice::from_raw_parts(*ptr, *len) }
            }
            Inner::Owned(bytes) => bytes,
        }
    }
}

impl Drop for FileBytes {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Inner::Mapped { ptr, len } = self.0 {
            // SAFETY: the mapping was made by `read` with this address and length.
            unsafe {
                libc::munmap(ptr as *mut libc::c_void, len);
            }
        }
    }
}

/// The bytes of the file at `path`: mapped on Unix-like systems (falling back
/// to a read when the mapping fails), read whole elsewhere.
///
/// # Safety
///
/// While the returned `FileBytes` lives, the file must not be truncated or
/// rewritten by anyone: a mapping is not a snapshot, and reading a page past
/// a new end of file faults. Engine loads hold it only for the load of one
/// model from the game's own assets.
pub unsafe fn read(path: &Path) -> std::io::Result<FileBytes> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len > 0 {
            // SAFETY: a private read-only mapping of an open file; the fd may be
            // closed afterwards, the mapping keeps the file.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr != libc::MAP_FAILED {
                return Ok(FileBytes(Inner::Mapped {
                    ptr: ptr as *const u8,
                    len,
                }));
            }
        }
    }
    std::fs::read(path).map(|bytes| FileBytes(Inner::Owned(bytes)))
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_mapped_file_reads_as_its_bytes() {
        let dir = std::env::temp_dir().join(format!("bloom-file-map-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bytes.bin");
        let bytes: Vec<u8> = (0..100_000u32).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        // SAFETY (all three): the test's own files, not changed while mapped.
        assert_eq!(&*unsafe { super::read(&path) }.unwrap(), &bytes[..]);
        std::fs::write(&path, b"").unwrap();
        assert!(unsafe { super::read(&path) }.unwrap().is_empty());
        assert!(unsafe { super::read(&dir.join("missing")) }.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
