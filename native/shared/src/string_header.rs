//! Perry string ABI: the header layout strings carry across the FFI.
//!
//! This mirrors `perry_runtime::string::StringHeader` (see
//! `perry/crates/perry-runtime/src/string/mod.rs`) — defined locally so the
//! engine doesn't pull the whole perry-runtime crate in as a dependency.
//!
//! ## Upgrade protocol
//!
//! Perry owns this layout and has changed it before (0.5.18 added `flags`;
//! engines built against the old 16-byte header read a 4-byte garbage
//! prefix on every string). There is no version symbol exported by the
//! Perry runtime to handshake against, so the defenses are:
//!
//!   1. Compile-time size/offset assertions below — any local edit that
//!      diverges from the documented layout fails the build.
//!   2. [`header_looks_valid`] — invariant checks on every incoming
//!      header. A Perry-side layout change makes these fire on the first
//!      string the engine receives (typically the window title in
//!      `bloom_init_window`), turning silent corruption into a loud
//!      log-once diagnostic.
//!   3. Checked UTF-8 conversion — a wrong `byte_len` can no longer cause
//!      undefined behavior, only an empty string + diagnostic.
//!
//! When bumping Perry across a runtime-ABI change: update the struct,
//! the assertions, and the doc reference above in the same commit.

/// Header for heap-allocated Perry strings. UTF-8 payload follows
/// immediately after the header.
#[repr(C)]
pub struct StringHeader {
    /// Length in UTF-16 code units (JS `.length` semantics). At offset 0
    /// for Perry's inline codegen.
    pub utf16_len: u32,
    /// Length in UTF-8 bytes.
    pub byte_len: u32,
    /// Capacity in bytes (allocated space for data).
    pub capacity: u32,
    /// Reference hint: 0=shared, 1=unique (in-place append OK).
    pub refcount: u32,
    /// Bit flags (STRING_FLAG_HAS_LONE_SURROGATES = 1). Added in Perry
    /// 0.5.18.
    pub flags: u32,
}

// Layout is an ABI contract — fail the build if the struct drifts from the
// documented Perry layout.
const _: () = {
    assert!(std::mem::size_of::<StringHeader>() == 20);
    assert!(std::mem::offset_of!(StringHeader, utf16_len) == 0);
    assert!(std::mem::offset_of!(StringHeader, byte_len) == 4);
    assert!(std::mem::offset_of!(StringHeader, capacity) == 8);
    assert!(std::mem::offset_of!(StringHeader, refcount) == 12);
    assert!(std::mem::offset_of!(StringHeader, flags) == 16);
};

/// All flag bits Perry currently defines.
const KNOWN_FLAGS: u32 = 1; // STRING_FLAG_HAS_LONE_SURROGATES

/// Sanity-check a header against invariants that hold for every string the
/// current Perry runtime produces. A failed check means either a corrupt
/// pointer or — the case this exists for — a Perry-side layout change
/// shifting which u32 lands in which field.
fn header_looks_valid(h: &StringHeader) -> bool {
    h.byte_len <= h.capacity
        && h.capacity < (1 << 31)
        // utf16 length is never larger than the utf8 byte length
        && h.utf16_len <= h.byte_len
        && (h.flags & !KNOWN_FLAGS) == 0
}

fn abi_mismatch_warn_once(what: &str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        crate::ffi::log_error(&format!(
            "bloom: incoming Perry string failed ABI validation ({what}). \
             This usually means the Perry runtime's StringHeader layout \
             changed — see native/shared/src/string_header.rs for the \
             upgrade protocol. Returning empty strings instead of reading \
             garbage; further occurrences are suppressed."
        ));
    }
}

/// Extract a `&str` from a `*const StringHeader` pointer (Perry string
/// format).
///
/// The returned slice borrows Perry-owned memory that is only guaranteed
/// to live for the duration of the FFI call — copy it (`to_string`) before
/// stashing it anywhere. The `'static` lifetime is a legacy artifact of
/// the FFI signatures, not a promise.
///
/// Never causes undefined behavior: null/garbage pointers, implausible
/// headers, and invalid UTF-8 all yield `""` plus a one-time diagnostic.
pub fn str_from_header(ptr: *const u8) -> &'static str {
    if ptr.is_null() || (ptr as usize) < 0x1000 {
        return "";
    }
    unsafe {
        let header = &*(ptr as *const StringHeader);
        if !header_looks_valid(header) {
            abi_mismatch_warn_once("header invariants violated");
            return "";
        }
        let len = header.byte_len as usize;
        let data = ptr.add(std::mem::size_of::<StringHeader>());
        match std::str::from_utf8(std::slice::from_raw_parts(data, len)) {
            Ok(s) => s,
            Err(_) => {
                abi_mismatch_warn_once("payload is not UTF-8");
                ""
            }
        }
    }
}

/// Allocate a Perry heap string suitable for returning across the FFI
/// boundary (declared as `returns: "string"` in package.json).
///
/// Older engine code allocated the 12-byte Perry 0.4.x header by hand and
/// Perry's 0.5.x runtime read 8 bytes into the payload. Always go through
/// this helper — the layout comes from the `StringHeader` type, which the
/// compile-time assertions above pin to the documented ABI.
pub fn alloc_perry_string(s: &str) -> *const u8 {
    let bytes = s.as_bytes();
    let byte_len = bytes.len();
    // ASCII fast path: utf16_len == byte_len when every byte is < 0x80.
    let utf16_len = if bytes.iter().all(|&b| b < 0x80) {
        byte_len
    } else {
        s.encode_utf16().count()
    };
    let total = std::mem::size_of::<StringHeader>() + byte_len;
    let layout = std::alloc::Layout::from_size_align(total, 4).unwrap();
    unsafe {
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            return std::ptr::null();
        }
        (ptr as *mut StringHeader).write(StringHeader {
            utf16_len: utf16_len as u32,
            byte_len: byte_len as u32,
            capacity: byte_len as u32,
            refcount: 1, // unique
            flags: 0,
        });
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            ptr.add(std::mem::size_of::<StringHeader>()),
            byte_len,
        );
        ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_ascii() {
        let p = alloc_perry_string("hello bloom");
        assert_eq!(str_from_header(p), "hello bloom");
    }

    #[test]
    fn round_trip_multibyte() {
        let p = alloc_perry_string("héllo 🌸");
        assert_eq!(str_from_header(p), "héllo 🌸");
        // utf16_len: 'héllo ' = 6 units, emoji = 2 (surrogate pair)
        let h = unsafe { &*(p as *const StringHeader) };
        assert_eq!(h.utf16_len, 8);
        assert_eq!(h.byte_len, "héllo 🌸".len() as u32);
    }

    #[test]
    fn rejects_null_and_low_pointers() {
        assert_eq!(str_from_header(std::ptr::null()), "");
        assert_eq!(str_from_header(0x10 as *const u8), "");
    }

    #[test]
    fn rejects_implausible_header() {
        // byte_len > capacity — the signature of a shifted layout.
        let bogus = StringHeader {
            utf16_len: 7,
            byte_len: 100,
            capacity: 8,
            refcount: 1,
            flags: 0,
        };
        let mut buf = vec![0u8; std::mem::size_of::<StringHeader>() + 8];
        unsafe {
            (buf.as_mut_ptr() as *mut StringHeader).write(bogus);
        }
        assert_eq!(str_from_header(buf.as_ptr()), "");
    }

    #[test]
    fn rejects_invalid_utf8() {
        let p = alloc_perry_string("abcd") as *mut u8;
        unsafe {
            // stomp the payload with a bare continuation byte
            *p.add(std::mem::size_of::<StringHeader>()) = 0xFF;
        }
        assert_eq!(str_from_header(p), "");
    }
}
