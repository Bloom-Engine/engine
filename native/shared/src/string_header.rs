/// Header for heap-allocated strings (mirrors perry_runtime::string::StringHeader).
/// Defined locally to avoid pulling in the entire perry-runtime crate as a dependency.
///
/// Perry 0.5.x changed the layout — utf16_len was added at offset 0 for inline
/// codegen, so byte_len moved to offset 4.
#[repr(C)]
pub struct StringHeader {
    /// Length in UTF-16 code units (JS `.length` semantics). Offset 0.
    pub utf16_len: u32,
    /// Length in UTF-8 bytes. Offset 4.
    pub byte_len: u32,
    /// Capacity in bytes.
    pub capacity: u32,
    /// Reference hint: 0=shared, 1=unique (in-place append OK).
    pub refcount: u32,
    /// Bit flags (STRING_FLAG_HAS_LONE_SURROGATES = 1). Added in
    /// Perry 0.5.18x. Engines built against older Perry headers
    /// skipped 16 bytes and ended up reading 4 bytes into the
    /// UTF-8 data; strings crossing the FFI came back with a 4-byte
    /// garbage prefix and trailing truncation.
    pub flags: u32,
}

/// Extract a &str from a *const StringHeader pointer (Perry string format).
pub fn str_from_header(ptr: *const u8) -> &'static str {
    if ptr.is_null() || (ptr as usize) < 0x1000 {
        return "";
    }
    unsafe {
        let header = ptr as *const StringHeader;
        let len = (*header).byte_len as usize;
        let data = ptr.add(std::mem::size_of::<StringHeader>());
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(data, len))
    }
}

/// Allocate a Perry 0.5.x heap string suitable for returning across the
/// FFI boundary (declared as `returns: "string"` in package.json). Layout
/// matches `StringHeader` above: 5×u32 header (utf16_len, byte_len,
/// capacity, refcount, flags) followed by UTF-8 data.
///
/// Older engine code allocated 12 bytes (Perry 0.4.x layout) and Perry's
/// 0.5.x runtime read 8 bytes into the payload — strings came back with a
/// garbage prefix and read past the end. Always go through this helper.
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
        if ptr.is_null() { return std::ptr::null(); }
        *(ptr as *mut u32)         = utf16_len as u32;
        *(ptr.add(4)  as *mut u32) = byte_len  as u32;
        *(ptr.add(8)  as *mut u32) = byte_len  as u32; // capacity
        *(ptr.add(12) as *mut u32) = 1;                // refcount=unique
        *(ptr.add(16) as *mut u32) = 0;                // flags
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(20), byte_len);
        ptr
    }
}
