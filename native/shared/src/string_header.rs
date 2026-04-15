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
