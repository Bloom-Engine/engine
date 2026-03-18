/// Header for heap-allocated strings (mirrors perry_runtime::string::StringHeader).
/// Defined locally to avoid pulling in the entire perry-runtime crate as a dependency.
#[repr(C)]
pub struct StringHeader {
    /// Length in bytes (not chars - we store UTF-8)
    pub length: u32,
    /// Capacity (allocated space for data)
    pub capacity: u32,
}

/// Extract a &str from a *const StringHeader pointer (Perry string format).
pub fn str_from_header(ptr: *const u8) -> &'static str {
    if ptr.is_null() || (ptr as usize) < 0x1000 {
        return "";
    }
    unsafe {
        let header = ptr as *const StringHeader;
        let len = (*header).length as usize;
        let data = ptr.add(std::mem::size_of::<StringHeader>());
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(data, len))
    }
}
