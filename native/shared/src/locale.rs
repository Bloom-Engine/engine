//! The user's preferred languages, most preferred first, as BCP 47 tags
//! ("de-DE", "en-US"). What `navigator.languages` reports in a browser.
//!
//! Explicit POSIX environment variables win (LANGUAGE, then LC_ALL >
//! LC_MESSAGES > LANG), so `LANG=de_DE.UTF-8 ./game` tests a language from a
//! terminal. GUI launches don't set them, so apps fall through to the
//! platform's own list:
//! - Apple: CFLocaleCopyPreferredLanguages, the ordered list from Settings.
//!   It includes the per-app language from iOS/tvOS Settings → the app →
//!   Language, which appears once the app's Info.plist lists more than one
//!   CFBundleLocalizations.
//! - Android: the system locale properties.
//! - Windows: the user's default locale.
//! Empty when nothing is known; callers default to English.

/// The preferred languages, most preferred first, deduplicated.
pub fn preferred_languages() -> Vec<String> {
    let mut raw: Vec<String> = Vec::new();
    if let Ok(list) = std::env::var("LANGUAGE") {
        raw.extend(list.split(':').map(str::to_string));
    }
    if let Some(v) = ["LC_ALL", "LC_MESSAGES", "LANG"].iter().find_map(|var| std::env::var(var).ok()) {
        raw.push(v);
    }
    let mut out: Vec<String> = Vec::new();
    let add = |out: &mut Vec<String>, list: Vec<String>| {
        for tag in list.iter().filter_map(|r| to_bcp47(r)) {
            if !out.contains(&tag) {
                out.push(tag);
            }
        }
    };
    add(&mut out, raw);
    if out.is_empty() {
        add(&mut out, platform::languages());
    }
    out
}

/// "de_DE.UTF-8" / "de-DE" / "zh-Hans-CN" / "de_DE@euro" -> "de-DE" etc.; None for "", "C", "POSIX".
fn to_bcp47(raw: &str) -> Option<String> {
    let s = raw.split(['.', '@']).next().unwrap_or("").trim();
    if s.is_empty() || s == "C" || s == "POSIX" {
        return None;
    }
    let tag = s.replace('_', "-");
    if !tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return None;
    }
    Some(tag)
}

#[cfg(target_vendor = "apple")]
mod platform {
    use std::ffi::c_void;

    type CFTypeRef = *const c_void;
    type CFIndex = isize;
    const UTF8: u32 = 0x0800_0100;

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFLocaleCopyPreferredLanguages() -> CFTypeRef;
        fn CFArrayGetCount(array: CFTypeRef) -> CFIndex;
        fn CFArrayGetValueAtIndex(array: CFTypeRef, idx: CFIndex) -> CFTypeRef;
        fn CFStringGetCString(string: CFTypeRef, buffer: *mut u8, size: CFIndex, encoding: u32) -> bool;
        fn CFRelease(cf: CFTypeRef);
    }

    pub fn languages() -> Vec<String> {
        let mut out = Vec::new();
        unsafe {
            let array = CFLocaleCopyPreferredLanguages();
            if array.is_null() {
                return out;
            }
            for i in 0..CFArrayGetCount(array) {
                let s = CFArrayGetValueAtIndex(array, i);
                let mut buf = [0u8; 64];
                if !s.is_null() && CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as CFIndex, UTF8) {
                    let len = buf.iter().position(|&b| b == 0).unwrap_or(0);
                    if let Ok(tag) = std::str::from_utf8(&buf[..len]) {
                        out.push(tag.to_string());
                    }
                }
            }
            CFRelease(array);
        }
        out
    }
}

#[cfg(target_os = "android")]
mod platform {
    extern "C" {
        fn __system_property_get(name: *const u8, value: *mut u8) -> i32;
    }

    pub fn languages() -> Vec<String> {
        let props: [&[u8]; 3] = [b"persist.sys.locale\0", b"ro.product.locale\0", b"persist.sys.language\0"];
        for prop in props {
            let mut buf = [0u8; 92]; // PROP_VALUE_MAX
            let len = unsafe { __system_property_get(prop.as_ptr(), buf.as_mut_ptr()) };
            if len > 0 {
                if let Ok(s) = std::str::from_utf8(&buf[..len as usize]) {
                    if !s.is_empty() {
                        return vec![s.to_string()];
                    }
                }
            }
        }
        Vec::new()
    }
}

#[cfg(target_os = "windows")]
mod platform {
    extern "system" {
        fn GetUserDefaultLocaleName(name: *mut u16, len: i32) -> i32;
    }

    pub fn languages() -> Vec<String> {
        let mut buf = [0u16; 85]; // LOCALE_NAME_MAX_LENGTH
        let len = unsafe { GetUserDefaultLocaleName(buf.as_mut_ptr(), buf.len() as i32) };
        if len > 1 { vec![String::from_utf16_lossy(&buf[..len as usize - 1])] } else { Vec::new() }
    }
}

#[cfg(not(any(target_vendor = "apple", target_os = "android", target_os = "windows")))]
mod platform {
    pub fn languages() -> Vec<String> { Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::to_bcp47;

    #[test]
    fn normalizes_posix_locales() {
        assert_eq!(to_bcp47("de_DE.UTF-8").as_deref(), Some("de-DE"));
        assert_eq!(to_bcp47("de_DE@euro").as_deref(), Some("de-DE"));
        assert_eq!(to_bcp47("zh-Hans-CN").as_deref(), Some("zh-Hans-CN"));
        assert_eq!(to_bcp47("en").as_deref(), Some("en"));
        assert_eq!(to_bcp47("C"), None);
        assert_eq!(to_bcp47("POSIX"), None);
        assert_eq!(to_bcp47(""), None);
    }

    #[test]
    fn lists_something_on_apple() {
        // the build machine always has a language list
        if cfg!(target_vendor = "apple") {
            assert!(!super::platform::languages().is_empty());
        }
    }
}
