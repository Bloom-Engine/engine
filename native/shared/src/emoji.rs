//! Color emoji rendered by the system's text stack.
//!
//! Bloom's text renderer draws outline glyphs; color emoji are bitmaps in
//! formats only the system reads in full (Apple Color Emoji keeps PNGs on
//! macOS but its own compressed `emjc` images on iOS, and composes skin
//! tones and ZWJ sequences with `morx` ligatures). On Apple platforms Core
//! Text draws an emoji cluster into a bitmap, which is written as a PNG for
//! the texture loader. Elsewhere there is no renderer and callers fall back.

/// Draw `text` (one emoji cluster) at `px` pixels into a PNG at `path`.
/// Returns the baseline's distance from the image top in pixels, or `None`
/// when this platform can't render emoji (or the text draws nothing).
pub fn render_emoji_png(text: &str, px: f64, path: &str) -> Option<f64> {
    #[cfg(target_vendor = "apple")]
    {
        apple::render(text, px, path)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (text, px, path);
        None
    }
}

#[cfg(target_vendor = "apple")]
mod apple {
    use std::ffi::c_void;

    type CFTypeRef = *const c_void;
    type CFStringRef = *const c_void;
    type CFDictionaryRef = *const c_void;
    type CFAllocatorRef = *const c_void;

    #[repr(C)]
    struct CFDictionaryKeyCallBacks {
        _private: [u8; 0],
    }
    #[repr(C)]
    struct CFDictionaryValueCallBacks {
        _private: [u8; 0],
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        static kCFTypeDictionaryKeyCallBacks: CFDictionaryKeyCallBacks;
        static kCFTypeDictionaryValueCallBacks: CFDictionaryValueCallBacks;
        fn CFStringCreateWithBytes(
            alloc: CFAllocatorRef,
            bytes: *const u8,
            len: isize,
            encoding: u32,
            external: u8,
        ) -> CFStringRef;
        fn CFDictionaryCreate(
            alloc: CFAllocatorRef,
            keys: *const CFTypeRef,
            values: *const CFTypeRef,
            n: isize,
            kcb: *const CFDictionaryKeyCallBacks,
            vcb: *const CFDictionaryValueCallBacks,
        ) -> CFDictionaryRef;
        fn CFAttributedStringCreate(
            alloc: CFAllocatorRef,
            s: CFStringRef,
            attrs: CFDictionaryRef,
        ) -> CFTypeRef;
        fn CFRelease(cf: CFTypeRef);
    }

    #[link(name = "CoreText", kind = "framework")]
    extern "C" {
        static kCTFontAttributeName: CFStringRef;
        fn CTFontCreateWithName(name: CFStringRef, size: f64, matrix: *const c_void) -> CFTypeRef;
        fn CTLineCreateWithAttributedString(s: CFTypeRef) -> CFTypeRef;
        fn CTLineGetTypographicBounds(
            line: CFTypeRef,
            ascent: *mut f64,
            descent: *mut f64,
            leading: *mut f64,
        ) -> f64;
        fn CTLineDraw(line: CFTypeRef, ctx: *mut c_void);
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGColorSpaceCreateDeviceRGB() -> *mut c_void;
        fn CGColorSpaceRelease(cs: *mut c_void);
        fn CGBitmapContextCreate(
            data: *mut c_void,
            width: usize,
            height: usize,
            bits_per_component: usize,
            bytes_per_row: usize,
            space: *mut c_void,
            bitmap_info: u32,
        ) -> *mut c_void;
        fn CGContextRelease(ctx: *mut c_void);
        fn CGContextSetTextPosition(ctx: *mut c_void, x: f64, y: f64);
    }

    const UTF8: u32 = 0x0800_0100;
    /// kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big: RGBA bytes.
    const RGBA: u32 = 1 | (4 << 12);

    fn cfstr(s: &str) -> CFStringRef {
        unsafe { CFStringCreateWithBytes(std::ptr::null(), s.as_ptr(), s.len() as isize, UTF8, 0) }
    }

    pub fn render(text: &str, px: f64, path: &str) -> Option<f64> {
        if text.is_empty() || !(px > 0.0 && px <= 1024.0) {
            return None;
        }
        unsafe {
            let name = cfstr("AppleColorEmoji");
            let font = CTFontCreateWithName(name, px, std::ptr::null());
            CFRelease(name);
            if font.is_null() {
                return None;
            }
            let s = cfstr(text);
            let keys = [kCTFontAttributeName as CFTypeRef];
            let values = [font];
            let attrs = CFDictionaryCreate(
                std::ptr::null(),
                keys.as_ptr(),
                values.as_ptr(),
                1,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            );
            let astr = CFAttributedStringCreate(std::ptr::null(), s, attrs);
            let line = CTLineCreateWithAttributedString(astr);
            let (mut ascent, mut descent, mut leading) = (0.0, 0.0, 0.0);
            let width = CTLineGetTypographicBounds(line, &mut ascent, &mut descent, &mut leading);
            let w = width.ceil().max(1.0) as usize;
            let h = (ascent + descent).ceil().max(1.0) as usize;
            let mut pixels = vec![0u8; w * h * 4];
            let cs = CGColorSpaceCreateDeviceRGB();
            let ctx =
                CGBitmapContextCreate(pixels.as_mut_ptr() as *mut c_void, w, h, 8, w * 4, cs, RGBA);
            let mut ok = false;
            if !ctx.is_null() {
                // baseline `descent` above the bottom (Core Graphics' origin is bottom-left)
                CGContextSetTextPosition(ctx, 0.0, descent);
                CTLineDraw(line, ctx);
                CGContextRelease(ctx);
                ok = true;
            }
            CGColorSpaceRelease(cs);
            CFRelease(line);
            CFRelease(astr);
            CFRelease(attrs);
            CFRelease(s);
            CFRelease(font);
            if !ok || pixels.chunks_exact(4).all(|p| p[3] == 0) {
                return None;
            }
            // premultiplied -> straight alpha for the texture loader
            for p in pixels.chunks_exact_mut(4) {
                let a = p[3] as u32;
                if a > 0 && a < 255 {
                    for c in &mut p[..3] {
                        *c = ((*c as u32 * 255 + a / 2) / a).min(255) as u8;
                    }
                }
            }
            let img = image::RgbaImage::from_raw(w as u32, h as u32, pixels)?;
            img.save_with_format(path, image::ImageFormat::Png).ok()?;
            Some(ascent)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_os = "macos")]
    fn renders_an_emoji_with_its_baseline() {
        let path = std::env::temp_dir().join("bloom-emoji-test.png");
        let ascent = super::render_emoji_png("👍🏽", 64.0, path.to_str().unwrap()).expect("rendered");
        assert!(ascent > 30.0 && ascent < 80.0, "ascent {ascent}");
        let img = image::open(&path).expect("png").to_rgba8();
        assert!(img.width() >= 48 && img.height() >= 48);
        // a skin-toned thumb: some opaque pixels, not one beige square
        assert!(img.pixels().filter(|p| p[3] == 255).count() > 500);
    }
}
