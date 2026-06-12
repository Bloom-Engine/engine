use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::{str_from_header, alloc_perry_string};
use bloom_shared::audio::{parse_wav, parse_ogg, parse_mp3};

use std::sync::OnceLock;
use std::os::unix::io::RawFd;

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();
static mut GAMEPAD_FD: RawFd = -1;

fn engine() -> &'static mut EngineState {
    unsafe { ENGINE.get_mut().expect("Engine not initialized") }
}
/// Asset-path hook for define_core_ffi! — identity on desktop, where game
/// asset paths are valid relative to the working directory.
fn bloom_resolve_asset_path(path: &str) -> std::borrow::Cow<'_, str> {
    std::borrow::Cow::Borrowed(path)
}

// The full shared (non-physics) FFI surface. See bloom_shared::ffi_core
// docs for the contract; tools/validate-ffi.js checks parity in CI.
bloom_shared::define_core_ffi!();


/// Map X11 keysym to Bloom key code.
fn map_keycode(keysym: u32) -> usize {
    match keysym {
        0x61..=0x7a => (keysym - 0x61 + 65) as usize, // a-z → A-Z (65-90)
        0x41..=0x5a => keysym as usize,                 // A-Z direct
        0x30..=0x39 => keysym as usize,                 // 0-9
        0xff52 => 256,  // XK_Up
        0xff54 => 257,  // XK_Down
        0xff51 => 258,  // XK_Left
        0xff53 => 259,  // XK_Right
        0x20 => 32,     // space
        0xff0d => 265,  // XK_Return → Bloom ENTER
        0xff1b => 27,   // XK_Escape
        0xff09 => 9,    // XK_Tab
        0xff08 => 8,    // XK_BackSpace
        0xffff => 127,  // XK_Delete
        0xff63 => 260,  // XK_Insert
        0xff50 => 261,  // XK_Home
        0xff57 => 262,  // XK_End
        0xff55 => 263,  // XK_Page_Up
        0xff56 => 264,  // XK_Page_Down
        0xffe1 => 280,  // XK_Shift_L
        0xffe2 => 281,  // XK_Shift_R
        0xffe3 => 282,  // XK_Control_L
        0xffe4 => 283,  // XK_Control_R
        0xffe9 => 284,  // XK_Alt_L
        0xffea => 285,  // XK_Alt_R
        0xffeb => 286,  // XK_Super_L
        0xffec => 287,  // XK_Super_R
        // Function keys
        0xffbe => 112,  // XK_F1
        0xffbf => 113,  // XK_F2
        0xffc0 => 114,  // XK_F3
        0xffc1 => 115,  // XK_F4
        0xffc2 => 116,  // XK_F5
        0xffc3 => 117,  // XK_F6
        0xffc4 => 118,  // XK_F7
        0xffc5 => 119,  // XK_F8
        0xffc6 => 120,  // XK_F9
        0xffc7 => 121,  // XK_F10
        0xffc8 => 122,  // XK_F11
        0xffc9 => 123,  // XK_F12
        _ => 0,
    }
}

#[cfg(target_os = "linux")]
mod x11_impl {
    use super::*;

    static mut DISPLAY: *mut x11::xlib::Display = std::ptr::null_mut();
    static mut X11_WINDOW: x11::xlib::Window = 0;
    static mut IS_FULLSCREEN: bool = false;
    static mut HEADLESS: bool = false;
    static mut NO_FULLSCREEN: bool = false;
    static mut CURSOR_HIDDEN: bool = false;
    static mut HIDDEN_CURSOR: x11::xlib::Cursor = 0;
    /// Cached XC_* shape cursors keyed by `cursor_shape` value 0..=6.
    /// Lazily created by `apply_cursor_shape`; reused across frames so
    /// we don't leak a Cursor handle every poll.
    static mut SHAPE_CURSORS: [x11::xlib::Cursor; 8] = [0; 8];
    static mut LAST_APPLIED_SHAPE: u32 = 0xFFFF_FFFF;
    /// When `cursor_disabled` (relative-mode) is on we keep warping the
    /// pointer back to window center each frame; remembering the last warp
    /// target lets motion handlers compute a reliable raw delta and ignore
    /// the synthetic motion event the warp itself generates.
    static mut WARP_CENTER_X: i32 = 0;
    static mut WARP_CENTER_Y: i32 = 0;
    static mut RELATIVE_MODE: bool = false;

    pub fn set_fullscreen(fullscreen: bool) {
        unsafe {
            // BLOOM_NO_FULLSCREEN=1 hard-disables the fullscreen path so
            // benchmark harnesses on a 4K monitor don't silently 4× their
            // pixel count when an inherited fullscreen Space leaks in.
            if NO_FULLSCREEN && fullscreen { return; }
            if DISPLAY.is_null() || X11_WINDOW == 0 { return; }

            let wm_state = x11::xlib::XInternAtom(
                DISPLAY,
                b"_NET_WM_STATE\0".as_ptr() as *const _,
                0,
            );
            let wm_fullscreen = x11::xlib::XInternAtom(
                DISPLAY,
                b"_NET_WM_STATE_FULLSCREEN\0".as_ptr() as *const _,
                0,
            );

            let action = if fullscreen { 1 } else { 0 }; // _NET_WM_STATE_ADD / _REMOVE

            let mut event: x11::xlib::XClientMessageEvent = std::mem::zeroed();
            event.type_ = x11::xlib::ClientMessage;
            event.window = X11_WINDOW;
            event.message_type = wm_state;
            event.format = 32;
            event.data.set_long(0, action);
            event.data.set_long(1, wm_fullscreen as i64);
            event.data.set_long(2, 0);
            event.data.set_long(3, 1); // source: normal application

            let root = x11::xlib::XDefaultRootWindow(DISPLAY);
            x11::xlib::XSendEvent(
                DISPLAY,
                root,
                0,
                x11::xlib::SubstructureRedirectMask | x11::xlib::SubstructureNotifyMask,
                &mut event as *mut x11::xlib::XClientMessageEvent as *mut x11::xlib::XEvent,
            );
            x11::xlib::XFlush(DISPLAY);
            IS_FULLSCREEN = fullscreen;
        }
    }

    pub fn toggle_fullscreen() {
        unsafe { set_fullscreen(!IS_FULLSCREEN); }
    }

    /// Returns (physical_w, physical_h). Caller's `width`/`height`
    /// are *logical*; on a HiDPI X11 display we multiply by the
    /// monitor's scale factor so the window appears the right size
    /// while its surface is at the screen's physical resolution.
    ///
    /// `headless = true` keeps the X11 window + Vulkan surface alive
    /// (wgpu needs a surface) but never maps the window — so it never
    /// appears on screen and never steals focus. Mirrors the macOS
    /// BLOOM_HEADLESS path for batch / CI rendering harnesses.
    pub fn create_window(width: f64, height: f64, title: &str, headless: bool) -> (u32, u32) {
        unsafe {
            HEADLESS = headless;
            DISPLAY = x11::xlib::XOpenDisplay(std::ptr::null());
            if DISPLAY.is_null() {
                panic!("Failed to open X11 display");
            }

            let screen = x11::xlib::XDefaultScreen(DISPLAY);
            let root = x11::xlib::XRootWindow(DISPLAY, screen);

            let scale = display_scale(DISPLAY, screen);
            let phys_w = (width * scale).round() as u32;
            let phys_h = (height * scale).round() as u32;

            // Off-screen origin is belt-and-braces: even if some WM
            // chooses to map the window despite us never calling
            // XMapWindow, it'll appear far off the visible desktop.
            let origin_x: i32 = if headless { -20000 } else { 0 };
            X11_WINDOW = x11::xlib::XCreateSimpleWindow(
                DISPLAY, root,
                origin_x, 0, phys_w, phys_h, 0,
                x11::xlib::XBlackPixel(DISPLAY, screen),
                x11::xlib::XWhitePixel(DISPLAY, screen),
            );

            let title_cstr = std::ffi::CString::new(title).unwrap();
            x11::xlib::XStoreName(DISPLAY, X11_WINDOW, title_cstr.as_ptr());

            x11::xlib::XSelectInput(DISPLAY, X11_WINDOW,
                x11::xlib::ExposureMask | x11::xlib::KeyPressMask | x11::xlib::KeyReleaseMask |
                x11::xlib::ButtonPressMask | x11::xlib::ButtonReleaseMask |
                x11::xlib::PointerMotionMask | x11::xlib::StructureNotifyMask);

            if !headless {
                x11::xlib::XMapWindow(DISPLAY, X11_WINDOW);
            }
            x11::xlib::XFlush(DISPLAY);
            (phys_w, phys_h)
        }
    }

    pub fn set_no_fullscreen(no_fs: bool) { unsafe { NO_FULLSCREEN = no_fs; } }
    pub fn is_headless() -> bool { unsafe { HEADLESS } }

    /// Read the current display's DPI scale factor. Computed from
    /// physical screen dimensions (pixels / mm). Snapped to the
    /// nearest 0.25 and clamped to [1.0, 4.0] — EDID often lies
    /// about millimetres so we want stable integer-ish steps. A
    /// real Wayland-style desktop environment with explicit scale
    /// settings is out of scope here; this matches what most X11
    /// users actually run (fractional scaling via Xft.dpi is the
    /// only thing GTK/Qt honour by default).
    pub fn display_scale(display: *mut x11::xlib::Display, screen: i32) -> f64 {
        unsafe {
            let pixels = x11::xlib::XDisplayWidth(display, screen) as f64;
            let mm = x11::xlib::XDisplayWidthMM(display, screen) as f64;
            if mm <= 0.0 || pixels <= 0.0 { return 1.0; }
            let dpi = pixels / (mm / 25.4);
            // 96 DPI = scale 1.0. Snap to 0.25 steps.
            let raw = (dpi / 96.0).max(1.0).min(4.0);
            (raw * 4.0).round() / 4.0
        }
    }

    pub fn display() -> *mut x11::xlib::Display { unsafe { DISPLAY } }
    pub fn window() -> x11::xlib::Window { unsafe { X11_WINDOW } }

    pub fn set_window_title(title: &str) {
        unsafe {
            if DISPLAY.is_null() || X11_WINDOW == 0 { return; }
            let cstr = match std::ffi::CString::new(title) { Ok(c) => c, Err(_) => return };
            x11::xlib::XStoreName(DISPLAY, X11_WINDOW, cstr.as_ptr());
            // Modern WMs honour _NET_WM_NAME (UTF-8) over the legacy
            // WM_NAME (Latin-1) set by XStoreName, so write both.
            let net_wm_name = x11::xlib::XInternAtom(
                DISPLAY, b"_NET_WM_NAME\0".as_ptr() as *const _, 0);
            let utf8_string = x11::xlib::XInternAtom(
                DISPLAY, b"UTF8_STRING\0".as_ptr() as *const _, 0);
            if net_wm_name != 0 && utf8_string != 0 {
                x11::xlib::XChangeProperty(
                    DISPLAY, X11_WINDOW, net_wm_name, utf8_string, 8,
                    x11::xlib::PropModeReplace,
                    title.as_ptr(), title.len() as i32,
                );
            }
            x11::xlib::XFlush(DISPLAY);
        }
    }

    /// Set the EWMH `_NET_WM_ICON` property from an image file. The X11
    /// icon format is a flat array of CARDINALs (`long`s on the wire):
    /// `[width, height, pixel0, pixel1, ...]` with each pixel as ARGB
    /// premultiplied-alpha packed into the low 32 bits of a long.
    pub fn set_window_icon(path: &str) {
        unsafe {
            if DISPLAY.is_null() || X11_WINDOW == 0 { return; }
            let img = match image::open(path) {
                Ok(i) => i.to_rgba8(),
                Err(_) => return,
            };
            let (w, h) = (img.width() as usize, img.height() as usize);
            let mut buf: Vec<std::os::raw::c_long> = Vec::with_capacity(2 + w * h);
            buf.push(w as std::os::raw::c_long);
            buf.push(h as std::os::raw::c_long);
            for px in img.chunks_exact(4) {
                let r = px[0] as u32;
                let g = px[1] as u32;
                let b = px[2] as u32;
                let a = px[3] as u32;
                let argb = (a << 24) | (r << 16) | (g << 8) | b;
                buf.push(argb as std::os::raw::c_long);
            }
            let net_wm_icon = x11::xlib::XInternAtom(
                DISPLAY, b"_NET_WM_ICON\0".as_ptr() as *const _, 0);
            let cardinal = x11::xlib::XInternAtom(
                DISPLAY, b"CARDINAL\0".as_ptr() as *const _, 0);
            x11::xlib::XChangeProperty(
                DISPLAY, X11_WINDOW, net_wm_icon, cardinal, 32,
                x11::xlib::PropModeReplace,
                buf.as_ptr() as *const u8,
                buf.len() as i32,
            );
            x11::xlib::XFlush(DISPLAY);
        }
    }

    /// Build (once) a 1x1 fully-transparent cursor — the standard X11
    /// trick for "hide the cursor". Subsequent calls reuse the cached
    /// cursor since X11 leaks Cursor handles otherwise.
    unsafe fn ensure_hidden_cursor() -> x11::xlib::Cursor {
        if HIDDEN_CURSOR != 0 { return HIDDEN_CURSOR; }
        let pixmap = x11::xlib::XCreatePixmap(DISPLAY, X11_WINDOW, 1, 1, 1);
        let mut color: x11::xlib::XColor = std::mem::zeroed();
        let cursor = x11::xlib::XCreatePixmapCursor(
            DISPLAY, pixmap, pixmap, &mut color, &mut color, 0, 0);
        x11::xlib::XFreePixmap(DISPLAY, pixmap);
        HIDDEN_CURSOR = cursor;
        cursor
    }

    pub fn hide_cursor() {
        unsafe {
            if DISPLAY.is_null() || X11_WINDOW == 0 || CURSOR_HIDDEN { return; }
            let c = ensure_hidden_cursor();
            x11::xlib::XDefineCursor(DISPLAY, X11_WINDOW, c);
            x11::xlib::XFlush(DISPLAY);
            CURSOR_HIDDEN = true;
        }
    }

    pub fn show_cursor() {
        unsafe {
            if DISPLAY.is_null() || X11_WINDOW == 0 || !CURSOR_HIDDEN { return; }
            x11::xlib::XUndefineCursor(DISPLAY, X11_WINDOW);
            x11::xlib::XFlush(DISPLAY);
            CURSOR_HIDDEN = false;
        }
    }

    /// Warp the pointer to window center and remember where we put it so
    /// the motion handler can compute deltas relative to the warp target.
    pub fn warp_to_center() {
        unsafe {
            if DISPLAY.is_null() || X11_WINDOW == 0 { return; }
            let mut attrs: x11::xlib::XWindowAttributes = std::mem::zeroed();
            x11::xlib::XGetWindowAttributes(DISPLAY, X11_WINDOW, &mut attrs);
            let cx = attrs.width / 2;
            let cy = attrs.height / 2;
            x11::xlib::XWarpPointer(DISPLAY, 0, X11_WINDOW, 0, 0, 0, 0, cx, cy);
            x11::xlib::XFlush(DISPLAY);
            WARP_CENTER_X = cx;
            WARP_CENTER_Y = cy;
        }
    }

    pub fn enter_relative_mode() {
        unsafe {
            RELATIVE_MODE = true;
            hide_cursor();
            warp_to_center();
        }
    }

    pub fn leave_relative_mode() {
        unsafe {
            RELATIVE_MODE = false;
            show_cursor();
        }
    }

    pub fn is_relative_mode() -> bool { unsafe { RELATIVE_MODE } }
    pub fn warp_center_x() -> i32 { unsafe { WARP_CENTER_X } }
    pub fn warp_center_y() -> i32 { unsafe { WARP_CENTER_Y } }

    /// Apply the requested cursor shape (the same 0..=6 enum macOS uses
    /// in NSCursor calls). XCreateFontCursor uses cursor-font glyph
    /// constants from <X11/cursorfont.h>; we cache one Cursor per shape
    /// so repeat calls don't leak server-side state.
    pub fn apply_cursor_shape(shape: u32) {
        unsafe {
            if DISPLAY.is_null() || X11_WINDOW == 0 || CURSOR_HIDDEN { return; }
            if shape == LAST_APPLIED_SHAPE { return; }
            // X11 cursor-font glyph indices (from cursorfont.h).
            // 0 = default arrow → XC_left_ptr (68)
            // 1 = pointing hand → XC_hand2     (60)
            // 2 = open hand     → XC_fleur     (52, "move")
            // 3 = I-beam        → XC_xterm     (152)
            // 4 = resize H      → XC_sb_h_double_arrow (108)
            // 5 = resize V      → XC_sb_v_double_arrow (116)
            // 6 = crosshair     → XC_crosshair (34)
            let glyph: u32 = match shape {
                1 => 60,
                2 => 52,
                3 => 152,
                4 => 108,
                5 => 116,
                6 => 34,
                _ => 68,
            };
            let idx = (shape as usize).min(SHAPE_CURSORS.len() - 1);
            if SHAPE_CURSORS[idx] == 0 {
                SHAPE_CURSORS[idx] = x11::xlib::XCreateFontCursor(DISPLAY, glyph);
            }
            if SHAPE_CURSORS[idx] != 0 {
                x11::xlib::XDefineCursor(DISPLAY, X11_WINDOW, SHAPE_CURSORS[idx]);
                x11::xlib::XFlush(DISPLAY);
                LAST_APPLIED_SHAPE = shape;
            }
        }
    }

    pub fn poll_events() {
        unsafe {
            while x11::xlib::XPending(DISPLAY) > 0 {
                let mut event = std::mem::zeroed::<x11::xlib::XEvent>();
                x11::xlib::XNextEvent(DISPLAY, &mut event);

                match event.type_ {
                    x11::xlib::KeyPress => {
                        let keysym = x11::xlib::XLookupKeysym(
                            &mut event.key as *mut _ as *mut _,
                            0,
                        );
                        let bloom_key = map_keycode(keysym as u32);
                        if bloom_key > 0 {
                            engine().input.set_key_down(bloom_key);
                        }
                        // Decode UTF-8 typed text via XLookupString so the
                        // editor's text-input widget receives characters.
                        let mut buf = [0u8; 32];
                        let mut ks: x11::xlib::KeySym = 0;
                        let len = x11::xlib::XLookupString(
                            &mut event.key as *mut _,
                            buf.as_mut_ptr() as *mut i8,
                            buf.len() as i32,
                            &mut ks,
                            std::ptr::null_mut(),
                        );
                        if len > 0 {
                            if let Ok(s) = std::str::from_utf8(&buf[..len as usize]) {
                                for c in s.chars() {
                                    let cp = c as u32;
                                    if cp >= 32 || cp == 8 || cp == 13 || cp == 9 {
                                        engine().input.push_char(cp);
                                    }
                                }
                            }
                        }
                    }
                    x11::xlib::KeyRelease => {
                        let keysym = x11::xlib::XLookupKeysym(
                            &mut event.key as *mut _ as *mut _,
                            0,
                        );
                        let bloom_key = map_keycode(keysym as u32);
                        if bloom_key > 0 {
                            engine().input.set_key_up(bloom_key);
                        }
                    }
                    x11::xlib::MotionNotify => {
                        let motion = event.motion;
                        if RELATIVE_MODE {
                            // Ignore the motion event we generated by warping
                            // back to center — it would otherwise add a stray
                            // -delta cancelling the user's actual movement.
                            if motion.x == WARP_CENTER_X && motion.y == WARP_CENTER_Y {
                                // synthetic warp event; skip
                            } else {
                                let dx = (motion.x - WARP_CENTER_X) as f64;
                                let dy = (motion.y - WARP_CENTER_Y) as f64;
                                engine().input.accumulate_mouse_delta(dx, dy);
                                warp_to_center();
                            }
                        } else {
                            engine().input.set_mouse_position(motion.x as f64, motion.y as f64);
                        }
                    }
                    x11::xlib::ButtonPress => {
                        let button = event.button.button;
                        match button {
                            1 => engine().input.set_mouse_button_down(0),
                            3 => engine().input.set_mouse_button_down(1),
                            2 => engine().input.set_mouse_button_down(2),
                            // X11 maps wheel up/down to button 4/5 and
                            // horizontal scroll to 6/7. macOS uses an
                            // accumulator with positive = scroll up; flip
                            // the sign on the down case to match.
                            4 => engine().input.accumulate_mouse_wheel(1.0),
                            5 => engine().input.accumulate_mouse_wheel(-1.0),
                            _ => {}
                        }
                    }
                    x11::xlib::ButtonRelease => {
                        let button = event.button.button;
                        match button {
                            1 => engine().input.set_mouse_button_up(0),
                            3 => engine().input.set_mouse_button_up(1),
                            2 => engine().input.set_mouse_button_up(2),
                            _ => {}
                        }
                    }
                    x11::xlib::ConfigureNotify => {
                        let configure = event.configure;
                        let phys_w = configure.width as u32;
                        let phys_h = configure.height as u32;
                        if phys_w > 0 && phys_h > 0 {
                            let eng = engine();
                            if phys_w != eng.renderer.physical_width()
                                || phys_h != eng.renderer.physical_height()
                            {
                                let screen = x11::xlib::XDefaultScreen(DISPLAY);
                                let scale = display_scale(DISPLAY, screen);
                                let log_w = ((phys_w as f64) / scale).round() as u32;
                                let log_h = ((phys_h as f64) / scale).round() as u32;
                                eng.renderer.resize(phys_w, phys_h, log_w, log_h);
                            }
                        }
                    }
                    x11::xlib::DestroyNotify => {
                        engine().should_close = true;
                    }
                    _ => {}
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_init_window(width: f64, height: f64, title_ptr: *const u8, fullscreen: f64) {
    let title = str_from_header(title_ptr);

    #[cfg(target_os = "linux")]
    {
        // Headless mode: BLOOM_HEADLESS=1 keeps the X11 window + Vulkan
        // surface alive (wgpu requires a real surface) but never maps
        // the window so it's invisible and never steals focus. Lets an
        // agent spin up the renderer in a batch loop without disturbing
        // the user's desktop.
        let headless = std::env::var("BLOOM_HEADLESS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        // BLOOM_NO_FULLSCREEN=1 hard-disables fullscreen capability for
        // benchmark harnesses where a 4K-display fullscreen path would
        // silently quadruple render cost.
        let no_fullscreen = std::env::var("BLOOM_NO_FULLSCREEN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        x11_impl::set_no_fullscreen(no_fullscreen);

        let (phys_w, phys_h) = x11_impl::create_window(width, height, title, headless);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let surface = unsafe {
            let raw_window = raw_window_handle::RawWindowHandle::Xlib(
                raw_window_handle::XlibWindowHandle::new(x11_impl::window())
            );
            let raw_display = raw_window_handle::RawDisplayHandle::Xlib(
                raw_window_handle::XlibDisplayHandle::new(
                    std::ptr::NonNull::new(x11_impl::display() as *mut _),
                    0,
                )
            );
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(raw_display),
                raw_window_handle: raw_window,
            }).expect("Failed to create surface")
        };

        let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })).expect("No adapter found");

        // Ticket 007b: HW ray-query via VK_KHR_ray_query on RT-capable
        // desktop Linux GPUs. Older integrated GPUs will fall back to
        // the SW path through this gate.
        let supported = adapter.features();
        let force_sw_gi = std::env::var("BLOOM_FORCE_SW_GI")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let rt_mask = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        let mut required_features = wgpu::Features::empty();
        // Ticket 011: request TIMESTAMP_QUERY when supported so the profiler
        // can record GPU timings. Optional — profiler falls back to CPU-only
        // when the adapter doesn't grant it.
        if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        // Cooked BC7 textures (bloom-cook) upload compressed when the
        // adapter has BC support; without it they CPU-decode at load.
        if supported.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
        }
        if !force_sw_gi && supported.contains(rt_mask) {
            required_features |= rt_mask;
        }
        let experimental_features = if required_features.intersects(rt_mask) {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        let mut required_limits = wgpu::Limits::default();
        // The material ABI declares 5 bind groups (PerFrame, PerView,
        // PerMaterial, PerDraw, SceneInputs). wgpu's default limit is
        // 4. Vulkan supports at least 7 here, so 5 is universally safe.
        required_limits.max_bind_groups = 5;
        if required_features.intersects(rt_mask) {
            required_limits = required_limits
                .using_minimum_supported_acceleration_structure_values();
        }
        let (device, queue) = pollster_block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("bloom_device"),
                required_features,
                required_limits,
                experimental_features,
                ..Default::default()
            },
        )).expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0];
        // Surface sized at the *physical* pixel count we computed
        // from XDisplayWidth/MM; renderer is told the caller's
        // logical size separately so screenWidth() etc. stay
        // DPI-independent.
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: phys_w,
            height: phys_h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let renderer = Renderer::new(device, queue, surface, surface_config, width as u32, height as u32);
        unsafe { let _ = ENGINE.set(EngineState::new(renderer)); }

        if fullscreen != 0.0 {
            x11_impl::set_fullscreen(true);
        }

        // Register Bloom's GPU screenshot capture with perry-geisterhand.
        // perry-runtime always exposes these symbols, so the link is direct
        // (not weak); the registry no-ops gracefully if the editor isn't
        // running.
        bloom_register_geisterhand_screenshot();
    }

    #[cfg(not(target_os = "linux"))]
    panic!("bloom-linux can only run on Linux");
}

#[no_mangle]
pub extern "C" fn bloom_close_window() {}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 {
    if engine().should_close { 1.0 } else { 0.0 }
}

#[cfg(target_os = "linux")]
fn poll_linux_gamepad() {
    unsafe {
        // Try to open gamepad if not already
        if GAMEPAD_FD < 0 {
            let path = b"/dev/input/js0\0";
            // Open non-blocking
            GAMEPAD_FD = libc::open(path.as_ptr() as *const libc::c_char, libc::O_RDONLY | libc::O_NONBLOCK);
            if GAMEPAD_FD >= 0 {
                engine().input.gamepad_available = true;
                engine().input.gamepad_axis_count = 6;
            }
        }
        if GAMEPAD_FD < 0 { return; }

        // Linux joystick event structure
        #[repr(C)]
        struct JsEvent {
            time: u32,
            value: i16,
            event_type: u8,
            number: u8,
        }

        loop {
            let mut event = std::mem::zeroed::<JsEvent>();
            let n = libc::read(
                GAMEPAD_FD,
                &mut event as *mut _ as *mut libc::c_void,
                std::mem::size_of::<JsEvent>(),
            );
            if n != std::mem::size_of::<JsEvent>() as isize { break; }

            let type_masked = event.event_type & 0x7f; // strip JS_EVENT_INIT
            if type_masked == 1 {
                // Button
                let idx = event.number as usize;
                if event.value != 0 {
                    engine().input.set_gamepad_button_down(idx);
                } else {
                    engine().input.set_gamepad_button_up(idx);
                }
            } else if type_masked == 2 {
                // Axis
                let idx = event.number as usize;
                let value = event.value as f32 / 32767.0;
                engine().input.set_gamepad_axis(idx, value);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    #[cfg(target_os = "linux")]
    {
        x11_impl::poll_events();
        poll_linux_gamepad();
        // Apply Q2 cursor shape — mirrors macOS NSCursor calls in its
        // event loop. set_cursor_shape just stores the value; the X11
        // cursor only changes when we actually call XDefineCursor.
        x11_impl::apply_cursor_shape(engine().input.cursor_shape);
    }
    engine().begin_frame();
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    // Pump geisterhand BEFORE end_frame. Mirrors macOS — the screenshot
    // function re-renders inline using the captured VP + vertex buffers.
    extern "C" { fn perry_geisterhand_pump(); }
    unsafe { perry_geisterhand_pump(); }

    engine().end_frame();
}

static AUDIO_RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[no_mangle]
pub extern "C" fn bloom_init_audio() {
    #[cfg(target_os = "linux")]
    {
        use std::sync::atomic::Ordering;
        AUDIO_RUNNING.store(true, Ordering::SeqCst);
        // Move the render half into the audio thread; the engine keeps
        // only the command-producing control half.
        let renderer = engine().audio.take_renderer();
        std::thread::spawn(move || {
            alsa_audio_thread(renderer);
        });
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_audio() {
    AUDIO_RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));
}

#[cfg(target_os = "linux")]
fn alsa_audio_thread(mut renderer: Option<bloom_shared::audio::AudioRenderer>) {
    use alsa::pcm::*;
    use alsa::{Direction, ValueOr};
    use std::sync::atomic::Ordering;

    let pcm = match PCM::new("default", Direction::Playback, false) {
        Ok(p) => p,
        Err(_) => return,
    };

    let sample_rate = 44100u32;
    let channels = 2u32;
    let period_size = 1024;

    {
        let hwp = HwParams::any(&pcm).unwrap();
        let _ = hwp.set_channels(channels);
        let _ = hwp.set_rate(sample_rate, ValueOr::Nearest);
        let _ = hwp.set_format(Format::float());
        let _ = hwp.set_access(Access::RWInterleaved);
        let _ = hwp.set_period_size(period_size, ValueOr::Nearest);
        let _ = hwp.set_buffer_size(period_size * 4);
        let _ = pcm.hw_params(&hwp);
    }

    let _ = pcm.prepare();

    let frames = period_size as usize;
    let mut mix_buf = vec![0.0f32; frames * channels as usize];

    while AUDIO_RUNNING.load(Ordering::SeqCst) {
        for s in mix_buf.iter_mut() { *s = 0.0; }

        // Renderer is moved into this thread at spawn — no shared
        // engine state is touched from here (see audio/mod.rs contract).
        if let Some(r) = renderer.as_mut() {
            r.mix(&mut mix_buf);
        }

        let io = pcm.io_f32().unwrap();
        match io.writei(&mix_buf) {
            Ok(_) => {}
            Err(e) => {
                let _ = pcm.try_recover(e, true);
            }
        }
    }

    let _ = pcm.drain();
}

// --- Texture FFI ---

// --- Camera FFI ---

// --- 3D Drawing FFI ---

// --- Model FFI ---

// ============================================================
// Phase 1c — material system FFI
// ============================================================

// --- Music FFI ---

// --- Gamepad FFI ---

// --- Skeletal Animation Debug ---

// --- Lighting ---

// --- Utility FFI ---

#[no_mangle]
pub extern "C" fn bloom_toggle_fullscreen() {
    #[cfg(target_os = "linux")]
    x11_impl::toggle_fullscreen();
}
#[no_mangle]
pub extern "C" fn bloom_set_window_title(title_ptr: *const u8) {
    let title = str_from_header(title_ptr);
    #[cfg(target_os = "linux")]
    x11_impl::set_window_title(title);
}
#[no_mangle]
pub extern "C" fn bloom_set_window_icon(path_ptr: *const u8) {
    let path = str_from_header(path_ptr);
    #[cfg(target_os = "linux")]
    x11_impl::set_window_icon(path);
}

#[no_mangle]
pub extern "C" fn bloom_disable_cursor() {
    let input = &mut engine().input;
    input.cursor_disabled = true;
    input.clear_mouse_delta();
    #[cfg(target_os = "linux")]
    x11_impl::enter_relative_mode();
}

#[no_mangle]
pub extern "C" fn bloom_enable_cursor() {
    engine().input.cursor_disabled = false;
    #[cfg(target_os = "linux")]
    x11_impl::leave_relative_mode();
}

// E4: Clipboard (arboard, X11/Wayland-aware)
#[no_mangle]
pub extern "C" fn bloom_set_clipboard_text(text_ptr: *const u8) {
    let text = str_from_header(text_ptr);
    if let Ok(mut clipboard) = arboard::Clipboard::new() {
        let _ = clipboard.set_text(text.to_string());
    }
}
#[no_mangle]
pub extern "C" fn bloom_get_clipboard_text() -> *const u8 {
    match arboard::Clipboard::new() {
        Ok(mut clipboard) => match clipboard.get_text() {
            Ok(text) => alloc_perry_string(&text),
            Err(_) => alloc_perry_string(""),
        },
        Err(_) => alloc_perry_string(""),
    }
}

// E5b: Native file dialogs (rfd → GTK/zenity/kdialog on Linux)
#[no_mangle]
pub extern "C" fn bloom_open_file_dialog(filter_ptr: *const u8, title_ptr: *const u8) -> *const u8 {
    let filter = str_from_header(filter_ptr);
    let title = str_from_header(title_ptr);
    let mut dialog = rfd::FileDialog::new().set_title(title);
    if !filter.is_empty() {
        dialog = dialog.add_filter("Files", &[filter]);
    }
    match dialog.pick_file() {
        Some(path) => alloc_perry_string(&path.to_string_lossy()),
        None => alloc_perry_string(""),
    }
}
#[no_mangle]
pub extern "C" fn bloom_save_file_dialog(default_name_ptr: *const u8, title_ptr: *const u8) -> *const u8 {
    let default_name = str_from_header(default_name_ptr);
    let title = str_from_header(title_ptr);
    let dialog = rfd::FileDialog::new()
        .set_title(title)
        .set_file_name(default_name);
    match dialog.save_file() {
        Some(path) => alloc_perry_string(&path.to_string_lossy()),
        None => alloc_perry_string(""),
    }
}
#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 { 4.0 }

/// Preferred OS language packed as `c0*256+c1` (ISO-639 primary subtag), from $LANG/$LC_*.
#[no_mangle]
pub extern "C" fn bloom_get_language() -> f64 {
    fn pack(code: &str) -> f64 { let l = code.to_ascii_lowercase(); let b = l.as_bytes(); if b.len() >= 2 { (b[0] as f64) * 256.0 + (b[1] as f64) } else { 25966.0 } }
    let v = std::env::var("LANG").or_else(|_| std::env::var("LC_ALL")).or_else(|_| std::env::var("LC_MESSAGES")).unwrap_or_default();
    if v.len() >= 2 && !v.starts_with('C') && !v.starts_with("POSIX") { pack(&v) } else { 25966.0 }
}

// ============================================================
// Frame callbacks
// ============================================================

// ============================================================
// Multiple lights
// ============================================================

// ============================================================
// Scene graph (retained mode)
// ============================================================

// ============================================================
// Geometry generation
// ============================================================

// ============================================================
// Shadow mapping
// ============================================================

// ============================================================
// Post-processing
// ============================================================

// ============================================================
// 3D→2D Projection (for UI overlays positioned in 3D space)
// ============================================================


// ============================================================
// Scene picking (raycasting)
// ============================================================


// ============================================================
// Thread-safe staging (for async asset loading via Perry threads)
// ============================================================

fn pollster_block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Wake, Waker};
    use std::pin::Pin;
    use std::sync::Arc;
    struct NoopWaker;
    impl Wake for NoopWaker { fn wake(self: Arc<Self>) {} }
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut future = unsafe { Pin::new_unchecked(Box::new(future)) };
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}



// Q6: Multi-hit picking
// ============================================================

// ============================================================
// Render quality toggles (individual + preset) — ticket 011
// Mirror of the macOS FFI surface added in commit 95da6af; previously
// macOS-only, now exposed on every native platform so non-macOS builds
// don't fail at runtime (missing symbol) when the TS API invokes them.
// ============================================================

// ============================================================
// Render scale / upscale / DRS / post-FX / screenshots / impulse
// Ports of the macOS FFI surface so the bloom/core TS layer links
// cleanly on Linux. EngineState in bloom-shared already exposes the
// underlying renderer methods, so these wrappers are platform-agnostic.
// ============================================================


// ============================================================
// Profiler — CPU phase timings (always available) + GPU timestamps
// (when the adapter supports TIMESTAMP_QUERY). Disabled by default.
// ============================================================

// ============================================================
// Geisterhand screenshot integration
// ============================================================

/// Register Bloom's GPU-based screenshot capture with perry-geisterhand.
/// Mirrors macOS — replaces the platform-default window-grabber with a
/// direct wgpu texture readback that works against the same Vulkan/GL
/// surface bloom is already drawing into.
fn bloom_register_geisterhand_screenshot() {
    extern "C" {
        fn perry_geisterhand_register_screenshot_capture(
            f: extern "C" fn(*mut usize) -> *mut u8,
        );
    }
    unsafe {
        perry_geisterhand_register_screenshot_capture(bloom_screenshot_capture);
    }
}

/// Capture the Bloom framebuffer as PNG. Called from the geisterhand pump
/// BEFORE end_frame in bloom_end_drawing. The vertices_3d/2d and VP matrix
/// from the game loop are still populated; we render to a fresh surface
/// texture with screenshot capture, producing the same visual output as
/// the real frame.
extern "C" fn bloom_screenshot_capture(out_len: *mut usize) -> *mut u8 {
    let eng = engine();

    eng.renderer.screenshot_requested = true;
    eng.scene.prepare(
        &eng.renderer.device,
        &eng.renderer.queue,
        &eng.renderer.vp_matrix(),
        &eng.renderer.prev_vp_matrix,
        eng.renderer.uniform_3d_layout(),
        // Screenshot capture renders everything the camera might see —
        // never occlusion-cull a one-shot capture.
        None,
    );
    eng.scene.prepare_materials(&eng.renderer);
    {
        let t = eng.get_time() as f32;
        let dt = eng.delta_time as f32;
        eng.renderer.material_system_begin_frame(t, dt);
    }
    eng.renderer.end_frame_with_scene(&mut eng.scene, &mut eng.profiler);

    match eng.renderer.screenshot_data.take() {
        Some((width, height, rgba)) => {
            match encode_png(width, height, &rgba) {
                Some(png_data) => {
                    let len = png_data.len();
                    let ptr = unsafe { libc::malloc(len) as *mut u8 };
                    if ptr.is_null() {
                        unsafe { *out_len = 0; }
                        return std::ptr::null_mut();
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(png_data.as_ptr(), ptr, len);
                        *out_len = len;
                    }
                    ptr
                }
                None => {
                    unsafe { *out_len = 0; }
                    std::ptr::null_mut()
                }
            }
        }
        None => {
            unsafe { *out_len = 0; }
            std::ptr::null_mut()
        }
    }
}

/// Minimal PNG encoder (no external dependency). Matches the macOS
/// implementation byte-for-byte so screenshots are identical across
/// platforms.
fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Option<Vec<u8>> {
    use std::io::Write;

    let mut png = Vec::new();
    png.write_all(&[137, 80, 78, 71, 13, 10, 26, 10]).ok()?;

    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8);
    ihdr.push(6);
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(0);
    write_png_chunk(&mut png, b"IHDR", &ihdr);

    let row_bytes = (width * 4) as usize;
    let mut raw = Vec::with_capacity((row_bytes + 1) * height as usize);
    for y in 0..height as usize {
        raw.push(0);
        let start = y * row_bytes;
        for x in 0..width as usize {
            let idx = start + x * 4;
            // wgpu Bgra8UnormSrgb: byte order is B, G, R, A
            raw.push(rgba[idx + 2]);
            raw.push(rgba[idx + 1]);
            raw.push(rgba[idx + 0]);
            raw.push(255);
        }
    }

    let deflated = deflate_store(&raw);
    write_png_chunk(&mut png, b"IDAT", &deflated);
    write_png_chunk(&mut png, b"IEND", &[]);
    Some(png)
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = crc32(&[chunk_type.as_slice(), data].concat());
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn deflate_store(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x78);
    out.push(0x01);

    let mut remaining = data.len();
    let mut offset = 0;
    while remaining > 0 {
        let block_size = remaining.min(65535);
        let is_last = remaining <= 65535;
        out.push(if is_last { 1 } else { 0 });
        let len = block_size as u16;
        let nlen = !len;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(&data[offset..offset + block_size]);
        offset += block_size;
        remaining -= block_size;
    }

    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

// ============================================================
// Physics (Jolt 5.x) — FFI surface generated from shared macro
// ============================================================

#[cfg(feature = "jolt")]
#[inline]
fn bloom_jolt_ffi_physics() -> &'static mut bloom_shared::physics_jolt::JoltPhysics {
    &mut engine().jolt
}

#[cfg(feature = "jolt")]
bloom_shared::define_physics_ffi!();
