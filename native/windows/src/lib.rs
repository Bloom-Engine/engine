use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::{alloc_perry_string, str_from_header};
use bloom_shared::audio::{parse_wav, parse_ogg, parse_mp3};

use std::sync::OnceLock;

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();

/// True when the engine renders into a host-provided child window (a Perry UI
/// `BloomView`) rather than a Bloom-owned top-level window. In embedded mode
/// the host owns the Win32 message loop, so `bloom_begin_drawing` must not pump
/// messages itself and `bloom_window_should_close` always reports "stay open".
#[cfg(windows)]
static mut EMBEDDED: bool = false;

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


/// Map Win32 virtual key code to Bloom key code.
fn map_keycode(vk: u32) -> usize {
    match vk {
        0x41..=0x5A => vk as usize,        // A-Z map directly (65-90)
        0x30..=0x39 => vk as usize,        // 0-9 map directly (48-57)
        0x70..=0x7B => (vk - 0x70 + 112) as usize, // F1-F12
        0x26 => 256,  // VK_UP
        0x28 => 257,  // VK_DOWN
        0x25 => 258,  // VK_LEFT
        0x27 => 259,  // VK_RIGHT
        0x20 => 32,   // VK_SPACE
        0x0D => 265,  // VK_RETURN → Bloom ENTER
        0x1B => 27,   // VK_ESCAPE
        0x09 => 9,    // VK_TAB
        0x08 => 8,    // VK_BACK
        0x2E => 127,  // VK_DELETE
        0x2D => 260,  // VK_INSERT
        0x24 => 261,  // VK_HOME
        0x23 => 262,  // VK_END
        0x21 => 263,  // VK_PRIOR (Page Up)
        0x22 => 264,  // VK_NEXT (Page Down)
        0xA0 => 280,  // VK_LSHIFT
        0xA1 => 281,  // VK_RSHIFT
        0xA2 => 282,  // VK_LCONTROL
        0xA3 => 283,  // VK_RCONTROL
        0xA4 => 284,  // VK_LMENU (Left Alt)
        0xA5 => 285,  // VK_RMENU (Right Alt)
        0x5B => 286,  // VK_LWIN
        0x5C => 287,  // VK_RWIN
        _ => 0,
    }
}

// Windows reports the GENERIC modifier VKs — VK_SHIFT (0x10), VK_CONTROL (0x11),
// VK_MENU (0x12) — in WM_KEYDOWN/UP for BOTH the left and right keys. The
// side-specific codes (VK_LSHIFT 0xA0, etc.) come back only from GetKeyState,
// never in the message wParam. map_keycode knows only the side-specific codes,
// so a raw generic VK mapped to 0 and was dropped: every Shift and Ctrl press
// vanished, which is why sprint (Shift) and dodge (Ctrl) silently never fired on
// Windows. Recover the side from lParam — scancode for Shift (LShift = 0x2A,
// RShift = 0x36), the extended-key bit for Ctrl/Alt (the right-hand ones are
// flagged extended).
fn resolve_modifier_vk(vk: u32, lparam: isize) -> u32 {
    let scancode = ((lparam >> 16) & 0xFF) as u32;
    let extended = (lparam >> 24) & 1 != 0;
    match vk {
        0x10 => if scancode == 0x36 { 0xA1 } else { 0xA0 }, // VK_SHIFT   -> R/L
        0x11 => if extended { 0xA3 } else { 0xA2 },          // VK_CONTROL -> R/L
        0x12 => if extended { 0xA5 } else { 0xA4 },          // VK_MENU    -> R/L
        _ => vk,
    }
}

// ---------------------------------------------------------------------------
// Crash reporting (title-freeze / EN-020 investigation): the game was dying
// with empty stderr, no WER event, and no dump — i.e. invisibly. Catch
// unhandled SEH exceptions, print code + module-relative address
// (symbolizable against main.pdb via llvm-symbolizer), and write our own
// minidump. Fast-fail (0xC0000409) bypasses this filter by design, so a
// death with no output at all narrows to fail-fast / TerminateProcess.
#[cfg(windows)]
mod crash_report {
    use std::os::windows::io::AsRawHandle;
    use std::sync::atomic::{AtomicBool, Ordering};
    use windows::core::{s, w};
    use windows::Win32::Foundation::BOOL;
    use windows::Win32::System::Diagnostics::Debug::{
        SetUnhandledExceptionFilter, EXCEPTION_POINTERS, MINIDUMP_EXCEPTION_INFORMATION,
    };
    use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
    use windows::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentProcessId, GetCurrentThreadId,
    };

    // dbghelp.dll is loaded at crash time — its import lib is not on
    // perry's link line, and adding manifest libs would force a
    // .perry-cache clear for every consumer of the engine.
    type MiniDumpWriteDumpFn = unsafe extern "system" fn(
        hprocess: *mut core::ffi::c_void,
        processid: u32,
        hfile: *mut core::ffi::c_void,
        dumptype: i32,
        exceptionparam: *const MINIDUMP_EXCEPTION_INFORMATION,
        userstreamparam: *const core::ffi::c_void,
        callbackparam: *const core::ffi::c_void,
    ) -> i32;

    // MiniDumpNormal | MiniDumpWithIndirectlyReferencedMemory | MiniDumpWithThreadInfo
    const DUMP_FLAGS: i32 = 0x0000_0040 | 0x0000_1000;

    static INSTALLED: AtomicBool = AtomicBool::new(false);

    pub fn install() {
        if INSTALLED.swap(true, Ordering::SeqCst) {
            return;
        }
        unsafe {
            SetUnhandledExceptionFilter(Some(filter));
        }
        eprintln!("bloom: crash filter installed (dumps -> tools/.testout/dumps)");
    }

    unsafe extern "system" fn filter(info: *const EXCEPTION_POINTERS) -> i32 {
        let (code, addr) = {
            let mut c = 0u32;
            let mut a = 0usize;
            if !info.is_null() {
                let rec = (*info).ExceptionRecord;
                if !rec.is_null() {
                    c = (*rec).ExceptionCode.0 as u32;
                    a = (*rec).ExceptionAddress as usize;
                }
            }
            (c, a)
        };
        let base = GetModuleHandleW(None).map(|m| m.0 as usize).unwrap_or(0);
        let rel = addr.wrapping_sub(base);
        eprintln!("bloom: FATAL unhandled exception {code:#010x} at {addr:#x} (main.exe+{rel:#x})");

        let _ = std::fs::create_dir_all("tools/.testout/dumps");
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = format!(
            "tools/.testout/dumps/crash_main_{}_{stamp:x}.dmp",
            GetCurrentProcessId()
        );
        let write_dump = LoadLibraryW(w!("dbghelp.dll"))
            .ok()
            .and_then(|h| GetProcAddress(h, s!("MiniDumpWriteDump")))
            .map(|p| std::mem::transmute::<_, MiniDumpWriteDumpFn>(p));
        match (std::fs::File::create(&path), write_dump) {
            (Ok(f), Some(dump_fn)) => {
                let exc = MINIDUMP_EXCEPTION_INFORMATION {
                    ThreadId: GetCurrentThreadId(),
                    ExceptionPointers: info as *mut EXCEPTION_POINTERS,
                    ClientPointers: BOOL(0),
                };
                let ok = dump_fn(
                    GetCurrentProcess().0,
                    GetCurrentProcessId(),
                    f.as_raw_handle() as *mut core::ffi::c_void,
                    DUMP_FLAGS,
                    &exc,
                    core::ptr::null(),
                    core::ptr::null(),
                );
                if ok != 0 {
                    eprintln!("bloom: minidump written: {path}");
                } else {
                    eprintln!("bloom: minidump FAILED (MiniDumpWriteDump returned FALSE)");
                }
            }
            (Err(e), _) => eprintln!("bloom: minidump file create failed: {e}"),
            (_, None) => eprintln!("bloom: minidump unavailable (dbghelp load failed)"),
        }
        1 // EXCEPTION_EXECUTE_HANDLER — let the process die
    }
}

// Win32 windowing implementation
#[cfg(windows)]
mod win32 {
    use super::*;
    use windows::Win32::UI::WindowsAndMessaging::*;
    use windows::Win32::UI::HiDpi::*;
    use windows::Win32::Foundation::*;
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::Graphics::Gdi::*;
    use windows::core::*;
    use raw_window_handle::{RawWindowHandle, Win32WindowHandle, RawDisplayHandle, WindowsDisplayHandle};

    static mut HWND_GLOBAL: Option<HWND> = None;
    static mut IS_FULLSCREEN: bool = false;
    static mut WINDOWED_STYLE: u32 = 0;
    static mut WINDOWED_RECT: RECT = RECT { left: 0, top: 0, right: 0, bottom: 0 };

    pub fn set_fullscreen(fullscreen: bool) {
        unsafe {
            let Some(hwnd) = HWND_GLOBAL else { return };

            if fullscreen && !IS_FULLSCREEN {
                // Save current style and window rect for restore
                WINDOWED_STYLE = GetWindowLongW(hwnd, GWL_STYLE) as u32;
                let _ = GetWindowRect(hwnd, &mut WINDOWED_RECT);

                // Get monitor dimensions
                let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
                let mut mi: MONITORINFO = std::mem::zeroed();
                mi.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
                let _ = GetMonitorInfoW(monitor, &mut mi);

                // Set borderless fullscreen
                SetWindowLongW(hwnd, GWL_STYLE, (WS_POPUP | WS_VISIBLE).0 as i32);
                let _ = SetWindowPos(
                    hwnd, HWND_TOP,
                    mi.rcMonitor.left, mi.rcMonitor.top,
                    mi.rcMonitor.right - mi.rcMonitor.left,
                    mi.rcMonitor.bottom - mi.rcMonitor.top,
                    SWP_FRAMECHANGED | SWP_NOOWNERZORDER,
                );
                IS_FULLSCREEN = true;
            } else if !fullscreen && IS_FULLSCREEN {
                // Restore windowed mode
                SetWindowLongW(hwnd, GWL_STYLE, WINDOWED_STYLE as i32);
                let _ = SetWindowPos(
                    hwnd, None,
                    WINDOWED_RECT.left, WINDOWED_RECT.top,
                    WINDOWED_RECT.right - WINDOWED_RECT.left,
                    WINDOWED_RECT.bottom - WINDOWED_RECT.top,
                    SWP_FRAMECHANGED | SWP_NOOWNERZORDER | SWP_NOZORDER,
                );
                IS_FULLSCREEN = false;
            }
        }
    }

    pub fn toggle_fullscreen() {
        unsafe { set_fullscreen(!IS_FULLSCREEN); }
    }

    unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        match msg {
            WM_CLOSE => {
                // Diagnostic (title-freeze investigation): who closes us?
                eprintln!("bloom: WM_CLOSE received");
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            WM_DESTROY => {
                eprintln!("bloom: WM_DESTROY — should_close set, main loop will exit cleanly");
                PostQuitMessage(0);
                if let Some(eng) = ENGINE.get_mut() {
                    eng.should_close = true;
                }
                LRESULT(0)
            }
            WM_KEYDOWN => {
                let bloom_key = map_keycode(resolve_modifier_vk(wparam.0 as u32, lparam.0));
                if bloom_key > 0 {
                    if let Some(eng) = ENGINE.get_mut() {
                        eng.input.set_key_down(bloom_key);
                    }
                }
                // F12 — native screenshot hotkey. Perry currently drops
                // the game-side takeScreenshot FFI call entirely
                // (PerryTS/perry#6087), so the capture is triggered here
                // where no compiler sits in the path. Initial press only
                // (lparam bit 30 = previous key state) so holding the key
                // doesn't machine-gun PNGs. The file lands in the working
                // directory with a timestamped name; the readback runs at
                // this frame's end_frame.
                if wparam.0 as u32 == 0x7B && (lparam.0 as u32 >> 30) & 1 == 0 {
                    if let Some(eng) = ENGINE.get_mut() {
                        let ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis())
                            .unwrap_or(0);
                        let path = format!("screenshot_{}.png", ms);
                        eprintln!("bloom: F12 screenshot -> {}", path);
                        eng.renderer.screenshot_requested = true;
                        eng.renderer.pending_screenshot_path = Some(path);
                    }
                }
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            WM_KEYUP => {
                let bloom_key = map_keycode(resolve_modifier_vk(wparam.0 as u32, lparam.0));
                if bloom_key > 0 {
                    if let Some(eng) = ENGINE.get_mut() {
                        eng.input.set_key_up(bloom_key);
                    }
                }
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            WM_MOUSEMOVE => {
                let x = (lparam.0 & 0xFFFF) as i16 as f64;
                let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as f64;
                if let Some(eng) = ENGINE.get_mut() {
                    if eng.input.cursor_disabled {
                        // FPS-style capture: accumulate the movement away from
                        // the window centre as a raw delta, then snap the OS
                        // cursor back to centre so look never hits the screen
                        // edge. (begin_frame consumes raw_delta when the cursor
                        // is disabled.) The recenter generates another
                        // WM_MOUSEMOVE at the centre with zero delta — no loop.
                        let mut rect = RECT::default();
                        let _ = GetClientRect(hwnd, &mut rect);
                        let cx = ((rect.right - rect.left) / 2) as f64;
                        let cy = ((rect.bottom - rect.top) / 2) as f64;
                        let dx = x - cx;
                        let dy = y - cy;
                        if dx != 0.0 || dy != 0.0 {
                            eng.input.accumulate_mouse_delta(dx, dy);
                            let mut pt = POINT { x: cx as i32, y: cy as i32 };
                            let _ = ClientToScreen(hwnd, &mut pt);
                            let _ = SetCursorPos(pt.x, pt.y);
                        }
                    } else {
                        eng.input.set_mouse_position(x, y);
                    }
                }
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            WM_LBUTTONDOWN => {
                if let Some(eng) = ENGINE.get_mut() {
                    eng.input.set_mouse_button_down(0);
                }
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            WM_LBUTTONUP => {
                if let Some(eng) = ENGINE.get_mut() {
                    eng.input.set_mouse_button_up(0);
                }
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            WM_RBUTTONDOWN => {
                if let Some(eng) = ENGINE.get_mut() {
                    eng.input.set_mouse_button_down(1);
                }
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            WM_RBUTTONUP => {
                if let Some(eng) = ENGINE.get_mut() {
                    eng.input.set_mouse_button_up(1);
                }
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            0x0005 /* WM_SIZE */ => {
                // lParam carries the new client-area size in *physical*
                // pixels (Per-Monitor-Aware-V2). Derive logical via
                // current DPI so the rest of the engine sees the same
                // logical/physical split macOS does.
                let phys_w = (lparam.0 & 0xFFFF) as u32;
                let phys_h = ((lparam.0 >> 16) & 0xFFFF) as u32;
                if phys_w > 0 && phys_h > 0 {
                    if let Some(eng) = ENGINE.get_mut() {
                        if phys_w != eng.renderer.physical_width()
                            || phys_h != eng.renderer.physical_height()
                        {
                            let scale = dpi_scale(hwnd);
                            let log_w = ((phys_w as f64) / scale).round() as u32;
                            let log_h = ((phys_h as f64) / scale).round() as u32;
                            eng.renderer.resize(phys_w, phys_h, log_w, log_h);
                        }
                    }
                }
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            0x02E0 /* WM_DPICHANGED */ => {
                // The window dragged onto a different-DPI monitor.
                // lParam holds a suggested RECT — Windows wants us to
                // accept this geometry to keep the window's apparent
                // size constant across the DPI change. The follow-up
                // WM_SIZE handles the renderer resize.
                let suggested = &*(lparam.0 as *const RECT);
                let _ = SetWindowPos(
                    hwnd, None,
                    suggested.left, suggested.top,
                    suggested.right - suggested.left,
                    suggested.bottom - suggested.top,
                    SWP_NOZORDER | SWP_NOACTIVATE,
                );
                LRESULT(0)
            }
            _ => DefWindowProcW(hwnd, msg, wparam, lparam),
        }
    }

    /// Returns (hwnd, physical_w, physical_h). Caller's `width`/`height`
    /// are *logical* — on a HiDPI monitor with scale 2.0 the window
    /// will appear logically the right size while its client area is
    /// scaled-up physical pixels for the renderer to fill.
    pub fn create_window(width: f64, height: f64, title: &str) -> (HWND, u32, u32) {
        unsafe {
            // Per-Monitor-Aware-V2: the window's DPI tracks the monitor
            // it's currently on, and Windows fires WM_DPICHANGED when
            // it moves between monitors of different DPI. Without this
            // call, Win32 silently virtualises us to 96 DPI on HiDPI
            // displays — a 4K monitor would render at ~2560×1440.
            // Safe to call early; ignored on pre-Win10 1703.
            let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

            let hmodule = GetModuleHandleW(None).unwrap();
            let hinstance: HINSTANCE = hmodule.into();
            let class_name = w!("BloomWindowClass");

            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(wndproc),
                hInstance: hinstance,
                lpszClassName: class_name,
                hCursor: LoadCursorW(None, IDC_ARROW).unwrap(),
                ..Default::default()
            };
            RegisterClassExW(&wc);

            let title_wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();

            // Initial window size in physical pixels. We don't have a
            // window handle yet, so use the system DPI — close enough,
            // and WM_DPICHANGED will resize on the first move if the
            // user lands on a different-DPI monitor.
            let system_dpi = GetDpiForSystem().max(96);
            let scale = system_dpi as f64 / 96.0;
            let phys_w = (width * scale).round() as i32;
            let phys_h = (height * scale).round() as i32;

            let hwnd = CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                class_name,
                PCWSTR(title_wide.as_ptr()),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                CW_USEDEFAULT, CW_USEDEFAULT,
                phys_w, phys_h,
                None, None, Some(&hinstance), None,
            ).unwrap();

            ShowWindow(hwnd, SW_SHOW);
            HWND_GLOBAL = Some(hwnd);

            // After the window exists, query the actual client-area
            // size. Includes whatever DWM trimmed for borders and
            // reflects this monitor's DPI rather than the system one.
            let mut rect = RECT::default();
            let _ = GetClientRect(hwnd, &mut rect);
            let actual_w = (rect.right - rect.left).max(1) as u32;
            let actual_h = (rect.bottom - rect.top).max(1) as u32;
            (hwnd, actual_w, actual_h)
        }
    }

    /// Current per-monitor DPI scale for this window (1.0 on a 96-DPI
    /// monitor, 2.0 on a Retina-class 192-DPI monitor, etc.). Falls
    /// back to 1.0 if the API is missing (pre-Win10 1607).
    pub fn dpi_scale(hwnd: HWND) -> f64 {
        unsafe {
            let dpi = GetDpiForWindow(hwnd);
            if dpi == 0 { 1.0 } else { dpi as f64 / 96.0 }
        }
    }

    pub fn poll_events() {
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).into() {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    // ---- Embedded mode (Perry UI BloomView host window) ----
    //
    // When Bloom renders into a Perry UI child window, that window already has
    // Perry's own WNDPROC. We classic-subclass it so Bloom sees the WM_SIZE /
    // keyboard / mouse it needs (its own `wndproc` above never runs for a
    // foreign-class window), then chain to the original proc. The host's
    // message loop dispatches these — Bloom never pumps in embedded mode.
    static mut EMBED_ORIG_WNDPROC: isize = 0;

    unsafe extern "system" fn embedded_wndproc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        match msg {
            0x0005 /* WM_SIZE */ => {
                let phys_w = (lparam.0 & 0xFFFF) as u32;
                let phys_h = ((lparam.0 >> 16) & 0xFFFF) as u32;
                if phys_w > 0 && phys_h > 0 {
                    if let Some(eng) = ENGINE.get_mut() {
                        if phys_w != eng.renderer.physical_width()
                            || phys_h != eng.renderer.physical_height()
                        {
                            let scale = dpi_scale(hwnd);
                            let log_w = ((phys_w as f64) / scale).round() as u32;
                            let log_h = ((phys_h as f64) / scale).round() as u32;
                            eng.renderer.resize(phys_w, phys_h, log_w, log_h);
                        }
                    }
                }
            }
            WM_KEYDOWN => {
                let k = map_keycode(resolve_modifier_vk(wparam.0 as u32, lparam.0));
                if k > 0 { if let Some(eng) = ENGINE.get_mut() { eng.input.set_key_down(k); } }
            }
            WM_KEYUP => {
                let k = map_keycode(resolve_modifier_vk(wparam.0 as u32, lparam.0));
                if k > 0 { if let Some(eng) = ENGINE.get_mut() { eng.input.set_key_up(k); } }
            }
            WM_MOUSEMOVE => {
                let x = (lparam.0 & 0xFFFF) as i16 as f64;
                let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as f64;
                if let Some(eng) = ENGINE.get_mut() { eng.input.set_mouse_position(x, y); }
            }
            WM_LBUTTONDOWN => { if let Some(eng) = ENGINE.get_mut() { eng.input.set_mouse_button_down(0); } }
            WM_LBUTTONUP   => { if let Some(eng) = ENGINE.get_mut() { eng.input.set_mouse_button_up(0);   } }
            WM_RBUTTONDOWN => { if let Some(eng) = ENGINE.get_mut() { eng.input.set_mouse_button_down(1); } }
            WM_RBUTTONUP   => { if let Some(eng) = ENGINE.get_mut() { eng.input.set_mouse_button_up(1);   } }
            _ => {}
        }
        let orig: WNDPROC = std::mem::transmute(EMBED_ORIG_WNDPROC);
        CallWindowProcW(orig, hwnd, msg, wparam, lparam)
    }

    /// Subclass a host-provided child window so Bloom receives resize/input.
    pub unsafe fn attach_subclass(hwnd: HWND) {
        let prev = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, embedded_wndproc as usize as isize);
        EMBED_ORIG_WNDPROC = prev;
        HWND_GLOBAL = Some(hwnd);
    }
}

#[no_mangle]
pub extern "C" fn bloom_init_window(width: f64, height: f64, title_ptr: *const u8, fullscreen: f64) {
    let title = str_from_header(title_ptr);

    #[cfg(windows)]
    {
        crash_report::install();
        let (hwnd, phys_w, phys_h) = win32::create_window(width, height, title);
        unsafe { init_engine_for_hwnd(hwnd, width as u32, height as u32, phys_w, phys_h); }
        if fullscreen != 0.0 {
            win32::set_fullscreen(true);
        }
    }

    #[cfg(not(windows))]
    {
        let _ = (width, height, title, fullscreen);
        panic!("bloom-windows can only run on Windows");
    }
}

/// Build the wgpu surface + engine on an existing HWND (top-level window or
/// host-provided child). Shared by `bloom_init_window` (Bloom owns the window)
/// and `bloom_attach_hwnd` (a Perry UI `BloomView` child window). `logical_*`
/// are DPI-independent sizes; `phys_*` are the surface's physical client size.
#[cfg(windows)]
unsafe fn init_engine_for_hwnd(
    hwnd: windows::Win32::Foundation::HWND,
    logical_w: u32,
    logical_h: u32,
    phys_w: u32,
    phys_h: u32,
) {
        // Compile shaders with DXC, not FXC.
        //
        // This is not a nicety. wgpu's DX12 backend reports the adapter's
        // shader model as min(device, compiler), and FXC — its default — caps
        // that at 5.1. Hardware ray query requires 6.5 (wgpu-hal
        // dx12/adapter.rs: `supports_ray_tracing`), so with FXC,
        // EXPERIMENTAL_RAY_QUERY is never exposed on DX12 on *any* GPU, no
        // matter how capable. Lumen then silently takes its software path and
        // the frame quietly loses its hardware-traced GI. That is what the
        // `ray_query=false` in the boot line has been telling us.
        //
        // DXC is loaded at runtime from `dxcompiler.dll` + `dxil.dll`. wgpu's
        // `static-dxc` feature would avoid the DLLs but needs MSVC's ATL,
        // which is not part of a default toolchain install. Both DLLs ship
        // with the Windows SDK; `tools/fetch-dxc.ps1` copies them next to the
        // binary. If they are missing, wgpu falls back to FXC on its own — we
        // lose HW ray query, exactly as before, and nothing else breaks.
        let mut backend_options = wgpu::BackendOptions::default();
        backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc {
            dxc_path: String::from("dxcompiler.dll"),
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
            backend_options,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let surface = unsafe {
            let mut handle = raw_window_handle::Win32WindowHandle::new(
                std::num::NonZeroIsize::new(hwnd.0 as isize).unwrap()
            );
            let raw = raw_window_handle::RawWindowHandle::Win32(handle);
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(raw_window_handle::RawDisplayHandle::Windows(
                    raw_window_handle::WindowsDisplayHandle::new()
                )),
                raw_window_handle: raw,
            }).expect("Failed to create surface")
        };

        // Pick the adapter that can actually trace rays.
        //
        // We used to take whatever `request_adapter` handed back, which on
        // Windows means DX12 — and wgpu's DX12 backend only reports
        // EXPERIMENTAL_RAY_QUERY when the driver advertises D3D12 raytracing
        // tier 1.1. The same GPU under Vulkan can expose VK_KHR_ray_query
        // when DX12 does not. The result was silent: Lumen's hardware trace
        // was never selected, the software SDF path ran instead, and nobody
        // saw a reason why. So enumerate the candidates, say out loud what
        // each one offers, and prefer one that supports ray query.
        //
        // BLOOM_FORCE_SW_GI keeps its meaning: it also stops us from picking
        // a backend *for* ray tracing, so the software path can be tested on
        // hardware that would otherwise take the fast route.
        let want_rt = !std::env::var("BLOOM_FORCE_SW_GI")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let candidates = pollster_block_on(
            instance.enumerate_adapters(wgpu::Backends::DX12 | wgpu::Backends::VULKAN),
        );
        let mut rt_adapter: Option<wgpu::Adapter> = None;
        for cand in candidates {
            let info = cand.get_info();
            let surf_ok = cand.is_surface_supported(&surface);
            let has_rt = cand
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
            eprintln!(
                "bloom: candidate '{}' ({:?}), ray_query={}, surface={}",
                info.name, info.backend, has_rt, surf_ok,
            );
            if want_rt && has_rt && surf_ok && rt_adapter.is_none() {
                rt_adapter = Some(cand);
            }
        }

        let adapter = match rt_adapter {
            Some(a) => a,
            None => pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
            .expect("No adapter found"),
        };

        {
            // One line of boot truth: which GPU we got and whether the
            // features that silently reshape the frame (HW ray query for
            // GI, timestamps for the profiler) are available on it.
            let info = adapter.get_info();
            eprintln!(
                "bloom: adapter '{}' ({:?}), ray_query={}, timestamps={}, tex_arrays={}",
                info.name,
                info.backend,
                adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
                adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
                adapter.features().contains(
                    wgpu::Features::TEXTURE_BINDING_ARRAY
                        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                ),
            );
        }

        // Ticket 007b: HW ray-query via DXR 1.1 / VK_KHR_ray_query.
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
        // PT-2: texture binding array + non-uniform indexing for textured
        // path-trace hit shading. Both or neither (the kernel indexes the
        // array with a per-thread material id).
        let pt_tex_mask = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        if supported.contains(pt_tex_mask) {
            required_features |= pt_tex_mask;
        }
        let experimental_features = if required_features.intersects(rt_mask) {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        let mut required_limits = wgpu::Limits::default();
        // Phase 1c: the material ABI declares 5 bind groups (PerFrame,
        // PerView, PerMaterial, PerDraw, SceneInputs). wgpu's default
        // limit is 4. Metal / Vulkan / D3D12 support at least 7, so 5 is
        // safely within every real backend's capabilities.
        required_limits.max_bind_groups = 5;
        // The refractive/translucent material profile binds up to 19
        // sampled textures in the fragment stage (5 material maps + env/
        // BRDF/3 shadow cascades/env-diffuse + planar reflection + 3 texture
        // arrays + the group-4 scene_color/scene_depth/impulse/motion inputs).
        // wgpu's default is 16. Raise to whatever the adapter actually
        // supports — every real D3D12/Vulkan/Metal GPU exposes ≥128 — so
        // opaque/transparent materials are unaffected and refractive ones link.
        let adapter_limits = adapter.limits();
        required_limits.max_sampled_textures_per_shader_stage =
            adapter_limits.max_sampled_textures_per_shader_stage;
        required_limits.max_samplers_per_shader_stage =
            adapter_limits.max_samplers_per_shader_stage;
        // PT-2: binding arrays have their own element budget, default 0.
        // Take whatever the adapter offers; the renderer checks the
        // granted value against its fixed array size before compiling
        // the textured kernel variant.
        if required_features.contains(pt_tex_mask) {
            required_limits.max_binding_array_elements_per_shader_stage =
                adapter_limits.max_binding_array_elements_per_shader_stage;
        }
        // PT-4: the path-trace kernel binds 9 storage buffers (accum +
        // moments + reservoir ping-pongs on top of instance/geo data);
        // the wgpu default limit is 8.
        required_limits.max_storage_buffers_per_shader_stage = required_limits
            .max_storage_buffers_per_shader_stage
            .max(adapter_limits.max_storage_buffers_per_shader_stage.min(16));
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
        // Surface is configured at the *physical* client-area size we
        // got back from create_window; Renderer::new takes the
        // caller's logical size separately so screenWidth() etc. keep
        // returning DPI-independent numbers.
        let surface_config = wgpu::SurfaceConfiguration {
            // COPY_SRC: bloom_take_screenshot reads the swapchain back;
            // without it the readback copy is a swallowed validation
            // error and screenshots silently produce nothing on Windows.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            format,
            width: phys_w,
            height: phys_h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // `Renderer::new` routes initial target creation through `resize`
        // itself, so the construction-time targets already honour
        // render_scale — no explicit resize needed here (it used to live
        // here, which left every non-Windows host with the bug).
        let renderer = Renderer::new(device, queue, surface, surface_config, logical_w, logical_h);
        let _ = ENGINE.set(EngineState::new(renderer));
}

/// Attach the engine to a host-owned `HWND` instead of creating its own
/// top-level window (PerryTS/perry#5519). `handle` is the child `HWND`
/// the host (Perry UI's `BloomView`) owns; `width`/`height` are its
/// client size in physical pixels. Returns 1.0 on success, 0.0 on a
/// null/invalid handle or surface bring-up failure. Idempotent once
/// attached.
#[no_mangle]
pub extern "C" fn bloom_attach_native(handle: i64, width: f64, height: f64) -> f64 {
    if handle == 0 {
        return 0.0;
    }
    if unsafe { ENGINE.get() }.is_some() {
        return 1.0;
    }

    #[cfg(windows)]
    {
        let Some(hwnd_nz) = std::num::NonZeroIsize::new(handle as isize) else {
            return 0.0;
        };
        let target = {
            let h = raw_window_handle::Win32WindowHandle::new(hwnd_nz);
            wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(raw_window_handle::RawDisplayHandle::Windows(
                    raw_window_handle::WindowsDisplayHandle::new(),
                )),
                raw_window_handle: raw_window_handle::RawWindowHandle::Win32(h),
            }
        };
        match unsafe {
            bloom_shared::attach::attach_engine(
                target,
                bloom_shared::attach::AttachParams {
                    backends: wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
                    logical_w: (width as u32).max(1),
                    logical_h: (height as u32).max(1),
                    physical_w: (width as u32).max(1),
                    physical_h: (height as u32).max(1),
                    format: bloom_shared::attach::FormatPreference::First,
                },
            )
        } {
            Ok(es) => {
                unsafe {
                    let _ = ENGINE.set(es);
                }
                1.0
            }
            Err(_) => 0.0,
        }
    }

    #[cfg(not(windows))]
    {
        let _ = (width, height);
        0.0
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_window() {}

/// Attach the engine to a host-provided child window (a Perry UI `BloomView`).
/// `hwnd_bits` is the raw HWND value as an integer (from `bloomViewGetHwnd`).
/// `width`/`height` are the logical viewport size. The engine builds its wgpu
/// surface on this window and subclasses it for resize/input; the host drives
/// frames via `bloom_begin_drawing` / `bloom_end_drawing`.
#[no_mangle]
pub extern "C" fn bloom_attach_hwnd(hwnd_bits: f64, width: f64, height: f64) {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Foundation::{HWND, RECT};
        use windows::Win32::UI::WindowsAndMessaging::GetClientRect;
        let hwnd = HWND(hwnd_bits as i64 as isize as *mut core::ffi::c_void);
        let mut rect = RECT::default();
        let _ = GetClientRect(hwnd, &mut rect);
        let phys_w = (rect.right - rect.left).max(1) as u32;
        let phys_h = (rect.bottom - rect.top).max(1) as u32;
        if ENGINE.get().is_none() {
            init_engine_for_hwnd(hwnd, width as u32, height as u32, phys_w, phys_h);
        }
        EMBEDDED = true;
        win32::attach_subclass(hwnd);
    }
    #[cfg(not(windows))]
    { let _ = (hwnd_bits, width, height); }
}

/// Resize the engine's surface. `phys_*` are physical pixels, `log_*` logical.
/// Embedded `BloomView`s resize automatically via the subclassed WM_SIZE; this
/// is exposed for hosts that need to drive the size explicitly.
#[no_mangle]
pub extern "C" fn bloom_resize(phys_w: f64, phys_h: f64, log_w: f64, log_h: f64) {
    #[cfg(windows)]
    unsafe {
        if let Some(eng) = ENGINE.get_mut() {
            eng.renderer.resize(phys_w as u32, phys_h as u32, log_w as u32, log_h as u32);
        }
    }
    #[cfg(not(windows))]
    { let _ = (phys_w, phys_h, log_w, log_h); }
}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 {
    #[cfg(windows)]
    unsafe { if EMBEDDED { return 0.0; } }
    if engine().should_close { 1.0 } else { 0.0 }
}

#[cfg(windows)]
fn poll_xinput_gamepad() {
    use windows::Win32::UI::Input::XboxController::*;
    let eng = engine();
    let mut state = XINPUT_STATE::default();
    let result = unsafe { XInputGetState(0, &mut state) };
    if result == 0 {
        // ERROR_SUCCESS
        eng.input.gamepad_available = true;
        let gp = &state.Gamepad;

        // Axes: left stick X/Y, right stick X/Y, triggers
        let normalize = |v: i16| -> f32 {
            if v > 0 { v as f32 / 32767.0 } else { v as f32 / 32768.0 }
        };
        eng.input.set_gamepad_axis(0, normalize(gp.sThumbLX));
        eng.input.set_gamepad_axis(1, -normalize(gp.sThumbLY)); // invert Y
        eng.input.set_gamepad_axis(2, normalize(gp.sThumbRX));
        eng.input.set_gamepad_axis(3, -normalize(gp.sThumbRY));
        eng.input.set_gamepad_axis(4, gp.bLeftTrigger as f32 / 255.0);
        eng.input.set_gamepad_axis(5, gp.bRightTrigger as f32 / 255.0);
        eng.input.gamepad_axis_count = 6;

        // Buttons
        let buttons = gp.wButtons;
        let mappings: &[(u16, usize)] = &[
            (0x1000, 0),  // A
            (0x2000, 1),  // B
            (0x4000, 2),  // X
            (0x8000, 3),  // Y
            (0x0100, 4),  // Left bumper
            (0x0200, 5),  // Right bumper
            (0x0020, 6),  // Back/Select
            (0x0010, 7),  // Start
            (0x0040, 8),  // Left stick press
            (0x0080, 9),  // Right stick press
            (0x0001, 10), // DPad Up
            (0x0002, 11), // DPad Down
            (0x0004, 12), // DPad Left
            (0x0008, 13), // DPad Right
        ];
        for &(mask, idx) in mappings {
            if buttons.0 & mask != 0 {
                eng.input.set_gamepad_button_down(idx);
            } else {
                eng.input.set_gamepad_button_up(idx);
            }
        }

        // EN-031 — rumble. The FFI writes (low, high, seconds) into shared
        // input state; we own the motors and the countdown. Only push a new
        // XInputSetState when the commanded level actually changes, since the
        // call goes out over USB/BT and doing it every frame is wasteful.
        let dt = eng.delta_time as f32;
        let (lo, hi, mut left) = (eng.input.rumble[0], eng.input.rumble[1], eng.input.rumble[2]);
        let (want_lo, want_hi) = if left > 0.0 { (lo, hi) } else { (0.0, 0.0) };
        unsafe {
            if (want_lo, want_hi) != LAST_RUMBLE {
                let mut vib = XINPUT_VIBRATION {
                    wLeftMotorSpeed: (want_lo * 65535.0) as u16,
                    wRightMotorSpeed: (want_hi * 65535.0) as u16,
                };
                let _ = XInputSetState(0, &mut vib);
                LAST_RUMBLE = (want_lo, want_hi);
            }
        }
        if left > 0.0 {
            left = (left - dt).max(0.0);
            eng.input.rumble[2] = left;
        }
    } else {
        eng.input.gamepad_available = false;
        // Pad unplugged mid-rumble: forget the commanded level so a
        // reconnecting pad doesn't inherit a stale "still buzzing" state.
        unsafe { LAST_RUMBLE = (0.0, 0.0); }
    }
}

/// Last vibration level actually pushed to the pad, so we only call
/// XInputSetState on change.
#[cfg(windows)]
static mut LAST_RUMBLE: (f32, f32) = (0.0, 0.0);

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    #[cfg(windows)]
    {
        // In embedded mode the host (Perry UI) owns the message loop and
        // dispatches our subclassed window's messages — pumping here would
        // steal messages from the host. Only poll when Bloom owns the window.
        unsafe { if !EMBEDDED { win32::poll_events(); } }
        poll_xinput_gamepad();
    }
    engine().begin_frame();
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() { engine().end_frame(); }

static AUDIO_RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[no_mangle]
pub extern "C" fn bloom_init_audio() {
    #[cfg(windows)]
    {
        use std::sync::atomic::Ordering;
        AUDIO_RUNNING.store(true, Ordering::SeqCst);

        // Move the render half into the audio thread; the engine keeps
        // only the command-producing control half.
        let renderer = engine().audio.take_renderer();
        std::thread::spawn(move || {
            unsafe { wasapi_audio_thread(renderer); }
        });
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_audio() {
    AUDIO_RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));
}

#[cfg(windows)]
unsafe fn wasapi_audio_thread(mut renderer: Option<bloom_shared::audio::AudioRenderer>) {
    use windows::Win32::Media::Audio::*;
    use windows::Win32::System::Com::*;
    use windows::core::*;
    use std::sync::atomic::Ordering;

    // Initialize COM on this thread
    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

    // Create device enumerator and get default output device
    let enumerator: IMMDeviceEnumerator = match CoCreateInstance(
        &MMDeviceEnumerator,
        None,
        CLSCTX_ALL,
    ) {
        Ok(e) => e,
        Err(_) => return,
    };

    let device = match enumerator.GetDefaultAudioEndpoint(eRender, eConsole) {
        Ok(d) => d,
        Err(_) => return,
    };

    let audio_client: IAudioClient = match device.Activate(CLSCTX_ALL, None) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Get mix format
    let mix_format_ptr = match audio_client.GetMixFormat() {
        Ok(f) => f,
        Err(_) => return,
    };
    let mix_format = &*mix_format_ptr;
    let sample_rate = mix_format.nSamplesPerSec;
    let channels = mix_format.nChannels as usize;

    // Initialize in shared mode with 20ms buffer
    let buffer_duration = 200_000; // 20ms in 100-nanosecond units
    if audio_client.Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        0,
        buffer_duration,
        0,
        mix_format_ptr,
        None,
    ).is_err() {
        return;
    }

    let buffer_size = match audio_client.GetBufferSize() {
        Ok(s) => s as usize,
        Err(_) => return,
    };

    let render_client: IAudioRenderClient = match audio_client.GetService() {
        Ok(r) => r,
        Err(_) => return,
    };

    let _ = audio_client.Start();

    // Temporary buffer for mixing (always stereo f32 from our mixer)
    let mut mix_buf = vec![0.0f32; buffer_size * 2];

    while AUDIO_RUNNING.load(Ordering::SeqCst) {
        let padding = audio_client.GetCurrentPadding().unwrap_or(0) as usize;
        let available = buffer_size - padding;
        if available == 0 {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }

        let buffer_ptr = match render_client.GetBuffer(available as u32) {
            Ok(p) => p,
            Err(_) => { std::thread::sleep(std::time::Duration::from_millis(2)); continue; }
        };

        // Mix audio
        let mix_samples = available * 2; // stereo
        for i in 0..mix_samples { mix_buf[i] = 0.0; }
        // Renderer is moved into this thread at spawn — no shared
        // engine state is touched from here (see audio/mod.rs contract).
        if let Some(r) = renderer.as_mut() {
            r.mix(&mut mix_buf[..mix_samples]);
        }

        // Write to WASAPI buffer (format is typically f32 or i16, assume float since we requested shared mode)
        let bits = mix_format.wBitsPerSample;
        let out_channels = channels;
        if bits == 32 {
            let out = std::slice::from_raw_parts_mut(buffer_ptr as *mut f32, available * out_channels);
            for i in 0..available {
                let l = mix_buf[i * 2];
                let r = if i * 2 + 1 < mix_samples { mix_buf[i * 2 + 1] } else { l };
                if out_channels >= 2 {
                    out[i * out_channels] = l;
                    out[i * out_channels + 1] = r;
                    for c in 2..out_channels { out[i * out_channels + c] = 0.0; }
                } else {
                    out[i] = (l + r) * 0.5;
                }
            }
        } else if bits == 16 {
            let out = std::slice::from_raw_parts_mut(buffer_ptr as *mut i16, available * out_channels);
            for i in 0..available {
                let l = mix_buf[i * 2];
                let r = if i * 2 + 1 < mix_samples { mix_buf[i * 2 + 1] } else { l };
                if out_channels >= 2 {
                    out[i * out_channels] = (l * 32767.0) as i16;
                    out[i * out_channels + 1] = (r * 32767.0) as i16;
                    for c in 2..out_channels { out[i * out_channels + c] = 0; }
                } else {
                    out[i] = ((l + r) * 0.5 * 32767.0) as i16;
                }
            }
        }

        let _ = render_client.ReleaseBuffer(available as u32, 0);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    let _ = audio_client.Stop();
    CoUninitialize();
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
    #[cfg(windows)]
    win32::toggle_fullscreen();
}
#[no_mangle]
pub extern "C" fn bloom_set_window_title(title_ptr: *const u8) { let _ = str_from_header(title_ptr); }
#[no_mangle]
pub extern "C" fn bloom_set_window_icon(path_ptr: *const u8) { let _ = str_from_header(path_ptr); }

#[no_mangle]
pub extern "C" fn bloom_disable_cursor() {
    engine().input.cursor_disabled = true;
    // Hide the OS cursor (ShowCursor keeps a counter; loop until it's hidden).
    unsafe { while windows::Win32::UI::WindowsAndMessaging::ShowCursor(false) >= 0 {} }
}

#[no_mangle]
pub extern "C" fn bloom_enable_cursor() {
    engine().input.cursor_disabled = false;
    unsafe { while windows::Win32::UI::WindowsAndMessaging::ShowCursor(true) < 0 {} }
}

// E4: Clipboard (stub on this platform)
#[no_mangle]
pub extern "C" fn bloom_set_clipboard_text(_text_ptr: *const u8) {}
#[no_mangle]
pub extern "C" fn bloom_get_clipboard_text() -> *const u8 { std::ptr::null() }

// E5b: File dialogs (stub on this platform)
#[no_mangle]
pub extern "C" fn bloom_open_file_dialog(_filter_ptr: *const u8, _title_ptr: *const u8) -> *const u8 { std::ptr::null() }
#[no_mangle]
pub extern "C" fn bloom_save_file_dialog(_default_name_ptr: *const u8, _title_ptr: *const u8) -> *const u8 { std::ptr::null() }
#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 { 3.0 }

/// Preferred OS language packed as `c0*256+c1` (ISO-639 primary subtag), from
/// `GetUserDefaultLocaleName` (e.g. "en-US" -> "en"). Falls back to "en".
#[no_mangle]
pub extern "C" fn bloom_get_language() -> f64 {
    use windows::Win32::Globalization::GetUserDefaultLocaleName;
    let mut buf = [0u16; 85]; // LOCALE_NAME_MAX_LENGTH
    let n = unsafe { GetUserDefaultLocaleName(&mut buf) };
    if n >= 2 {
        let lc = |c: u16| -> u8 { let b = c as u8; if b.is_ascii_uppercase() { b + 32 } else { b } };
        let (c0, c1) = (lc(buf[0]), lc(buf[1]));
        if c0.is_ascii_alphabetic() && c1.is_ascii_alphabetic() {
            return (c0 as f64) * 256.0 + (c1 as f64);
        }
    }
    25966.0
}

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
// Ports of the macOS / Linux FFI surface so the bloom/core TS layer
// links cleanly on Windows. EngineState in bloom-shared already
// exposes the underlying renderer methods, so these wrappers are
// platform-agnostic.
// ============================================================

// ============================================================
// Profiler — CPU phase timings (always available) + GPU timestamps
// (when the adapter supports TIMESTAMP_QUERY). Disabled by default.
// ============================================================

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
