use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::{alloc_perry_string, str_from_header};
use bloom_shared::audio::{parse_wav, parse_ogg, parse_mp3};

use std::sync::OnceLock;

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();

fn engine() -> &'static mut EngineState {
    unsafe { ENGINE.get_mut().expect("Engine not initialized") }
}

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
            WM_DESTROY => {
                PostQuitMessage(0);
                if let Some(eng) = ENGINE.get_mut() {
                    eng.should_close = true;
                }
                LRESULT(0)
            }
            WM_KEYDOWN => {
                let bloom_key = map_keycode(wparam.0 as u32);
                if bloom_key > 0 {
                    if let Some(eng) = ENGINE.get_mut() {
                        eng.input.set_key_down(bloom_key);
                    }
                }
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
            WM_KEYUP => {
                let bloom_key = map_keycode(wparam.0 as u32);
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
                    eng.input.set_mouse_position(x, y);
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
}

#[no_mangle]
pub extern "C" fn bloom_init_window(width: f64, height: f64, title_ptr: *const u8, fullscreen: f64) {
    let title = str_from_header(title_ptr);

    #[cfg(windows)]
    {
        let (hwnd, phys_w, phys_h) = win32::create_window(width, height, title);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
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

        let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })).expect("No adapter found");

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
        if !force_sw_gi && supported.contains(rt_mask) {
            required_features |= rt_mask;
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
            win32::set_fullscreen(true);
        }
    }

    #[cfg(not(windows))]
    {
        panic!("bloom-windows can only run on Windows");
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_window() {}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 {
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
    } else {
        eng.input.gamepad_available = false;
    }
}

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    #[cfg(windows)]
    {
        win32::poll_events();
        poll_xinput_gamepad();
    }
    engine().begin_frame();
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() { engine().end_frame(); }

#[no_mangle]
pub extern "C" fn bloom_clear_background(r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.set_clear_color(r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_set_target_fps(fps: f64) { engine().target_fps = fps; }

#[no_mangle]
pub extern "C" fn bloom_set_direct_2d_mode(on: f64) { engine().direct_2d_mode = on > 0.5; }

#[no_mangle]
pub extern "C" fn bloom_get_delta_time() -> f64 { engine().delta_time }

#[no_mangle]
pub extern "C" fn bloom_get_fps() -> f64 { engine().get_fps() }

#[no_mangle]
pub extern "C" fn bloom_get_screen_width() -> f64 { engine().screen_width() }

#[no_mangle]
pub extern "C" fn bloom_get_screen_height() -> f64 { engine().screen_height() }

#[no_mangle]
pub extern "C" fn bloom_is_key_pressed(key: f64) -> f64 {
    if engine().input.is_key_pressed(key as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_key_down(key: f64) -> f64 {
    if engine().input.is_key_down(key as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_key_released(key: f64) -> f64 {
    if engine().input.is_key_released(key as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_get_mouse_x() -> f64 { engine().input.mouse_x }

#[no_mangle]
pub extern "C" fn bloom_get_mouse_y() -> f64 { engine().input.mouse_y }

#[no_mangle]
pub extern "C" fn bloom_is_mouse_button_pressed(btn: f64) -> f64 {
    if engine().input.is_mouse_button_pressed(btn as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_mouse_button_down(btn: f64) -> f64 {
    if engine().input.is_mouse_button_down(btn as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_is_mouse_button_released(btn: f64) -> f64 {
    if engine().input.is_mouse_button_released(btn as usize) { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_draw_line(x1: f64, y1: f64, x2: f64, y2: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_line(x1, y1, x2, y2, thickness, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_rect(x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_rect(x, y, w, h, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_rect_lines(x: f64, y: f64, w: f64, h: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_rect_lines(x, y, w, h, thickness, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_circle(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_circle(cx, cy, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_circle_lines(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_circle_lines(cx, cy, radius, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_triangle(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_triangle(x1, y1, x2, y2, x3, y3, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_poly(cx: f64, cy: f64, sides: f64, radius: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_poly(cx, cy, sides, radius, rotation, r, g, b, a);
}

#[no_mangle]
pub extern "C" fn bloom_draw_text(text_ptr: *const u8, x: f64, y: f64, size: f64, r: f64, g: f64, b: f64, a: f64) {
    let text = str_from_header(text_ptr);
    let eng = engine();
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::empty());
    text_renderer.draw_text(&mut eng.renderer, text, x, y, size as u32, r, g, b, a);
    eng.text = text_renderer;
}

#[no_mangle]
pub extern "C" fn bloom_measure_text(text_ptr: *const u8, size: f64) -> f64 {
    let text = str_from_header(text_ptr);
    engine().text.measure_text(text, size as u32)
}

#[no_mangle]
pub extern "C" fn bloom_load_font(path_ptr: *const u8, size: f64) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => engine().text.load_font(&data) as f64,
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_unload_font(font_handle: f64) {
    engine().text.unload_font(font_handle as usize);
}

#[no_mangle]
pub extern "C" fn bloom_draw_text_ex(font_handle: f64, text_ptr: *const u8, x: f64, y: f64, size: f64, spacing: f64, r: f64, g: f64, b: f64, a: f64) {
    let text = str_from_header(text_ptr);
    let eng = engine();
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::empty());
    text_renderer.draw_text_ex(&mut eng.renderer, font_handle as usize, text, x, y, size as u32, spacing as f32, r, g, b, a);
    eng.text = text_renderer;
}

#[no_mangle]
pub extern "C" fn bloom_measure_text_ex(font_handle: f64, text_ptr: *const u8, size: f64, spacing: f64) -> f64 {
    let text = str_from_header(text_ptr);
    engine().text.measure_text_ex(font_handle as usize, text, size as u32, spacing as f32)
}

static AUDIO_RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[no_mangle]
pub extern "C" fn bloom_init_audio() {
    #[cfg(windows)]
    {
        use std::sync::atomic::Ordering;
        AUDIO_RUNNING.store(true, Ordering::SeqCst);

        std::thread::spawn(|| {
            unsafe { wasapi_audio_thread(); }
        });
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_audio() {
    AUDIO_RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));
}

#[cfg(windows)]
unsafe fn wasapi_audio_thread() {
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
        ENGINE.get_mut().map(|eng| {
            eng.audio.mix_output(&mut mix_buf[..mix_samples]);
        });

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

#[no_mangle]
pub extern "C" fn bloom_load_sound(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => {
            if let Some(s) = parse_wav(&data) { engine().audio.load_sound(s) }
            else if let Some(s) = parse_ogg(&data) { engine().audio.load_sound(s) }
            else if let Some(s) = parse_mp3(&data) { engine().audio.load_sound(s) }
            else { 0.0 }
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_play_sound(handle: f64) { engine().audio.play_sound(handle); }

#[no_mangle]
pub extern "C" fn bloom_stop_sound(handle: f64) { engine().audio.stop_sound(handle); }

#[no_mangle]
pub extern "C" fn bloom_set_sound_volume(handle: f64, volume: f64) {
    engine().audio.set_sound_volume(handle, volume as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_master_volume(volume: f64) {
    engine().audio.master_volume = volume as f32;
}

#[no_mangle]
pub extern "C" fn bloom_play_sound_3d(handle: f64, x: f64, y: f64, z: f64) {
    engine().audio.play_sound_3d(handle, x as f32, y as f32, z as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_listener_position(x: f64, y: f64, z: f64, fx: f64, fy: f64, fz: f64) {
    engine().audio.set_listener_position(x as f32, y as f32, z as f32, fx as f32, fy as f32, fz as f32);
}

// --- Texture FFI ---

#[no_mangle]
pub extern "C" fn bloom_load_texture(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => {
            let eng = engine();
            let renderer_ptr = &mut eng.renderer as *mut bloom_shared::renderer::Renderer;
            eng.textures.load_texture(unsafe { &mut *renderer_ptr }, &data)
        }
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_unload_texture(handle: f64) {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut bloom_shared::renderer::Renderer;
    eng.textures.unload_texture(handle, unsafe { &mut *renderer_ptr });
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture(handle: f64, x: f64, y: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let idx = tex.bind_group_idx;
        eng.renderer.draw_texture(idx, x, y, r, g, b, a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_rec(handle: f64, src_x: f64, src_y: f64, src_w: f64, src_h: f64, dst_x: f64, dst_y: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let idx = tex.bind_group_idx;
        eng.renderer.draw_texture_rec(idx, src_x, src_y, src_w, src_h, dst_x, dst_y, r, g, b, a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_draw_texture_pro(handle: f64, src_x: f64, src_y: f64, src_w: f64, src_h: f64, dst_x: f64, dst_y: f64, dst_w: f64, dst_h: f64, origin_x: f64, origin_y: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let idx = tex.bind_group_idx;
        eng.renderer.draw_texture_pro(idx, src_x, src_y, src_w, src_h, dst_x, dst_y, dst_w, dst_h, origin_x, origin_y, rotation, r, g, b, a);
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_width(handle: f64) -> f64 {
    engine().textures.get(handle).map(|t| t.width as f64).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn bloom_get_texture_height(handle: f64) -> f64 {
    engine().textures.get(handle).map(|t| t.height as f64).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn bloom_gen_texture_mipmaps(_handle: f64) {
    // No-op: wgpu handles mipmaps internally
}

#[no_mangle]
pub extern "C" fn bloom_set_texture_filter(handle: f64, mode: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let bind_group_idx = tex.bind_group_idx;
        eng.renderer.set_texture_filter(bind_group_idx, mode > 0.5);
    }
}

#[no_mangle]
pub extern "C" fn bloom_load_image(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) { Ok(data) => engine().textures.load_image(&data), Err(_) => 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_image_resize(handle: f64, w: f64, h: f64) {
    engine().textures.image_resize(handle, w as u32, h as u32);
}

#[no_mangle]
pub extern "C" fn bloom_image_crop(handle: f64, x: f64, y: f64, w: f64, h: f64) {
    engine().textures.image_crop(handle, x as u32, y as u32, w as u32, h as u32);
}

#[no_mangle]
pub extern "C" fn bloom_image_flip_h(handle: f64) {
    engine().textures.image_flip_h(handle);
}

#[no_mangle]
pub extern "C" fn bloom_image_flip_v(handle: f64) {
    engine().textures.image_flip_v(handle);
}

#[no_mangle]
pub extern "C" fn bloom_load_texture_from_image(handle: f64) -> f64 {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut bloom_shared::renderer::Renderer;
    eng.textures.load_texture_from_image(handle, unsafe { &mut *renderer_ptr })
}

// --- Camera FFI ---

#[no_mangle]
pub extern "C" fn bloom_begin_mode_2d(offset_x: f64, offset_y: f64, target_x: f64, target_y: f64, rotation: f64, zoom: f64) {
    engine().renderer.begin_mode_2d(offset_x as f32, offset_y as f32, target_x as f32, target_y as f32, rotation as f32, zoom as f32);
}
#[no_mangle]
pub extern "C" fn bloom_end_mode_2d() { engine().renderer.end_mode_2d(); }

#[no_mangle]
pub extern "C" fn bloom_begin_mode_3d(pos_x: f64, pos_y: f64, pos_z: f64, target_x: f64, target_y: f64, target_z: f64, up_x: f64, up_y: f64, up_z: f64, fovy: f64, projection: f64) {
    engine().renderer.begin_mode_3d(pos_x as f32, pos_y as f32, pos_z as f32, target_x as f32, target_y as f32, target_z as f32, up_x as f32, up_y as f32, up_z as f32, fovy as f32, projection as f32);
}
#[no_mangle]
pub extern "C" fn bloom_end_mode_3d() { engine().renderer.end_mode_3d(); }

// --- 3D Drawing FFI ---

#[no_mangle]
pub extern "C" fn bloom_draw_cube(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube(x, y, z, w, h, d, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_cube_wires(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube_wires(x, y, z, w, h, d, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_sphere(x: f64, y: f64, z: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere(x, y, z, radius, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_sphere_wires(x: f64, y: f64, z: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere_wires(x, y, z, radius, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_cylinder(x: f64, y: f64, z: f64, rt: f64, rb: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cylinder(x, y, z, rt, rb, h, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_plane(x: f64, y: f64, z: f64, w: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_plane(x, y, z, w, d, r, g, b, a);
}
#[no_mangle]
pub extern "C" fn bloom_draw_grid(slices: f64, spacing: f64) {
    engine().renderer.draw_grid(slices as i32, spacing);
}
#[no_mangle]
pub extern "C" fn bloom_draw_ray(ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_ray(ox, oy, oz, dx, dy, dz, r, g, b, a);
}

// --- Model FFI ---

#[no_mangle]
pub extern "C" fn bloom_load_model(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => {
            let eng = engine();
            let renderer_ptr = &mut eng.renderer as *mut Renderer;
            eng.models.load_model_with_textures(&data, unsafe { &mut *renderer_ptr })
        }
        Err(_) => 0.0,
    }
}
#[no_mangle]
pub extern "C" fn bloom_draw_model(handle: f64, x: f64, y: f64, z: f64, scale: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(model) = eng.models.get(handle) {
        let tint = [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32];
        let position = [x as f32, y as f32, z as f32];
        let handle_bits = handle.to_bits();
        if eng.renderer.cache_model_if_static(handle_bits, &model.meshes) {
            eng.renderer.draw_model_cached(handle_bits, position, scale as f32, tint);
        } else {
            for mesh in &model.meshes {
                let tex_idx = mesh.texture_idx.unwrap_or(0);
                eng.renderer.draw_model_mesh_tinted(&mesh.vertices, &mesh.indices, position, scale as f32, tint, tex_idx);
            }
        }
    }
}
#[no_mangle]
pub extern "C" fn bloom_draw_model_rotated(
    handle: f64, x: f64, y: f64, z: f64,
    scale: f64, rot_y: f64,
    color_packed_argb: f64,
) {
    let bits = color_packed_argb as u32;
    let a = ((bits >> 24) & 0xff) as f32 / 255.0;
    let r = ((bits >> 16) & 0xff) as f32 / 255.0;
    let g = ((bits >>  8) & 0xff) as f32 / 255.0;
    let b = ( bits        & 0xff) as f32 / 255.0;
    let eng = engine();
    if let Some(model) = eng.models.get(handle) {
        let position = [x as f32, y as f32, z as f32];
        let scale = scale as f32;
        let tint = [r, g, b, a];
        for mesh in &model.meshes {
            let tex_idx = mesh.texture_idx.unwrap_or(0);
            eng.renderer.draw_model_mesh_tinted_rotated(
                &mesh.vertices, &mesh.indices, position, scale, tint, tex_idx, rot_y as f32,
            );
        }
    }
}
#[no_mangle]
pub extern "C" fn bloom_unload_model(handle: f64) { engine().models.unload_model(handle); }

#[no_mangle]
pub extern "C" fn bloom_get_model_mesh_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_model_material_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_gen_mesh_cube(w: f64, h: f64, d: f64) -> f64 {
    engine().models.gen_mesh_cube(w as f32, h as f32, d as f32)
}

#[no_mangle]
pub extern "C" fn bloom_gen_mesh_heightmap(image_handle: f64, size_x: f64, size_y: f64, size_z: f64) -> f64 {
    let eng = engine();
    if let Some(img) = eng.textures.images.get(image_handle) {
        let data = img.data.clone();
        let w = img.width;
        let h = img.height;
        eng.models.gen_mesh_heightmap(&data, w, h, size_x as f32, size_y as f32, size_z as f32)
    } else {
        0.0
    }
}

#[no_mangle]
pub extern "C" fn bloom_load_shader(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    engine().renderer.load_custom_shader(source) as f64
}

#[no_mangle]
pub extern "C" fn bloom_create_mesh(vertex_ptr: *const f32, vertex_count: f64, index_ptr: *const u32, index_count: f64) -> f64 {
    if vertex_ptr.is_null() || index_ptr.is_null() { return 0.0; }
    let vcount = vertex_count as usize;
    let icount = index_count as usize;
    let vertex_data = unsafe { std::slice::from_raw_parts(vertex_ptr, vcount * 12) }; // 12 floats per vertex
    let index_data = unsafe { std::slice::from_raw_parts(index_ptr, icount) };
    engine().models.create_mesh(vertex_data, index_data)
}

// ============================================================
// Phase 1c — material system FFI
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_material_params(
    handle: f64,
    params_ptr: *const f64,
    param_count: f64,
) {
    let count = param_count as usize;
    if count > 64 {
        eprintln!("[material] set_material_params: param_count {} > 64 (256-byte UBO cap)", count);
        return;
    }
    let mut bytes = vec![0u8; count * 4];
    if !params_ptr.is_null() && count > 0 {
        let slots = unsafe { std::slice::from_raw_parts(params_ptr, count) };
        for (i, &v) in slots.iter().enumerate() {
            let f = v as f32;
            bytes[i*4..i*4+4].copy_from_slice(&f.to_le_bytes());
        }
    }
    let eng = engine();
    if let Err(e) = eng.renderer.material_system.set_user_params(
        &eng.renderer.device, &eng.renderer.queue,
        handle as u32, &bytes,
    ) {
        eprintln!("[material] set_material_params failed: {}", e);
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material(source) {
        Ok(handle) => handle as f64,
        Err(e) => {
            eprintln!("[material] compile failed: {:?}", e);
            0.0
        }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_refractive(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Refractive, true, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[refractive] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_transparent(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Transparent, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_additive(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Additive, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_cutout(source_ptr: *const u8) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Opaque, Bucket::Cutout, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_instanced(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.compile_material_instanced(source) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] instanced compile failed: {:?}", e); 0.0 }
    }
}

#[no_mangle]
pub extern "C" fn bloom_create_instance_buffer(
    data_ptr: *const f64, instance_count: f64,
) -> f64 {
    if data_ptr.is_null() || instance_count <= 0.0 { return 0.0; }
    let count = instance_count as u32;
    let slot_count = (count as usize) * 9;
    let raw_f64 = unsafe { std::slice::from_raw_parts(data_ptr, slot_count) };
    let raw_f32: Vec<f32> = raw_f64.iter().map(|&v| v as f32).collect();
    engine().renderer.create_instance_buffer(&raw_f32, count) as f64
}

#[no_mangle]
pub extern "C" fn bloom_submit_material_draw_instanced(
    material: f64, mesh_handle: f64, mesh_idx: f64,
    instance_buffer: f64, instance_count: f64,
) {
    let eng = engine();
    let handle_bits = mesh_handle.to_bits();
    if let Some(model) = eng.models.get(mesh_handle) {
        eng.renderer.cache_model_if_static(handle_bits, &model.meshes);
    }
    eng.renderer.submit_material_draw_instanced(
        material as u32,
        handle_bits,
        mesh_idx as usize,
        instance_buffer as u32,
        instance_count as u32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_destroy_instance_buffer(handle: f64) {
    engine().renderer.destroy_instance_buffer(handle as u32);
}

/// EN-011 — create a planar reflection probe. See macOS lib.rs for the
/// full doc comment; this entry-point exists on every native platform
/// so games can target the same FFI surface across iOS/tvOS/Windows/
/// Linux/Android.
#[no_mangle]
pub extern "C" fn bloom_create_planar_reflection(
    plane_y: f64, nx: f64, ny: f64, nz: f64, resolution: f64,
) -> f64 {
    engine().renderer.create_planar_reflection(
        plane_y as f32,
        [nx as f32, ny as f32, nz as f32],
        resolution as u32,
    ) as f64
}

/// EN-011 — link a material to a planar reflection probe. `probe = 0`
/// reverts the binding to the engine's default 1×1 black texture.
#[no_mangle]
pub extern "C" fn bloom_set_material_reflection_probe(
    material: f64, probe: f64,
) {
    engine().renderer.set_material_reflection_probe(material as u32, probe as u32);
}

/// EN-014 — create a texture array from concatenated RGBA8 byte data.
/// See macOS lib.rs for the full doc comment; this entry-point exists
/// on every native platform so a TS game targets the same FFI across
/// iOS / tvOS / Windows / Linux / Android.
#[no_mangle]
pub extern "C" fn bloom_create_texture_array(
    data_ptr:    *const u8,
    data_len:    f64,
    width:       f64,
    height:      f64,
    layer_count: f64,
) -> f64 {
    // EN-014 V2 — V1 forwards to _ex with default sRGB / no mips.
    bloom_create_texture_array_ex(data_ptr, data_len, width, height, layer_count, 0.0, 1.0)
}

/// EN-014 V2 — explicit format + mip control. See macOS lib.rs for docs.
#[no_mangle]
pub extern "C" fn bloom_create_texture_array_ex(
    data_ptr:    *const u8,
    data_len:    f64,
    width:       f64,
    height:      f64,
    layer_count: f64,
    format:      f64,
    mip_levels:  f64,
) -> f64 {
    if data_ptr.is_null() || data_len <= 0.0 { return 0.0; }
    let w = width as u32;
    let h = height as u32;
    if w == 0 || h == 0 { return 0.0; }
    let layers_count = (layer_count as u32)
        .min(bloom_shared::renderer::material_system::MAX_TEXTURE_ARRAY_LAYERS);
    if layers_count == 0 { return 0.0; }
    let layer_size = (w as usize) * (h as usize) * 4;
    let total_bytes = (data_len as usize)
        .min(layers_count as usize * layer_size);
    let bytes = unsafe { std::slice::from_raw_parts(data_ptr, total_bytes) };
    let mut layers: Vec<(&[u8], u32, u32)> = Vec::with_capacity(layers_count as usize);
    for i in 0..(layers_count as usize) {
        let start = i * layer_size;
        let end   = start + layer_size;
        if end > bytes.len() { break; }
        layers.push((&bytes[start..end], w, h));
    }
    engine().renderer.create_texture_array_ex(&layers, format as u32, mip_levels as u32) as f64
}

/// EN-014 — link a texture-array handle to a material at one of three
/// slots: 0 = albedo (binding 14), 1 = normal (binding 15),
/// 2 = MR (binding 16). Pass `array = 0` to revert to the stub.
#[no_mangle]
pub extern "C" fn bloom_set_material_texture_array(
    material: f64, slot: f64, array: f64,
) {
    engine().renderer.set_material_texture_array(
        material as u32, slot as u32, array as u32,
    );
}

/// EN-012 — set the shading model for a material (0=default lit,
/// 1=foliage, 2=subsurface V2 stub).
#[no_mangle]
pub extern "C" fn bloom_set_material_shading_model(
    material: f64, model: f64,
) {
    engine().renderer.set_material_shading_model(material as u32, model as u32);
}

/// EN-012 — set the foliage shading parameters for a material.
/// Only takes effect when shading_model == 1 (foliage).
#[no_mangle]
pub extern "C" fn bloom_set_material_foliage(
    material: f64,
    trans_r: f64, trans_g: f64, trans_b: f64,
    trans_amount: f64, wrap_factor: f64,
) {
    engine().renderer.set_material_foliage(
        material as u32,
        [trans_r as f32, trans_g as f32, trans_b as f32],
        trans_amount as f32, wrap_factor as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_compile_material_from_file(
    path_ptr: *const u8,
    bucket_kind: f64,
) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let path = str_from_header(path_ptr);
    let (profile, bucket, reads_scene) = match bucket_kind as u32 {
        0 => (FragmentProfile::Opaque,      Bucket::Opaque,      false),
        1 => (FragmentProfile::Translucent, Bucket::Transparent, false),
        2 => (FragmentProfile::Translucent, Bucket::Refractive,  true),
        3 => (FragmentProfile::Translucent, Bucket::Additive,    false),
        4 => (FragmentProfile::Opaque,      Bucket::Cutout,      false),
        _ => {
            eprintln!("[material] from_file: unknown bucket_kind {bucket_kind}");
            return 0.0;
        }
    };
    match engine().renderer.compile_material_from_file(
        std::path::Path::new(path), profile, bucket, reads_scene,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => { eprintln!("[material] from_file failed: {e}"); 0.0 }
    }
}

/// EN-017 — compile + install a fullscreen post-pass material.
/// See `bloom-macos::bloom_set_post_pass` for the full ABI.
#[no_mangle]
pub extern "C" fn bloom_set_post_pass(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.set_post_pass(source) {
        Ok(()) => 1.0,
        Err(e) => { eprintln!("[post_pass] compile failed: {:?}", e); 0.0 }
    }
}

/// EN-017 — uninstall the active post-pass.
#[no_mangle]
pub extern "C" fn bloom_clear_post_pass() {
    engine().renderer.clear_post_pass();
}

/// EN-017 V2 — append a post-pass to the stack.
/// See `bloom-macos::bloom_add_post_pass` for the full ABI.
#[no_mangle]
pub extern "C" fn bloom_add_post_pass(source_ptr: *const u8) -> f64 {
    let source = str_from_header(source_ptr);
    match engine().renderer.add_post_pass(source) {
        Ok(h) => h as f64,
        Err(e) => { eprintln!("[post_pass] compile failed: {:?}", e); 0.0 }
    }
}

/// EN-017 V2 — wipe the entire post-pass stack.
#[no_mangle]
pub extern "C" fn bloom_clear_all_post_passes() {
    engine().renderer.clear_all_post_passes();
}

#[no_mangle]
pub extern "C" fn bloom_draw_material(
    material: f64,
    mesh_handle: f64,
    mesh_idx: f64,
    x: f64, y: f64, z: f64, scale: f64,
    r: f64, g: f64, b: f64, a: f64,
) {
    let eng = engine();
    let handle_bits = mesh_handle.to_bits();
    if let Some(model) = eng.models.get(mesh_handle) {
        eng.renderer.cache_model_if_static(handle_bits, &model.meshes);
    }
    eng.renderer.submit_material_draw(
        material as u32,
        handle_bits,
        mesh_idx as usize,
        [x as f32, y as f32, z as f32],
        scale as f32,
        [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32],
    );
}

#[no_mangle]
pub extern "C" fn bloom_load_model_animation(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => engine().models.load_model_animation(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_update_model_animation(handle: f64, anim_index: f64, time: f64, scale: f64, px: f64, py: f64, pz: f64, rot_sin: f64, rot_cos: f64) {
    let eng = engine();
    eng.models.update_model_animation(handle, anim_index as usize, time as f32);
    if let Some(anim) = eng.models.get_animation(handle) {
        if !anim.joint_matrices.is_empty() {
            eng.renderer.set_joint_matrices_scaled(&anim.joint_matrices, scale as f32, [px as f32, py as f32, pz as f32], rot_sin as f32, rot_cos as f32);
        }
    }
}

// --- Music FFI ---

#[no_mangle]
pub extern "C" fn bloom_load_music(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => {
            if let Some(s) = parse_ogg(&data) { engine().audio.load_music(s) }
            else if let Some(s) = parse_wav(&data) { engine().audio.load_music(s) }
            else if let Some(s) = parse_mp3(&data) { engine().audio.load_music(s) }
            else { 0.0 }
        }
        Err(_) => 0.0,
    }
}
#[no_mangle]
pub extern "C" fn bloom_play_music(handle: f64) { engine().audio.play_music(handle); }
#[no_mangle]
pub extern "C" fn bloom_stop_music(handle: f64) { engine().audio.stop_music(handle); }
#[no_mangle]
pub extern "C" fn bloom_update_music_stream(handle: f64) { engine().audio.update_music_stream(handle); }
#[no_mangle]
pub extern "C" fn bloom_set_music_volume(handle: f64, volume: f64) { engine().audio.set_music_volume(handle, volume as f32); }
#[no_mangle]
pub extern "C" fn bloom_is_music_playing(handle: f64) -> f64 { if engine().audio.is_music_playing(handle) { 1.0 } else { 0.0 } }

// --- Gamepad FFI ---

#[no_mangle]
pub extern "C" fn bloom_is_gamepad_available() -> f64 { if engine().input.is_gamepad_available() { 1.0 } else { 0.0 } }
#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis(axis: f64) -> f64 { engine().input.get_gamepad_axis(axis as usize) as f64 }
#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_pressed(btn: f64) -> f64 { if engine().input.is_gamepad_button_pressed(btn as usize) { 1.0 } else { 0.0 } }
#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_down(btn: f64) -> f64 { if engine().input.is_gamepad_button_down(btn as usize) { 1.0 } else { 0.0 } }
#[no_mangle]
pub extern "C" fn bloom_is_gamepad_button_released(btn: f64) -> f64 { if engine().input.is_gamepad_button_released(btn as usize) { 1.0 } else { 0.0 } }
#[no_mangle]
pub extern "C" fn bloom_get_gamepad_axis_count() -> f64 { engine().input.get_gamepad_axis_count() as f64 }

// --- Skeletal Animation Debug ---

#[no_mangle]
pub extern "C" fn bloom_set_joint_test(_joint: f64, _angle: f64) {
    // No-op for now — skeletal animation testing
}

// --- Lighting ---

#[no_mangle]
pub extern "C" fn bloom_set_ambient_light(r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_ambient_light(r, g, b, intensity);
}

#[no_mangle]
pub extern "C" fn bloom_set_directional_light(dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_directional_light(dx, dy, dz, r, g, b, intensity);
}

#[no_mangle]
pub extern "C" fn bloom_set_procedural_sky(enabled: f64, rayleigh_density: f64, mie_density: f64, ground_albedo: f64) {
    engine().renderer.set_procedural_sky(
        enabled != 0.0,
        rayleigh_density as f32,
        mie_density as f32,
        ground_albedo as f32,
    );
}

#[no_mangle]
pub extern "C" fn bloom_set_sun_direction(dx: f64, dy: f64, dz: f64, intensity: f64) {
    engine().renderer.set_sun_direction(dx as f32, dy as f32, dz as f32, intensity as f32);
}

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
}

#[no_mangle]
pub extern "C" fn bloom_enable_cursor() {
    engine().input.cursor_disabled = false;
}

#[no_mangle]
pub extern "C" fn bloom_get_mouse_delta_x() -> f64 {
    engine().input.mouse_delta_x
}

#[no_mangle]
pub extern "C" fn bloom_get_mouse_delta_y() -> f64 {
    engine().input.mouse_delta_y
}

// Accumulated scroll wheel delta since the last call. Reading consumes the
// value (returns 0 on the next call until the user scrolls again). Used by
// the editor's orbit camera and any scrollable UI panel.
#[no_mangle]
pub extern "C" fn bloom_get_mouse_wheel() -> f64 {
    engine().input.consume_mouse_wheel()
}

#[no_mangle]
pub extern "C" fn bloom_get_char_pressed() -> f64 {
    engine().input.pop_char() as f64
}

// Q2: Cursor shape
#[no_mangle]
pub extern "C" fn bloom_set_cursor_shape(shape: f64) {
    engine().input.cursor_shape = shape as u32;
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

// Model bounds accessors. Return the axis-aligned bounding box of a loaded
// model in model-local coordinates. Editors use these to size gizmos, auto-
// frame the camera on selection, and snap placed entities onto terrain.
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_min_x(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[0] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_min_y(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[1] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_min_z(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).0[2] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_max_x(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[0] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_max_y(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[1] as f64
}
#[no_mangle]
pub extern "C" fn bloom_get_model_bounds_max_z(model_handle: f64) -> f64 {
    engine().models.get_bounds(model_handle).1[2] as f64
}

#[no_mangle]
pub extern "C" fn bloom_write_file(path_ptr: *const u8, data_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = str_from_header(data_ptr);
    match std::fs::write(path, data.as_bytes()) {
        Ok(_) => 1.0,
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_file_exists(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    if std::path::Path::new(path).exists() { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_read_file(path_ptr: *const u8) -> *const u8 {
    let path = str_from_header(path_ptr);
    // Always return a valid Perry string. A null pointer would NaN-box into a
    // string-typed JS value pointing at address 0; subsequent `.length` /
    // `.charCodeAt` reads dereference the bogus StringHeader and segfault.
    // Callers detect "missing file" via `data.length === 0` (e.g. the
    // jump game's discoverLevels probe across level1..level10 / custom_*).
    match std::fs::read_to_string(path) {
        Ok(contents) => alloc_perry_string(&contents),
        Err(_)       => alloc_perry_string(""),
    }
}

#[no_mangle]
pub extern "C" fn bloom_get_touch_x(index: f64) -> f64 { engine().input.get_touch_x(index as usize) }
#[no_mangle]
pub extern "C" fn bloom_get_touch_y(index: f64) -> f64 { engine().input.get_touch_y(index as usize) }
#[no_mangle]
pub extern "C" fn bloom_get_touch_count() -> f64 { engine().input.get_touch_count() as f64 }
#[no_mangle]
pub extern "C" fn bloom_get_time() -> f64 { engine().get_time() }

// Input injection + platform detection
#[no_mangle]
pub extern "C" fn bloom_inject_key_down(key: f64) {
    engine().input.set_key_down(key as usize);
}
#[no_mangle]
pub extern "C" fn bloom_inject_key_up(key: f64) {
    engine().input.set_key_up(key as usize);
}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_axis(axis: f64, value: f64) {
    engine().input.set_gamepad_axis(axis as usize, value as f32);
}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_button_down(button: f64) {
    engine().input.set_gamepad_button_down(button as usize);
}
#[no_mangle]
pub extern "C" fn bloom_inject_gamepad_button_up(button: f64) {
    engine().input.set_gamepad_button_up(button as usize);
}
#[no_mangle]
pub extern "C" fn bloom_get_platform() -> f64 { 3.0 }
#[no_mangle]
pub extern "C" fn bloom_is_any_input_pressed() -> f64 {
    if engine().input.is_any_input_pressed() { 1.0 } else { 0.0 }
}
#[no_mangle]
pub extern "C" fn bloom_get_crown_rotation() -> f64 {
    engine().input.consume_crown_rotation()
}

// ============================================================
// Thread-safe staging (for async asset loading via Perry threads)
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_stage_texture(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => bloom_shared::staging::decode_and_stage_texture(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_stage_model(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return 0.0,
    };
    match bloom_shared::models::load_gltf_staged(&data) {
        Some(staged) => bloom_shared::staging::stage_model(staged),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_stage_sound(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return 0.0,
    };
    let sound_data = if path.ends_with(".ogg") || path.ends_with(".OGG") {
        parse_ogg(&data)
    } else if path.ends_with(".mp3") || path.ends_with(".MP3") {
        parse_mp3(&data)
    } else {
        parse_wav(&data)
    };
    match sound_data {
        Some(sd) => bloom_shared::staging::stage_sound(sd),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_commit_texture(staging_handle: f64) -> f64 {
    let staged = match bloom_shared::staging::take_texture(staging_handle) {
        Some(s) => s,
        None => return 0.0,
    };
    let eng = engine();
    let bind_group_idx = eng.renderer.register_texture(staged.width, staged.height, &staged.data);
    eng.textures.textures.alloc(bloom_shared::textures::TextureData {
        bind_group_idx, width: staged.width, height: staged.height,
    })
}

#[no_mangle]
pub extern "C" fn bloom_commit_model(staging_handle: f64) -> f64 {
    let staged = match bloom_shared::staging::take_model(staging_handle) {
        Some(s) => s,
        None => return 0.0,
    };
    let eng = engine();
    let mut tex_map: Vec<u32> = Vec::with_capacity(staged.textures.len());
    for tex in &staged.textures {
        tex_map.push(eng.renderer.register_texture(tex.width, tex.height, &tex.data));
    }
    let mut model = staged.model;
    for mesh in &mut model.meshes {
        if let Some(ref mut idx) = mesh.texture_idx {
            let staged_idx = *idx as usize;
            if staged_idx > 0 && staged_idx <= tex_map.len() {
                *idx = tex_map[staged_idx - 1];
            } else {
                mesh.texture_idx = None;
            }
        }
    }
    eng.models.models.alloc(model)
}

#[no_mangle]
pub extern "C" fn bloom_commit_sound(staging_handle: f64) -> f64 {
    match bloom_shared::staging::take_sound(staging_handle) {
        Some(sd) => engine().audio.load_sound(sd),
        None => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_commit_music(staging_handle: f64) -> f64 {
    match bloom_shared::staging::take_sound(staging_handle) {
        Some(sd) => engine().audio.load_music(sd),
        None => 0.0,
    }
}

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

// 3D engine stubs — not yet implemented on Windows
#[no_mangle] pub extern "C" fn bloom_register_frame_callback(_priority: f64, _callback: extern "C" fn(f64)) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_unregister_frame_callback(_id: f64) {}

#[no_mangle]
pub extern "C" fn bloom_run_game(_callback: extern "C" fn(f64)) {
    // No-op on native. The TypeScript runGame() helper provides the while loop.
}

#[no_mangle] pub extern "C" fn bloom_add_directional_light(_dx: f64, _dy: f64, _dz: f64, _r: f64, _g: f64, _b: f64, _intensity: f64) {}
#[no_mangle] pub extern "C" fn bloom_add_point_light(_x: f64, _y: f64, _z: f64, _range: f64, _r: f64, _g: f64, _b: f64, _intensity: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_create_node() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_scene_destroy_node(_handle: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_set_visible(_handle: f64, _visible: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_set_cast_shadow(_handle: f64, _cast: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_set_receive_shadow(_handle: f64, _receive: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_set_parent(_handle: f64, _parent: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_set_transform(_handle: f64, _mat_ptr: *const f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_update_geometry(_handle: f64, _vert_ptr: *const f64, _vert_count: f64, _idx_ptr: *const f64, _idx_count: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_set_material_color(_handle: f64, _r: f64, _g: f64, _b: f64, _a: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_set_material_pbr(_handle: f64, _roughness: f64, _metalness: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_set_material_texture(_handle: f64, _texture_idx: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_node_count() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_scene_node_vertex_count(_handle: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_scene_node_index_count(_handle: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_scene_extrude_polygon(_handle: f64, _polygon_ptr: *const f64, _polygon_count: f64, _depth: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_subtract_box(_handle: f64, _min_x: f64, _min_y: f64, _min_z: f64, _max_x: f64, _max_y: f64, _max_z: f64) {}
#[no_mangle] pub extern "C" fn bloom_enable_shadows() {}
#[no_mangle] pub extern "C" fn bloom_disable_shadows() {}
#[no_mangle] pub extern "C" fn bloom_enable_postfx() {}
#[no_mangle] pub extern "C" fn bloom_disable_postfx() {}
#[no_mangle] pub extern "C" fn bloom_postfx_set_selected(_handle: f64) {}
#[no_mangle] pub extern "C" fn bloom_postfx_set_hovered(_handle: f64) {}
#[no_mangle] pub extern "C" fn bloom_postfx_set_outline_color(_r: f64, _g: f64, _b: f64, _a: f64) {}
#[no_mangle] pub extern "C" fn bloom_postfx_set_outline_thickness(_thickness: f64) {}
#[no_mangle] pub extern "C" fn bloom_project_to_screen(_wx: f64, _wy: f64, _wz: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_project_screen_y() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_scene_attach_model(_node_handle: f64, _model_handle: f64, _mesh_index: f64) {}
#[no_mangle] pub extern "C" fn bloom_scene_pick(_screen_x: f64, _screen_y: f64) -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_pick_hit_handle() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_pick_hit_distance() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_pick_hit_x() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_pick_hit_y() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_pick_hit_z() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_pick_hit_normal_x() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_pick_hit_normal_y() -> f64 { 0.0 }
#[no_mangle] pub extern "C" fn bloom_pick_hit_normal_z() -> f64 { 0.0 }


// Q6: Multi-hit picking
static mut LAST_PICK_ALL: Vec<bloom_shared::picking::PickResult> = Vec::new();

#[no_mangle]
pub extern "C" fn bloom_scene_pick_all(screen_x: f64, screen_y: f64, max_results: f64) -> f64 {
    let eng = engine();
    let inv_vp = eng.renderer.inverse_vp_matrix();
    let cam_pos = eng.renderer.camera_pos();
    let w = eng.renderer.width() as f32;
    let h = eng.renderer.height() as f32;
    let (origin, direction) = bloom_shared::picking::screen_to_ray(
        screen_x as f32, screen_y as f32, w, h, &inv_vp, &cam_pos,
    );
    let results = bloom_shared::picking::raycast_scene_all(&eng.scene, &origin, &direction, max_results as usize);
    let count = results.len();
    unsafe { LAST_PICK_ALL = results; }
    count as f64
}
#[no_mangle]
pub extern "C" fn bloom_pick_all_handle(index: f64) -> f64 {
    let i = index as usize;
    unsafe { LAST_PICK_ALL.get(i).map(|r| r.handle).unwrap_or(0.0) }
}
#[no_mangle]
pub extern "C" fn bloom_pick_all_distance(index: f64) -> f64 {
    let i = index as usize;
    unsafe { LAST_PICK_ALL.get(i).map(|r| r.distance as f64).unwrap_or(0.0) }
}
// ============================================================

// ============================================================
// Render quality toggles (individual + preset) — ticket 011
// Mirror of the macOS FFI surface added in commit 95da6af; previously
// macOS-only, now exposed on every native platform so non-macOS builds
// don't fail at runtime (missing symbol) when the TS API invokes them.
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_quality_preset(preset: f64) {
    engine().renderer.apply_quality_preset(preset as u32);
}
#[no_mangle]
pub extern "C" fn bloom_set_shadows_enabled(on: f64) {
    engine().renderer.set_shadows_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_shadows_always_fresh(on: f64) {
    engine().renderer.set_shadows_always_fresh(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_bloom_enabled(on: f64) {
    engine().renderer.set_bloom_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssao_enabled(on: f64) {
    engine().renderer.set_ssao_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssao_intensity(value: f64) {
    engine().renderer.set_ssao_strength(value as f32);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssao_radius(world_radius: f64) {
    engine().renderer.set_ssao_radius(world_radius as f32);
}
#[no_mangle]
pub extern "C" fn bloom_set_wind(dir_x: f64, dir_z: f64, amplitude: f64, frequency: f64) {
    engine().renderer.set_wind(dir_x as f32, dir_z as f32, amplitude as f32, frequency as f32);
}
#[no_mangle]
pub extern "C" fn bloom_set_ssr_enabled(on: f64) {
    engine().renderer.set_ssr_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_motion_blur_enabled(on: f64) {
    engine().renderer.set_motion_blur_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_set_sss_enabled(on: f64) {
    engine().renderer.set_sss_enabled(on != 0.0);
}

// ============================================================
// Render scale / upscale / DRS / post-FX / screenshots / impulse
// Ports of the macOS / Linux FFI surface so the bloom/core TS layer
// links cleanly on Windows. EngineState in bloom-shared already
// exposes the underlying renderer methods, so these wrappers are
// platform-agnostic.
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_take_screenshot(path_ptr: *const u8) {
    let path = str_from_header(path_ptr).to_string();
    let eng = engine();
    eng.renderer.screenshot_requested = true;
    eng.renderer.pending_screenshot_path = Some(path);
}

#[no_mangle]
pub extern "C" fn bloom_set_env_clear_from_hdr(path_ptr: *const u8) {
    use image::ImageDecoder;
    let path = str_from_header(path_ptr).to_string();
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let decoder = match image::codecs::hdr::HdrDecoder::new(std::io::BufReader::new(file)) {
        Ok(d) => d,
        Err(_) => return,
    };
    let (w, h) = decoder.dimensions();
    let byte_len = (w as usize) * (h as usize) * 3 * 4;
    let mut buf = vec![0u8; byte_len];
    if decoder.read_image(&mut buf).is_err() {
        return;
    }
    let rgb_f32: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    engine().renderer.load_env_from_hdr(w, h, &rgb_f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_fog(r: f64, g: f64, b: f64, density: f64, height_ref: f64, height_falloff: f64) {
    let r_ = engine();
    r_.renderer.set_fog_color(r as f32, g as f32, b as f32);
    r_.renderer.set_fog_density(density as f32);
    r_.renderer.set_fog_height_falloff(height_ref as f32, height_falloff as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_chromatic_aberration(strength: f64) {
    engine().renderer.set_chromatic_aberration(strength as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_vignette(strength: f64, softness: f64) {
    engine().renderer.set_vignette(strength as f32, softness as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_film_grain(strength: f64) {
    engine().renderer.set_film_grain(strength as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_sun_shafts(strength: f64, decay: f64, r: f64, g: f64, b: f64) {
    let eng = engine();
    eng.renderer.set_sun_shaft_strength(strength as f32);
    eng.renderer.set_sun_shaft_decay(decay as f32);
    eng.renderer.set_sun_shaft_color(r as f32, g as f32, b as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_auto_exposure(on: f64) {
    engine().renderer.set_auto_exposure(on != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_set_taa_enabled(on: f64) {
    engine().renderer.set_taa_enabled(on != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_set_render_scale(scale: f64) {
    engine().renderer.set_render_scale(scale as f32);
}

#[no_mangle]
pub extern "C" fn bloom_get_render_scale() -> f64 {
    engine().renderer.render_scale() as f64
}

#[no_mangle]
pub extern "C" fn bloom_set_upscale_mode(mode: f64) {
    engine().renderer.set_upscale_mode(mode as u32);
}

#[no_mangle]
pub extern "C" fn bloom_set_cas_strength(strength: f64) {
    engine().renderer.set_cas_strength(strength as f32);
}

#[no_mangle]
pub extern "C" fn bloom_get_physical_width() -> f64 {
    engine().renderer.physical_width() as f64
}

#[no_mangle]
pub extern "C" fn bloom_get_physical_height() -> f64 {
    engine().renderer.physical_height() as f64
}

#[no_mangle]
pub extern "C" fn bloom_set_auto_resolution(target_hz: f64, enabled: f64) {
    let eng = engine();
    if enabled != 0.0 {
        let current = eng.renderer.render_scale();
        eng.drs.enable(target_hz as f32, current);
    } else {
        eng.drs.disable();
    }
}

#[no_mangle]
pub extern "C" fn bloom_set_manual_exposure(value: f64) {
    engine().renderer.set_manual_exposure(value as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_env_intensity(intensity: f64) {
    engine().renderer.set_env_intensity(intensity as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_ssgi_enabled(enabled: f64) {
    engine().renderer.set_ssgi_enabled(enabled != 0.0);
}

#[no_mangle]
pub extern "C" fn bloom_set_ssgi_intensity(intensity: f64) {
    engine().renderer.set_ssgi_intensity(intensity as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_ssgi_radius(radius: f64) {
    engine().renderer.set_ssgi_radius(radius as f32);
}

#[no_mangle]
pub extern "C" fn bloom_set_dof(enabled: f64, focus_distance: f64, aperture: f64) {
    let r = &mut engine().renderer;
    r.set_dof_enabled(enabled != 0.0);
    r.set_dof_focus_distance(focus_distance as f32);
    r.set_dof_aperture(aperture as f32);
}

#[no_mangle]
pub extern "C" fn bloom_splat_impulse(x: f64, z: f64, radius: f64, strength: f64) {
    engine().renderer.impulse_field.submit_splat(
        x as f32, z as f32, radius as f32, strength as f32,
    );
}

// Render texture FFI (stub — GPU implementation deferred).
#[no_mangle]
pub extern "C" fn bloom_load_render_texture(width: f64, height: f64) -> f64 {
    engine().textures.load_render_texture(width as u32, height as u32)
}
#[no_mangle]
pub extern "C" fn bloom_unload_render_texture(handle: f64) {
    engine().textures.unload_render_texture(handle);
}
#[no_mangle]
pub extern "C" fn bloom_begin_texture_mode(_handle: f64) {
    // Stub: no-op until GPU render-to-texture is wired.
}
#[no_mangle]
pub extern "C" fn bloom_end_texture_mode() {
    // Stub: no-op.
}
#[no_mangle]
pub extern "C" fn bloom_get_render_texture_texture(handle: f64) -> f64 {
    engine().textures.get_render_texture_texture(handle)
}

#[no_mangle]
pub extern "C" fn bloom_profiler_frame_history() -> *const u8 {
    let hist = engine().profiler.frame_history();
    let mut s = String::with_capacity(hist.len() * 24);
    for (cpu, gpu) in &hist {
        s.push_str(&format!("{:.2}|{:.2}\n", cpu, gpu));
    }
    alloc_perry_string(&s)
}

#[no_mangle]
pub extern "C" fn bloom_profiler_overlay_text() -> *const u8 {
    let snap = engine().profiler.snapshot();
    let mut s = String::with_capacity(snap.len() * 48);
    for (label, cpu, gpu) in &snap {
        s.push_str(label);
        s.push('|');
        s.push_str(&format!("{:.2}", cpu));
        s.push('|');
        match gpu {
            Some(g) => s.push_str(&format!("{:.2}", g)),
            None    => s.push_str("-1"),
        }
        s.push('\n');
    }
    alloc_perry_string(&s)
}

// ============================================================
// Profiler — CPU phase timings (always available) + GPU timestamps
// (when the adapter supports TIMESTAMP_QUERY). Disabled by default.
// ============================================================

#[no_mangle]
pub extern "C" fn bloom_set_profiler_enabled(on: f64) {
    engine().profiler.set_enabled(on != 0.0);
}
#[no_mangle]
pub extern "C" fn bloom_get_profiler_frame_cpu_us() -> f64 {
    engine().profiler.avg_frame_cpu_us()
}
#[no_mangle]
pub extern "C" fn bloom_get_profiler_frame_gpu_us() -> f64 {
    engine().profiler.avg_frame_gpu_us()
}
#[no_mangle]
pub extern "C" fn bloom_print_profiler_summary() {
    print!("{}", engine().profiler.summary());
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
