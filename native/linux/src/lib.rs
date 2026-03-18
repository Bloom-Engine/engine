use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::str_from_header;
use bloom_shared::audio::{parse_wav, parse_ogg, parse_mp3};

use std::sync::OnceLock;
use std::os::unix::io::RawFd;

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();
static mut GAMEPAD_FD: RawFd = -1;

fn engine() -> &'static mut EngineState {
    unsafe { ENGINE.get_mut().expect("Engine not initialized") }
}

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

    pub fn create_window(width: f64, height: f64, title: &str) {
        unsafe {
            DISPLAY = x11::xlib::XOpenDisplay(std::ptr::null());
            if DISPLAY.is_null() {
                panic!("Failed to open X11 display");
            }

            let screen = x11::xlib::XDefaultScreen(DISPLAY);
            let root = x11::xlib::XRootWindow(DISPLAY, screen);

            X11_WINDOW = x11::xlib::XCreateSimpleWindow(
                DISPLAY, root,
                0, 0, width as u32, height as u32, 0,
                x11::xlib::XBlackPixel(DISPLAY, screen),
                x11::xlib::XWhitePixel(DISPLAY, screen),
            );

            let title_cstr = std::ffi::CString::new(title).unwrap();
            x11::xlib::XStoreName(DISPLAY, X11_WINDOW, title_cstr.as_ptr());

            x11::xlib::XSelectInput(DISPLAY, X11_WINDOW,
                x11::xlib::ExposureMask | x11::xlib::KeyPressMask | x11::xlib::KeyReleaseMask |
                x11::xlib::ButtonPressMask | x11::xlib::ButtonReleaseMask |
                x11::xlib::PointerMotionMask | x11::xlib::StructureNotifyMask);

            x11::xlib::XMapWindow(DISPLAY, X11_WINDOW);
            x11::xlib::XFlush(DISPLAY);
        }
    }

    pub fn display() -> *mut x11::xlib::Display { unsafe { DISPLAY } }
    pub fn window() -> x11::xlib::Window { unsafe { X11_WINDOW } }

    pub fn poll_events() {
        unsafe {
            while x11::xlib::XPending(DISPLAY) > 0 {
                let mut event = std::mem::zeroed::<x11::xlib::XEvent>();
                x11::xlib::XNextEvent(DISPLAY, &mut event);

                match event.type_ {
                    x11::xlib::KeyPress => {
                        let key_event = event.key;
                        let keysym = x11::xlib::XLookupKeysym(
                            &mut event.key as *mut _ as *mut _,
                            0,
                        );
                        let bloom_key = map_keycode(keysym as u32);
                        if bloom_key > 0 {
                            engine().input.set_key_down(bloom_key);
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
                        engine().input.set_mouse_position(motion.x as f64, motion.y as f64);
                    }
                    x11::xlib::ButtonPress => {
                        let button = event.button.button;
                        match button {
                            1 => engine().input.set_mouse_button_down(0),
                            3 => engine().input.set_mouse_button_down(1),
                            2 => engine().input.set_mouse_button_down(2),
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
                        let new_w = configure.width as u32;
                        let new_h = configure.height as u32;
                        if new_w > 0 && new_h > 0 {
                            let eng = engine();
                            if new_w != eng.renderer.width() || new_h != eng.renderer.height() {
                                eng.renderer.resize(new_w, new_h);
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
pub extern "C" fn bloom_init_window(width: f64, height: f64, title_ptr: *const u8) {
    let title = str_from_header(title_ptr);

    #[cfg(target_os = "linux")]
    {
        x11_impl::create_window(width, height, title);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = unsafe {
            let raw_window = raw_window_handle::RawWindowHandle::Xlib(
                raw_window_handle::XlibWindowHandle::new(x11_impl::window() as u32)
            );
            let raw_display = raw_window_handle::RawDisplayHandle::Xlib(
                raw_window_handle::XlibDisplayHandle::new(
                    std::ptr::NonNull::new(x11_impl::display() as *mut _),
                    0,
                )
            );
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: raw_display,
                raw_window_handle: raw_window,
            }).expect("Failed to create surface")
        };

        let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })).expect("No adapter found");

        let (device, queue) = pollster_block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(), None,
        )).expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0];
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width as u32,
            height: height as u32,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let renderer = Renderer::new(device, queue, surface, surface_config);
        unsafe { let _ = ENGINE.set(EngineState::new(renderer)); }
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
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::new());
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
    match std::fs::read(path) { Ok(data) => engine().text.load_font(&data) as f64, Err(_) => 0.0 }
}

#[no_mangle]
pub extern "C" fn bloom_unload_font(font_handle: f64) {
    engine().text.unload_font(font_handle as usize);
}

#[no_mangle]
pub extern "C" fn bloom_draw_text_ex(font_handle: f64, text_ptr: *const u8, x: f64, y: f64, size: f64, spacing: f64, r: f64, g: f64, b: f64, a: f64) {
    let text = str_from_header(text_ptr);
    let eng = engine();
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::new());
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
    #[cfg(target_os = "linux")]
    {
        use std::sync::atomic::Ordering;
        AUDIO_RUNNING.store(true, Ordering::SeqCst);
        std::thread::spawn(|| {
            alsa_audio_thread();
        });
    }
}

#[no_mangle]
pub extern "C" fn bloom_close_audio() {
    AUDIO_RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));
}

#[cfg(target_os = "linux")]
fn alsa_audio_thread() {
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

        unsafe {
            ENGINE.get_mut().map(|eng| {
                eng.audio.mix_output(&mut mix_buf);
            });
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
pub extern "C" fn bloom_set_sound_volume(handle: f64, volume: f64) { engine().audio.set_sound_volume(handle, volume as f32); }
#[no_mangle]
pub extern "C" fn bloom_set_master_volume(volume: f64) { engine().audio.master_volume = volume as f32; }

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
    match std::fs::read(path) { Ok(data) => engine().models.load_model(&data), Err(_) => 0.0 }
}
#[no_mangle]
pub extern "C" fn bloom_draw_model(handle: f64, x: f64, y: f64, z: f64, scale: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(model) = eng.models.get(handle) {
        let tint = [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32];
        for mesh in &model.meshes {
            eng.renderer.draw_model_mesh_tinted(&mesh.vertices, &mesh.indices, [x as f32, y as f32, z as f32], scale as f32, tint);
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

#[no_mangle]
pub extern "C" fn bloom_load_model_animation(path_ptr: *const u8) -> f64 {
    let path = str_from_header(path_ptr);
    match std::fs::read(path) {
        Ok(data) => engine().models.load_model_animation(&data),
        Err(_) => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn bloom_update_model_animation(handle: f64, anim_index: f64, time: f64) {
    engine().models.update_model_animation(handle, anim_index as usize, time as f32);
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

// --- Utility FFI ---

#[no_mangle]
pub extern "C" fn bloom_toggle_fullscreen() {}
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
    match std::fs::read_to_string(path) {
        Ok(contents) => {
            let c_str = std::ffi::CString::new(contents).unwrap_or_default();
            c_str.into_raw() as *const u8
        }
        Err(_) => std::ptr::null(),
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
