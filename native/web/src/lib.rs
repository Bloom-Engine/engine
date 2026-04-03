use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;

use wasm_bindgen::prelude::*;
use std::sync::OnceLock;

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();
static mut LAST_PROJECT: (f64, f64) = (0.0, 0.0);
static mut LAST_PICK: Option<bloom_shared::picking::PickResult> = None;

fn engine() -> &'static mut EngineState {
    unsafe { ENGINE.get_mut().expect("Engine not initialized") }
}

/// Log to the browser console.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn console_log(s: &str);
}

// ============================================================
// Window
// ============================================================

#[wasm_bindgen]
pub fn bloom_init_window(width: f64, height: f64, _title: f64, fullscreen: f64) {
    // Set up panic hook for better error messages in the browser console
    console_error_panic_hook::set_once();

    let w = width as u32;
    let h = height as u32;
    let _fullscreen = fullscreen != 0.0;

    wasm_bindgen_futures::spawn_local(async move {
        let window = web_sys::window().expect("no global window");
        let document = window.document().expect("no document");
        let canvas = document
            .get_element_by_id("bloom-canvas")
            .expect("no element with id 'bloom-canvas'")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("element is not a canvas");

        canvas.set_width(w);
        canvas.set_height(h);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .expect("Failed to create surface from canvas");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("No WebGPU/WebGL adapter found");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bloom_device"),
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let renderer = Renderer::new(device, queue, surface, surface_config);
        let engine_state = EngineState::new(renderer);

        unsafe {
            let _ = ENGINE.set(engine_state);
        }

        console_log("Bloom engine initialized (WebGPU)");
    });
}

#[wasm_bindgen]
pub fn bloom_close_window() {
    // TODO: Phase 3 — clean up resources if needed
}

#[wasm_bindgen]
pub fn bloom_window_should_close() -> f64 {
    // Web windows don't close the same way; always return false
    0.0
}

#[wasm_bindgen]
pub fn bloom_toggle_fullscreen() {
    // Handled by JS glue — uses Fullscreen API on the canvas element
}

#[wasm_bindgen]
pub fn bloom_set_window_title(_title: f64) {
    // TODO: Phase 3 — set document.title (need string conversion)
}

#[wasm_bindgen]
pub fn bloom_set_window_icon(_path: f64) {
    // TODO: Phase 3 — set favicon (need string conversion + fetch)
}

// ============================================================
// Drawing
// ============================================================

#[wasm_bindgen]
pub fn bloom_begin_drawing() {
    // No event polling needed on web — events are injected from JS via bloom_inject_*
    engine().begin_frame();
}

#[wasm_bindgen]
pub fn bloom_end_drawing() {
    engine().end_frame();
}

#[wasm_bindgen]
pub fn bloom_clear_background(r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.set_clear_color(r, g, b, a);
}

// ============================================================
// Timing
// ============================================================

#[wasm_bindgen]
pub fn bloom_set_target_fps(fps: f64) {
    engine().target_fps = fps;
}

#[wasm_bindgen]
pub fn bloom_get_delta_time() -> f64 {
    engine().delta_time
}

#[wasm_bindgen]
pub fn bloom_get_fps() -> f64 {
    engine().get_fps()
}

#[wasm_bindgen]
pub fn bloom_get_screen_width() -> f64 {
    engine().screen_width()
}

#[wasm_bindgen]
pub fn bloom_get_screen_height() -> f64 {
    engine().screen_height()
}

#[wasm_bindgen]
pub fn bloom_get_time() -> f64 {
    engine().get_time()
}

// ============================================================
// Input - Keyboard
// ============================================================

#[wasm_bindgen]
pub fn bloom_is_key_pressed(key: f64) -> f64 {
    if engine().input.is_key_pressed(key as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_key_down(key: f64) -> f64 {
    if engine().input.is_key_down(key as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_key_released(key: f64) -> f64 {
    if engine().input.is_key_released(key as usize) { 1.0 } else { 0.0 }
}

// ============================================================
// Input - Mouse
// ============================================================

#[wasm_bindgen]
pub fn bloom_get_mouse_x() -> f64 {
    engine().input.mouse_x
}

#[wasm_bindgen]
pub fn bloom_get_mouse_y() -> f64 {
    engine().input.mouse_y
}

#[wasm_bindgen]
pub fn bloom_is_mouse_button_pressed(btn: f64) -> f64 {
    if engine().input.is_mouse_button_pressed(btn as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_mouse_button_down(btn: f64) -> f64 {
    if engine().input.is_mouse_button_down(btn as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_mouse_button_released(btn: f64) -> f64 {
    if engine().input.is_mouse_button_released(btn as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_mouse_delta_x() -> f64 {
    engine().input.mouse_delta_x
}

#[wasm_bindgen]
pub fn bloom_get_mouse_delta_y() -> f64 {
    engine().input.mouse_delta_y
}

// ============================================================
// Input - Gamepad
// ============================================================

#[wasm_bindgen]
pub fn bloom_is_gamepad_available(gamepad: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_available() { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_gamepad_axis(gamepad: f64, axis: f64) -> f64 {
    let _ = gamepad;
    engine().input.get_gamepad_axis(axis as usize) as f64
}

#[wasm_bindgen]
pub fn bloom_is_gamepad_button_pressed(gamepad: f64, button: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_button_pressed(button as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_gamepad_button_down(gamepad: f64, button: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_button_down(button as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_is_gamepad_button_released(gamepad: f64, button: f64) -> f64 {
    let _ = gamepad;
    if engine().input.is_gamepad_button_released(button as usize) { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_gamepad_axis_count(gamepad: f64) -> f64 {
    let _ = gamepad;
    engine().input.get_gamepad_axis_count() as f64
}

// ============================================================
// Input - Touch
// ============================================================

#[wasm_bindgen]
pub fn bloom_get_touch_x() -> f64 {
    engine().input.get_touch_x(0)
}

#[wasm_bindgen]
pub fn bloom_get_touch_y() -> f64 {
    engine().input.get_touch_y(0)
}

#[wasm_bindgen]
pub fn bloom_get_touch_count() -> f64 {
    engine().input.get_touch_count() as f64
}

// ============================================================
// Input injection (called from JS event listeners)
// ============================================================

#[wasm_bindgen]
pub fn bloom_inject_key_down(key: f64) {
    engine().input.set_key_down(key as usize);
}

#[wasm_bindgen]
pub fn bloom_inject_key_up(key: f64) {
    engine().input.set_key_up(key as usize);
}

#[wasm_bindgen]
pub fn bloom_inject_gamepad_axis(axis: f64, value: f64) {
    engine().input.set_gamepad_axis(axis as usize, value as f32);
}

#[wasm_bindgen]
pub fn bloom_inject_gamepad_button_down(button: f64) {
    engine().input.set_gamepad_button_down(button as usize);
}

#[wasm_bindgen]
pub fn bloom_inject_gamepad_button_up(button: f64) {
    engine().input.set_gamepad_button_up(button as usize);
}

// ============================================================
// 2D Drawing - Shapes
// ============================================================

#[wasm_bindgen]
pub fn bloom_draw_line(x1: f64, y1: f64, x2: f64, y2: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_line(x1, y1, x2, y2, thickness, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_rect(x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_rect(x, y, w, h, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_rect_lines(x: f64, y: f64, w: f64, h: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_rect_lines(x, y, w, h, thickness, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_circle(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_circle(cx, cy, radius, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_circle_lines(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_circle_lines(cx, cy, radius, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_triangle(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_triangle(x1, y1, x2, y2, x3, y3, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_poly(cx: f64, cy: f64, sides: f64, radius: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_poly(cx, cy, sides, radius, rotation, r, g, b, a);
}

// ============================================================
// Camera 2D
// ============================================================

#[wasm_bindgen]
pub fn bloom_begin_mode_2d(offset_x: f64, offset_y: f64, target_x: f64, target_y: f64, rotation: f64, zoom: f64) {
    engine().renderer.begin_mode_2d(
        offset_x as f32, offset_y as f32,
        target_x as f32, target_y as f32,
        rotation as f32, zoom as f32,
    );
}

#[wasm_bindgen]
pub fn bloom_end_mode_2d() {
    engine().renderer.end_mode_2d();
}

// ============================================================
// Camera 3D and 3D Drawing
// ============================================================

#[wasm_bindgen]
pub fn bloom_begin_mode_3d(
    pos_x: f64, pos_y: f64, pos_z: f64,
    target_x: f64, target_y: f64, target_z: f64,
    up_x: f64, up_y: f64, up_z: f64,
    fovy: f64, projection: f64,
) {
    engine().renderer.begin_mode_3d(
        pos_x as f32, pos_y as f32, pos_z as f32,
        target_x as f32, target_y as f32, target_z as f32,
        up_x as f32, up_y as f32, up_z as f32,
        fovy as f32, projection as f32,
    );
}

#[wasm_bindgen]
pub fn bloom_end_mode_3d() {
    engine().renderer.end_mode_3d();
}

#[wasm_bindgen]
pub fn bloom_draw_cube(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube(x, y, z, w, h, d, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_cube_wires(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cube_wires(x, y, z, w, h, d, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_sphere(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere(cx, cy, cz, radius, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_sphere_wires(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_sphere_wires(cx, cy, cz, radius, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_cylinder(x: f64, y: f64, z: f64, radius_top: f64, radius_bottom: f64, height: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_cylinder(x, y, z, radius_top, radius_bottom, height, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_plane(cx: f64, cy: f64, cz: f64, w: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_plane(cx, cy, cz, w, d, r, g, b, a);
}

#[wasm_bindgen]
pub fn bloom_draw_grid(slices: f64, spacing: f64) {
    engine().renderer.draw_grid(slices as i32, spacing);
}

#[wasm_bindgen]
pub fn bloom_draw_ray(origin_x: f64, origin_y: f64, origin_z: f64, dir_x: f64, dir_y: f64, dir_z: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().renderer.draw_ray(origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, r, g, b, a);
}

// ============================================================
// Text
// ============================================================

// The original FFI functions accept f64 (NaN-boxed string handles from Perry WASM).
// The JS glue intercepts these and calls the _str variants below with actual strings.
#[wasm_bindgen]
pub fn bloom_draw_text(_text: f64, _x: f64, _y: f64, _size: f64, _r: f64, _g: f64, _b: f64, _a: f64) {
    // No-op: JS glue calls bloom_draw_text_str instead
}

#[wasm_bindgen]
pub fn bloom_draw_text_str(text: &str, x: f64, y: f64, size: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::empty());
    text_renderer.draw_text(&mut eng.renderer, text, x, y, size as u32, r, g, b, a);
    eng.text = text_renderer;
}

#[wasm_bindgen]
pub fn bloom_measure_text(_text: f64, _size: f64) -> f64 { 0.0 }

#[wasm_bindgen]
pub fn bloom_measure_text_str(text: &str, size: f64) -> f64 {
    engine().text.measure_text(text, size as u32)
}

#[wasm_bindgen]
pub fn bloom_load_font(_path: f64, _size: f64) -> f64 { 0.0 }

/// Load a font from raw bytes (fetched by JS glue via fetch()).
#[wasm_bindgen]
pub fn bloom_load_font_bytes(data: &[u8]) -> f64 {
    engine().text.load_font(data) as f64
}

#[wasm_bindgen]
pub fn bloom_unload_font(font_handle: f64) {
    engine().text.unload_font(font_handle as usize);
}

#[wasm_bindgen]
pub fn bloom_draw_text_ex(_font_handle: f64, _text: f64, _x: f64, _y: f64, _size: f64, _spacing: f64, _r: f64, _g: f64, _b: f64, _a: f64) {
    // No-op: JS glue calls bloom_draw_text_ex_str instead
}

#[wasm_bindgen]
pub fn bloom_draw_text_ex_str(font_handle: f64, text: &str, x: f64, y: f64, size: f64, spacing: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    let mut text_renderer = std::mem::replace(&mut eng.text, bloom_shared::text_renderer::TextRenderer::empty());
    text_renderer.draw_text_ex(&mut eng.renderer, font_handle as usize, text, x, y, size as u32, spacing as f32, r, g, b, a);
    eng.text = text_renderer;
}

#[wasm_bindgen]
pub fn bloom_measure_text_ex(_font_handle: f64, _text: f64, _size: f64, _spacing: f64) -> f64 { 0.0 }

#[wasm_bindgen]
pub fn bloom_measure_text_ex_str(font_handle: f64, text: &str, size: f64, spacing: f64) -> f64 {
    engine().text.measure_text_ex(font_handle as usize, text, size as u32, spacing as f32)
}

// --- Texture loading from bytes (fetched by JS glue) ---

/// Load a texture from raw image bytes (PNG/JPEG/etc.), fetched by JS glue via fetch().
#[wasm_bindgen]
pub fn bloom_load_texture_bytes(data: &[u8]) -> f64 {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut bloom_shared::renderer::Renderer;
    eng.textures.load_texture(unsafe { &mut *renderer_ptr }, data)
}

// ============================================================
// Textures
// ============================================================

#[wasm_bindgen]
pub fn bloom_load_texture(_path: f64) -> f64 {
    // TODO: Phase 3 — need fetch() for texture file + string conversion
    0.0
}

#[wasm_bindgen]
pub fn bloom_unload_texture(handle: f64) {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut Renderer;
    eng.textures.unload_texture(handle, unsafe { &mut *renderer_ptr });
}

#[wasm_bindgen]
pub fn bloom_draw_texture(handle: f64, x: f64, y: f64, tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let bind_group_idx = tex.bind_group_idx;
        eng.renderer.draw_texture(bind_group_idx, x, y, tint_r, tint_g, tint_b, tint_a);
    }
}

#[wasm_bindgen]
pub fn bloom_draw_texture_pro(
    handle: f64,
    src_x: f64, src_y: f64, src_w: f64, src_h: f64,
    dst_x: f64, dst_y: f64, dst_w: f64, dst_h: f64,
    origin_x: f64, origin_y: f64, rotation: f64,
    tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let bind_group_idx = tex.bind_group_idx;
        eng.renderer.draw_texture_pro(
            bind_group_idx,
            src_x, src_y, src_w, src_h,
            dst_x, dst_y, dst_w, dst_h,
            origin_x, origin_y, rotation,
            tint_r, tint_g, tint_b, tint_a,
        );
    }
}

#[wasm_bindgen]
pub fn bloom_draw_texture_rec(
    handle: f64,
    src_x: f64, src_y: f64, src_w: f64, src_h: f64,
    dst_x: f64, dst_y: f64,
    tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let bind_group_idx = tex.bind_group_idx;
        eng.renderer.draw_texture_rec(
            bind_group_idx,
            src_x, src_y, src_w, src_h,
            dst_x, dst_y,
            tint_r, tint_g, tint_b, tint_a,
        );
    }
}

#[wasm_bindgen]
pub fn bloom_get_texture_width(handle: f64) -> f64 {
    engine().textures.get(handle).map(|t| t.width as f64).unwrap_or(0.0)
}

#[wasm_bindgen]
pub fn bloom_get_texture_height(handle: f64) -> f64 {
    engine().textures.get(handle).map(|t| t.height as f64).unwrap_or(0.0)
}

#[wasm_bindgen]
pub fn bloom_load_image(_path: f64) -> f64 { 0.0 }

#[wasm_bindgen]
pub fn bloom_load_image_bytes(data: &[u8]) -> f64 {
    engine().textures.load_image(data)
}

#[wasm_bindgen]
pub fn bloom_image_resize(handle: f64, w: f64, h: f64) {
    engine().textures.image_resize(handle, w as u32, h as u32);
}

#[wasm_bindgen]
pub fn bloom_image_crop(handle: f64, x: f64, y: f64, w: f64, h: f64) {
    engine().textures.image_crop(handle, x as u32, y as u32, w as u32, h as u32);
}

#[wasm_bindgen]
pub fn bloom_image_flip_h(handle: f64) {
    engine().textures.image_flip_h(handle);
}

#[wasm_bindgen]
pub fn bloom_image_flip_v(handle: f64) {
    engine().textures.image_flip_v(handle);
}

#[wasm_bindgen]
pub fn bloom_load_texture_from_image(handle: f64) -> f64 {
    let eng = engine();
    let renderer_ptr = &mut eng.renderer as *mut Renderer;
    eng.textures.load_texture_from_image(handle, unsafe { &mut *renderer_ptr })
}

#[wasm_bindgen]
pub fn bloom_gen_texture_mipmaps(_handle: f64) {
    // Mipmap generation is handled by the GPU texture creation pipeline
}

#[wasm_bindgen]
pub fn bloom_set_texture_filter(handle: f64, mode: f64) {
    let eng = engine();
    if let Some(tex) = eng.textures.get(handle) {
        let bind_group_idx = tex.bind_group_idx;
        eng.renderer.set_texture_filter(bind_group_idx, mode > 0.5);
    }
}

// ============================================================
// Models
// ============================================================

#[wasm_bindgen]
pub fn bloom_load_model(_path: f64) -> f64 { 0.0 }

#[wasm_bindgen]
pub fn bloom_load_model_bytes(data: &[u8]) -> f64 {
    let eng = engine();
    eng.models.load_model_with_textures(data, &mut eng.renderer)
}

#[wasm_bindgen]
pub fn bloom_draw_model(handle: f64, x: f64, y: f64, z: f64, scale: f64, r: f64, g: f64, b: f64, a: f64) {
    let eng = engine();
    if let Some(model) = eng.models.get(handle) {
        let position = [x as f32, y as f32, z as f32];
        let scale = scale as f32;
        let tint = [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32];
        let handle_bits = handle.to_bits();
        if eng.renderer.cache_model_if_static(handle_bits, &model.meshes) {
            eng.renderer.draw_model_cached(handle_bits, position, scale, tint);
        } else {
            for mesh in &model.meshes {
                let tex_idx = mesh.texture_idx.unwrap_or(0);
                eng.renderer.draw_model_mesh_tinted(&mesh.vertices, &mesh.indices, position, scale, tint, tex_idx);
            }
        }
    }
}

#[wasm_bindgen]
pub fn bloom_unload_model(handle: f64) {
    engine().models.unload_model(handle);
}

#[wasm_bindgen]
pub fn bloom_gen_mesh_cube(w: f64, h: f64, d: f64) -> f64 {
    engine().models.gen_mesh_cube(w as f32, h as f32, d as f32)
}

#[wasm_bindgen]
pub fn bloom_gen_mesh_heightmap(image_handle: f64, size_x: f64, size_y: f64, size_z: f64) -> f64 {
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

#[wasm_bindgen]
pub fn bloom_load_shader(_source: f64) -> f64 {
    // TODO: Phase 4 — need string conversion from NaN-boxed f64 handle
    0.0
}

#[wasm_bindgen]
pub fn bloom_create_mesh(_vertex_ptr: f64, _vertex_count: f64, _index_ptr: f64, _index_count: f64) -> f64 {
    // TODO: Phase 4 — need to handle pointer passing from WASM linear memory
    0.0
}

#[wasm_bindgen]
pub fn bloom_load_model_animation(_path: f64) -> f64 { 0.0 }

#[wasm_bindgen]
pub fn bloom_load_model_animation_bytes(data: &[u8]) -> f64 {
    engine().models.load_model_animation(data)
}

#[wasm_bindgen]
pub fn bloom_update_model_animation(_handle: f64, _anim_index: f64, _time: f64, _scale: f64, _px: f64, _py: f64, _pz: f64, _rot_sin: f64, _rot_cos: f64) {
    // TODO: Phase 4 — depends on bloom_load_model_animation
}

#[wasm_bindgen]
pub fn bloom_get_model_mesh_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

#[wasm_bindgen]
pub fn bloom_get_model_material_count(handle: f64) -> f64 {
    match engine().models.get(handle) {
        Some(model) => model.meshes.len() as f64,
        None => 0.0,
    }
}

// ============================================================
// Lighting
// ============================================================

#[wasm_bindgen]
pub fn bloom_set_ambient_light(r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_ambient_light(r, g, b, intensity);
}

#[wasm_bindgen]
pub fn bloom_set_directional_light(dx: f64, dy: f64, dz: f64, r: f64, g: f64, b: f64, intensity: f64) {
    engine().renderer.set_directional_light(dx, dy, dz, r, g, b, intensity);
}

#[wasm_bindgen]
pub fn bloom_set_joint_test(joint_index: f64, angle: f64) {
    engine().renderer.set_joint_test(joint_index as usize, angle as f32);
}

#[wasm_bindgen]
pub fn bloom_add_directional_light(
    dx: f64, dy: f64, dz: f64,
    r: f64, g: f64, b: f64,
    intensity: f64,
) {
    engine().renderer.add_directional_light(
        dx as f32, dy as f32, dz as f32,
        r as f32, g as f32, b as f32,
        intensity as f32,
    );
}

#[wasm_bindgen]
pub fn bloom_add_point_light(
    x: f64, y: f64, z: f64, range: f64,
    r: f64, g: f64, b: f64,
    intensity: f64,
) {
    engine().renderer.add_point_light(
        x as f32, y as f32, z as f32, range as f32,
        r as f32, g as f32, b as f32,
        intensity as f32,
    );
}

// ============================================================
// Audio
// ============================================================

#[wasm_bindgen]
pub fn bloom_init_audio() {
    // Audio initialization is handled by JS glue (Web Audio API AudioContext).
    // The Rust AudioMixer is already initialized as part of EngineState.
}

#[wasm_bindgen]
pub fn bloom_close_audio() {
    // AudioContext cleanup handled by JS glue.
}

#[wasm_bindgen]
pub fn bloom_load_sound(_path: f64) -> f64 { 0.0 }

/// Load a sound from raw file bytes (WAV or OGG). Fetched by JS glue via fetch().
#[wasm_bindgen]
pub fn bloom_load_sound_bytes(data: &[u8]) -> f64 {
    if let Some(sound) = bloom_shared::audio::parse_wav(data)
        .or_else(|| bloom_shared::audio::parse_ogg(data))
    {
        engine().audio.load_sound(sound)
    } else {
        0.0
    }
}

#[wasm_bindgen]
pub fn bloom_play_sound(handle: f64) {
    engine().audio.play_sound(handle);
}

#[wasm_bindgen]
pub fn bloom_stop_sound(handle: f64) {
    engine().audio.stop_sound(handle);
}

#[wasm_bindgen]
pub fn bloom_set_sound_volume(handle: f64, volume: f64) {
    engine().audio.set_sound_volume(handle, volume as f32);
}

#[wasm_bindgen]
pub fn bloom_set_master_volume(volume: f64) {
    engine().audio.master_volume = volume as f32;
}

#[wasm_bindgen]
pub fn bloom_play_sound_3d(handle: f64, x: f64, y: f64, z: f64) {
    engine().audio.play_sound_3d(handle, x as f32, y as f32, z as f32);
}

#[wasm_bindgen]
pub fn bloom_set_listener_position(x: f64, y: f64, z: f64, fx: f64, fy: f64, fz: f64) {
    engine().audio.set_listener_position(x as f32, y as f32, z as f32, fx as f32, fy as f32, fz as f32);
}

#[wasm_bindgen]
pub fn bloom_load_music(_path: f64) -> f64 { 0.0 }

/// Load music from raw file bytes (WAV or OGG). Fetched by JS glue via fetch().
#[wasm_bindgen]
pub fn bloom_load_music_bytes(data: &[u8]) -> f64 {
    if let Some(sound) = bloom_shared::audio::parse_wav(data)
        .or_else(|| bloom_shared::audio::parse_ogg(data))
    {
        engine().audio.load_music(sound)
    } else {
        0.0
    }
}

/// Mix audio into an interleaved stereo f32 buffer.
/// Called by JS Web Audio ScriptProcessorNode/AudioWorklet each audio frame.
#[wasm_bindgen]
pub fn bloom_audio_mix(output: &mut [f32]) {
    engine().audio.mix_output(output);
}

#[wasm_bindgen]
pub fn bloom_play_music(handle: f64) {
    engine().audio.play_music(handle);
}

#[wasm_bindgen]
pub fn bloom_stop_music(handle: f64) {
    engine().audio.stop_music(handle);
}

#[wasm_bindgen]
pub fn bloom_update_music_stream(handle: f64) {
    engine().audio.update_music_stream(handle);
}

#[wasm_bindgen]
pub fn bloom_set_music_volume(handle: f64, volume: f64) {
    engine().audio.set_music_volume(handle, volume as f32);
}

#[wasm_bindgen]
pub fn bloom_is_music_playing(handle: f64) -> f64 {
    if engine().audio.is_music_playing(handle) { 1.0 } else { 0.0 }
}

// ============================================================
// Scene graph (retained mode)
// ============================================================

#[wasm_bindgen]
pub fn bloom_scene_create_node() -> f64 {
    engine().scene.create_node()
}

#[wasm_bindgen]
pub fn bloom_scene_destroy_node(handle: f64) {
    engine().scene.destroy_node(handle);
}

#[wasm_bindgen]
pub fn bloom_scene_set_visible(handle: f64, visible: f64) {
    engine().scene.set_visible(handle, visible != 0.0);
}

#[wasm_bindgen]
pub fn bloom_scene_set_cast_shadow(handle: f64, cast: f64) {
    engine().scene.set_cast_shadow(handle, cast != 0.0);
}

#[wasm_bindgen]
pub fn bloom_scene_set_receive_shadow(handle: f64, receive: f64) {
    engine().scene.set_receive_shadow(handle, receive != 0.0);
}

#[wasm_bindgen]
pub fn bloom_scene_set_parent(handle: f64, parent: f64) {
    engine().scene.set_parent(handle, parent);
}

#[wasm_bindgen]
pub fn bloom_scene_set_transform(
    handle: f64,
    m00: f64, m01: f64, m02: f64, m03: f64,
    m10: f64, m11: f64, m12: f64, m13: f64,
    m20: f64, m21: f64, m22: f64, m23: f64,
    m30: f64, m31: f64, m32: f64, m33: f64,
) {
    // On web we pass the 16 matrix elements as individual f64 args
    // (no raw pointer passing from WASM)
    let mat = [
        [m00 as f32, m01 as f32, m02 as f32, m03 as f32],
        [m10 as f32, m11 as f32, m12 as f32, m13 as f32],
        [m20 as f32, m21 as f32, m22 as f32, m23 as f32],
        [m30 as f32, m31 as f32, m32 as f32, m33 as f32],
    ];
    engine().scene.set_transform(handle, mat);
}

#[wasm_bindgen]
pub fn bloom_scene_update_geometry(
    _handle: f64,
    _vert_ptr: f64,
    _vert_count: f64,
    _idx_ptr: f64,
    _idx_count: f64,
) {
    // TODO: Phase 4 — need to handle pointer/buffer passing from WASM linear memory.
    // On web, vertex and index data will need to be passed via a different mechanism
    // (e.g., typed arrays through JS interop).
}

#[wasm_bindgen]
pub fn bloom_scene_set_material_color(handle: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().scene.set_material_color(handle, r as f32, g as f32, b as f32, a as f32);
}

#[wasm_bindgen]
pub fn bloom_scene_set_material_pbr(handle: f64, roughness: f64, metalness: f64) {
    engine().scene.set_material_pbr(handle, roughness as f32, metalness as f32);
}

#[wasm_bindgen]
pub fn bloom_scene_set_material_texture(handle: f64, texture_idx: f64) {
    engine().scene.set_material_texture(handle, texture_idx as u32);
}

#[wasm_bindgen]
pub fn bloom_scene_node_count() -> f64 {
    engine().scene.node_count() as f64
}

#[wasm_bindgen]
pub fn bloom_scene_node_vertex_count(handle: f64) -> f64 {
    match engine().scene.nodes.get(handle) {
        Some(node) => node.vertices.len() as f64,
        None => -1.0,
    }
}

#[wasm_bindgen]
pub fn bloom_scene_node_index_count(handle: f64) -> f64 {
    match engine().scene.nodes.get(handle) {
        Some(node) => node.indices.len() as f64,
        None => -1.0,
    }
}

#[wasm_bindgen]
pub fn bloom_scene_attach_model(node_handle: f64, model_handle: f64, mesh_index: f64) {
    let eng = engine();
    let mi = mesh_index as usize;

    let model_data = match eng.models.models.get(model_handle) {
        Some(md) => md,
        None => return,
    };
    if mi >= model_data.meshes.len() { return; }
    let mesh = &model_data.meshes[mi];

    let vertices = mesh.vertices.clone();
    let indices = mesh.indices.clone();
    eng.scene.update_geometry(node_handle, vertices, indices);

    if let Some(tex_idx) = mesh.texture_idx {
        eng.scene.set_material_texture(node_handle, tex_idx);
    }
}

// ============================================================
// Geometry generation
// ============================================================

#[wasm_bindgen]
pub fn bloom_scene_extrude_polygon(
    _handle: f64,
    _polygon_ptr: f64,
    _polygon_count: f64,
    _depth: f64,
) {
    // TODO: Phase 4 — need to handle pointer/buffer passing from WASM linear memory
}

#[wasm_bindgen]
pub fn bloom_scene_subtract_box(
    handle: f64,
    min_x: f64, min_y: f64, min_z: f64,
    max_x: f64, max_y: f64, max_z: f64,
) {
    let eng = engine();
    if let Some(node) = eng.scene.nodes.get(handle) {
        let current = bloom_shared::geometry::GeometryData {
            vertices: node.vertices.clone(),
            indices: node.indices.clone(),
        };
        let result = bloom_shared::geometry::subtract_box(
            &current,
            [min_x as f32, min_y as f32, min_z as f32],
            [max_x as f32, max_y as f32, max_z as f32],
        );
        eng.scene.update_geometry(handle, result.vertices, result.indices);
    }
}

// ============================================================
// Shadow mapping
// ============================================================

#[wasm_bindgen]
pub fn bloom_enable_shadows() {
    engine().renderer.shadow_map.enable();
}

#[wasm_bindgen]
pub fn bloom_disable_shadows() {
    engine().renderer.shadow_map.disable();
}

// ============================================================
// Post-processing
// ============================================================

#[wasm_bindgen]
pub fn bloom_enable_postfx() {
    let eng = engine();
    let w = eng.renderer.width();
    let h = eng.renderer.height();
    let fmt = eng.renderer.surface_format();
    eng.postfx = Some(bloom_shared::postfx::PostFxPipeline::new(
        &eng.renderer.device, w, h, fmt,
    ));
}

#[wasm_bindgen]
pub fn bloom_disable_postfx() {
    engine().postfx = None;
}

#[wasm_bindgen]
pub fn bloom_postfx_set_selected(handle: f64) {
    if let Some(pfx) = &mut engine().postfx {
        if handle == 0.0 {
            pfx.set_selected(Vec::new());
        } else {
            pfx.set_selected(vec![handle]);
        }
    }
}

#[wasm_bindgen]
pub fn bloom_postfx_set_hovered(handle: f64) {
    if let Some(pfx) = &mut engine().postfx {
        pfx.set_hovered(handle);
    }
}

#[wasm_bindgen]
pub fn bloom_postfx_set_outline_color(r: f64, g: f64, b: f64, a: f64) {
    if let Some(pfx) = &mut engine().postfx {
        pfx.outline_params.color_selected = [r as f32, g as f32, b as f32, a as f32];
    }
}

#[wasm_bindgen]
pub fn bloom_postfx_set_outline_thickness(thickness: f64) {
    if let Some(pfx) = &mut engine().postfx {
        pfx.outline_params.thickness[0] = thickness as f32;
    }
}

// ============================================================
// Picking
// ============================================================

#[wasm_bindgen]
pub fn bloom_scene_pick(screen_x: f64, screen_y: f64) -> f64 {
    let eng = engine();
    let inv_vp = eng.renderer.inverse_vp_matrix();
    let cam_pos = eng.renderer.camera_pos();
    let w = eng.renderer.width() as f32;
    let h = eng.renderer.height() as f32;

    let (origin, direction) = bloom_shared::picking::screen_to_ray(
        screen_x as f32, screen_y as f32,
        w, h, &inv_vp, &cam_pos,
    );

    let result = bloom_shared::picking::raycast_scene(&eng.scene, &origin, &direction);
    let hit = result.hit;
    unsafe { LAST_PICK = Some(result); }
    if hit { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_pick_hit_handle() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.handle).unwrap_or(0.0) }
}

#[wasm_bindgen]
pub fn bloom_pick_hit_distance() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.distance as f64).unwrap_or(0.0) }
}

#[wasm_bindgen]
pub fn bloom_pick_hit_x() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.point[0] as f64).unwrap_or(0.0) }
}

#[wasm_bindgen]
pub fn bloom_pick_hit_y() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.point[1] as f64).unwrap_or(0.0) }
}

#[wasm_bindgen]
pub fn bloom_pick_hit_z() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.point[2] as f64).unwrap_or(0.0) }
}

#[wasm_bindgen]
pub fn bloom_pick_hit_normal_x() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.normal[0] as f64).unwrap_or(0.0) }
}

#[wasm_bindgen]
pub fn bloom_pick_hit_normal_y() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.normal[1] as f64).unwrap_or(0.0) }
}

#[wasm_bindgen]
pub fn bloom_pick_hit_normal_z() -> f64 {
    unsafe { LAST_PICK.as_ref().map(|r| r.normal[2] as f64).unwrap_or(0.0) }
}

// ============================================================
// 3D -> 2D Projection
// ============================================================

#[wasm_bindgen]
pub fn bloom_project_to_screen(wx: f64, wy: f64, wz: f64) -> f64 {
    let eng = engine();
    let vp = eng.renderer.vp_matrix();
    let w = eng.renderer.width() as f32;
    let h = eng.renderer.height() as f32;

    let x = wx as f32;
    let y = wy as f32;
    let z = wz as f32;
    let clip_x = vp[0][0]*x + vp[1][0]*y + vp[2][0]*z + vp[3][0];
    let clip_y = vp[0][1]*x + vp[1][1]*y + vp[2][1]*z + vp[3][1];
    let clip_w = vp[0][3]*x + vp[1][3]*y + vp[2][3]*z + vp[3][3];

    if clip_w <= 0.0 {
        unsafe { LAST_PROJECT = (-9999.0, -9999.0); }
        return -9999.0;
    }

    let ndc_x = clip_x / clip_w;
    let ndc_y = clip_y / clip_w;
    let screen_x = ((ndc_x + 1.0) * 0.5 * w) as f64;
    let screen_y = ((1.0 - ndc_y) * 0.5 * h) as f64;

    unsafe { LAST_PROJECT = (screen_x, screen_y); }
    screen_x
}

#[wasm_bindgen]
pub fn bloom_project_screen_y() -> f64 {
    unsafe { LAST_PROJECT.1 }
}

// ============================================================
// File I/O
// ============================================================

#[wasm_bindgen]
pub fn bloom_write_file(_path: f64, _data: f64) -> f64 {
    // Handled by JS glue — uses localStorage
    0.0
}

#[wasm_bindgen]
pub fn bloom_file_exists(_path: f64) -> f64 {
    // Handled by JS glue — checks localStorage
    0.0
}

#[wasm_bindgen]
pub fn bloom_read_file(_path: f64) -> f64 {
    // Handled by JS glue — reads from localStorage
    0.0
}

// ============================================================
// Cursor
// ============================================================

#[wasm_bindgen]
pub fn bloom_disable_cursor() {
    let input = &mut engine().input;
    input.cursor_disabled = true;
    input.clear_mouse_delta();
    // JS glue also calls canvas.requestPointerLock()
}

#[wasm_bindgen]
pub fn bloom_enable_cursor() {
    engine().input.cursor_disabled = false;
    // JS glue also calls document.exitPointerLock()
}

// ============================================================
// Frame callbacks & game loop
// ============================================================

#[wasm_bindgen]
pub fn bloom_register_frame_callback(_priority: f64, _callback: f64) -> f64 {
    // On web, frame callbacks are managed by the JS glue layer (bloom_glue.js)
    // since the callback is a Perry WASM closure that can only be invoked
    // through Perry's runtime. The JS glue intercepts this call.
    0.0
}

#[wasm_bindgen]
pub fn bloom_unregister_frame_callback(_id: f64) {
    // Managed by JS glue layer
}

/// Emscripten-style game loop entry point.
/// On native: blocks in a while loop calling begin_drawing/callback/end_drawing.
/// On web: returns immediately. The JS glue layer (bloom_glue.js) intercepts
/// this call and drives the game loop via requestAnimationFrame.
#[wasm_bindgen]
pub fn bloom_run_game(_callback: f64) {
    // On web, this is a no-op — the JS glue intercepts the FFI call
    // before it reaches here and sets up the rAF loop with the callback.
    // The game's while(!windowShouldClose()) loop should exit after this
    // (bloom_glue.js makes windowShouldClose return 1.0 once runGame is called).
}

// ============================================================
// Staging (async asset loading)
// ============================================================

#[wasm_bindgen]
pub fn bloom_stage_texture(_path: f64) -> f64 {
    // TODO: Phase 4 — fetch() texture then decode_and_stage_texture
    0.0
}

#[wasm_bindgen]
pub fn bloom_stage_model(_path: f64) -> f64 {
    // TODO: Phase 4 — fetch() model then stage
    0.0
}

#[wasm_bindgen]
pub fn bloom_stage_sound(_path: f64) -> f64 {
    // TODO: Phase 4 — fetch() sound then stage
    0.0
}

#[wasm_bindgen]
pub fn bloom_commit_texture(staging_handle: f64) -> f64 {
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

#[wasm_bindgen]
pub fn bloom_commit_model(staging_handle: f64) -> f64 {
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

#[wasm_bindgen]
pub fn bloom_commit_sound(staging_handle: f64) -> f64 {
    match bloom_shared::staging::take_sound(staging_handle) {
        Some(sd) => engine().audio.load_sound(sd),
        None => 0.0,
    }
}

#[wasm_bindgen]
pub fn bloom_commit_music(staging_handle: f64) -> f64 {
    match bloom_shared::staging::take_sound(staging_handle) {
        Some(sd) => engine().audio.load_music(sd),
        None => 0.0,
    }
}

// ============================================================
// Platform detection
// ============================================================

#[wasm_bindgen]
pub fn bloom_get_platform() -> f64 {
    7.0 // Web platform ID
}

#[wasm_bindgen]
pub fn bloom_is_any_input_pressed() -> f64 {
    if engine().input.is_any_input_pressed() { 1.0 } else { 0.0 }
}
