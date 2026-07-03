use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;

use wasm_bindgen::prelude::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();
static mut LAST_PROJECT: (f64, f64) = (0.0, 0.0);
static mut LAST_PICK: Option<bloom_shared::picking::PickResult> = None;
// Guards against double wgpu init when bloom_init_window is called twice
// (once by the JS orchestrator before Perry boots, and once by Perry's main).
static INIT_STARTED: AtomicBool = AtomicBool::new(false);

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

/// Returns 1.0 once `ENGINE` has been populated by the async wgpu setup, 0.0
/// before then. Non-panicking — safe to call at any time. Used by the JS
/// orchestrator to gate Perry boot until the engine is actually usable.
#[wasm_bindgen]
pub fn bloom_is_initialized() -> f64 {
    unsafe { if ENGINE.get().is_some() { 1.0 } else { 0.0 } }
}

/// Host-surface attach (PerryTS/perry#5519). Not applicable on web: the
/// engine builds its surface from the `bloom-canvas` DOM element in
/// `bloom_init_window`, and a wasm module can't be handed a native view
/// pointer. Present for FFI-surface uniformity; always returns 0.0 so web
/// hosts use the canvas-based `bloom_init_window` path instead.
#[wasm_bindgen]
pub fn bloom_attach_native(_handle: f64, _width: f64, _height: f64) -> f64 {
    0.0
}

#[wasm_bindgen]
pub fn bloom_init_window(width: f64, height: f64, _title: f64, fullscreen: f64) {
    // Set up panic hook for better error messages in the browser console
    console_error_panic_hook::set_once();

    // Idempotent: once an init is in flight (or done), later calls are no-ops.
    // Perry's main() typically calls initWindow after the JS orchestrator has
    // already kicked off wgpu setup — the second call must not start a new one.
    if INIT_STARTED.swap(true, Ordering::SeqCst) {
        return;
    }

    let log_w = width as u32;
    let log_h = height as u32;
    let _fullscreen = fullscreen != 0.0;

    wasm_bindgen_futures::spawn_local(async move {
        let window = web_sys::window().expect("no global window");
        let document = window.document().expect("no document");
        let canvas = document
            .get_element_by_id("bloom-canvas")
            .expect("no element with id 'bloom-canvas'")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("element is not a canvas");

        // HiDPI: size the canvas *backing store* (set_width/set_height,
        // i.e. the WebGPU surface) to logical × devicePixelRatio so a
        // 4K Retina display actually renders at 4K. CSS layout dimensions
        // stay in logical pixels — index.html keeps `width: 100%` on
        // the canvas — so the document layout is unchanged. Clamp to
        // [1, 3] because higher dprs (some phones report 4) are wasted
        // shading work and can crash low-end mobile GPUs.
        let dpr = window.device_pixel_ratio().max(1.0).min(3.0);
        let phys_w = ((log_w as f64) * dpr).round() as u32;
        let phys_h = ((log_h as f64) * dpr).round() as u32;
        canvas.set_width(phys_w);
        canvas.set_height(phys_h);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
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

        // BC feature -> cooked BC7 textures upload compressed (else CPU-decode).
        let required_features =
            adapter.features() & wgpu::Features::TEXTURE_COMPRESSION_BC;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bloom_device"),
                    required_features,
                    ..Default::default()
                },
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
            width: phys_w,
            height: phys_h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let renderer = Renderer::new(device, queue, surface, surface_config, log_w, log_h);
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

/// Web-only resize hook. Called from index.html when a ResizeObserver
/// or matchMedia(`(resolution: ...)`) signals the canvas's CSS box
/// changed or devicePixelRatio shifted (e.g. browser zoom). The JS
/// glue does the dpr math; we just receive both sizes and forward to
/// the renderer.
#[wasm_bindgen]
pub fn bloom_resize(physical_w: f64, physical_h: f64, logical_w: f64, logical_h: f64) {
    let pw = (physical_w as u32).max(1);
    let ph = (physical_h as u32).max(1);
    let lw = (logical_w as u32).max(1);
    let lh = (logical_h as u32).max(1);
    unsafe {
        if let Some(eng) = ENGINE.get_mut() {
            if pw != eng.renderer.physical_width() || ph != eng.renderer.physical_height() {
                eng.renderer.resize(pw, ph, lw, lh);
            }
        }
    }
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
pub fn bloom_set_direct_2d_mode(on: f64) {
    engine().direct_2d_mode = on > 0.5;
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

// Input FFI (keyboard/mouse/gamepad/touch + JS-event injection) lives
// in input_ffi.rs (2000-line file policy).
mod input_ffi;
pub use input_ffi::*;

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
pub fn bloom_draw_model_rotated(
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

// ============================================================
// Phase 1c — material system FFI
// ============================================================

#[wasm_bindgen]
pub fn bloom_set_material_params(_handle: f64, _params_ptr: f64, _param_count: f64) {
    // No-op: JS glue calls bloom_set_material_params_floats instead
}

#[wasm_bindgen]
pub fn bloom_set_material_params_floats(handle: f64, params: &[f32]) {
    let count = params.len();
    if count > 64 {
        web_sys::console::error_1(&format!(
            "[material] set_material_params: param_count {} > 64 (256-byte UBO cap)",
            count
        ).into());
        return;
    }
    let mut bytes = vec![0u8; count * 4];
    for (i, &v) in params.iter().enumerate() {
        bytes[i*4..i*4+4].copy_from_slice(&v.to_le_bytes());
    }
    let eng = engine();
    if let Err(e) = eng.renderer.material_system.set_user_params(
        &eng.renderer.device, &eng.renderer.queue,
        handle as u32, &bytes,
    ) {
        web_sys::console::error_1(&format!("[material] set_material_params failed: {}", e).into());
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_str(source: &str) -> f64 {
    match engine().renderer.compile_material(source) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_refractive(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_refractive_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_refractive_str(source: &str) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Refractive, true, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[refractive] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_transparent(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_transparent_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_transparent_str(source: &str) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Transparent, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_additive(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_additive_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_additive_str(source: &str) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Translucent, Bucket::Additive, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_cutout(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_cutout_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_cutout_str(source: &str) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    match engine().renderer.compile_material_with_options(
        source, FragmentProfile::Opaque, Bucket::Cutout, false, false,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_compile_material_instanced(_source: f64) -> f64 {
    // No-op: JS glue calls bloom_compile_material_instanced_str instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_instanced_str(source: &str) -> f64 {
    match engine().renderer.compile_material_instanced(source) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] instanced compile failed: {:?}", e).into());
            0.0
        }
    }
}

#[wasm_bindgen]
pub fn bloom_create_instance_buffer(_data_ptr: f64, _instance_count: f64) -> f64 {
    // No-op: JS glue calls bloom_create_instance_buffer_floats instead
    0.0
}

#[wasm_bindgen]
pub fn bloom_create_instance_buffer_floats(data: &[f32], instance_count: f64) -> f64 {
    if instance_count <= 0.0 { return 0.0; }
    let count = instance_count as u32;
    engine().renderer.create_instance_buffer(data, count) as f64
}

#[wasm_bindgen]
pub fn bloom_submit_material_draw_instanced(
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

#[wasm_bindgen]
pub fn bloom_destroy_instance_buffer(handle: f64) {
    engine().renderer.destroy_instance_buffer(handle as u32);
}

/// EN-011 — create a planar reflection probe. See macOS lib.rs for the
/// full doc comment. Web ports the same FFI surface so a TypeScript
/// game targets one API across native + browser.
#[wasm_bindgen]
pub fn bloom_create_planar_reflection(
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
#[wasm_bindgen]
pub fn bloom_set_material_reflection_probe(
    material: f64, probe: f64,
) {
    engine().renderer.set_material_reflection_probe(material as u32, probe as u32);
}

/// EN-014 — pointer-shaped variant exists only so the FFI manifest
/// validates against the Perry surface; JS glue calls
/// `bloom_create_texture_array_bytes` with a `Uint8Array` instead.
/// Same precedent as `bloom_create_instance_buffer` (see EN-001).
#[wasm_bindgen]
pub fn bloom_create_texture_array(
    _data_ptr: f64, _data_len: f64, _width: f64, _height: f64, _layer_count: f64,
) -> f64 {
    0.0
}

/// EN-014 — create a texture array from concatenated RGBA8 bytes.
/// JS glue passes a `Uint8Array` of `layer_count × width × height × 4`
/// bytes (each layer back-to-back). Layer count is capped at 16; the
/// rest are silently dropped. Returns a 1-based handle (0 on failure).
#[wasm_bindgen]
pub fn bloom_create_texture_array_bytes(
    data: &[u8],
    width: f64, height: f64, layer_count: f64,
) -> f64 {
    // EN-014 V2 — V1 forwards to _ex with default sRGB / no mips.
    bloom_create_texture_array_ex_bytes(data, width, height, layer_count, 0.0, 1.0)
}

/// EN-014 V2 — pointer-shaped Ex variant exists only so the FFI manifest
/// validates against the Perry surface; JS glue uses the `_bytes` form.
#[wasm_bindgen]
pub fn bloom_create_texture_array_ex(
    _data_ptr: f64, _data_len: f64, _width: f64, _height: f64,
    _layer_count: f64, _format: f64, _mip_levels: f64,
) -> f64 {
    0.0
}

/// EN-014 V2 — bytes form of `_ex`. See `MaterialSystem::create_texture_array_ex`
/// for `format` (0 = sRGB, 1 = linear) and `mip_levels` (1 = none, 0 =
/// auto-generate via point-sample copies) semantics.
#[wasm_bindgen]
pub fn bloom_create_texture_array_ex_bytes(
    data: &[u8],
    width: f64, height: f64, layer_count: f64,
    format: f64, mip_levels: f64,
) -> f64 {
    let w = width as u32;
    let h = height as u32;
    if w == 0 || h == 0 { return 0.0; }
    let layers_count = (layer_count as u32)
        .min(bloom_shared::renderer::material_system::MAX_TEXTURE_ARRAY_LAYERS);
    if layers_count == 0 { return 0.0; }
    let layer_size = (w as usize) * (h as usize) * 4;
    let mut layers: Vec<(&[u8], u32, u32)> = Vec::with_capacity(layers_count as usize);
    for i in 0..(layers_count as usize) {
        let start = i * layer_size;
        let end   = start + layer_size;
        if end > data.len() { break; }
        layers.push((&data[start..end], w, h));
    }
    engine().renderer.create_texture_array_ex(&layers, format as u32, mip_levels as u32) as f64
}

/// EN-014 — link a texture-array handle to a material slot
/// (0 = albedo / 1 = normal / 2 = MR). `array = 0` reverts to the
/// engine's 1×1×1 stub.
#[wasm_bindgen]
pub fn bloom_set_material_texture_array(
    material: f64, slot: f64, array: f64,
) {
    engine().renderer.set_material_texture_array(
        material as u32, slot as u32, array as u32,
    );
}

/// EN-012 — set the shading model for a material (0=default lit,
/// 1=foliage, 2=subsurface V2 stub).
#[wasm_bindgen]
pub fn bloom_set_material_shading_model(
    material: f64, model: f64,
) {
    engine().renderer.set_material_shading_model(material as u32, model as u32);
}

/// EN-012 — set the foliage shading parameters for a material.
/// Only takes effect when shading_model == 1 (foliage).
#[wasm_bindgen]
pub fn bloom_set_material_foliage(
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

#[wasm_bindgen]
pub fn bloom_compile_material_from_file(_path: f64, _bucket_kind: f64) -> f64 {
    // No-op: web has no filesystem — JS glue would have to fetch + call
    // bloom_compile_material_from_file_str instead.
    0.0
}

#[wasm_bindgen]
pub fn bloom_compile_material_from_file_str(path: &str, bucket_kind: f64) -> f64 {
    use bloom_shared::renderer::material_pipeline::{FragmentProfile, Bucket};
    let (profile, bucket, reads_scene) = match bucket_kind as u32 {
        0 => (FragmentProfile::Opaque,      Bucket::Opaque,      false),
        1 => (FragmentProfile::Translucent, Bucket::Transparent, false),
        2 => (FragmentProfile::Translucent, Bucket::Refractive,  true),
        3 => (FragmentProfile::Translucent, Bucket::Additive,    false),
        4 => (FragmentProfile::Opaque,      Bucket::Cutout,      false),
        _ => {
            web_sys::console::error_1(&format!(
                "[material] from_file: unknown bucket_kind {}", bucket_kind
            ).into());
            return 0.0;
        }
    };
    match engine().renderer.compile_material_from_file(
        std::path::Path::new(path), profile, bucket, reads_scene,
    ) {
        Ok(handle) => handle as f64,
        Err(e) => {
            web_sys::console::error_1(&format!("[material] from_file failed: {}", e).into());
            0.0
        }
    }
}

/// EN-017 — stub: JS glue calls `bloom_set_post_pass_str` instead.
#[wasm_bindgen]
pub fn bloom_set_post_pass(_source: f64) -> f64 { 0.0 }

/// EN-017 — compile + install a fullscreen post-pass material on web.
/// See `bloom-macos::bloom_set_post_pass` for the full ABI. Returns
/// 1.0 on success, 0.0 on compile failure.
#[wasm_bindgen]
pub fn bloom_set_post_pass_str(source: &str) -> f64 {
    match engine().renderer.set_post_pass(source) {
        Ok(()) => 1.0,
        Err(e) => {
            web_sys::console::error_1(
                &format!("[post_pass] compile failed: {:?}", e).into(),
            );
            0.0
        }
    }
}

/// EN-017 — uninstall the active post-pass.
#[wasm_bindgen]
pub fn bloom_clear_post_pass() {
    engine().renderer.clear_post_pass();
}

/// EN-017 V2 — stub: JS glue calls `bloom_add_post_pass_str` instead.
#[wasm_bindgen]
pub fn bloom_add_post_pass(_source: f64) -> f64 { 0.0 }

/// EN-017 V2 — append a fullscreen post-pass to the stack on web.
/// See `bloom-macos::bloom_add_post_pass` for the full ABI. Returns
/// 1-based handle on success, 0.0 on compile failure.
#[wasm_bindgen]
pub fn bloom_add_post_pass_str(source: &str) -> f64 {
    match engine().renderer.add_post_pass(source) {
        Ok(h) => h as f64,
        Err(e) => {
            web_sys::console::error_1(
                &format!("[post_pass] compile failed: {:?}", e).into(),
            );
            0.0
        }
    }
}

/// EN-017 V2 — wipe the entire post-pass stack.
#[wasm_bindgen]
pub fn bloom_clear_all_post_passes() {
    engine().renderer.clear_all_post_passes();
}

#[wasm_bindgen]
pub fn bloom_draw_material(
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
pub fn bloom_set_procedural_sky(enabled: f64, rayleigh_density: f64, mie_density: f64, ground_albedo: f64) {
    engine().renderer.set_procedural_sky(
        enabled != 0.0,
        rayleigh_density as f32,
        mie_density as f32,
        ground_albedo as f32,
    );
}

#[wasm_bindgen]
pub fn bloom_set_sun_direction(dx: f64, dy: f64, dz: f64, intensity: f64) {
    engine().renderer.set_sun_direction(dx as f32, dy as f32, dz as f32, intensity as f32);
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
    engine().audio.set_master_volume(volume as f32);
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
    let base_color_tex = mesh.texture_idx;
    let normal_tex = mesh.normal_texture_idx;
    let mr_tex = mesh.metallic_roughness_texture_idx;
    let emissive_tex = mesh.emissive_texture_idx;
    let emissive_factor = mesh.emissive_factor;
    eng.scene.update_geometry(node_handle, vertices, indices);

    if let Some(tex_idx) = base_color_tex {
        eng.scene.set_material_texture(node_handle, tex_idx);
    }
    if let Some(tex_idx) = normal_tex {
        eng.scene.set_material_normal_texture(node_handle, tex_idx);
    }
    if let Some(tex_idx) = mr_tex {
        eng.scene.set_material_metallic_roughness_texture(node_handle, tex_idx);
    }
    if let Some(tex_idx) = emissive_tex {
        eng.scene.set_material_emissive_texture(node_handle, tex_idx);
    }
    eng.scene.set_material_emissive_factor(
        node_handle,
        emissive_factor[0],
        emissive_factor[1],
        emissive_factor[2],
    );
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
pub fn bloom_get_language() -> f64 {
    // navigator.language (e.g. "en-US" / "zh-Hans") -> packed 2-letter code.
    let lang = web_sys::window().and_then(|w| w.navigator().language()).unwrap_or_default();
    let b = lang.to_ascii_lowercase().into_bytes();
    if b.len() >= 2 { (b[0] as f64) * 256.0 + (b[1] as f64) } else { 25966.0 }
}

#[wasm_bindgen]
pub fn bloom_is_any_input_pressed() -> f64 {
    if engine().input.is_any_input_pressed() { 1.0 } else { 0.0 }
}

#[wasm_bindgen]
pub fn bloom_get_crown_rotation() -> f64 {
    engine().input.consume_crown_rotation()
}


// Scene graph QoL — Q4/Q5/Q6/Q7
#[wasm_bindgen]
pub fn bloom_scene_get_transform(handle: f64, index: f64) -> f64 {
    let mat = engine().scene.get_transform(handle);
    let i = index as usize;
    let col = i / 4;
    let row = i % 4;
    if col < 4 && row < 4 { mat[col][row] as f64 } else { 0.0 }
}
#[wasm_bindgen]
pub fn bloom_scene_get_bounds_min_x(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[0] as f64 }
#[wasm_bindgen]
pub fn bloom_scene_get_bounds_min_y(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[1] as f64 }
#[wasm_bindgen]
pub fn bloom_scene_get_bounds_min_z(handle: f64) -> f64 { engine().scene.get_bounds(handle).0[2] as f64 }
#[wasm_bindgen]
pub fn bloom_scene_get_bounds_max_x(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[0] as f64 }
#[wasm_bindgen]
pub fn bloom_scene_get_bounds_max_y(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[1] as f64 }
#[wasm_bindgen]
pub fn bloom_scene_get_bounds_max_z(handle: f64) -> f64 { engine().scene.get_bounds(handle).1[2] as f64 }
#[wasm_bindgen]
pub fn bloom_scene_set_user_data(handle: f64, data: f64) { engine().scene.set_user_data(handle, data as i64); }
#[wasm_bindgen]
pub fn bloom_scene_get_user_data(handle: f64) -> f64 { engine().scene.get_user_data(handle) as f64 }

// Q1: Render texture FFI (stub)
#[wasm_bindgen]
pub fn bloom_load_render_texture(width: f64, height: f64) -> f64 {
    engine().textures.load_render_texture(width as u32, height as u32)
}
#[wasm_bindgen]
pub fn bloom_unload_render_texture(handle: f64) { engine().textures.unload_render_texture(handle); }
#[wasm_bindgen]
pub fn bloom_begin_texture_mode(_handle: f64) { /* stub */ }
#[wasm_bindgen]
pub fn bloom_end_texture_mode() { /* stub */ }
#[wasm_bindgen]
pub fn bloom_get_render_texture_texture(handle: f64) -> f64 { engine().textures.get_render_texture_texture(handle) }

// Q8: Water material
#[wasm_bindgen]
pub fn bloom_scene_set_material_water(handle: f64, wave_amp: f64, wave_speed: f64, r: f64, g: f64, b: f64, a: f64) {
    engine().scene.set_material_water(handle, wave_amp as f32, wave_speed as f32, r as f32, g as f32, b as f32, a as f32);
}

// Q9: Spline ribbon mesh
#[wasm_bindgen]
pub fn bloom_gen_mesh_spline_ribbon(points_ptr: *const u8, point_count: f64, widths_ptr: *const u8, width_count: f64) -> f64 {
    let n = point_count as usize;
    let wn = width_count as usize;
    let points = unsafe { std::slice::from_raw_parts(points_ptr as *const f32, n * 3) };
    let widths = unsafe { std::slice::from_raw_parts(widths_ptr as *const f32, wn) };
    engine().models.gen_mesh_spline_ribbon(points, widths)
}

// Q6: Multi-hit picking
static mut LAST_PICK_ALL: Vec<bloom_shared::picking::PickResult> = Vec::new();

#[wasm_bindgen]
pub fn bloom_scene_pick_all(screen_x: f64, screen_y: f64, max_results: f64) -> f64 {
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
#[wasm_bindgen]
pub fn bloom_pick_all_handle(index: f64) -> f64 {
    let i = index as usize;
    unsafe { LAST_PICK_ALL.get(i).map(|r| r.handle).unwrap_or(0.0) }
}
#[wasm_bindgen]
pub fn bloom_pick_all_distance(index: f64) -> f64 {
    let i = index as usize;
    unsafe { LAST_PICK_ALL.get(i).map(|r| r.distance as f64).unwrap_or(0.0) }
}

// Q2: Cursor shape
#[wasm_bindgen]
pub fn bloom_set_cursor_shape(shape: f64) {
    engine().input.cursor_shape = shape as u32;
}

// E4: Clipboard (stub)
#[wasm_bindgen]
pub fn bloom_set_clipboard_text(_text_ptr: *const u8) {}
#[wasm_bindgen]
pub fn bloom_get_clipboard_text() -> f64 { 0.0 }

// E5b: File dialogs (stub)
#[wasm_bindgen]
pub fn bloom_open_file_dialog(_filter_ptr: *const u8, _title_ptr: *const u8) -> f64 { 0.0 }
#[wasm_bindgen]
pub fn bloom_save_file_dialog(_default_name_ptr: *const u8, _title_ptr: *const u8) -> f64 { 0.0 }

// ============================================================
// Render quality toggles (individual + preset) — ticket 011
// Mirror of the macOS FFI surface added in commit 95da6af, exposed to
// the browser via wasm_bindgen. Without these the TS API's
// setQualityPreset / setShadowsEnabled / etc. would fail with
// "bloom_set_quality_preset is not a function" on the web target.
// ============================================================

#[wasm_bindgen]
pub fn bloom_set_quality_preset(preset: f64) {
    engine().renderer.apply_quality_preset(preset as u32);
}
#[wasm_bindgen]
pub fn bloom_set_shadows_enabled(on: f64) {
    engine().renderer.set_shadows_enabled(on != 0.0);
}
#[wasm_bindgen]
pub fn bloom_set_shadows_always_fresh(on: f64) {
    engine().renderer.set_shadows_always_fresh(on != 0.0);
}
#[wasm_bindgen]
pub fn bloom_set_bloom_enabled(on: f64) {
    engine().renderer.set_bloom_enabled(on != 0.0);
}
#[wasm_bindgen]
pub fn bloom_set_ssao_enabled(on: f64) {
    engine().renderer.set_ssao_enabled(on != 0.0);
}
#[wasm_bindgen]
pub fn bloom_set_ssao_intensity(value: f64) {
    engine().renderer.set_ssao_strength(value as f32);
}
#[wasm_bindgen]
pub fn bloom_set_ssao_radius(world_radius: f64) {
    engine().renderer.set_ssao_radius(world_radius as f32);
}
#[wasm_bindgen]
pub fn bloom_set_wind(dir_x: f64, dir_z: f64, amplitude: f64, frequency: f64) {
    engine().renderer.set_wind(dir_x as f32, dir_z as f32, amplitude as f32, frequency as f32);
}
#[wasm_bindgen]
pub fn bloom_set_ssr_enabled(on: f64) {
    engine().renderer.set_ssr_enabled(on != 0.0);
}
#[wasm_bindgen]
pub fn bloom_set_motion_blur_enabled(on: f64) {
    engine().renderer.set_motion_blur_enabled(on != 0.0);
}
#[wasm_bindgen]
pub fn bloom_set_sss_enabled(on: f64) {
    engine().renderer.set_sss_enabled(on != 0.0);
}

// ============================================================
// Profiler — CPU phase timings (always available). TIMESTAMP_QUERY
// is not part of the WebGPU spec as of wgpu 29, so the GPU path
// returns 0 on web regardless of adapter — the profiler stays
// CPU-only, which is still useful for frame-phase profiling.
// ============================================================

#[wasm_bindgen]
pub fn bloom_set_profiler_enabled(on: f64) {
    engine().profiler.set_enabled(on != 0.0);
}
#[wasm_bindgen]
pub fn bloom_get_profiler_frame_cpu_us() -> f64 {
    engine().profiler.avg_frame_cpu_us()
}
#[wasm_bindgen]
pub fn bloom_get_profiler_frame_gpu_us() -> f64 {
    engine().profiler.avg_frame_gpu_us()
}
#[wasm_bindgen]
pub fn bloom_print_profiler_summary() {
    // No stdout in the browser — route through the existing
    // console_log binding so the summary lands in devtools.
    console_log(&engine().profiler.summary());
}
// ============================================================

// Physics (Jolt via JoltPhysics.js) — the whole bloom_physics_* surface
// lives in physics_ffi.rs, kept in lockstep with jolt_bridge.js and the
// package.json manifest (tools/validate-ffi.js checks the name coverage).
mod physics_ffi;
pub use physics_ffi::*;

// Renderer settings + LOD surface live in render_settings.rs (2000-line
// file policy); same shared engine methods the native macro wraps.
mod render_settings;
pub use render_settings::*;
