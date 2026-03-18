use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
use bloom_shared::string_header::str_from_header;
use bloom_shared::audio::{parse_wav, parse_ogg, parse_mp3};

use std::sync::OnceLock;

static mut ENGINE: OnceLock<EngineState> = OnceLock::new();

fn engine() -> &'static mut EngineState {
    unsafe { ENGINE.get_mut().expect("Engine not initialized") }
}

// Android uses NativeActivity. The window (ANativeWindow) is provided by the OS.
// bloom_init_window will be called after the native activity is created and the
// ANativeWindow is available. Touch events are routed through AInputEvent.

#[no_mangle]
pub extern "C" fn bloom_init_window(width: f64, height: f64, title_ptr: *const u8) {
    let _title = str_from_header(title_ptr);

    // Android windowing requires ANativeWindow from the NativeActivity.
    // The actual window handle is obtained from the Android runtime.
    // Full implementation will:
    // 1. Get ANativeWindow from NativeActivity
    // 2. Create wgpu surface using AndroidNdkWindowHandle
    // 3. Set up Vulkan/GLES backend via wgpu
    todo!("Android windowing implementation pending NativeActivity integration")
}

#[no_mangle]
pub extern "C" fn bloom_close_window() {}

#[no_mangle]
pub extern "C" fn bloom_window_should_close() -> f64 { 0.0 }

#[no_mangle]
pub extern "C" fn bloom_begin_drawing() {
    engine().begin_frame();
}

#[no_mangle]
pub extern "C" fn bloom_end_drawing() {
    engine().end_frame();
}

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

#[no_mangle]
pub extern "C" fn bloom_init_audio() {}
#[no_mangle]
pub extern "C" fn bloom_close_audio() {}

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
pub extern "C" fn bloom_get_touch_x(index: f64) -> f64 { engine().input.get_touch_x(index as usize) }
#[no_mangle]
pub extern "C" fn bloom_get_touch_y(index: f64) -> f64 { engine().input.get_touch_y(index as usize) }
#[no_mangle]
pub extern "C" fn bloom_get_touch_count() -> f64 { engine().input.get_touch_count() as f64 }
#[no_mangle]
pub extern "C" fn bloom_get_time() -> f64 { engine().get_time() }
