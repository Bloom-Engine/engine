//! 2D/3D immediate-mode drawing, render modes, text.
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi) — see
//! `ffi_core/mod.rs` for the architecture and the invoking-crate contract.
//! Internal: platform crates must invoke `define_core_ffi!()`, never the
//! section macros directly.

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_draw {
    () => {

        // bloom_clear_background  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_clear_background(r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_clear_background", move || {
                engine().renderer.set_clear_color(r, g, b, a);
        })
        }

        // bloom_draw_line  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_line(x1: f64, y1: f64, x2: f64, y2: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_line", move || {
                engine().renderer.draw_line(x1, y1, x2, y2, thickness, r, g, b, a);
        })
        }

        // bloom_draw_rect  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_rect(x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_rect", move || {
                engine().renderer.draw_rect(x, y, w, h, r, g, b, a);
        })
        }

        // bloom_draw_rect_lines  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_rect_lines(x: f64, y: f64, w: f64, h: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_rect_lines", move || {
                engine().renderer.draw_rect_lines(x, y, w, h, thickness, r, g, b, a);
        })
        }

        // bloom_draw_circle  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_circle(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_circle", move || {
                engine().renderer.draw_circle(cx, cy, radius, r, g, b, a);
        })
        }

        // bloom_draw_circle_lines  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_circle_lines(cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_circle_lines", move || {
                engine().renderer.draw_circle_lines(cx, cy, radius, r, g, b, a);
        })
        }

        // bloom_draw_triangle  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_triangle(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_triangle", move || {
                engine().renderer.draw_triangle(x1, y1, x2, y2, x3, y3, r, g, b, a);
        })
        }

        // bloom_draw_poly  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_poly(cx: f64, cy: f64, sides: f64, radius: f64, rotation: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_poly", move || {
                engine().renderer.draw_poly(cx, cy, sides, radius, rotation, r, g, b, a);
        })
        }

        // bloom_draw_text  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_text(text_ptr: *const u8, x: f64, y: f64, size: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_text", move || {
                let text = $crate::string_header::str_from_header(text_ptr);
                let eng = engine();
                // Need to split borrow: take text out temporarily
                let mut text_renderer = std::mem::replace(&mut eng.text, $crate::text_renderer::TextRenderer::empty());
                text_renderer.draw_text(&mut eng.renderer, text, x, y, size as u32, r, g, b, a);
                eng.text = text_renderer;
        })
        }

        // bloom_measure_text  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_measure_text(text_ptr: *const u8, size: f64) -> f64 {
            $crate::ffi::guard("bloom_measure_text", move || {
                let text = $crate::string_header::str_from_header(text_ptr);
                engine().text.measure_text(text, size as u32)
        })
        }

        // bloom_draw_text_ex  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_text_ex(font_handle: f64, text_ptr: *const u8, x: f64, y: f64, size: f64, spacing: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_text_ex", move || {
                let text = $crate::string_header::str_from_header(text_ptr);
                let eng = engine();
                let mut text_renderer = std::mem::replace(&mut eng.text, $crate::text_renderer::TextRenderer::empty());
                text_renderer.draw_text_ex(&mut eng.renderer, font_handle as usize, text, x, y, size as u32, spacing as f32, r, g, b, a);
                eng.text = text_renderer;
        })
        }

        // bloom_measure_text_ex  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_measure_text_ex(font_handle: f64, text_ptr: *const u8, size: f64, spacing: f64) -> f64 {
            $crate::ffi::guard("bloom_measure_text_ex", move || {
                let text = $crate::string_header::str_from_header(text_ptr);
                engine().text.measure_text_ex(font_handle as usize, text, size as u32, spacing as f32)
        })
        }

        // bloom_draw_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_texture(handle: f64, x: f64, y: f64, tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64) {
            $crate::ffi::guard("bloom_draw_texture", move || {
                let eng = engine();
                if let Some(tex) = eng.textures.get(handle) {
                    let bind_group_idx = tex.bind_group_idx;
                    eng.renderer.draw_texture(bind_group_idx, x, y, tint_r, tint_g, tint_b, tint_a);
                }
        })
        }

        // bloom_draw_texture_rec  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_texture_rec(
            handle: f64,
            src_x: f64, src_y: f64, src_w: f64, src_h: f64,
            dst_x: f64, dst_y: f64,
            tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
        ) {
            $crate::ffi::guard("bloom_draw_texture_rec", move || {
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
        })
        }

        // bloom_draw_texture_pro  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_texture_pro(
            handle: f64,
            src_x: f64, src_y: f64, src_w: f64, src_h: f64,
            dst_x: f64, dst_y: f64, dst_w: f64, dst_h: f64,
            origin_x: f64, origin_y: f64, rotation: f64,
            tint_r: f64, tint_g: f64, tint_b: f64, tint_a: f64,
        ) {
            $crate::ffi::guard("bloom_draw_texture_pro", move || {
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
        })
        }

        // bloom_begin_mode_2d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_begin_mode_2d(offset_x: f64, offset_y: f64, target_x: f64, target_y: f64, rotation: f64, zoom: f64) {
            $crate::ffi::guard("bloom_begin_mode_2d", move || {
                engine().renderer.begin_mode_2d(
                    offset_x as f32, offset_y as f32,
                    target_x as f32, target_y as f32,
                    rotation as f32, zoom as f32,
                );
        })
        }

        // bloom_end_mode_2d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_end_mode_2d() {
            $crate::ffi::guard("bloom_end_mode_2d", move || {
                engine().renderer.end_mode_2d();
        })
        }

        // bloom_begin_mode_3d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_begin_mode_3d(
            pos_x: f64, pos_y: f64, pos_z: f64,
            target_x: f64, target_y: f64, target_z: f64,
            up_x: f64, up_y: f64, up_z: f64,
            fovy: f64, projection: f64,
        ) {
            $crate::ffi::guard("bloom_begin_mode_3d", move || {
                engine().renderer.begin_mode_3d(
                    pos_x as f32, pos_y as f32, pos_z as f32,
                    target_x as f32, target_y as f32, target_z as f32,
                    up_x as f32, up_y as f32, up_z as f32,
                    fovy as f32, projection as f32,
                );
        })
        }

        // bloom_end_mode_3d  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_end_mode_3d() {
            $crate::ffi::guard("bloom_end_mode_3d", move || {
                engine().renderer.end_mode_3d();
        })
        }

        // bloom_draw_cube  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_cube(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_cube", move || {
                engine().renderer.draw_cube(x, y, z, w, h, d, r, g, b, a);
        })
        }

        // bloom_draw_cube_wires  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_cube_wires(x: f64, y: f64, z: f64, w: f64, h: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_cube_wires", move || {
                engine().renderer.draw_cube_wires(x, y, z, w, h, d, r, g, b, a);
        })
        }

        // bloom_draw_sphere  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_sphere(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_sphere", move || {
                engine().renderer.draw_sphere(cx, cy, cz, radius, r, g, b, a);
        })
        }

        // bloom_draw_sphere_wires  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_sphere_wires(cx: f64, cy: f64, cz: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_sphere_wires", move || {
                engine().renderer.draw_sphere_wires(cx, cy, cz, radius, r, g, b, a);
        })
        }

        // bloom_draw_cylinder  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_cylinder(x: f64, y: f64, z: f64, radius_top: f64, radius_bottom: f64, height: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_cylinder", move || {
                engine().renderer.draw_cylinder(x, y, z, radius_top, radius_bottom, height, r, g, b, a);
        })
        }

        // bloom_draw_plane  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_plane(cx: f64, cy: f64, cz: f64, w: f64, d: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_plane", move || {
                engine().renderer.draw_plane(cx, cy, cz, w, d, r, g, b, a);
        })
        }

        // bloom_draw_grid  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_grid(slices: f64, spacing: f64) {
            $crate::ffi::guard("bloom_draw_grid", move || {
                engine().renderer.draw_grid(slices as i32, spacing);
        })
        }

        // bloom_draw_ray  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_draw_ray(origin_x: f64, origin_y: f64, origin_z: f64, dir_x: f64, dir_y: f64, dir_z: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_ray", move || {
                engine().renderer.draw_ray(origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, r, g, b, a);
        })
        }

        // bloom_begin_texture_mode  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_begin_texture_mode(handle: f64) {
            $crate::ffi::guard("bloom_begin_texture_mode", move || {
                let eng = engine();
                let (w, h, bg_idx) = match eng.textures.render_textures.get(handle) {
                    Some(rt) => {
                        let tex_handle = rt.texture_handle;
                        match eng.textures.textures.get(tex_handle) {
                            Some(td) => (rt.width, rt.height, td.bind_group_idx as usize),
                            None => return,
                        }
                    }
                    None => return,
                };
                if let Some(texture) = eng.renderer.get_texture_ref(bg_idx) {
                    // We need to call begin_texture_mode with a reference to the texture,
                    // but get_texture_ref borrows renderer immutably. Clone the texture view
                    // data we need first, then call the mutable method.
                    let color_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    // Create depth texture for this RT.
                    let depth_tex = eng.renderer.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("rt_depth"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    });
                    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    eng.renderer.rt_color_view = Some(color_view);
                    eng.renderer.rt_depth_view = Some(depth_view);
                    eng.renderer.rt_depth_texture = Some(depth_tex);
                    eng.renderer.rt_width = w;
                    eng.renderer.rt_height = h;
                }
        })
        }

        // bloom_end_texture_mode  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_end_texture_mode() {
            $crate::ffi::guard("bloom_end_texture_mode", move || {
                engine().renderer.end_texture_mode();
        })
        }

    };
}
