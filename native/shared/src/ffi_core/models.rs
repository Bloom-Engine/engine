//! 3D model + mesh surface (gated on the `models3d` cargo feature).
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi) — see
//! `ffi_core/mod.rs` for the architecture and the invoking-crate contract.
//! Internal: platform crates must invoke `define_core_ffi!()`, never the
//! section macros directly.

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_models {
    () => {

        // bloom_load_model  [source: curated; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_load_model(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_model", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => {
                        let eng = engine();
                        let $crate::engine::EngineState { ref mut models, ref mut renderer, .. } = *eng;
                        models.load_model_with_textures(&data, renderer)
                    }
                    Err(_) => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_load_model(_path_ptr: *const u8) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_load_model", "models3d");
            0.0
        }

        // bloom_draw_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_draw_model(handle: f64, x: f64, y: f64, z: f64, scale: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_draw_model", move || {
                let eng = engine();
                if let Some(model) = eng.models.get(handle) {
                    let tint = [(r / 255.0) as f32, (g / 255.0) as f32, (b / 255.0) as f32, (a / 255.0) as f32];
                    let position = [x as f32, y as f32, z as f32];
                    let handle_bits = handle.to_bits();
                    if eng.renderer.cache_model_if_static(handle_bits, &model.meshes) {
                        eng.renderer.draw_model_cached(handle_bits, position, scale as f32, tint);
                    } else {
                        // Skinned model (uncacheable): ONE staged pose per
                        // drawModel call, shared by every primitive. Letting
                        // each primitive pop its own pose starved the second
                        // primitive of multi-primitive models onto joint
                        // offset 0 — another model's matrices.
                        let joint_offset = eng.renderer.take_staged_skin_offset();
                        for mesh in &model.meshes {
                            let tex_idx = mesh.texture_idx.unwrap_or(0);
                            eng.renderer.draw_model_mesh_tinted_with_joints(&mesh.vertices, &mesh.indices, position, scale as f32, tint, tex_idx, joint_offset);
                        }
                    }
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_draw_model(_handle: f64, _x: f64, _y: f64, _z: f64, _scale: f64, _r: f64, _g: f64, _b: f64, _a: f64) {
            $crate::ffi::feature_off_warn_once("bloom_draw_model", "models3d");
        }

        // bloom_draw_model_rotated  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_draw_model_rotated(
            handle: f64, x: f64, y: f64, z: f64,
            scale: f64, rot_y: f64,
            color_packed_argb: f64,
        ) {
            $crate::ffi::guard("bloom_draw_model_rotated", move || {
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
                    let handle_bits = handle.to_bits();
                    // Static models go through the cached scene pipeline
                    // (alpha cutout, normal/MR maps, foliage wind +
                    // transmission, cutout shadows, planar reflections) —
                    // the immediate path below has none of that and used
                    // to render cutout foliage as opaque cards. Skinned
                    // models stay on the immediate fallback, matching
                    // bloom_draw_model.
                    if eng.renderer.cache_model_if_static(handle_bits, &model.meshes) {
                        eng.renderer.draw_model_cached_rotated(
                            handle_bits, position, scale, rot_y as f32, tint,
                        );
                    } else {
                        // Same one-pose-per-model rule as bloom_draw_model
                        // (see there) — primitives of a skinned model share
                        // the staged pose.
                        let joint_offset = eng.renderer.take_staged_skin_offset();
                        for mesh in &model.meshes {
                            let tex_idx = mesh.texture_idx.unwrap_or(0);
                            eng.renderer.draw_model_mesh_tinted_rotated_with_joints(
                                &mesh.vertices, &mesh.indices, position, scale, tint, tex_idx, rot_y as f32, joint_offset,
                            );
                        }
                    }
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_draw_model_rotated(_handle: f64, _x: f64, _y: f64, _z: f64, _scale: f64, _rot_y: f64, _color_packed_argb: f64) {
            $crate::ffi::feature_off_warn_once("bloom_draw_model_rotated", "models3d");
        }

        // bloom_unload_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_unload_model(handle: f64) {
            $crate::ffi::guard("bloom_unload_model", move || {
                let eng = engine();
                // Evict cached GPU meshes (keyed by handle bits) before the
                // handle dies — without this the buffers leak and a slot-
                // reusing future model would render the stale geometry.
                eng.renderer.evict_model_cache(handle.to_bits());
                eng.models.unload_model(handle);
            })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_unload_model(_handle: f64) {
            $crate::ffi::feature_off_warn_once("bloom_unload_model", "models3d");
        }

        // bloom_gen_mesh_cube  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_cube(w: f64, h: f64, d: f64) -> f64 {
            $crate::ffi::guard("bloom_gen_mesh_cube", move || {
                engine().models.gen_mesh_cube(w as f32, h as f32, d as f32)
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_cube(_w: f64, _h: f64, _d: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_gen_mesh_cube", "models3d");
            0.0
        }

        // bloom_gen_mesh_heightmap  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_heightmap(image_handle: f64, size_x: f64, size_y: f64, size_z: f64) -> f64 {
            $crate::ffi::guard("bloom_gen_mesh_heightmap", move || {
                let eng = engine();
                if let Some(img) = eng.textures.images.get(image_handle) {
                    let data = img.data.clone();
                    let w = img.width;
                    let h = img.height;
                    eng.models.gen_mesh_heightmap(&data, w, h, size_x as f32, size_y as f32, size_z as f32)
                } else {
                    0.0
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_heightmap(_image_handle: f64, _size_x: f64, _size_y: f64, _size_z: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_gen_mesh_heightmap", "models3d");
            0.0
        }

        // bloom_compile_material  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material(source) {
                    Ok(handle) => handle as f64,
                    Err(e) => {
                        eprintln!("[material] compile failed: {:?}", e);
                        0.0
                    }
                }
        })
        }

        // bloom_compile_material_refractive  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_refractive(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_refractive", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_with_options(
                    source, FragmentProfile::Translucent, Bucket::Refractive, true, false,
                ) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[refractive] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_compile_material_transparent  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_transparent(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_transparent", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_with_options(
                    source, FragmentProfile::Translucent, Bucket::Transparent, false, false,
                ) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_compile_material_additive  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_additive(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_additive", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_with_options(
                    source, FragmentProfile::Translucent, Bucket::Additive, false, false,
                ) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_compile_material_cutout  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_cutout(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_cutout", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_with_options(
                    source, FragmentProfile::Opaque, Bucket::Cutout, false, false,
                ) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[material] compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_compile_material_instanced  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_instanced(source_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_compile_material_instanced", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_instanced(source) {
                    Ok(handle) => handle as f64,
                    Err(e) => { eprintln!("[material] instanced compile failed: {:?}", e); 0.0 }
                }
        })
        }

        // bloom_submit_material_draw_instanced  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_submit_material_draw_instanced(
            material: f64, mesh_handle: f64, mesh_idx: f64,
            instance_buffer: f64, instance_count: f64,
        ) {
            $crate::ffi::guard("bloom_submit_material_draw_instanced", move || {
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
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_submit_material_draw_instanced(_material: f64, _mesh_handle: f64, _mesh_idx: f64, _instance_buffer: f64, _instance_count: f64) {
            $crate::ffi::feature_off_warn_once("bloom_submit_material_draw_instanced", "models3d");
        }

        // bloom_draw_material  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_draw_material(
            material: f64,
            mesh_handle: f64,
            mesh_idx: f64,
            x: f64, y: f64, z: f64, scale: f64,
            r: f64, g: f64, b: f64, a: f64,
        ) {
            $crate::ffi::guard("bloom_draw_material", move || {
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
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_draw_material(_material: f64, _mesh_handle: f64, _mesh_idx: f64, _x: f64, _y: f64, _z: f64, _scale: f64, _r: f64, _g: f64, _b: f64, _a: f64) {
            $crate::ffi::feature_off_warn_once("bloom_draw_material", "models3d");
        }

        // bloom_load_model_animation  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_load_model_animation(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_load_model_animation", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                match std::fs::read(path) {
                    Ok(data) => engine().models.load_model_animation(&data),
                    Err(_) => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_load_model_animation(_path_ptr: *const u8) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_load_model_animation", "models3d");
            0.0
        }

        // bloom_update_model_animation  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_update_model_animation(handle: f64, anim_index: f64, time: f64, scale: f64, px: f64, py: f64, pz: f64, rot_y: f64) {
            $crate::ffi::guard("bloom_update_model_animation", move || {
                // Take a single Y-axis angle (radians) instead of sin/cos, so the
                // engine reconstructs both with full precision + correct signs.
                // Older callers that passed (rot_sin, rot_cos) hit a Perry-ARM64
                // 9th-arg garbling bug AND a sqrt(1-sin²) workaround that lost
                // the sign of cos — model rotation was correct only on half the
                // circle. 8-arg signature dodges both issues. Matches macOS.
                let rot_y_f = rot_y as f32;
                let rot_sin = rot_y_f.sin();
                let rot_cos = rot_y_f.cos();
                let eng = engine();
                eng.models.update_model_animation(handle, anim_index as usize, time as f32);
                if let Some(anim) = eng.models.get_animation(handle) {
                    if !anim.joint_matrices.is_empty() {
                        eng.renderer.set_joint_matrices_scaled(&anim.joint_matrices, scale as f32, [px as f32, py as f32, pz as f32], rot_sin, rot_cos);
                    }
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_update_model_animation(_handle: f64, _anim_index: f64, _time: f64, _scale: f64, _px: f64, _py: f64, _pz: f64, _rot_y: f64) {
            $crate::ffi::feature_off_warn_once("bloom_update_model_animation", "models3d");
        }

        // bloom_create_mesh  [source: curated; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_create_mesh(vertex_ptr: *const f64, vertex_count: f64, index_ptr: *const f64, index_count: f64) -> f64 {
            $crate::ffi::guard("bloom_create_mesh", move || {
                // Perry arrays are pointers to inline f64 data (8-byte ArrayHeader
                // skipped by the compiler before the call) — see perry-codegen
                // lower_call/extern_func.rs. Reading f32/u32 here was a latent
                // misread on windows/android/ios/tvos before the FFI unification.
                if vertex_ptr.is_null() || index_ptr.is_null() { return 0.0; }
                let vcount = vertex_count as usize;
                let icount = index_count as usize;
                let vertex_f64 = unsafe { std::slice::from_raw_parts(vertex_ptr, vcount * 12) };
                let index_f64 = unsafe { std::slice::from_raw_parts(index_ptr, icount) };
                let vertices: Vec<f32> = vertex_f64.iter().map(|&v| v as f32).collect();
                let indices: Vec<u32> = index_f64.iter().map(|&v| v as u32).collect();
                engine().models.create_mesh(&vertices, &indices)
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_create_mesh(_vertex_ptr: *const f64, _vertex_count: f64, _index_ptr: *const f64, _index_count: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_create_mesh", "models3d");
            0.0
        }

        // Array-free mesh upload (Perry 0.5.1171 rejects number[] -> i64 pointer
        // params; see ModelManager::scratch_f32). Push the 12-float vertex
        // records + u32 indices one scalar at a time (all-f64 ABI), then build.
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_mesh_scratch_reset() {
            $crate::ffi::guard("bloom_mesh_scratch_reset", move || {
                engine().models.mesh_scratch_reset();
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_mesh_scratch_reset() {
            $crate::ffi::feature_off_warn_once("bloom_mesh_scratch_reset", "models3d");
        }

        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_mesh_scratch_push_f32(v: f64) {
            $crate::ffi::guard("bloom_mesh_scratch_push_f32", move || {
                engine().models.mesh_scratch_push_f32(v as f32);
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_mesh_scratch_push_f32(_v: f64) {
            $crate::ffi::feature_off_warn_once("bloom_mesh_scratch_push_f32", "models3d");
        }

        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_mesh_scratch_push_u32(v: f64) {
            $crate::ffi::guard("bloom_mesh_scratch_push_u32", move || {
                engine().models.mesh_scratch_push_u32(v as u32);
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_mesh_scratch_push_u32(_v: f64) {
            $crate::ffi::feature_off_warn_once("bloom_mesh_scratch_push_u32", "models3d");
        }

        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_create_mesh_scratch(vertex_count: f64, index_count: f64) -> f64 {
            $crate::ffi::guard("bloom_create_mesh_scratch", move || {
                engine().models.create_mesh_from_scratch(vertex_count as u32, index_count as u32)
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_create_mesh_scratch(_vertex_count: f64, _index_count: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_create_mesh_scratch", "models3d");
            0.0
        }

        // bloom_get_model_mesh_count  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_mesh_count(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_mesh_count", move || {
                match engine().models.get(handle) {
                    Some(model) => model.meshes.len() as f64,
                    None => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_mesh_count(_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_mesh_count", "models3d");
            0.0
        }

        // bloom_get_model_material_count  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_material_count(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_material_count", move || {
                match engine().models.get(handle) {
                    Some(model) => model.meshes.len() as f64,
                    None => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_material_count(_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_material_count", "models3d");
            0.0
        }

        // bloom_get_model_bounds_min_x  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_x(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_min_x", move || {
                engine().models.get_bounds(model_handle).0[0] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_x(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_min_x", "models3d");
            0.0
        }

        // bloom_get_model_bounds_min_y  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_y(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_min_y", move || {
                engine().models.get_bounds(model_handle).0[1] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_y(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_min_y", "models3d");
            0.0
        }

        // bloom_get_model_bounds_min_z  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_z(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_min_z", move || {
                engine().models.get_bounds(model_handle).0[2] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_min_z(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_min_z", "models3d");
            0.0
        }

        // bloom_get_model_bounds_max_x  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_x(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_max_x", move || {
                engine().models.get_bounds(model_handle).1[0] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_x(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_max_x", "models3d");
            0.0
        }

        // bloom_get_model_bounds_max_y  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_y(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_max_y", move || {
                engine().models.get_bounds(model_handle).1[1] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_y(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_max_y", "models3d");
            0.0
        }

        // bloom_get_model_bounds_max_z  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_z(model_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_get_model_bounds_max_z", move || {
                engine().models.get_bounds(model_handle).1[2] as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_get_model_bounds_max_z(_model_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_get_model_bounds_max_z", "models3d");
            0.0
        }

        // bloom_gen_mesh_spline_ribbon  [source: curated; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_spline_ribbon(points_ptr: *const f64, point_count: f64, widths_ptr: *const f64, width_count: f64) -> f64 {
            $crate::ffi::guard("bloom_gen_mesh_spline_ribbon", move || {
                // Perry arrays are f64 buffers (see bloom_create_mesh note).
                if points_ptr.is_null() || widths_ptr.is_null() { return 0.0; }
                let n = point_count as usize;
                let wn = width_count as usize;
                let points_f64 = unsafe { std::slice::from_raw_parts(points_ptr, n * 3) };
                let widths_f64 = unsafe { std::slice::from_raw_parts(widths_ptr, wn) };
                let points: Vec<f32> = points_f64.iter().map(|&v| v as f32).collect();
                let widths: Vec<f32> = widths_f64.iter().map(|&v| v as f32).collect();
                engine().models.gen_mesh_spline_ribbon(&points, &widths)
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_spline_ribbon(_points_ptr: *const f64, _point_count: f64, _widths_ptr: *const f64, _width_count: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_gen_mesh_spline_ribbon", "models3d");
            0.0
        }

        // bloom_stage_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_stage_model(path_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_stage_model", move || {
                let path = $crate::string_header::str_from_header(path_ptr);
                let path: &str = &bloom_resolve_asset_path(path);
                let data = match std::fs::read(path) {
                    Ok(d) => d,
                    Err(_) => return 0.0,
                };
                match $crate::models::load_gltf_staged(&data) {
                    Some(staged) => $crate::staging::stage_model(staged),
                    None => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_stage_model(_path_ptr: *const u8) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_stage_model", "models3d");
            0.0
        }

        // bloom_commit_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_commit_model(staging_handle: f64) -> f64 {
            $crate::ffi::guard("bloom_commit_model", move || {
                let staged = match $crate::staging::take_model(staging_handle) {
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
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_commit_model(_staging_handle: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_commit_model", "models3d");
            0.0
        }

        // bloom_compile_material_from_file  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_from_file(
            path_ptr: *const u8,
            bucket_kind: f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_compile_material_from_file", move || {
                use $crate::renderer::material_pipeline::{FragmentProfile, Bucket};
                let path = $crate::string_header::str_from_header(path_ptr);
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
        })
        }

        // bloom_set_material_params_scratch — params arrive via the mesh
        // scratch (bloom_mesh_scratch_reset + push_f32 × N) instead of the
        // pointer param below, which Perry 0.5.x rejects for JS arrays.
        // Same idiom as bloom_create_instance_buffer_scratch. ≤ 64 floats.
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_params_scratch(handle: f64, param_count: f64) {
            $crate::ffi::guard("bloom_set_material_params_scratch", move || {
                let eng = engine();
                let count = param_count as usize;
                if count > 64 {
                    eprintln!("[material] set_material_params_scratch: param_count {} > 64 (256-byte UBO cap)", count);
                    return;
                }
                if eng.models.scratch_f32.len() < count { return; }
                let mut bytes = vec![0u8; count * 4];
                for i in 0..count {
                    bytes[i*4..i*4+4].copy_from_slice(&eng.models.scratch_f32[i].to_le_bytes());
                }
                eng.models.mesh_scratch_reset();
                if let Err(e) = eng.renderer.material_system.set_user_params(
                    &eng.renderer.device, &eng.renderer.queue,
                    handle as u32, &bytes,
                ) {
                    eprintln!("[material] set_material_params_scratch failed: {}", e);
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_params_scratch(_handle: f64, _param_count: f64) {
            $crate::ffi::feature_off_warn_once("bloom_set_material_params_scratch", "models3d");
        }

        // bloom_set_material_params  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_params(
            handle: f64,
            params_ptr: *const f64,
            param_count: f64,
        ) {
            $crate::ffi::guard("bloom_set_material_params", move || {
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
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_set_material_params(_handle: f64, _params_ptr: *const f64, _param_count: f64) {
            $crate::ffi::feature_off_warn_once("bloom_set_material_params", "models3d");
        }

    };
}
