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
                        // Skinned models cache too now (bind-pose VB with raw
                        // joint indices, skinned in the scene VS) — routed to
                        // the skinned cached draw, which pops the staged pose
                        // ONCE for the whole model and shares the offset
                        // across every primitive (per-primitive pops starved
                        // multi-primitive models onto joint offset 0 —
                        // another model's matrices).
                        if eng.renderer.is_model_skinned(handle_bits) {
                            eng.renderer.draw_model_cached_skinned(handle_bits, position, scale as f32, tint);
                        } else {
                            eng.renderer.draw_model_cached(handle_bits, position, scale as f32, tint);
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
                    // the old immediate path had none of that and used
                    // to render cutout foliage as opaque cards. Skinned
                    // models take the skinned cached draw and IGNORE the
                    // rotation — their joint matrices bake orientation,
                    // exactly as the old immediate fallback behaved.
                    if eng.renderer.cache_model_if_static(handle_bits, &model.meshes) {
                        if eng.renderer.is_model_skinned(handle_bits) {
                            eng.renderer.draw_model_cached_skinned(
                                handle_bits, position, scale, tint,
                            );
                        } else {
                            eng.renderer.draw_model_cached_rotated(
                                handle_bits, position, scale, rot_y as f32, tint,
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

        // bloom_draw_model_transform16 — EN-039. [gated: models3d]
        //
        // The immediate-mode twin of bloom_scene_set_transform16: a full
        // column-major 4x4, so an immediate draw can finally pitch and roll.
        // bloom_draw_model_rotated only takes a Y rotation, which is why the
        // shooter's gun cannot tilt with the aim.
        //
        // The ticket proposed routing this through the mesh scratch on the
        // grounds that "Perry can't pass 16 f64 args". It can:
        // bloom_scene_set_transform16 already passes 17 and works. Spelling the
        // matrix out keeps this STATELESS — no scratch to reset, no ordering
        // hazard between a reset and a draw — which is the same reasoning
        // recorded on the scene twin.
        //
        // SKINNED models are rejected rather than silently mis-drawn: their
        // joint matrices already bake world orientation, so a caller-supplied
        // model matrix would double-transform them. bloom_draw_model_rotated
        // has the same carve-out (it ignores rot_y for skinned).
        #[cfg(feature = "models3d")]
        #[no_mangle]
        #[allow(clippy::too_many_arguments)]
        pub extern "C" fn bloom_draw_model_transform16(
            handle: f64,
            m0: f64, m1: f64, m2: f64, m3: f64,
            m4: f64, m5: f64, m6: f64, m7: f64,
            m8: f64, m9: f64, m10: f64, m11: f64,
            m12: f64, m13: f64, m14: f64, m15: f64,
            color_packed_argb: f64,
        ) {
            $crate::ffi::guard("bloom_draw_model_transform16", move || {
                let bits = color_packed_argb as u32;
                let a = ((bits >> 24) & 0xff) as f32 / 255.0;
                let r = ((bits >> 16) & 0xff) as f32 / 255.0;
                let g = ((bits >>  8) & 0xff) as f32 / 255.0;
                let b = ( bits        & 0xff) as f32 / 255.0;
                let s = [
                    m0, m1, m2, m3,
                    m4, m5, m6, m7,
                    m8, m9, m10, m11,
                    m12, m13, m14, m15,
                ];
                let mut mat = [[0.0f32; 4]; 4];
                for col in 0..4 {
                    for row in 0..4 {
                        mat[col][row] = s[col * 4 + row] as f32;
                    }
                }
                let eng = engine();
                if let Some(model) = eng.models.get(handle) {
                    let handle_bits = handle.to_bits();
                    if eng.renderer.cache_model_if_static(handle_bits, &model.meshes) {
                        if eng.renderer.is_model_skinned(handle_bits) {
                            return;   // see the note above
                        }
                        eng.renderer.draw_model_cached_transform(
                            handle_bits, mat, [r, g, b, a],
                        );
                    }
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        #[allow(clippy::too_many_arguments)]
        pub extern "C" fn bloom_draw_model_transform16(
            _handle: f64,
            _m0: f64, _m1: f64, _m2: f64, _m3: f64,
            _m4: f64, _m5: f64, _m6: f64, _m7: f64,
            _m8: f64, _m9: f64, _m10: f64, _m11: f64,
            _m12: f64, _m13: f64, _m14: f64, _m15: f64,
            _color_packed_argb: f64,
        ) {
            $crate::ffi::feature_off_warn_once("bloom_draw_model_transform16", "models3d");
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

        // bloom_compile_material_instanced_bucket  [EN-026/027]
        // bucket: 0 = opaque, 1 = cutout, 2 = additive, 3 = transparent.
        // reads_scene: bind the scene colour/depth snapshot (soft particles).
        // The plain instanced compile above is hardcoded to opaque, which is
        // right for grass and wrong for the two things that most want
        // instancing: particles (additive) and decals (cutout).
        #[no_mangle]
        pub extern "C" fn bloom_compile_material_instanced_bucket(
            source_ptr: *const u8, bucket: f64, reads_scene: f64,
        ) -> f64 {
            $crate::ffi::guard("bloom_compile_material_instanced_bucket", move || {
                let source = $crate::string_header::str_from_header(source_ptr);
                match engine().renderer.compile_material_instanced_bucket(
                    source, bucket as u32, reads_scene != 0.0)
                {
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
                        // PT-7: the anim handle keys the prev-palette
                        // pairing for skinned motion vectors.
                        eng.renderer.set_joint_matrices_scaled(handle.to_bits(), &anim.joint_matrices, scale as f32, [px as f32, py as f32, pz as f32], rot_sin, rot_cos);
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

        // Array-free scene-geometry upload. Same motivation as the mesh scratch
        // above: `bloom_scene_update_geometry` takes `i64` pointers, which
        // Perry cannot produce from a `number[]`. Push the vertex floats and
        // u32 indices through the mesh scratch, then re-upload them into an
        // existing scene node (the editor's terrain re-meshes every brush
        // stroke this way).
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_scene_update_geometry_scratch(
            handle: f64,
            vertex_count: f64,
            index_count: f64,
        ) {
            $crate::ffi::guard("bloom_scene_update_geometry_scratch", move || {
                let taken = engine()
                    .models
                    .take_scratch_geometry(vertex_count as u32, index_count as u32);
                let (vert_floats, indices) = match taken {
                    Some(v) => v,
                    None => return,
                };
                let nv = vertex_count as usize;
                let mut vertices = Vec::with_capacity(nv);
                for i in 0..nv {
                    let base = i * 12;
                    vertices.push($crate::renderer::Vertex3D {
                        position: [vert_floats[base], vert_floats[base + 1], vert_floats[base + 2]],
                        normal: [vert_floats[base + 3], vert_floats[base + 4], vert_floats[base + 5]],
                        color: [
                            vert_floats[base + 6],
                            vert_floats[base + 7],
                            vert_floats[base + 8],
                            vert_floats[base + 9],
                        ],
                        uv: [vert_floats[base + 10], vert_floats[base + 11]],
                        joints: [0.0; 4],
                        weights: [0.0; 4],
                        tangent: [0.0; 4],
                    });
                }
                engine().scene.update_geometry(handle, vertices, indices);
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_scene_update_geometry_scratch(
            _handle: f64,
            _vertex_count: f64,
            _index_count: f64,
        ) {
            $crate::ffi::feature_off_warn_once("bloom_scene_update_geometry_scratch", "models3d");
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

        // Array-free spline ribbon. Same reason as the mesh scratch above: the
        // pointer form is unreachable from TypeScript (Perry won't pass a
        // `number[]` into an i64 param). Push `point_count * 3` position floats
        // followed by `width_count` width floats into the mesh scratch, then
        // call this.
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_spline_ribbon_scratch(point_count: f64, width_count: f64) -> f64 {
            $crate::ffi::guard("bloom_gen_mesh_spline_ribbon_scratch", move || {
                let n = point_count as usize;
                let wn = width_count as usize;
                let scratch = engine().models.scratch_floats();
                if n < 2 || wn == 0 || scratch.len() < n * 3 + wn {
                    return 0.0;
                }
                let points: Vec<f32> = scratch[..n * 3].to_vec();
                let widths: Vec<f32> = scratch[n * 3..n * 3 + wn].to_vec();
                engine().models.gen_mesh_spline_ribbon(&points, &widths)
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_gen_mesh_spline_ribbon_scratch(_point_count: f64, _width_count: f64) -> f64 {
            $crate::ffi::feature_off_warn_once("bloom_gen_mesh_spline_ribbon_scratch", "models3d");
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
                // Every other asset loader routes through the platform's
                // resolve hook; this one read the raw relative path straight
                // off the working directory, which is only the asset root on
                // the desktop hosts. On iOS the CWD is not the app bundle, so
                // every from-file material failed to canonicalize and the whole
                // scene lost its shaders.
                let path: &str = &bloom_resolve_asset_path(path);
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

        // ==================================================================
        // EN-028 — animation mixer (crossfade + masked layer + root motion)
        // EN-033 — bone sockets
        //
        // All numeric ABI, <= 8 f64 args per call (the 9th arg garbles on
        // Perry/ARM64 — see bloom_update_model_animation's note).
        // ==================================================================

        // bloom_anim_play  [gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_anim_play(handle: f64, clip: f64, fade: f64, speed: f64, looping: f64) {
            $crate::ffi::guard("bloom_anim_play", move || {
                engine().models.anim_play(
                    handle, clip as usize, fade as f32, speed as f32, looping != 0.0);
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_anim_play(_h: f64, _c: f64, _f: f64, _s: f64, _l: f64) {
            $crate::ffi::feature_off_warn_once("bloom_anim_play", "models3d");
        }

        // bloom_anim_set_layer  [gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_anim_set_layer(handle: f64, clip: f64, weight: f64, mask_root: f64, speed: f64, looping: f64) {
            $crate::ffi::guard("bloom_anim_set_layer", move || {
                engine().models.anim_set_layer(
                    handle, clip as i32, weight as f32, mask_root as i32,
                    speed as f32, looping != 0.0);
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_anim_set_layer(_h: f64, _c: f64, _w: f64, _m: f64, _s: f64, _l: f64) {
            $crate::ffi::feature_off_warn_once("bloom_anim_set_layer", "models3d");
        }

        // bloom_anim_set_root_motion  [gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_anim_set_root_motion(handle: f64, on: f64) {
            $crate::ffi::guard("bloom_anim_set_root_motion", move || {
                engine().models.anim_set_root_motion(handle, on != 0.0);
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_anim_set_root_motion(_h: f64, _o: f64) {
            $crate::ffi::feature_off_warn_once("bloom_anim_set_root_motion", "models3d");
        }

        // bloom_anim_update  [gated: models3d]
        // Advances every clock on the model by dt, rebuilds the blended pose,
        // and uploads the joint matrices — the mixer-driven replacement for
        // bloom_update_model_animation (which stays for single-clip callers).
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_anim_update(handle: f64, dt: f64, scale: f64, px: f64, py: f64, pz: f64, rot_y: f64) {
            $crate::ffi::guard("bloom_anim_update", move || {
                let rot_y_f = rot_y as f32;
                let rot_sin = rot_y_f.sin();
                let rot_cos = rot_y_f.cos();
                let eng = engine();
                eng.models.advance_and_update(handle, dt as f32);
                if let Some(anim) = eng.models.get_animation(handle) {
                    if !anim.joint_matrices.is_empty() {
                        // PT-7: anim handle = prev-palette pairing key.
                        eng.renderer.set_joint_matrices_scaled(
                            handle.to_bits(), &anim.joint_matrices, scale as f32,
                            [px as f32, py as f32, pz as f32], rot_sin, rot_cos);
                    }
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_anim_update(_h: f64, _d: f64, _s: f64, _x: f64, _y: f64, _z: f64, _r: f64) {
            $crate::ffi::feature_off_warn_once("bloom_anim_update", "models3d");
        }

        // bloom_anim_finished  [gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_anim_finished(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_anim_finished", move || {
                if engine().models.anim_finished(handle) { 1.0 } else { 0.0 }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_anim_finished(_h: f64) -> f64 { 1.0 }

        // bloom_anim_clip_duration  [gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_anim_clip_duration(handle: f64, clip: f64) -> f64 {
            $crate::ffi::guard("bloom_anim_clip_duration", move || {
                engine().models.anim_clip_duration(handle, clip as usize) as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_anim_clip_duration(_h: f64, _c: f64) -> f64 { 0.0 }

        // bloom_anim_root_delta  [gated: models3d]  axis: 0=x 1=y 2=z
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_anim_root_delta(handle: f64, axis: f64) -> f64 {
            $crate::ffi::guard("bloom_anim_root_delta", move || {
                let d = engine().models.anim_root_delta(handle);
                let i = axis as usize;
                if i < 3 { d[i] as f64 } else { 0.0 }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_anim_root_delta(_h: f64, _a: f64) -> f64 { 0.0 }

        // bloom_model_find_joint  [gated: models3d]
        // Load-time only (it parses a string) — cache the index, per
        // perry-quirks #5.
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_model_find_joint(handle: f64, name_ptr: *const u8) -> f64 {
            $crate::ffi::guard("bloom_model_find_joint", move || {
                let name = $crate::string_header::str_from_header(name_ptr);
                engine().models.find_joint(handle, &name) as f64
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_model_find_joint(_h: f64, _n: *const u8) -> f64 { -1.0 }

        // bloom_model_joint_world  [gated: models3d]
        // Model-space 4x4 of a joint, column-major, component 0..15.
        // Translation is components 12/13/14. The caller applies the same
        // (scale, position, yaw) it passed to bloom_anim_update to lift this
        // into world space — the engine deliberately does not, so a socket
        // costs no extra state.
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_model_joint_world(handle: f64, joint: f64, comp: f64) -> f64 {
            $crate::ffi::guard("bloom_model_joint_world", move || {
                let j = joint as i64;
                let c = comp as usize;
                if j < 0 || c > 15 { return 0.0; }
                match engine().models.joint_world(handle, j as usize) {
                    Some(m) => m[c / 4][c % 4] as f64,
                    None => 0.0,
                }
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_model_joint_world(_h: f64, _j: f64, _c: f64) -> f64 { 0.0 }

    };
}
