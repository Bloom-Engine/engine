//! Retained scene graph, picking, and screen projection.
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi) — see
//! `ffi_core/mod.rs` for the architecture and the invoking-crate contract.
//! Internal: platform crates must invoke `define_core_ffi!()`, never the
//! section macros directly.

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_scene {
    () => {

        // bloom_scene_create_node  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_create_node() -> f64 {
            $crate::ffi::guard("bloom_scene_create_node", move || {
                engine().scene.create_node()
        })
        }

        // bloom_scene_destroy_node  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_destroy_node(handle: f64) {
            $crate::ffi::guard("bloom_scene_destroy_node", move || {
                engine().scene.destroy_node(handle);
        })
        }

        // bloom_scene_set_visible  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_visible(handle: f64, visible: f64) {
            $crate::ffi::guard("bloom_scene_set_visible", move || {
                engine().scene.set_visible(handle, visible != 0.0);
        })
        }

        // bloom_scene_set_trs — position + Y-rotation + uniform scale as six
        // f64 scalars. The full-matrix bloom_scene_set_transform passes a JS
        // array into an i64 pointer param, which Perry 0.5.x rejects; this
        // covers the common placement case without touching that ABI.
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_trs(handle: f64, px: f64, py: f64, pz: f64, yaw: f64, scale: f64) {
            $crate::ffi::guard("bloom_scene_set_trs", move || {
                engine().scene.set_trs(handle, px as f32, py as f32, pz as f32, yaw as f32, scale as f32);
        })
        }

        // bloom_scene_set_cast_shadow  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_cast_shadow(handle: f64, cast: f64) {
            $crate::ffi::guard("bloom_scene_set_cast_shadow", move || {
                engine().scene.set_cast_shadow(handle, cast != 0.0);
        })
        }

        // bloom_scene_set_gi_only — mark a node as a GI proxy: it feeds
        // BLAS/TLAS, mesh cards, and the SDF clipmap (SSGI off-screen
        // bounce) but is skipped by the main render, planar reflections,
        // and the sun-shadow pass. For material-system games whose world
        // never becomes scene nodes.
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_gi_only(handle: f64, gi_only: f64) {
            $crate::ffi::guard("bloom_scene_set_gi_only", move || {
                engine().scene.set_gi_only(handle, gi_only != 0.0);
        })
        }

        // bloom_scene_set_receive_shadow  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_receive_shadow(handle: f64, receive: f64) {
            $crate::ffi::guard("bloom_scene_set_receive_shadow", move || {
                engine().scene.set_receive_shadow(handle, receive != 0.0);
        })
        }

        // bloom_scene_set_parent  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_parent(handle: f64, parent: f64) {
            $crate::ffi::guard("bloom_scene_set_parent", move || {
                engine().scene.set_parent(handle, parent);
        })
        }

        // bloom_scene_set_transform  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_transform(handle: f64, mat_ptr: *const f64) {
            $crate::ffi::guard("bloom_scene_set_transform", move || {
                if mat_ptr.is_null() { return; }
                let slice = unsafe { std::slice::from_raw_parts(mat_ptr, 16) };
                let mut mat = [[0.0f32; 4]; 4];
                for col in 0..4 {
                    for row in 0..4 {
                        mat[col][row] = slice[col * 4 + row] as f32;
                    }
                }
                engine().scene.set_transform(handle, mat);
        })
        }

        // bloom_scene_update_geometry  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_update_geometry(
            handle: f64,
            vert_ptr: *const f64,
            vert_count: f64,
            idx_ptr: *const f64,
            idx_count: f64,
        ) {
            $crate::ffi::guard("bloom_scene_update_geometry", move || {
                if vert_ptr.is_null() || idx_ptr.is_null() { return; }
                let nv = vert_count as usize;
                let ni = idx_count as usize;

                let vert_floats = unsafe { std::slice::from_raw_parts(vert_ptr, nv * 12) };
                let idx_floats = unsafe { std::slice::from_raw_parts(idx_ptr, ni) };

                let mut vertices = Vec::with_capacity(nv);
                for i in 0..nv {
                    let base = i * 12;
                    vertices.push($crate::renderer::Vertex3D {
                        position: [vert_floats[base] as f32, vert_floats[base+1] as f32, vert_floats[base+2] as f32],
                        normal: [vert_floats[base+3] as f32, vert_floats[base+4] as f32, vert_floats[base+5] as f32],
                        color: [vert_floats[base+6] as f32, vert_floats[base+7] as f32, vert_floats[base+8] as f32, vert_floats[base+9] as f32],
                        uv: [vert_floats[base+10] as f32, vert_floats[base+11] as f32],
                        joints: [0.0; 4],
                        weights: [0.0; 4],
                        tangent: [0.0; 4],
                    });
                }

                let indices: Vec<u32> = idx_floats.iter().map(|&v| v as u32).collect();

                engine().scene.update_geometry(handle, vertices, indices);
        })
        }

        // bloom_scene_set_lod — reduced-detail variant for a node. Same
        // 12-float vertex layout as bloom_scene_update_geometry; the base
        // geometry is the finest level. max_coverage = screen-coverage
        // threshold below which this level renders.
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_lod(
            handle: f64,
            lod_index: f64,
            vert_ptr: *const f64,
            vert_count: f64,
            idx_ptr: *const f64,
            idx_count: f64,
            max_coverage: f64,
        ) {
            $crate::ffi::guard("bloom_scene_set_lod", move || {
                if vert_ptr.is_null() || idx_ptr.is_null() { return; }
                let nv = vert_count as usize;
                let ni = idx_count as usize;
                let vert_floats = unsafe { std::slice::from_raw_parts(vert_ptr, nv * 12) };
                let idx_floats = unsafe { std::slice::from_raw_parts(idx_ptr, ni) };
                let mut vertices = Vec::with_capacity(nv);
                for i in 0..nv {
                    let base = i * 12;
                    vertices.push($crate::renderer::Vertex3D {
                        position: [vert_floats[base] as f32, vert_floats[base+1] as f32, vert_floats[base+2] as f32],
                        normal: [vert_floats[base+3] as f32, vert_floats[base+4] as f32, vert_floats[base+5] as f32],
                        color: [vert_floats[base+6] as f32, vert_floats[base+7] as f32, vert_floats[base+8] as f32, vert_floats[base+9] as f32],
                        uv: [vert_floats[base+10] as f32, vert_floats[base+11] as f32],
                        joints: [0.0; 4],
                        weights: [0.0; 4],
                        tangent: [0.0; 4],
                    });
                }
                let indices: Vec<u32> = idx_floats.iter().map(|&v| v as u32).collect();
                engine().scene.set_lod_geometry(handle, lod_index as usize, vertices, indices, max_coverage as f32);
        })
        }

        // bloom_scene_attach_model_lod — model mesh as a reduced-detail
        // variant (LOD counterpart of bloom_scene_attach_model).
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_scene_attach_model_lod(node_handle: f64, model_handle: f64, mesh_index: f64, lod_index: f64, max_coverage: f64) {
            $crate::ffi::guard("bloom_scene_attach_model_lod", move || {
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
                eng.scene.set_lod_geometry(node_handle, lod_index as usize, vertices, indices, max_coverage as f32);
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_scene_attach_model_lod(_node_handle: f64, _model_handle: f64, _mesh_index: f64, _lod_index: f64, _max_coverage: f64) {
            $crate::ffi::feature_off_warn_once("bloom_scene_attach_model_lod", "models3d");
        }

        // bloom_scene_set_material_color  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_material_color(handle: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_scene_set_material_color", move || {
                engine().scene.set_material_color(handle, r as f32, g as f32, b as f32, a as f32);
        })
        }

        // bloom_scene_set_material_pbr  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_material_pbr(handle: f64, roughness: f64, metalness: f64) {
            $crate::ffi::guard("bloom_scene_set_material_pbr", move || {
                engine().scene.set_material_pbr(handle, roughness as f32, metalness as f32);
        })
        }

        // bloom_scene_set_material_texture  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_material_texture(handle: f64, texture_idx: f64) {
            $crate::ffi::guard("bloom_scene_set_material_texture", move || {
                engine().scene.set_material_texture(handle, texture_idx as u32);
        })
        }

        // bloom_scene_node_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_node_count() -> f64 {
            $crate::ffi::guard("bloom_scene_node_count", move || {
                engine().scene.node_count() as f64
        })
        }

        // bloom_scene_node_vertex_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_node_vertex_count(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_node_vertex_count", move || {
                match engine().scene.nodes.get(handle) {
                    Some(node) => node.vertices.len() as f64,
                    None => -1.0,
                }
        })
        }

        // bloom_scene_node_index_count  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_node_index_count(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_node_index_count", move || {
                match engine().scene.nodes.get(handle) {
                    Some(node) => node.indices.len() as f64,
                    None => -1.0,
                }
        })
        }

        // bloom_scene_pick_all  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_pick_all(screen_x: f64, screen_y: f64, max_results: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_pick_all", move || {
                let eng = engine();
                let inv_vp = eng.renderer.inverse_vp_matrix();
                let cam_pos = eng.renderer.camera_pos();
                let w = eng.renderer.width() as f32;
                let h = eng.renderer.height() as f32;
                let (origin, direction) = $crate::picking::screen_to_ray(
                    screen_x as f32, screen_y as f32, w, h, &inv_vp, &cam_pos,
                );
                let results = $crate::picking::raycast_scene_all(&eng.scene, &origin, &direction, max_results as usize);
                let count = results.len();
                eng.last_pick_all = results;
                count as f64
        })
        }

        // bloom_pick_all_handle  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_all_handle(index: f64) -> f64 {
            $crate::ffi::guard("bloom_pick_all_handle", move || {
                let i = index as usize;
                engine().last_pick_all.get(i).map(|r| r.handle).unwrap_or(0.0)
        })
        }

        // bloom_pick_all_distance  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_all_distance(index: f64) -> f64 {
            $crate::ffi::guard("bloom_pick_all_distance", move || {
                let i = index as usize;
                engine().last_pick_all.get(i).map(|r| r.distance as f64).unwrap_or(0.0)
        })
        }

        // bloom_scene_set_material_water  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_material_water(handle: f64, wave_amp: f64, wave_speed: f64, r: f64, g: f64, b: f64, a: f64) {
            $crate::ffi::guard("bloom_scene_set_material_water", move || {
                engine().scene.set_material_water(handle, wave_amp as f32, wave_speed as f32, r as f32, g as f32, b as f32, a as f32);
        })
        }

        // bloom_scene_get_transform  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_transform(handle: f64, index: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_transform", move || {
                let mat = engine().scene.get_transform(handle);
                let i = index as usize;
                let col = i / 4;
                let row = i % 4;
                if col < 4 && row < 4 { mat[col][row] as f64 } else { 0.0 }
        })
        }

        // bloom_scene_get_bounds_min_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_min_x(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_min_x", move || {     engine().scene.get_bounds(handle).0[0] as f64 })
        }

        // bloom_scene_get_bounds_min_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_min_y(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_min_y", move || {     engine().scene.get_bounds(handle).0[1] as f64 })
        }

        // bloom_scene_get_bounds_min_z  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_min_z(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_min_z", move || {     engine().scene.get_bounds(handle).0[2] as f64 })
        }

        // bloom_scene_get_bounds_max_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_max_x(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_max_x", move || {     engine().scene.get_bounds(handle).1[0] as f64 })
        }

        // bloom_scene_get_bounds_max_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_max_y(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_max_y", move || {     engine().scene.get_bounds(handle).1[1] as f64 })
        }

        // bloom_scene_get_bounds_max_z  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_bounds_max_z(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_bounds_max_z", move || {     engine().scene.get_bounds(handle).1[2] as f64 })
        }

        // bloom_scene_set_user_data  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_set_user_data(handle: f64, data: f64) {
            $crate::ffi::guard("bloom_scene_set_user_data", move || {
                engine().scene.set_user_data(handle, data as i64);
        })
        }

        // bloom_scene_get_user_data  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_get_user_data(handle: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_get_user_data", move || {
                engine().scene.get_user_data(handle) as f64
        })
        }

        // bloom_scene_extrude_polygon  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_extrude_polygon(
            handle: f64,
            polygon_ptr: *const f64,
            polygon_count: f64,
            depth: f64,
        ) {
            $crate::ffi::guard("bloom_scene_extrude_polygon", move || {
                if polygon_ptr.is_null() { return; }
                let n = polygon_count as usize;
                let polygon = unsafe { std::slice::from_raw_parts(polygon_ptr, n * 2) };

                let geo = $crate::geometry::extrude_polygon(polygon, &[], depth);
                engine().scene.update_geometry(handle, geo.vertices, geo.indices);
        })
        }

        // bloom_scene_subtract_box  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_subtract_box(
            handle: f64,
            min_x: f64, min_y: f64, min_z: f64,
            max_x: f64, max_y: f64, max_z: f64,
        ) {
            $crate::ffi::guard("bloom_scene_subtract_box", move || {
                let eng = engine();
                if let Some(node) = eng.scene.nodes.get(handle) {
                    let current = $crate::geometry::GeometryData {
                        vertices: node.vertices.clone(),
                        indices: node.indices.clone(),
                    };
                    let result = $crate::geometry::subtract_box(
                        &current,
                        [min_x as f32, min_y as f32, min_z as f32],
                        [max_x as f32, max_y as f32, max_z as f32],
                    );
                    eng.scene.update_geometry(handle, result.vertices, result.indices);
                }
        })
        }

        // bloom_scene_attach_model  [source: linux; gated: models3d]
        #[cfg(feature = "models3d")]
        #[no_mangle]
        pub extern "C" fn bloom_scene_attach_model(node_handle: f64, model_handle: f64, mesh_index: f64) {
            $crate::ffi::guard("bloom_scene_attach_model", move || {
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
                let roughness_factor = mesh.roughness_factor;
                let metallic_factor = mesh.metallic_factor;
                let alpha_cutoff = mesh.alpha_cutoff;
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
                // Carry the glTF material factors + MASK cutoff too —
                // previously dropped, which left attached foliage opaque
                // (solid cards) and every attached mesh at the default
                // roughness 0.8 regardless of its authored material.
                eng.scene.set_material_pbr(node_handle, roughness_factor, metallic_factor);
                eng.scene.set_material_alpha_cutoff(node_handle, alpha_cutoff);
        })
        }
        #[cfg(not(feature = "models3d"))]
        #[no_mangle]
        pub extern "C" fn bloom_scene_attach_model(_node_handle: f64, _model_handle: f64, _mesh_index: f64) {
            $crate::ffi::feature_off_warn_once("bloom_scene_attach_model", "models3d");
        }

        // bloom_project_to_screen  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_project_to_screen(wx: f64, wy: f64, wz: f64) -> f64 {
            $crate::ffi::guard("bloom_project_to_screen", move || {
                let eng = engine();
                let vp = eng.renderer.vp_matrix();
                let w = eng.renderer.width() as f32;
                let h = eng.renderer.height() as f32;

                // Multiply by VP matrix
                let x = wx as f32;
                let y = wy as f32;
                let z = wz as f32;
                let clip_x = vp[0][0]*x + vp[1][0]*y + vp[2][0]*z + vp[3][0];
                let clip_y = vp[0][1]*x + vp[1][1]*y + vp[2][1]*z + vp[3][1];
                let clip_w = vp[0][3]*x + vp[1][3]*y + vp[2][3]*z + vp[3][3];

                if clip_w <= 0.0 {
                    engine().last_project = (-9999.0, -9999.0);
                    return -9999.0;
                }

                // NDC to screen
                let ndc_x = clip_x / clip_w;
                let ndc_y = clip_y / clip_w;
                let screen_x = ((ndc_x + 1.0) * 0.5 * w) as f64;
                let screen_y = ((1.0 - ndc_y) * 0.5 * h) as f64;

                engine().last_project = (screen_x, screen_y);
                screen_x
        })
        }

        // bloom_project_screen_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_project_screen_y() -> f64 {
            $crate::ffi::guard("bloom_project_screen_y", move || {
                engine().last_project.1
        })
        }

        // bloom_scene_pick  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_scene_pick(screen_x: f64, screen_y: f64) -> f64 {
            $crate::ffi::guard("bloom_scene_pick", move || {
                let eng = engine();
                let inv_vp = eng.renderer.inverse_vp_matrix();
                let cam_pos = eng.renderer.camera_pos();
                let w = eng.renderer.width() as f32;
                let h = eng.renderer.height() as f32;

                let (origin, direction) = $crate::picking::screen_to_ray(
                    screen_x as f32, screen_y as f32,
                    w, h, &inv_vp, &cam_pos,
                );

                let result = $crate::picking::raycast_scene(&eng.scene, &origin, &direction);
                let hit = result.hit;
                engine().last_pick = Some(result);
                if hit { 1.0 } else { 0.0 }
        })
        }

        // bloom_pick_hit_handle  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_handle() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_handle", move || {
                engine().last_pick.as_ref().map(|r| r.handle).unwrap_or(0.0)
        })
        }

        // bloom_pick_hit_distance  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_distance() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_distance", move || {
                engine().last_pick.as_ref().map(|r| r.distance as f64).unwrap_or(0.0)
        })
        }

        // bloom_pick_hit_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_x() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_x", move || {
                engine().last_pick.as_ref().map(|r| r.point[0] as f64).unwrap_or(0.0)
        })
        }

        // bloom_pick_hit_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_y() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_y", move || {
                engine().last_pick.as_ref().map(|r| r.point[1] as f64).unwrap_or(0.0)
        })
        }

        // bloom_pick_hit_z  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_z() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_z", move || {
                engine().last_pick.as_ref().map(|r| r.point[2] as f64).unwrap_or(0.0)
        })
        }

        // bloom_pick_hit_normal_x  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_normal_x() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_normal_x", move || {
                engine().last_pick.as_ref().map(|r| r.normal[0] as f64).unwrap_or(0.0)
        })
        }

        // bloom_pick_hit_normal_y  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_normal_y() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_normal_y", move || {
                engine().last_pick.as_ref().map(|r| r.normal[1] as f64).unwrap_or(0.0)
        })
        }

        // bloom_pick_hit_normal_z  [source: macos]
        #[no_mangle]
        pub extern "C" fn bloom_pick_hit_normal_z() -> f64 {
            $crate::ffi::guard("bloom_pick_hit_normal_z", move || {
                engine().last_pick.as_ref().map(|r| r.normal[2] as f64).unwrap_or(0.0)
        })
        }

    };
}
