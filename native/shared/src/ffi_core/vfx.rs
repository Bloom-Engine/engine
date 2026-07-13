//! EN-026 particles + EN-027 decals.
//!
//! Section of [`define_core_ffi!`](crate::define_core_ffi) — see
//! `ffi_core/mod.rs` for the architecture and the invoking-crate contract.
//!
//! Both subsystems follow the same shape: the engine simulates and owns a
//! dynamic instance buffer; the game supplies the material + quad mesh and
//! issues one instanced draw per system. The FFI traffic is therefore
//! proportional to *events* (a burst, an impact), never to the number of live
//! particles — which is the whole reason the sim is native (a 2 000-particle
//! pool pushed float-by-float across the FFI would be ~24 000 calls a frame).

#[doc(hidden)]
#[macro_export]
macro_rules! __bloom_ffi_vfx {
    () => {

        // ---- EN-026 particles ------------------------------------------

        // bloom_particles_create — pool + dynamic instance buffer.
        // Returns a 1-based system handle (0 = failure).
        #[no_mangle]
        pub extern "C" fn bloom_particles_create(capacity: f64) -> f64 {
            $crate::ffi::guard("bloom_particles_create", move || {
                let cap = (capacity as usize).clamp(1, 100_000);
                let eng = engine();
                let ib = eng.renderer.material_system.create_dynamic_instance_buffer(
                    &eng.renderer.device, cap as u32);
                eng.particles.create(cap, ib) as f64
        })
        }

        // bloom_particles_configure — behaviour, pushed through the mesh
        // scratch (same idiom as create_instance_buffer_scratch: Perry 0.5.x
        // rejects JS arrays in i64 pointer params). Startup-only, so the
        // per-float call cost does not matter. See particles.rs for the slot
        // order.
        #[no_mangle]
        pub extern "C" fn bloom_particles_configure(sys: f64) {
            $crate::ffi::guard("bloom_particles_configure", move || {
                let eng = engine();
                #[cfg(feature = "models3d")]
                {
                    let params: Vec<f32> = eng.models.scratch_f32.clone();
                    eng.models.mesh_scratch_reset();
                    if let Some(s) = eng.particles.get_mut(sys as u32) {
                        s.configure_from_slice(&params);
                    }
                }
                #[cfg(not(feature = "models3d"))]
                { let _ = sys; }
        })
        }

        // bloom_particles_emit — one burst. 8 args (the ARM64 ceiling).
        #[no_mangle]
        pub extern "C" fn bloom_particles_emit(
            sys: f64, x: f64, y: f64, z: f64,
            dx: f64, dy: f64, dz: f64, count: f64,
        ) {
            $crate::ffi::guard("bloom_particles_emit", move || {
                if let Some(s) = engine().particles.get_mut(sys as u32) {
                    s.emit(
                        [x as f32, y as f32, z as f32],
                        [dx as f32, dy as f32, dz as f32],
                        (count as usize).min(4096),
                    );
                }
        })
        }

        // bloom_particles_update — integrate + upload. Returns the live count,
        // which is what the caller passes as instanceCount to the draw.
        #[no_mangle]
        pub extern "C" fn bloom_particles_update(sys: f64, dt: f64) -> f64 {
            $crate::ffi::guard("bloom_particles_update", move || {
                let eng = engine();
                let (live, ib) = match eng.particles.get_mut(sys as u32) {
                    Some(s) => (s.update(dt as f32), s.instance_buffer),
                    None => return 0.0,
                };
                if live > 0 {
                    // Re-borrow: the packed slice and the renderer are disjoint
                    // fields, but the borrow checker cannot see that through
                    // two method calls.
                    let packed: Vec<f32> = match eng.particles.get_mut(sys as u32) {
                        Some(s) => s.packed()[..(live as usize) * 12].to_vec(),
                        None => return 0.0,
                    };
                    eng.renderer.material_system.update_instance_buffer(
                        &eng.renderer.queue, ib, &packed, live);
                }
                live as f64
        })
        }

        // bloom_particles_instance_buffer — the handle to hand to
        // drawMeshWithMaterialInstanced.
        #[no_mangle]
        pub extern "C" fn bloom_particles_instance_buffer(sys: f64) -> f64 {
            $crate::ffi::guard("bloom_particles_instance_buffer", move || {
                engine().particles.get_mut(sys as u32)
                    .map(|s| s.instance_buffer as f64)
                    .unwrap_or(0.0)
        })
        }

        #[no_mangle]
        pub extern "C" fn bloom_particles_clear(sys: f64) {
            $crate::ffi::guard("bloom_particles_clear", move || {
                if let Some(s) = engine().particles.get_mut(sys as u32) { s.clear(); }
        })
        }

        #[no_mangle]
        pub extern "C" fn bloom_particles_live(sys: f64) -> f64 {
            $crate::ffi::guard("bloom_particles_live", move || {
                engine().particles.get_mut(sys as u32)
                    .map(|s| s.live as f64)
                    .unwrap_or(0.0)
        })
        }

        // ---- EN-027 decals ---------------------------------------------

        // bloom_decals_init — ring capacity + its dynamic instance buffer.
        #[no_mangle]
        pub extern "C" fn bloom_decals_init(capacity: f64) -> f64 {
            $crate::ffi::guard("bloom_decals_init", move || {
                let cap = (capacity as usize).clamp(1, 8192);
                let eng = engine();
                let ib = eng.renderer.material_system.create_dynamic_instance_buffer(
                    &eng.renderer.device, cap as u32);
                eng.decals.init(cap, ib);
                ib as f64
        })
        }

        // bloom_decals_spawn — position + normal + size + roll. Colour, atlas
        // frame and lifetime come from bloom_decals_set_style, which the caller
        // sets once per decal *type* rather than paying for them on every hit.
        #[no_mangle]
        pub extern "C" fn bloom_decals_spawn(
            x: f64, y: f64, z: f64,
            nx: f64, ny: f64, nz: f64,
            size: f64, roll: f64,
        ) {
            $crate::ffi::guard("bloom_decals_spawn", move || {
                engine().decals.spawn_styled(
                    [x as f32, y as f32, z as f32],
                    [nx as f32, ny as f32, nz as f32],
                    size as f32, roll as f32,
                );
        })
        }

        // bloom_decals_set_style — frame, rgba, life, fade for subsequent
        // spawns. 7 args.
        #[no_mangle]
        pub extern "C" fn bloom_decals_set_style(
            frame: f64, r: f64, g: f64, b: f64, a: f64, life: f64, fade: f64,
        ) {
            $crate::ffi::guard("bloom_decals_set_style", move || {
                engine().decals.style = $crate::decals::DecalStyle {
                    frame: frame as f32,
                    color: [r as f32, g as f32, b as f32, a as f32],
                    life: life as f32,
                    fade: fade as f32,
                };
        })
        }

        #[no_mangle]
        pub extern "C" fn bloom_decals_update(dt: f64) -> f64 {
            $crate::ffi::guard("bloom_decals_update", move || {
                let eng = engine();
                let live = eng.decals.update(dt as f32);
                let ib = eng.decals.instance_buffer;
                if live > 0 {
                    let packed: Vec<f32> = eng.decals.packed()[..(live as usize) * 12].to_vec();
                    eng.renderer.material_system.update_instance_buffer(
                        &eng.renderer.queue, ib, &packed, live);
                }
                live as f64
        })
        }

        #[no_mangle]
        pub extern "C" fn bloom_decals_instance_buffer() -> f64 {
            $crate::ffi::guard("bloom_decals_instance_buffer", move || {
                engine().decals.instance_buffer as f64
        })
        }

        #[no_mangle]
        pub extern "C" fn bloom_decals_clear() {
            $crate::ffi::guard("bloom_decals_clear", move || {
                engine().decals.clear();
        })
        }

    };
}
