//! Cascaded shadow-map pass: cascade fitting from the primary
//! directional light, the ticket-004 cache-hit skip, and the per-cascade
//! depth renders. Split from end_frame_with_scene (2000-line file policy
//! + render-graph migration prep).

use super::*;

impl Renderer {
    pub(super) fn record_shadow_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        scene: &mut crate::scene::SceneGraph,
    ) {
    // Shadow pass: render scene nodes from light's perspective into
    // cascaded shadow maps (3 cascades).
    //
    // Cache hit path (ticket 004): if no caster moved, the light
    // didn't move, and the freshly-computed cascade VPs match the
    // ones the cached depth textures were rendered with, we skip
    // the whole pass. The depth textures retain their content and
    // the main pass samples from them as if we had redrawn.
    profiler.begin("shadow_pass");
    if self.shadow_map.enabled {
        // EN-043 — take last frame's caster transforms out of `self` up front: the
        // caster lists below hold immutable borrows of self.model_gpu_cache for the
        // rest of the function, so this map cannot be touched through `self` again
        // until they are dead.
        let prev_caster_tf = std::mem::take(&mut self.shadow_caster_tf);
        let mut caster_tf_now: std::collections::HashMap<u64, u64> =
            std::collections::HashMap::with_capacity(prev_caster_tf.len() + 64);
        // Nth-draw-of-this-(handle, mesh) counter — see entry_key.
        let mut caster_occ: std::collections::HashMap<u64, u32> =
            std::collections::HashMap::with_capacity(64);
        // Compute cascade VPs from the primary directional light and camera.
        let light_dir = [
            self.lighting_uniforms.light_dir[0],
            self.lighting_uniforms.light_dir[1],
            self.lighting_uniforms.light_dir[2],
        ];
        // Auto-fit: compute world-space AABB across every visible,
        // cast-shadow node so the ortho volume always covers the
        // scene regardless of what's loaded. No per-scene magic
        // numbers.
        let scene_bounds = scene.compute_shadow_bounds();
        self.shadow_map.compute_cascade_vps(
            light_dir,
            self.current_camera_pos,
            self.current_view_matrix,
            // Use the pre-jitter projection so the cascade VPs
            // stay byte-stable when the camera is actually
            // stationary (the shadow cache compares them exactly).
            self.current_proj_matrix_unjittered,
            0.5,   // near — start cascades slightly past the camera
            80.0,  // far — shadow coverage range
            scene_bounds,
        );

        // Re-upload lighting uniforms with cascade VPs and splits.
        // Always write these — even on a cache hit the cascade
        // split distances and view matrix track camera movement
        // (they drive per-pixel cascade selection in the main
        // shader), which is independent of shadow texture content.
        self.lighting_uniforms.shadow_cascade_vps = self.shadow_map.light_vps;
        self.lighting_uniforms.shadow_cascade_splits = [
            self.shadow_map.cascade_splits[0],
            self.shadow_map.cascade_splits[1],
            self.shadow_map.cascade_splits[2],
            // .w = mip-LOD bias for material textures. Bias finer
            // by log2(render_scale) — recovers -1.0 at 0.5 (one
            // mip finer to offset hardware's coarser selection
            // at half-res), ~-0.42 at 0.75, 0 at native.
            if self.render_scale < 0.999 { self.render_scale.log2() } else { 0.0 },
        ];
        self.lighting_uniforms.shadow_view_matrix = self.current_view_matrix;
        self.queue.write_buffer(
            &self.lighting_buffer,
            0,
            bytemuck::bytes_of(&self.lighting_uniforms),
        );
        // Shadow-flicker fix: the material system's PerView buffer was
        // uploaded before this fit ran and still carries LAST frame's
        // cascade VPs. Patch its shadow fields so material-path
        // receivers sample the depth maps with the same matrices the
        // maps are rendered with this frame. Keep .w at 0.0 — that slot
        // is the TSR mip-LOD bias, which the material path historically
        // never received (begin_mode_3d resets it before the material
        // PerView upload); delivering -1.0 here makes hardware mip
        // selection flip per-panel under TAA jitter (visible texture
        // detail popping).
        let mut splits = self.lighting_uniforms.shadow_cascade_splits;
        splits[3] = 0.0;
        self.material_system.refresh_shadow_uniforms(
            &self.queue,
            splits,
            self.lighting_uniforms.shadow_view_matrix,
            self.shadow_map.light_vps,
        );

        // Cache gate, stage 1 — whole-pass invalidators. Per-cascade
        // staleness (VP compare + caster-content signature) is decided
        // after the caster lists are built. Texel-snap + radius
        // quantization + re-fit slack in `compute_cascade_vps` make the
        // per-cascade VP compare exact, so a kept VP + unchanged content
        // means the cascade's cached depth texture is still valid.
        let scene_ver = scene.shadow_version;
        let light_changed = self.shadow_map.rendered_light_dir
            .map(|cached| cached != light_dir)
            .unwrap_or(true);
        let force_all = self.shadow_map.always_fresh
            || self.shadow_map.dirty
            || light_changed
            || self.shadow_map.rendered_light_vps.is_none()
            || self.shadow_map.rendered_scene_version != scene_ver;

        // Build a shared caster list + buffer-ref vectors, then
        // filter per cascade against that cascade's ortho frustum.
        // A caster outside cascade N's frustum can't write pixels
        // into cascade N; near/far pancaking already covers
        // behind-camera casters via the cascade's own far plane.
        struct ShadowDrawEntry {
            vb_idx: usize,
            ib_idx: usize,
            index_start: u32,
            index_count: u32,
            transform: [[f32; 4]; 4],
            wmin: [f32; 3],
            wmax: [f32; 3],
            // Index into `cutout_bgs` for an alpha-tested caster (cutout
            // foliage), or -1 for an opaque caster (plain depth pipeline).
            cutout_idx: i32,
            // Immediate-mode segment containing skinned characters —
            // rendered with the skinning-aware shadow pipeline so animated
            // player/enemies cast a posed shadow instead of a rest pose at
            // the origin. (Mixed segments: non-skinned verts still
            // transform by the model matrix via the shader's weight branch.)
            skinned: bool,
            // Content identity for the per-cascade cache: stable across
            // frames for static casters, salted with `frame_nonce` for
            // animated ones so their cascades re-render every frame.
            sig: u64,
            // Immediate-batch content (animated characters, per-frame
            // primitives). Dynamic casters render into the live cascade
            // texture every frame on top of the cached static depth;
            // they never invalidate the static cache.
            dynamic: bool,
            // Base slot of a skinned cached draw's pose in the shared
            // joint buffer (the cached VB keeps RAW joint indices, so
            // vs_shadow_skinned adds this via ShadowUniforms.misc.x).
            // 0.0 for everything else — including immediate-batch
            // skinned segments, whose vertex joints are pre-offset.
            joint_offset: f32,
            // Foliage wind amount for this caster (0 = rigid). Non-zero makes the
            // caster MOVE, which is why it also forces `dynamic` — a swaying tree
            // cannot reuse its cached static shadow depth.
            foliage: f32,
            // EN-043 — stable identity (NOT including the transform), so a caster
            // that moved can be told apart from a caster that appeared.
            key: u64,
        }
        fn entry_sig(kind: u8, id: u64, idx: u64, transform: &[[f32; 4]; 4]) -> u64 {
            let mut h = FNV_OFFSET;
            h = fnv1a_bytes(h, &[kind]);
            h = fnv1a_bytes(h, &id.to_le_bytes());
            h = fnv1a_bytes(h, &idx.to_le_bytes());
            fnv1a_bytes(h, bytemuck::bytes_of(transform))
        }
        // EN-043 — a caster's IDENTITY, without its transform, so "this caster
        // moved" can be told apart from "a different caster appeared".
        //
        // `occ` is load-bearing, and the reason this is not just (handle, mesh_idx):
        // the forest is 88 trees sharing THREE model handles, so every tree would
        // collide on one key, each would be compared against some other tree's
        // transform, and all 88 would be declared movers. That is not theoretical —
        // it is what the first cut did: the whole forest went dynamic, overflowed
        // the 64-slot budget, and every shadow in the game vanished while the fps
        // went UP. Exactly the EN-042 trap.
        //
        // `occ` is the Nth draw of this (handle, mesh) this frame; the game submits
        // its draws in a stable order, so occurrence N is the same tree every frame.
        fn entry_key(kind: u8, id: u64, idx: u64, occ: u32) -> u64 {
            let mut h = FNV_OFFSET;
            h = fnv1a_bytes(h, &[kind]);
            h = fnv1a_bytes(h, &id.to_le_bytes());
            h = fnv1a_bytes(h, &idx.to_le_bytes());
            fnv1a_bytes(h, &occ.to_le_bytes())
        }
        fn tf_hash(transform: &[[f32; 4]; 4]) -> u64 {
            fnv1a_bytes(FNV_OFFSET, bytemuck::bytes_of(transform))
        }
        // Per-frame nonce for animated casters' signatures. Bumped
        // whenever shadows render — skinned CACHED model draws need it
        // even when the immediate batch is empty, which is the norm now
        // that skinned models draw through the cache.
        self.shadow_map.frame_nonce = self.shadow_map.frame_nonce.wrapping_add(1);
        let nonce = self.shadow_map.frame_nonce;

        let mut shadow_nodes: Vec<ShadowDrawEntry> = Vec::new();
        let mut shadow_vbs: Vec<&wgpu::Buffer> = Vec::new();
        let mut shadow_ibs: Vec<&wgpu::Buffer> = Vec::new();
        let mut cutout_bgs: Vec<&wgpu::BindGroup> = Vec::new();
        for (i, (_handle, node)) in scene.nodes.iter().enumerate() {
            // gi_only proxies duplicate geometry that already casts through
            // the material-command path below — including them would
            // double-render every caster.
            if !node.visible || node.gi_only || !node.cast_shadow || node.indices.is_empty() {
                continue;
            }
            let Some(vb) = &node.gpu_vb else { continue };
            let Some(ib) = &node.gpu_ib else { continue };
            let vb_idx = shadow_vbs.len();
            shadow_vbs.push(vb);
            shadow_ibs.push(ib);
            // MASK-material nodes carry an alpha-test shadow bind group so
            // foliage casts dappled shadows (same as the cached-model path).
            let cutout_idx = match &node.gpu_shadow_cutout_bg {
                Some(bg) => { let i = cutout_bgs.len(); cutout_bgs.push(bg); i as i32 }
                None => -1,
            };
            shadow_nodes.push(ShadowDrawEntry {
                vb_idx,
                ib_idx: vb_idx,
                index_start: 0,
                index_count: node.gpu_index_count,
                transform: node.transform,
                wmin: node.world_bounds_min,
                wmax: node.world_bounds_max,
                cutout_idx,
                skinned: false,
                sig: entry_sig(0, i as u64, node.gpu_index_count as u64, &node.transform),
                dynamic: false,
                joint_offset: 0.0,
                foliage: 0.0,
                key: { let k0 = entry_key(0, i as u64, node.gpu_index_count as u64, 0);
                       let o = caster_occ.entry(k0).or_insert(0); let v = *o; *o += 1;
                       entry_key(0, i as u64, node.gpu_index_count as u64, v) },
            });
        }

        // Immediate-mode 3D batch (drawCube/drawSphere/non-cached models),
        // one entry per segment. These verts are already in WORLD space, so
        // the model matrix is identity. Model draws maintain per-segment
        // bounds inline (skinned via joint-transformed rest AABBs);
        // primitive-only segments are scanned here. Segments with skinned
        // content take a per-frame nonce as their signature (animation
        // means their rendered output changes every frame); static
        // segments hash their vertex positions, so e.g. pickups
        // re-submitted identically each frame don't dirty their cascades.
        if !self.indices_3d.is_empty() {
            self.scan_unbounded_segments_3d();
            let vb_idx = shadow_vbs.len();
            shadow_vbs.push(&self.persistent_vb_3d);
            shadow_ibs.push(&self.persistent_ib_3d);
            if self.draw_calls_3d.is_empty() {
                // Fallback: vertices without segment tracking — one
                // unbounded, always-dirty entry (pre-segmentation shape).
                shadow_nodes.push(ShadowDrawEntry {
                    vb_idx,
                    ib_idx: vb_idx,
                    index_start: 0,
                    index_count: self.indices_3d.len() as u32,
                    transform: IDENTITY_MAT4,
                    wmin: [1.0, 1.0, 1.0],
                    wmax: [-1.0, -1.0, -1.0],
                    cutout_idx: -1,
                    skinned: true,
                    sig: nonce,
                    dynamic: true,
                    joint_offset: 0.0,
                    foliage: 0.0,
                    key: 0,
                });
            } else {
                let num_calls = self.draw_calls_3d.len();
                for ci in 0..num_calls {
                    let call = &self.draw_calls_3d[ci];
                    let next_start = if ci + 1 < num_calls {
                        self.draw_calls_3d[ci + 1].index_start
                    } else {
                        self.indices_3d.len() as u32
                    };
                    let count = next_start - call.index_start;
                    if count == 0 { continue; }
                    shadow_nodes.push(ShadowDrawEntry {
                        vb_idx,
                        ib_idx: vb_idx,
                        index_start: call.index_start,
                        index_count: count,
                        transform: IDENTITY_MAT4,
                        wmin: call.wmin,
                        wmax: call.wmax,
                        cutout_idx: -1,
                        skinned: call.has_skinned,
                        sig: if call.has_skinned { nonce } else { call.content_hash },
                        dynamic: true,
                        joint_offset: 0.0,
                        foliage: 0.0,
                        key: 0,
                    });
                }
            }
        }

        // Foliage promoted to the dynamic set this frame. Capped well below
        // SHADOW_MAX_DYNAMIC so the characters — whose shadows are the ones a
        // player actually looks at — always keep their slots.
        const MAX_FOLIAGE_DYNAMIC: u32 = 24;
        let mut foliage_dynamic: u32 = 0;

        // Cached models (drawModel: trees, characters, etc.) — each is a
        // GpuMesh plus its object→world matrix. World AABB from the
        // cache-time local AABB so per-cascade culling rejects casters
        // outside a cascade's ortho frustum (the forest was previously
        // re-drawn into every cascade every frame). Skinned cached draws
        // render through the skinning pipeline as dynamic casters (pose
        // changes every frame → nonce signature) with the joint-union
        // AABB computed at submit time.
        for cmd in self.model_draw_commands.iter() {
            if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.cache_handle) {
                if cmd.mesh_idx < meshes.len() {
                    let mesh = &meshes[cmd.mesh_idx];
                    let vb_idx = shadow_vbs.len();
                    shadow_vbs.push(&mesh.vb);
                    shadow_ibs.push(&mesh.ib);
                    // Cutout foliage → alpha-tested shadow pipeline.
                    let cutout_idx = match &mesh.shadow_cutout_bg {
                        Some(bg) => { let i = cutout_bgs.len(); cutout_bgs.push(bg); i as i32 }
                        None => -1,
                    };
                    if cmd.skinned {
                        // Sentinel bounds (min > max) when the submit-time
                        // AABB was empty → uncullable, never lost.
                        let (wmin, wmax) = cmd.bounds_override
                            .unwrap_or(([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]));
                        shadow_nodes.push(ShadowDrawEntry {
                            vb_idx,
                            ib_idx: vb_idx,
                            index_start: 0,
                            index_count: mesh.index_count,
                            transform: cmd.model,
                            wmin,
                            wmax,
                            cutout_idx,
                            skinned: true,
                            sig: nonce,
                            dynamic: true,
                            joint_offset: cmd.joint_offset,
                            foliage: 0.0,
                            key: 0,
                        });
                    } else {
                        // Only sway the shadow if the game asked for it AND there is
                        // room in the dynamic-caster budget.
                        //
                        // That second condition is not paranoia. A swaying caster
                        // cannot reuse the cached static depth, so it must move to
                        // the DYNAMIC set — and that set holds SHADOW_MAX_DYNAMIC
                        // (64) entries. The shooter's forest alone is 88 trees x 4
                        // primitives = 352. Marking them all dynamic overflows the
                        // budget, and the overflow is dropped — which does not merely
                        // cost frames, it silently DELETES shadows. Measured: turning
                        // this on removed every tree shadow AND the player's own
                        // shadow from under their feet, while reporting a higher fps.
                        //
                        // So: sway as many as fit, leave the rest rigid. A slightly
                        // stale canopy shadow is invisible; a missing one is not.
                        let fol = if self.foliage_shadow_motion
                            && foliage_dynamic < MAX_FOLIAGE_DYNAMIC
                        {
                            let f = self.foliage_wind.get(&cmd.cache_handle).copied().unwrap_or(0.0);
                            if f > 0.0 { foliage_dynamic += 1; }
                            f
                        } else { 0.0 };
                        let (wmin, wmax) =
                            transform_aabb(&cmd.model, mesh.local_min, mesh.local_max);
                        shadow_nodes.push(ShadowDrawEntry {
                            vb_idx,
                            ib_idx: vb_idx,
                            index_start: 0,
                            index_count: mesh.index_count,
                            transform: cmd.model,
                            wmin,
                            wmax,
                            cutout_idx,
                            skinned: false,
                            // A swaying caster changes shape every frame, so it
                            // cannot share the cached static depth: signature goes
                            // to the per-frame nonce and it renders as dynamic.
                            // That is exactly the cost `foliage_shadow_motion`
                            // gates, which is why it defaults off.
                            sig: if fol > 0.0 { nonce }
                                 else { entry_sig(1, cmd.cache_handle, cmd.mesh_idx as u64, &cmd.model) },
                            dynamic: fol > 0.0,
                            joint_offset: 0.0,
                            foliage: fol,
                            key: { let k0 = entry_key(1, cmd.cache_handle, cmd.mesh_idx as u64, 0);
                                let o = caster_occ.entry(k0).or_insert(0); let v = *o; *o += 1;
                                entry_key(1, cmd.cache_handle, cmd.mesh_idx as u64, v) },
                        });
                    }
                }
            }
        }

        // Material-system draws (terrain / building / trees rendered through
        // compiled materials). Same GpuMesh cache as drawModel — the command
        // carries a CPU-side copy of its model matrix precisely for this
        // pass. `commands` holds only the opaque + cutout buckets, so water /
        // glass / additive effects never cast. Instanced draws (the 20k-blade
        // grass field) are skipped deliberately: vs_shadow has no instance
        // stream, and per-blade grass shadows are sub-texel noise at these
        // cascade resolutions anyway.
        for cmd in self.material_system.commands.iter() {
            if cmd.instance.is_some() { continue; }
            if let Some(Some(meshes)) = self.model_gpu_cache.get(&cmd.mesh_handle) {
                if cmd.mesh_idx < meshes.len() {
                    let mesh = &meshes[cmd.mesh_idx];
                    let vb_idx = shadow_vbs.len();
                    shadow_vbs.push(&mesh.vb);
                    shadow_ibs.push(&mesh.ib);
                    // MASK-material meshes (leaf cards) keep their dappled
                    // alpha-tested shadows, same as the other two paths.
                    let cutout_idx = match &mesh.shadow_cutout_bg {
                        Some(bg) => { let i = cutout_bgs.len(); cutout_bgs.push(bg); i as i32 }
                        None => -1,
                    };
                    let (wmin, wmax) =
                        transform_aabb(&cmd.model, mesh.local_min, mesh.local_max);
                    shadow_nodes.push(ShadowDrawEntry {
                        vb_idx,
                        ib_idx: vb_idx,
                        index_start: 0,
                        index_count: mesh.index_count,
                        transform: cmd.model,
                        wmin,
                        wmax,
                        cutout_idx,
                        skinned: false,
                        sig: entry_sig(2, cmd.mesh_handle, cmd.mesh_idx as u64, &cmd.model),
                        dynamic: false,
                        joint_offset: 0.0,
                        foliage: 0.0,
                        key: { let k0 = entry_key(2, cmd.mesh_handle, cmd.mesh_idx as u64, 0);
                            let o = caster_occ.entry(k0).or_insert(0); let v = *o; *o += 1;
                            entry_key(2, cmd.mesh_handle, cmd.mesh_idx as u64, v) },
                    });
                }
            }
        }

        // EN-043 — promote MOVERS to the dynamic set.
        //
        // A non-skinned cached caster whose transform changed since last frame used
        // to stay in the STATIC set with a different content signature. That
        // invalidated the cascade's cached depth, so every tree, wall and terrain
        // tile in the world re-rendered into all three cascades — every frame —
        // because one pickup was bobbing. Measured on the shooter's title screen:
        // shadow_pass GPU 6.0-7.0 ms against the 0.1-1.7 ms the cache was built to
        // deliver.
        //
        // A caster that moves is DYNAMIC, by definition. Dynamic casters draw on
        // top of the cached static depth every frame and never invalidate it, which
        // is exactly what a moving object needs and costs one draw instead of a
        // thousand.
        for e in shadow_nodes.iter_mut() {
            if e.dynamic { continue; }
            let tf = tf_hash(&e.transform);
            caster_tf_now.insert(e.key, tf);
            if let Some(&prev) = prev_caster_tf.get(&e.key) {
                if prev != tf {
                    e.dynamic = true;
                    e.sig = nonce;
                }
            }
        }

        let cascade_planes: [[[f32; 4]; 6]; crate::shadows::NUM_CASCADES] =
            std::array::from_fn(|c| {
                crate::scene::extract_frustum_planes(&self.shadow_map.light_vps[c])
            });
        let mut cascade_indices: [Vec<usize>; crate::shadows::NUM_CASCADES] =
            std::array::from_fn(|_| Vec::with_capacity(shadow_nodes.len()));
        for (i, entry) in shadow_nodes.iter().enumerate() {
            let has_bounds = entry.wmin[0] <= entry.wmax[0];
            for c in 0..crate::shadows::NUM_CASCADES {
                if has_bounds
                    && crate::scene::aabb_outside_frustum(&cascade_planes[c], entry.wmin, entry.wmax)
                {
                    continue;
                }
                cascade_indices[c].push(i);
            }
        }
        // Per-cascade STATIC content signature: fold every surviving
        // non-dynamic caster's identity, in draw order. The static depth
        // cache re-renders only when its cascade's VP changed, this
        // signature changed, or a whole-pass invalidator fired. Dynamic
        // casters are excluded — they draw on top of the cached static
        // depth every frame and never invalidate it.
        let mut cascade_sigs = [0u64; crate::shadows::NUM_CASCADES];
        for c in 0..crate::shadows::NUM_CASCADES {
            let mut h = FNV_OFFSET;
            for &ei in cascade_indices[c].iter() {
                if shadow_nodes[ei].dynamic { continue; }
                h = fnv1a_bytes(h, &shadow_nodes[ei].sig.to_le_bytes());
            }
            cascade_sigs[c] = h;
        }

        // Render each cascade. Static casters live in a cached depth
        // texture ("cached whole-scene shadows") re-rendered only when
        // the cascade's VP or static content changes; every frame the
        // live texture is refreshed by copy and the few dynamic casters
        // draw on top with Load. A cascade with no change is skipped
        // entirely. Uniform slots: static casters use the head of the
        // cascade's region, dynamic casters the reserved tail — the
        // ranges are disjoint because every write_buffer lands at
        // submit, before any encoded pass executes.
        for cascade in 0..crate::shadows::NUM_CASCADES {
            let stride = crate::shadows::SHADOW_UNIFORM_STRIDE as usize;
            let max = crate::shadows::SHADOW_MAX_NODES as usize;
            let max_dynamic = crate::shadows::SHADOW_MAX_DYNAMIC as usize;
            let max_static = max - max_dynamic;
            let cascade_base = cascade * stride * max;
            let cascade_vp = self.shadow_map.light_vps[cascade];
            let entries = &cascade_indices[cascade];

            let vp_changed = self.shadow_map.rendered_light_vps
                .map(|vps| vps[cascade] != self.shadow_map.light_vps[cascade])
                .unwrap_or(true);
            let static_stale = force_all || vp_changed
                || self.shadow_map.rendered_cascade_sig[cascade] != cascade_sigs[cascade];
            let dyn_now = entries.iter().any(|&ei| shadow_nodes[ei].dynamic);
            if !static_stale && !dyn_now && !self.shadow_map.had_dynamic[cascade] {
                // Live texture already holds exactly this content.
                continue;
            }

            if static_stale {
                let static_entries: Vec<usize> = entries.iter().copied()
                    .filter(|&ei| !shadow_nodes[ei].dynamic)
                    .take(max_static)
                    .collect();
                let mut uniform_data: Vec<u8> =
                    vec![0u8; stride * static_entries.len().max(1)];
                for (slot, &ei) in static_entries.iter().enumerate() {
                    let uniforms = crate::shadows::ShadowUniforms {
                        light_vp: cascade_vp,
                        model: shadow_nodes[ei].transform,
                        misc: [shadow_nodes[ei].joint_offset, 0.0, shadow_nodes[ei].foliage, 0.0],
                        wind: self.lighting_uniforms.wind,
                    };
                    let off = slot * stride;
                    uniform_data[off..off + std::mem::size_of::<crate::shadows::ShadowUniforms>()]
                        .copy_from_slice(bytemuck::bytes_of(&uniforms));
                }
                // Each cascade owns its own slice of the uniform buffer —
                // all write_buffer calls execute at submit, BEFORE any of
                // the encoded passes run, so sharing one region would
                // leave every cascade rendering with the last cascade's
                // matrices (the no-near-shadows bug).
                if !static_entries.is_empty() {
                    self.queue.write_buffer(
                        &self.shadow_map.uniform_buffer,
                        cascade_base as u64,
                        &uniform_data[..static_entries.len() * stride],
                    );
                }
                {
                    let shadow_ts = profiler.pass_timestamp_writes("shadow_pass");
                    let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("shadow_pass_static"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.shadow_map.static_depth_views[cascade],
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: shadow_ts,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    // Pipeline kind per caster: 0 opaque, 1 cutout, 2 skinned.
                    // Only switch when the kind changes.
                    let mut cur_kind: u8 = 0;
                    shadow_pass.set_pipeline(&self.shadow_map.pipeline);
                    for (slot, &ei) in static_entries.iter().enumerate() {
                        let entry = &shadow_nodes[ei];
                        let offset = (cascade_base + slot * stride) as u32;
                        let kind: u8 = if entry.skinned { 2 }
                            else if entry.cutout_idx >= 0 { 1 }
                            else { 0 };
                        if kind != cur_kind {
                            shadow_pass.set_pipeline(match kind {
                                1 => &self.shadow_map.pipeline_cutout,
                                2 => &self.shadow_map.pipeline_skinned,
                                _ => &self.shadow_map.pipeline,
                            });
                            cur_kind = kind;
                        }
                        shadow_pass.set_bind_group(0, &self.shadow_map.uniform_bind_group, &[offset]);
                        if kind == 1 {
                            shadow_pass.set_bind_group(1, cutout_bgs[entry.cutout_idx as usize], &[]);
                        } else if kind == 2 {
                            shadow_pass.set_bind_group(1, &self.joint_bind_group, &[]);
                        }
                        shadow_pass.set_vertex_buffer(0, shadow_vbs[entry.vb_idx].slice(..));
                        shadow_pass.set_index_buffer(shadow_ibs[entry.ib_idx].slice(..), wgpu::IndexFormat::Uint32);
                        shadow_pass.draw_indexed(
                            entry.index_start..entry.index_start + entry.index_count,
                            0,
                            0..1,
                        );
                    }
                }
                self.shadow_map.rendered_cascade_sig[cascade] = cascade_sigs[cascade];
            }

            // Refresh the live texture from the static cache, then draw
            // dynamic casters on top.
            encoder.copy_texture_to_texture(
                self.shadow_map.static_depth_textures[cascade].as_image_copy(),
                self.shadow_map.depth_textures[cascade].as_image_copy(),
                wgpu::Extent3d {
                    width: crate::shadows::CASCADE_MAP_SIZE,
                    height: crate::shadows::CASCADE_MAP_SIZE,
                    depth_or_array_layers: 1,
                },
            );
            if dyn_now {
                let dyn_base = cascade_base + stride * max_static;
                // EN-042 — the dynamic budget can overflow, and the overflow IS
                // dropped. Which caster gets dropped must not be an accident of
                // queue order. It was, and it cost this project twice: both times
                // the thing that silently vanished was the player's own shadow, and
                // both times the frame rate went UP and looked like a win.
                //
                // Rank them, so if we must lose a shadow we lose one nobody misses:
                // characters first (the shadow a player actually looks at), then
                // other movers, then foliage — a swaying canopy shadow is soft and
                // dappled and the most forgiving thing in the frame.
                let mut dyn_entries: Vec<usize> = entries.iter().copied()
                    .filter(|&ei| shadow_nodes[ei].dynamic)
                    .collect();
                if dyn_entries.len() > max_dynamic {
                    dyn_entries.sort_by_key(|&ei| {
                        let e = &shadow_nodes[ei];
                        if e.skinned { 0u8 }
                        else if e.foliage > 0.0 { 2u8 }
                        else { 1u8 }
                    });
                    dyn_entries.truncate(max_dynamic);
                }
                let mut uniform_data: Vec<u8> =
                    vec![0u8; stride * dyn_entries.len().max(1)];
                for (slot, &ei) in dyn_entries.iter().enumerate() {
                    let uniforms = crate::shadows::ShadowUniforms {
                        light_vp: cascade_vp,
                        model: shadow_nodes[ei].transform,
                        misc: [shadow_nodes[ei].joint_offset, 0.0, shadow_nodes[ei].foliage, 0.0],
                        wind: self.lighting_uniforms.wind,
                    };
                    let off = slot * stride;
                    uniform_data[off..off + std::mem::size_of::<crate::shadows::ShadowUniforms>()]
                        .copy_from_slice(bytemuck::bytes_of(&uniforms));
                }
                self.queue.write_buffer(
                    &self.shadow_map.uniform_buffer,
                    dyn_base as u64,
                    &uniform_data[..dyn_entries.len() * stride],
                );
                {
                    let shadow_ts = profiler.pass_timestamp_writes("shadow_pass");
                    let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("shadow_pass_dynamic"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.shadow_map.depth_views[cascade],
                            depth_ops: Some(wgpu::Operations {
                                // Refreshed static depth is the base.
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: shadow_ts,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    let mut cur_kind: u8 = 0;
                    shadow_pass.set_pipeline(&self.shadow_map.pipeline);
                    for (slot, &ei) in dyn_entries.iter().enumerate() {
                        let entry = &shadow_nodes[ei];
                        let offset = (dyn_base + slot * stride) as u32;
                        let kind: u8 = if entry.skinned { 2 }
                            else if entry.cutout_idx >= 0 { 1 }
                            else { 0 };
                        if kind != cur_kind {
                            shadow_pass.set_pipeline(match kind {
                                1 => &self.shadow_map.pipeline_cutout,
                                2 => &self.shadow_map.pipeline_skinned,
                                _ => &self.shadow_map.pipeline,
                            });
                            cur_kind = kind;
                        }
                        shadow_pass.set_bind_group(0, &self.shadow_map.uniform_bind_group, &[offset]);
                        if kind == 1 {
                            shadow_pass.set_bind_group(1, cutout_bgs[entry.cutout_idx as usize], &[]);
                        } else if kind == 2 {
                            // Joint matrices for skinning the animated
                            // characters in the immediate-mode batch.
                            shadow_pass.set_bind_group(1, &self.joint_bind_group, &[]);
                        }
                        shadow_pass.set_vertex_buffer(0, shadow_vbs[entry.vb_idx].slice(..));
                        shadow_pass.set_index_buffer(shadow_ibs[entry.ib_idx].slice(..), wgpu::IndexFormat::Uint32);
                        shadow_pass.draw_indexed(
                            entry.index_start..entry.index_start + entry.index_count,
                            0,
                            0..1,
                        );
                    }
                }
            }
            self.shadow_map.had_dynamic[cascade] = dyn_now;
        }

        // Cache bookkeeping — next frame skips every cascade whose VP
        // and caster content stay put.
        self.shadow_caster_tf = caster_tf_now;
        self.shadow_map.rendered_light_vps = Some(self.shadow_map.light_vps);
        self.shadow_map.rendered_light_dir = Some(light_dir);
        self.shadow_map.rendered_scene_version = scene_ver;
        self.shadow_map.dirty = false;
    }

    profiler.end("shadow_pass");
    }
}
