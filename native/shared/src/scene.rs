//! Retained scene graph for Bloom Engine.
//!
//! Unlike immediate-mode drawing (drawCube, drawModel), the scene graph holds
//! persistent meshes that survive across frames. Systems update geometry and
//! transforms; the renderer draws all visible nodes each frame automatically.

use wgpu::util::DeviceExt;
use crate::handles::HandleRegistry;
use crate::renderer::Vertex3D;

// ============================================================
// PBR Material
// ============================================================

#[derive(Clone, Debug)]
pub struct PbrMaterial {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metalness: f32,
    pub opacity: f32,
    pub emissive: [f32; 3],
    pub double_sided: bool,
    /// glTF MASK alpha cutoff. 0.0 = OPAQUE. Non-zero routes the node
    /// through the scene shader's alpha-cutout path (discard below the
    /// cutoff, two-sided foliage shading, wind sway).
    pub alpha_cutoff: f32,
    pub texture_idx: u32,
    /// Normal-map texture. 0 means "no normal map" — scene shader falls
    /// back to the geometric normal. Stored as a texture index rather
    /// than bind group so the renderer can build per-material bind
    /// groups lazily without SceneGraph holding GPU references.
    pub normal_texture_idx: u32,
    pub metallic_roughness_texture_idx: u32,
    pub emissive_texture_idx: u32,
    pub occlusion_texture_idx: u32,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            roughness: 0.8,
            metalness: 0.0,
            opacity: 1.0,
            emissive: [0.0, 0.0, 0.0],
            double_sided: false,
            alpha_cutoff: 0.0,
            texture_idx: 0,
            normal_texture_idx: 0,
            metallic_roughness_texture_idx: 0,
            emissive_texture_idx: 0,
            occlusion_texture_idx: 0,
        }
    }
}

// ============================================================
// Scene Node Uniforms (matches Uniforms3D in renderer)
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NodeUniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    prev_mvp: [[f32; 4]; 4],
    model_tint: [f32; 4],
}

// ============================================================
// Scene Node
// ============================================================

pub struct SceneNode {
    // Geometry (CPU-side, updated by systems)
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
    // Material
    pub material: PbrMaterial,
    // Transform
    pub transform: [[f32; 4]; 4],
    /// Previous frame's world transform — used to compute per-mesh
    /// screen-space velocity for motion blur and TAA reprojection.
    pub prev_transform: [[f32; 4]; 4],
    // Flags
    pub visible: bool,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    pub parent: f64,
    // Editor user data — an arbitrary i64 attached to the node. The editor
    // uses this to store the entity id directly on the scene node so picking
    // can return the entity id without a handle → id map lookup (Q7).
    pub user_data: i64,
    // Cached world-space AABB, recomputed when geometry changes (Q5).
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// World-space AABB cached each frame by `prepare()` — the result of
    /// transforming the local `bounds_min`/`bounds_max` box by the node
    /// transform. Consumed by the shadow pass for per-cascade frustum
    /// culling so it doesn't repeat the 8-corner transform. Sentinel
    /// `world_bounds_min[0] > world_bounds_max[0]` means "not yet valid"
    /// (node never passed through `prepare()`, or local bounds empty).
    pub world_bounds_min: [f32; 3],
    pub world_bounds_max: [f32; 3],
    // GPU resources (lazily created)
    pub gpu_vb: Option<wgpu::Buffer>,
    pub gpu_ib: Option<wgpu::Buffer>,
    pub gpu_index_count: u32,
    /// Vertex count, cached at VB upload so the ray-tracing BLAS build
    /// can reference it without re-reading the full `vertices` Vec.
    pub gpu_vertex_count: u32,
    /// Ticket 007b — bottom-level acceleration structure built at the
    /// same geo_dirty flush that creates `gpu_vb`/`gpu_ib`. `None` on
    /// non-RT adapters or until the first upload. Build is scheduled
    /// by `prepare()`, committed to the GPU by the renderer's main
    /// encoder in `build_acceleration_structures`.
    pub blas: Option<wgpu::Blas>,
    /// Ticket 013 — first of 6 consecutive card-atlas slots assigned
    /// at BLAS creation. Slots are laid out per axis:
    ///   first_slot + 0 → +X, +1 → -X, +2 → +Y,
    ///   +3 → -Y,       +4 → +Z, +5 → -Z.
    /// `None` until the capture pass has allocated the run.
    pub card_first_slot: Option<u32>,
    /// Ticket 013 V3 — when true, the mesh is re-queued into
    /// `pending_card_captures` every frame so the card atlas stays
    /// in sync with animated geometry. Off by default — static meshes
    /// pay the capture cost once and never again.
    pub card_dynamic: bool,
    /// Ticket 014 — per-mesh unsigned distance field baked by a
    /// compute pass at geo-upload time. 3D R16Float texture, fixed
    /// resolution (`MESH_SDF_RES`³). Used later by the SW probe
    /// trace for sphere-marching when the adapter lacks HW RT.
    /// `None` on non-RT-capable adapters or until the bake lands.
    pub mesh_sdf: Option<wgpu::Texture>,
    pub mesh_sdf_view: Option<wgpu::TextureView>,
    /// Content hash of (positions, indices) computed at upload time.
    /// Set whenever `mesh_sdf` exists; the renderer reads it back when
    /// flushing cache writes after a fresh bake. `None` until the
    /// first geo upload — and on non-RT-capable adapters that never
    /// allocate a per-mesh SDF.
    pub mesh_hash: Option<crate::sdf_cache::MeshHash>,
    /// Flat mesh-average world-space normal, cached on BLAS build so
    /// the per-instance GI data buffer can be populated without
    /// re-reading the vertex array. Rough heuristic — for walls and
    /// floors this tracks the surface closely; for radially-symmetric
    /// meshes (columns) it averages to near-zero and the trace falls
    /// back to a fixed up-vector. Phase-2 Mesh Cards (ticket 013)
    /// upgrades this to textured per-hit normals.
    pub flat_normal_ws: [f32; 3],
    /// Mean world-space albedo cached alongside `flat_normal_ws`.
    /// For the hit-lighting-lite path in 007b HW trace.
    pub flat_albedo: [f32; 3],
    /// Index into the scene-graph's shared node-uniform pool. `None`
    /// until this node's first `prepare()` assigns a slot. Cleared when
    /// the pool reallocates so the next prepare re-assigns + rebuilds
    /// `gpu_uniform_bg`.
    uniform_slot: Option<u32>,
    gpu_uniform_bg: Option<wgpu::BindGroup>,
    /// Transient: set by `prepare()` based on the camera frustum, read
    /// by `render()` to skip off-screen nodes. Shadow pass ignores this
    /// flag — off-screen geometry can still cast shadows into view.
    in_view_frustum: bool,
    /// Hidden behind other geometry per the Hi-Z occlusion grid (one
    /// frame of latency, conservative). Only gates the main camera
    /// pass — shadows/picking/TLAS never read it.
    occluded: bool,
    /// Material bind group for the scene pipeline — holds base color,
    /// normal, metallic-roughness and emissive texture views in one
    /// group. Rebuilt whenever one of the material texture indices
    /// changes (tracked via `mat_dirty`).
    pub gpu_material_bg: Option<wgpu::BindGroup>,
    pub gpu_material_uniform_buf: Option<wgpu::Buffer>,
    /// Alpha-tested shadow-caster bind group for MASK materials (base
    /// colour + sampler + cutoff). None for opaque casters. Built with
    /// the material bind group; consumed by the shadow pass's cutout
    /// pipeline so scene-node foliage casts dappled, not solid, shadows.
    pub gpu_shadow_cutout_bg: Option<wgpu::BindGroup>,
    pub mat_dirty: bool,
    geo_dirty: bool,
    /// Reduced-detail geometry variants, ordered coarser and coarser
    /// (descending max_coverage). Selected per frame in prepare() by
    /// projected screen coverage; the base geometry above is "LOD 0".
    /// Shadows, picking, BLAS, and SDF always use the base geometry —
    /// LODs only affect the main camera rasterization.
    pub lods: Vec<LodLevel>,
    /// Active LOD this frame: -1 = base geometry, otherwise an index
    /// into `lods`. Driven by prepare(); render() binds accordingly.
    active_lod: i32,
}

/// One reduced-detail variant for a SceneNode.
pub struct LodLevel {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
    /// Use this level when the node's projected screen coverage (longest
    /// NDC extent of its world AABB, 0..1) drops below this value.
    pub max_coverage: f32,
    gpu_vb: Option<wgpu::Buffer>,
    gpu_ib: Option<wgpu::Buffer>,
    gpu_index_count: u32,
    dirty: bool,
}

impl SceneNode {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            material: PbrMaterial::default(),
            transform: crate::renderer::IDENTITY_MAT4,
            prev_transform: crate::renderer::IDENTITY_MAT4,
            visible: true,
            cast_shadow: true,
            receive_shadow: true,
            parent: 0.0,
            user_data: 0,
            bounds_min: [0.0; 3],
            bounds_max: [0.0; 3],
            world_bounds_min: [f32::MAX; 3],
            world_bounds_max: [f32::MIN; 3],
            gpu_vb: None,
            gpu_ib: None,
            gpu_index_count: 0,
            gpu_vertex_count: 0,
            blas: None,
            card_first_slot: None,
            card_dynamic: false,
            mesh_sdf: None,
            mesh_sdf_view: None,
            mesh_hash: None,
            flat_normal_ws: [0.0, 1.0, 0.0],
            flat_albedo: [1.0, 1.0, 1.0],
            uniform_slot: None,
            gpu_uniform_bg: None,
            in_view_frustum: true,
            occluded: false,
            gpu_material_bg: None,
            gpu_material_uniform_buf: None,
            gpu_shadow_cutout_bg: None,
            mat_dirty: true,
            geo_dirty: true,
            lods: Vec::new(),
            active_lod: -1,
        }
    }
}

// ============================================================
// Scene Graph
// ============================================================

/// Stride between per-node uniform slots in the shared pool buffer.
/// Must be >= sizeof(NodeUniforms) (208B) and a multiple of the device's
/// `min_uniform_buffer_offset_alignment`. 256 is safe on every platform.
const NODE_UNIFORM_STRIDE: u64 = 256;

pub struct SceneGraph {
    pub nodes: HandleRegistry<SceneNode>,
    /// Shared uniform buffer holding one 256B slot per scene node. All
    /// per-node uniforms get written to this buffer in a single
    /// `queue.write_buffer` call per frame, replacing what used to be
    /// one write per node. Grows on demand; bind groups referencing
    /// the old buffer get invalidated when that happens.
    uniform_pool: Option<wgpu::Buffer>,
    uniform_pool_capacity: u32,
    /// Next free slot index. Slots are never released — they only grow
    /// with the high-water-mark node count. Sufficient for the current
    /// workload (scenes with < 10k retained nodes).
    next_slot: u32,
    /// Scratch buffer reused across frames for building the packed
    /// uniform payload. Sized to `uniform_pool_capacity * STRIDE`.
    scratch: Vec<u8>,
    /// Monotonic counter bumped whenever a change lands that affects
    /// what gets drawn into the directional shadow map — transform,
    /// cast_shadow toggle, visibility (of a caster), geometry update,
    /// or destruction of a visible caster. The renderer's shadow-map
    /// cache compares this against its last-rendered version to decide
    /// if it can reuse the cached cascades. Writes that can't affect
    /// shadows (materials, user_data, etc.) deliberately don't bump it.
    pub shadow_version: u64,

    /// Ticket 007b — true when the host renderer's device was created
    /// with `Features::EXPERIMENTAL_RAY_QUERY`. Controls whether the
    /// scene bakes `BLAS_INPUT` buffer usage + a per-node BLAS at
    /// `prepare()` time. Off on non-RT adapters (web, most Android,
    /// Intel integrated GPUs) so the cost is never paid there.
    pub hw_rt_enabled: bool,
    /// Monotonic counter bumped when any change that would require a
    /// TLAS rebuild lands — geometry upload, transform, visibility
    /// toggle, node add/destroy. The renderer compares against its
    /// cached `tlas_built_version` and rebuilds TLAS + instance-data
    /// buffer when they differ. Mirror of `shadow_version`'s pattern.
    pub tlas_version: u64,
    /// Node handles whose BLAS was (re)created this frame and has not
    /// been built yet. `prepare()` pushes entries; the renderer drains
    /// this list and submits the builds in its main frame encoder via
    /// `CommandEncoder::build_acceleration_structures`.
    pub pending_blas_builds: Vec<f64>,
    /// Ticket 013 — node handles waiting for their mesh card to be
    /// rasterised into the shared atlas. Renderer drains this list
    /// at frame start (via a dedicated capture pass) before the
    /// probe chain runs. Populated alongside BLAS creation so each
    /// new mesh gets exactly one card.
    pub pending_card_captures: Vec<f64>,
    /// Bump allocator for card-atlas slots. Grows monotonically
    /// across the lifetime of the scene — free'd slots aren't
    /// reclaimed in V1 (Sponza fits comfortably in 1024 slots; loop
    /// back when scenes start exceeding capacity).
    pub next_card_slot: u32,
    /// 6-slot card blocks returned by destroyed nodes, reused before
    /// next_card_slot grows — without recycling, create/destroy cycles
    /// exhaust the fixed-size card atlas.
    free_card_blocks: Vec<u32>,
    /// Ticket 014 — node handles whose per-mesh SDF still needs to
    /// be baked. Populated alongside BLAS creation; renderer drains
    /// in a per-frame budget via a compute pass. Static meshes
    /// never re-bake once their SDF lands.
    pub pending_sdf_bakes: Vec<f64>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: HandleRegistry::new(),
            uniform_pool: None,
            uniform_pool_capacity: 0,
            next_slot: 0,
            scratch: Vec::new(),
            // Start at 1 so the shadow-cache's initial 0 always differs
            // on the first frame and forces an initial render.
            shadow_version: 1,
            hw_rt_enabled: false,
            tlas_version: 1,
            pending_blas_builds: Vec::new(),
            pending_card_captures: Vec::new(),
            next_card_slot: 0,
            free_card_blocks: Vec::new(),
            pending_sdf_bakes: Vec::new(),
        }
    }

    pub fn create_node(&mut self) -> f64 {
        // New nodes default to `cast_shadow = true`, so the first
        // `set_transform` + `update_geometry` will dirty shadows
        // anyway. Bumping here too costs nothing and keeps the
        // invalidation story simple.
        self.shadow_version = self.shadow_version.wrapping_add(1);
        self.tlas_version = self.tlas_version.wrapping_add(1);
        self.nodes.alloc(SceneNode::new())
    }

    pub fn destroy_node(&mut self, handle: f64) {
        if let Some(node) = self.nodes.get(handle) {
            if node.visible && node.cast_shadow {
                self.shadow_version = self.shadow_version.wrapping_add(1);
            }
            if node.visible {
                self.tlas_version = self.tlas_version.wrapping_add(1);
            }
            // Recycle the node's 6-slot card block. The freed node's GPU
            // buffers/BLAS/SDF drop with the SceneNode itself (wgpu
            // releases them once in-flight work completes).
            if let Some(first) = node.card_first_slot {
                self.free_card_blocks.push(first);
            }
        }
        self.nodes.free(handle);
    }

    /// Position + Y-rotation + uniform-scale convenience setter. Exists
    /// because the full-matrix `set_transform` crosses the FFI as an
    /// i64 pointer parameter, which Perry 0.5.x rejects for JS arrays;
    /// six f64 scalars stay register-friendly on every ABI. The yaw
    /// convention matches `draw_model_rotated`:
    /// world.x = c·x + s·z, world.z = −s·x + c·z.
    pub fn set_trs(&mut self, handle: f64, px: f32, py: f32, pz: f32, yaw: f32, scale: f32) {
        let (s, c) = yaw.sin_cos();
        let m = [
            [c * scale, 0.0, -s * scale, 0.0],
            [0.0, scale, 0.0, 0.0],
            [s * scale, 0.0, c * scale, 0.0],
            [px, py, pz, 1.0],
        ];
        self.set_transform(handle, m);
    }

    pub fn set_transform(&mut self, handle: f64, matrix: [[f32; 4]; 4]) {
        if let Some(node) = self.nodes.get_mut(handle) {
            // Only dirty shadows when the transform actually changed on
            // a shadow-casting node. Skeletal animation leaves the node
            // transform untouched (joints drive the mesh), so static
            // scenes with animated characters still cache well.
            let changed = node.transform != matrix;
            if changed && node.cast_shadow && node.visible {
                self.shadow_version = self.shadow_version.wrapping_add(1);
            }
            if changed && node.visible {
                self.tlas_version = self.tlas_version.wrapping_add(1);
            }
            node.transform = matrix;
        }
    }

    pub fn set_visible(&mut self, handle: f64, visible: bool) {
        if let Some(node) = self.nodes.get_mut(handle) {
            if node.visible != visible {
                self.tlas_version = self.tlas_version.wrapping_add(1);
            }
            if node.cast_shadow && node.visible != visible {
                self.shadow_version = self.shadow_version.wrapping_add(1);
            }
            node.visible = visible;
        }
    }

    pub fn set_cast_shadow(&mut self, handle: f64, cast: bool) {
        if let Some(node) = self.nodes.get_mut(handle) {
            if node.visible && node.cast_shadow != cast {
                self.shadow_version = self.shadow_version.wrapping_add(1);
            }
            node.cast_shadow = cast;
        }
    }

    pub fn set_receive_shadow(&mut self, handle: f64, receive: bool) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.receive_shadow = receive;
        }
    }

    /// Ticket 013 V3 — mark a mesh as dynamic so its card atlas slots
    /// get re-captured every frame. Off by default; call once per
    /// skeletal / morph-target / procedurally-animated node so its
    /// indirect-bounce contribution tracks the animation.
    pub fn set_mesh_dynamic(&mut self, handle: f64, dynamic: bool) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.card_dynamic = dynamic;
        }
    }

    /// Ticket 014 V2 — gather every visible mesh's triangles into a
    /// single world-space buffer so the scene-wide SDF clipmap bake
    /// can treat them as one big mesh. Vertex layout matches the
    /// shader's hard-coded 12-f32 Vertex3D stride; only position is
    /// meaningful for the UDF (normals/colour/UV are left zero).
    /// Returns (vertex_buf, index_buf, total_triangle_count).
    pub fn build_world_triangles(&self) -> (Vec<f32>, Vec<u32>, u32) {
        // 12 floats per vertex to match `Vertex3D` stride the bake
        // shader indexes with. position + zero-padding.
        const STRIDE: usize = 12;
        let mut vbuf: Vec<f32> = Vec::new();
        let mut ibuf: Vec<u32> = Vec::new();
        let mut tri_count: u32 = 0;
        for (_, node) in self.nodes.iter() {
            if !node.visible || node.vertices.is_empty() || node.indices.is_empty() {
                continue;
            }
            let base = (vbuf.len() / STRIDE) as u32;
            let t = &node.transform;
            for v in &node.vertices {
                let px = v.position[0];
                let py = v.position[1];
                let pz = v.position[2];
                let wx = t[0][0]*px + t[1][0]*py + t[2][0]*pz + t[3][0];
                let wy = t[0][1]*px + t[1][1]*py + t[2][1]*pz + t[3][1];
                let wz = t[0][2]*px + t[1][2]*py + t[2][2]*pz + t[3][2];
                // Zero-pad the remaining 9 floats — only position is
                // read by `SDF_BAKE_WGSL`.
                vbuf.extend_from_slice(&[wx, wy, wz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            }
            for &idx in &node.indices {
                ibuf.push(base + idx);
            }
            tri_count += (node.indices.len() / 3) as u32;
        }
        (vbuf, ibuf, tri_count)
    }

    pub fn set_parent(&mut self, handle: f64, parent: f64) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.parent = parent;
        }
    }

    /// Set (or replace) a reduced-detail variant. `lod_index` is 0-based
    /// into the reduced set (the node's base geometry is implicitly the
    /// finest level). `max_coverage` is the screen-coverage threshold
    /// below which this level is used; give coarser levels smaller
    /// thresholds. Pass empty vertices to remove the level.
    pub fn set_lod_geometry(
        &mut self,
        handle: f64,
        lod_index: usize,
        vertices: Vec<Vertex3D>,
        indices: Vec<u32>,
        max_coverage: f32,
    ) {
        let Some(node) = self.nodes.get_mut(handle) else { return };
        if vertices.is_empty() {
            if lod_index < node.lods.len() {
                node.lods.remove(lod_index);
            }
            return;
        }
        while node.lods.len() <= lod_index {
            node.lods.push(LodLevel {
                vertices: Vec::new(),
                indices: Vec::new(),
                max_coverage: 0.0,
                gpu_vb: None,
                gpu_ib: None,
                gpu_index_count: 0,
                dirty: true,
            });
        }
        let lod = &mut node.lods[lod_index];
        lod.vertices = vertices;
        lod.indices = indices;
        lod.max_coverage = max_coverage;
        lod.dirty = true;
    }

    pub fn update_geometry(&mut self, handle: f64, vertices: Vec<Vertex3D>, indices: Vec<u32>) {
        if let Some(node) = self.nodes.get_mut(handle) {
            // Recompute bounds from vertex positions (Q5).
            let mut bmin = [f32::MAX; 3];
            let mut bmax = [f32::MIN; 3];
            for v in &vertices {
                for k in 0..3 {
                    if v.position[k] < bmin[k] { bmin[k] = v.position[k]; }
                    if v.position[k] > bmax[k] { bmax[k] = v.position[k]; }
                }
            }
            if vertices.is_empty() {
                bmin = [0.0; 3];
                bmax = [0.0; 3];
            }
            node.bounds_min = bmin;
            node.bounds_max = bmax;
            node.vertices = vertices;
            node.indices = indices;
            node.geo_dirty = true;
            if node.cast_shadow && node.visible {
                self.shadow_version = self.shadow_version.wrapping_add(1);
            }
            if node.visible {
                self.tlas_version = self.tlas_version.wrapping_add(1);
            }
        }
    }

    // ---- Q4: transform read-back -------------------------------------------

    /// Read back the current 4x4 transform matrix of a scene node.
    pub fn get_transform(&self, handle: f64) -> [[f32; 4]; 4] {
        match self.nodes.get(handle) {
            Some(node) => node.transform,
            None => crate::renderer::IDENTITY_MAT4,
        }
    }

    // ---- Q5: world-space bounds query --------------------------------------

    /// Return the cached AABB of a scene node's geometry (local space).
    pub fn get_bounds(&self, handle: f64) -> ([f32; 3], [f32; 3]) {
        match self.nodes.get(handle) {
            Some(node) => (node.bounds_min, node.bounds_max),
            None => ([0.0; 3], [0.0; 3]),
        }
    }

    /// World-space AABB of every visible, shadow-casting node.
    /// Used to auto-fit the directional shadow ortho volume — no
    /// scene-specific magic numbers, works for Sponza / Bistro /
    /// anything a user loads.
    ///
    /// Returns `None` if the scene is empty (caller should fall back
    /// to a safe default).
    pub fn compute_shadow_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut bmin = [f32::MAX; 3];
        let mut bmax = [f32::MIN; 3];
        let mut any = false;
        for (_h, node) in self.nodes.iter() {
            if !node.visible || !node.cast_shadow {
                continue;
            }
            if node.bounds_min[0] > node.bounds_max[0] {
                continue; // empty bounds
            }
            // Transform the 8 local-AABB corners by the node's world matrix
            // and union into the running bounds.
            let t = &node.transform;
            for ix in 0..2 {
                for iy in 0..2 {
                    for iz in 0..2 {
                        let lx = if ix == 0 { node.bounds_min[0] } else { node.bounds_max[0] };
                        let ly = if iy == 0 { node.bounds_min[1] } else { node.bounds_max[1] };
                        let lz = if iz == 0 { node.bounds_min[2] } else { node.bounds_max[2] };
                        // column-major mat4 * vec4(x,y,z,1)
                        let wx = t[0][0]*lx + t[1][0]*ly + t[2][0]*lz + t[3][0];
                        let wy = t[0][1]*lx + t[1][1]*ly + t[2][1]*lz + t[3][1];
                        let wz = t[0][2]*lx + t[1][2]*ly + t[2][2]*lz + t[3][2];
                        if wx < bmin[0] { bmin[0] = wx; }
                        if wy < bmin[1] { bmin[1] = wy; }
                        if wz < bmin[2] { bmin[2] = wz; }
                        if wx > bmax[0] { bmax[0] = wx; }
                        if wy > bmax[1] { bmax[1] = wy; }
                        if wz > bmax[2] { bmax[2] = wz; }
                        any = true;
                    }
                }
            }
        }
        if any { Some((bmin, bmax)) } else { None }
    }

    // ---- Q7: user data -----------------------------------------------------

    pub fn set_user_data(&mut self, handle: f64, data: i64) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.user_data = data;
        }
    }

    pub fn get_user_data(&self, handle: f64) -> i64 {
        match self.nodes.get(handle) {
            Some(node) => node.user_data,
            None => 0,
        }
    }

    pub fn set_material_color(&mut self, handle: f64, r: f32, g: f32, b: f32, a: f32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.color = [r, g, b];
            node.material.opacity = a;
        }
    }

    pub fn set_material_pbr(&mut self, handle: f64, roughness: f32, metalness: f32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.roughness = roughness;
            node.material.metalness = metalness;
            // Factors live in the material uniform, which is only rebuilt
            // together with the bind group — without dirtying, factor
            // changes after the first render never applied.
            node.mat_dirty = true;
        }
    }

    /// glTF MASK alpha cutoff for the node's material (0 = opaque).
    /// Routes the node through the scene shader's alpha-cutout path.
    pub fn set_material_alpha_cutoff(&mut self, handle: f64, cutoff: f32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.alpha_cutoff = cutoff;
            node.mat_dirty = true;
        }
    }

    /// Q8: Set a water-like material on a scene node. The actual animated
    /// wave shader requires a dedicated WGSL pipeline pass (deferred).
    /// For now, this sets a translucent tinted material that approximates water.
    pub fn set_material_water(&mut self, handle: f64, _wave_amp: f32, _wave_speed: f32, r: f32, g: f32, b: f32, a: f32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.color = [r, g, b];
            node.material.opacity = a;
            node.material.roughness = 0.1;
            node.material.metalness = 0.3;
        }
    }

    pub fn set_material_texture(&mut self, handle: f64, texture_idx: u32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.texture_idx = texture_idx;
            node.mat_dirty = true;
        }
    }

    pub fn set_material_normal_texture(&mut self, handle: f64, texture_idx: u32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.normal_texture_idx = texture_idx;
            node.mat_dirty = true;
        }
    }

    pub fn set_material_metallic_roughness_texture(&mut self, handle: f64, texture_idx: u32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.metallic_roughness_texture_idx = texture_idx;
            node.mat_dirty = true;
        }
    }

    pub fn set_material_emissive_texture(&mut self, handle: f64, texture_idx: u32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.emissive_texture_idx = texture_idx;
            node.mat_dirty = true;
        }
    }

    pub fn set_material_emissive_factor(&mut self, handle: f64, r: f32, g: f32, b: f32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.emissive = [r, g, b];
            node.mat_dirty = true;
        }
    }

    /// Prepare GPU resources for all visible nodes. Must be called before render().
    /// Creates/updates vertex buffers, index buffers, and uniform bind groups.
    /// `prev_vp_matrix` is the previous frame's view-projection — used together
    /// with each node's `prev_transform` to compute `prev_mvp` for the velocity
    /// buffer (motion blur + TAA per-object reprojection).
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vp_matrix: &[[f32; 4]; 4],
        prev_vp_matrix: &[[f32; 4]; 4],
        uniform_layout: &wgpu::BindGroupLayout,
        occlusion: Option<&crate::renderer::OcclusionCuller>,
    ) {
        let frustum = extract_frustum_planes(vp_matrix);

        // Phase 1: upload geometry for any freshly-added or dirty nodes,
        // assign uniform slots, and count how many nodes we'll draw.
        // Borrow splits so we can push to `pending_blas_builds` while
        // iterating `nodes` mutably.
        let hw_rt = self.hw_rt_enabled;
        let pending_blas = &mut self.pending_blas_builds;
        let pending_cards = &mut self.pending_card_captures;
        let pending_sdf = &mut self.pending_sdf_bakes;
        let next_card_slot = &mut self.next_card_slot;
        let free_card_blocks = &mut self.free_card_blocks;
        let mut visible_count: u32 = 0;
        for (handle, node) in self.nodes.iter_mut() {
            if !node.visible || node.indices.is_empty() {
                continue;
            }
            // Frustum cull against world-space AABB. Transform the local
            // bounds into world space by applying the node's transform
            // to the 8 corners; use the min/max of the result.
            if node.bounds_min[0] <= node.bounds_max[0] {
                let t = &node.transform;
                let mut wmin = [f32::MAX; 3];
                let mut wmax = [f32::MIN; 3];
                for ix in 0..2 {
                    for iy in 0..2 {
                        for iz in 0..2 {
                            let lx = if ix == 0 { node.bounds_min[0] } else { node.bounds_max[0] };
                            let ly = if iy == 0 { node.bounds_min[1] } else { node.bounds_max[1] };
                            let lz = if iz == 0 { node.bounds_min[2] } else { node.bounds_max[2] };
                            let wx = t[0][0]*lx + t[1][0]*ly + t[2][0]*lz + t[3][0];
                            let wy = t[0][1]*lx + t[1][1]*ly + t[2][1]*lz + t[3][1];
                            let wz = t[0][2]*lx + t[1][2]*ly + t[2][2]*lz + t[3][2];
                            if wx < wmin[0] { wmin[0] = wx; }
                            if wy < wmin[1] { wmin[1] = wy; }
                            if wz < wmin[2] { wmin[2] = wz; }
                            if wx > wmax[0] { wmax[0] = wx; }
                            if wy > wmax[1] { wmax[1] = wy; }
                            if wz > wmax[2] { wmax[2] = wz; }
                        }
                    }
                }
                node.in_view_frustum = !aabb_outside_frustum(&frustum, wmin, wmax);
                node.world_bounds_min = wmin;
                node.world_bounds_max = wmax;
                // Hi-Z occlusion: only worth testing what survived the
                // frustum; every uncertain case inside test_aabb
                // resolves to visible.
                node.occluded = node.in_view_frustum
                    && occlusion.is_some_and(|o| !o.test_aabb(wmin, wmax));

                // LOD selection by projected screen coverage: longest
                // NDC extent of the world AABB. Corners at/behind the
                // near plane force the base level (huge on screen).
                if !node.lods.is_empty() && node.in_view_frustum && !node.occluded {
                    let coverage = aabb_screen_coverage(vp_matrix, wmin, wmax);
                    let current = node.active_lod;
                    let mut chosen: i32 = -1;
                    for (i, lod) in node.lods.iter().enumerate() {
                        // 10% hysteresis: stepping coarser needs coverage
                        // clearly below the threshold, stepping finer
                        // clearly above — kills boundary flicker.
                        let t = if current >= i as i32 {
                            lod.max_coverage * 1.05
                        } else {
                            lod.max_coverage * 0.95
                        };
                        if coverage < t {
                            chosen = i as i32;
                        }
                    }
                    node.active_lod = chosen;
                } else {
                    node.active_lod = -1;
                }
            } else {
                // Empty or uninitialized bounds — can't cull, play safe.
                node.in_view_frustum = true;
                node.occluded = false;
                node.world_bounds_min = [f32::MAX; 3];
                node.world_bounds_max = [f32::MIN; 3];
            }
            // Upload any dirty LOD variants (plain vertex/index buffers —
            // LODs never feed BLAS/SDF, those read the base geometry).
            for lod in node.lods.iter_mut().filter(|l| l.dirty) {
                lod.gpu_vb = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene_node_lod_vb"),
                    contents: bytemuck::cast_slice(&lod.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }));
                lod.gpu_ib = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene_node_lod_ib"),
                    contents: bytemuck::cast_slice(&lod.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }));
                lod.gpu_index_count = lod.indices.len() as u32;
                lod.dirty = false;
            }

            if node.geo_dirty || node.gpu_vb.is_none() {
                // Ticket 007b: widen buffer usage when HW RT is on so
                // the same buffer can back both the raster draw and
                // the BLAS build. Cheap — no measurable cost when RT
                // is off.
                // Ticket 014 V3 — STORAGE is unconditional now so the
                // scene-wide SDF clipmap can be baked on SW-only
                // adapters too (web, older Android, Intel iGPUs). The
                // cost is a buffer-usage-flag bit — no runtime
                // overhead on non-RT adapters that don't read it.
                // BLAS_INPUT stays gated on `hw_rt` since it's wgpu-29
                // ray-tracing-only.
                let vb_usage = if hw_rt {
                    wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::BLAS_INPUT
                        | wgpu::BufferUsages::STORAGE
                } else {
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE
                };
                let ib_usage = if hw_rt {
                    wgpu::BufferUsages::INDEX
                        | wgpu::BufferUsages::BLAS_INPUT
                        | wgpu::BufferUsages::STORAGE
                } else {
                    wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE
                };
                node.gpu_vb = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene_node_vb"),
                    contents: bytemuck::cast_slice(&node.vertices),
                    usage: vb_usage,
                }));
                node.gpu_ib = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene_node_ib"),
                    contents: bytemuck::cast_slice(&node.indices),
                    usage: ib_usage,
                }));
                node.gpu_index_count = node.indices.len() as u32;
                node.gpu_vertex_count = node.vertices.len() as u32;
                node.geo_dirty = false;

                // Ticket 014 V4 — flat normal/albedo + card-slot
                // allocation now happen regardless of `hw_rt`. The SW
                // SDF sphere-trace's broad-phase hit uses these same
                // per-instance fields to sample the card radiance
                // atlas, so they need to be populated even when no
                // BLAS / TLAS is built. BLAS + per-mesh SDF bake
                // remain hw-rt-gated because only the HW trace reads
                // those (and the per-mesh SDFs are currently dormant).
                if !node.indices.is_empty() {
                    let mut n_sum = [0.0_f32; 3];
                    for v in &node.vertices {
                        n_sum[0] += v.normal[0];
                        n_sum[1] += v.normal[1];
                        n_sum[2] += v.normal[2];
                    }
                    let t = &node.transform;
                    let nx = t[0][0]*n_sum[0] + t[1][0]*n_sum[1] + t[2][0]*n_sum[2];
                    let ny = t[0][1]*n_sum[0] + t[1][1]*n_sum[1] + t[2][1]*n_sum[2];
                    let nz = t[0][2]*n_sum[0] + t[1][2]*n_sum[1] + t[2][2]*n_sum[2];
                    let len = (nx*nx + ny*ny + nz*nz).sqrt();
                    if len > 1e-4 {
                        node.flat_normal_ws = [nx / len, ny / len, nz / len];
                    } else {
                        // Radially symmetric — fall back to world-up.
                        node.flat_normal_ws = [0.0, 1.0, 0.0];
                    }
                    node.flat_albedo = node.material.color;

                    // Ticket 013 V2 — allocate 6 consecutive slots per
                    // signed AABB axis and schedule the capture pass.
                    // Runs on both HW and SW paths so the SDF trace's
                    // broad-phase hit can sample textured radiance.
                    if node.card_first_slot.is_none() {
                        let first = match free_card_blocks.pop() {
                            Some(reused) => reused,
                            None => {
                                let fresh = *next_card_slot;
                                *next_card_slot += 6;
                                fresh
                            }
                        };
                        node.card_first_slot = Some(first);
                        pending_cards.push(handle);
                    }

                    if hw_rt {
                        // BLAS creation only on HW-RT adapters.
                        let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                            vertex_format: wgpu::VertexFormat::Float32x3,
                            vertex_count: node.gpu_vertex_count,
                            index_format: Some(wgpu::IndexFormat::Uint32),
                            index_count: Some(node.gpu_index_count),
                            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                        };
                        node.blas = Some(device.create_blas(
                            &wgpu::CreateBlasDescriptor {
                                label: Some("scene_node_blas"),
                                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                            },
                            wgpu::BlasGeometrySizeDescriptors::Triangles {
                                descriptors: vec![size_desc],
                            },
                        ));
                        pending_blas.push(handle);

                        // Per-mesh SDF texture — currently dormant (V4
                        // dynamic-scene merge will consume it), but
                        // cheap to allocate alongside the BLAS.
                        if node.mesh_sdf.is_none()
                            && node.bounds_min[0] < node.bounds_max[0]
                            && node.bounds_min[1] < node.bounds_max[1]
                            && node.bounds_min[2] < node.bounds_max[2]
                        {
                            let (sdf_tex, sdf_view) = crate::renderer::create_mesh_sdf_texture_public(
                                device,
                                "scene_node_sdf",
                            );

                            // Ticket 022 — content-hash the geometry and
                            // try the on-disk SDF cache before scheduling
                            // a GPU bake. Vertex layout is interleaved;
                            // pull the position prefix out as a
                            // [[f32; 3]] slice so the hash only sees
                            // geometry-relevant bytes.
                            let positions: Vec<[f32; 3]> =
                                node.vertices.iter().map(|v| v.position).collect();
                            let hash = crate::sdf_cache::compute_mesh_hash(
                                &positions, &node.indices,
                            );
                            node.mesh_hash = Some(hash);

                            if let Some(bytes) = crate::sdf_cache::load(hash) {
                                // Cache hit — pad the tightly-packed
                                // 128 B/row payload to 256 B/row so it
                                // clears wgpu's COPY_BYTES_PER_ROW
                                // alignment, then upload directly and
                                // skip the bake. Native cache size
                                // stays compact (128 KB/mesh on disk);
                                // the 128 KB padding allocation is
                                // free'd immediately after the call.
                                const RES: u32 = crate::sdf_cache::VOXEL_RES;
                                let row_tight = (RES * 4) as usize;
                                let row_padded = ((row_tight + 255) & !255) as u32;
                                let mut padded = vec![
                                    0u8;
                                    (row_padded as usize) * (RES as usize) * (RES as usize)
                                ];
                                for z in 0..RES as usize {
                                    for y in 0..RES as usize {
                                        let src_off = (z * RES as usize + y) * row_tight;
                                        let dst_off = (z * RES as usize + y) * row_padded as usize;
                                        padded[dst_off..dst_off + row_tight]
                                            .copy_from_slice(&bytes[src_off..src_off + row_tight]);
                                    }
                                }
                                queue.write_texture(
                                    wgpu::TexelCopyTextureInfo {
                                        texture: &sdf_tex,
                                        mip_level: 0,
                                        origin: wgpu::Origin3d::ZERO,
                                        aspect: wgpu::TextureAspect::All,
                                    },
                                    &padded,
                                    wgpu::TexelCopyBufferLayout {
                                        offset: 0,
                                        bytes_per_row: Some(row_padded),
                                        rows_per_image: Some(RES),
                                    },
                                    wgpu::Extent3d {
                                        width: RES,
                                        height: RES,
                                        depth_or_array_layers: RES,
                                    },
                                );
                            } else {
                                pending_sdf.push(handle);
                            }

                            node.mesh_sdf = Some(sdf_tex);
                            node.mesh_sdf_view = Some(sdf_view);
                        }
                    }
                }
            }
            if node.uniform_slot.is_none() {
                node.uniform_slot = Some(self.next_slot);
                self.next_slot += 1;
            }

            // Ticket 013 V3 — re-queue dynamic meshes every frame so
            // their card atlas slots get re-captured to track
            // animated geometry. Static meshes (default) stay out of
            // the queue after their one-shot first-frame capture.
            if node.card_dynamic && node.card_first_slot.is_some() {
                pending_cards.push(handle);
            }

            visible_count += 1;
        }

        // Phase 2: ensure pool is large enough. Grow with 2x + padding
        // so this branch is rare once the scene has stabilized.
        let needed_capacity = self.next_slot.max(32);
        if needed_capacity > self.uniform_pool_capacity {
            let new_cap = needed_capacity.next_power_of_two().max(64);
            let byte_size = (new_cap as u64) * NODE_UNIFORM_STRIDE;
            self.uniform_pool = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scene_node_uniform_pool"),
                size: byte_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.uniform_pool_capacity = new_cap;
            self.scratch.resize(byte_size as usize, 0);
            // Invalidate every per-node bind group — they reference
            // the old buffer.
            for (_handle, node) in self.nodes.iter_mut() {
                node.gpu_uniform_bg = None;
            }
        }

        let Some(pool_buf) = self.uniform_pool.as_ref() else { return };
        let uniform_size = std::mem::size_of::<NodeUniforms>();
        let stride = NODE_UNIFORM_STRIDE as usize;

        // Phase 3: build the packed payload + create any missing bind
        // groups. Bind-group creation is rare (only on first sight of
        // a slot or after pool grow); the hot per-frame path is just
        // the memcpy into scratch.
        let mut max_byte_offset: usize = 0;
        for (_handle, node) in self.nodes.iter_mut() {
            if !node.visible || node.indices.is_empty() {
                continue;
            }
            let Some(slot) = node.uniform_slot else { continue };

            let mvp = mat4_mul(vp_matrix, &node.transform);
            let prev_mvp = mat4_mul(prev_vp_matrix, &node.prev_transform);
            // Guard against NaN/Inf opacity (Perry TS passes NaN
            // when a default-arg alpha isn't provided) — a single
            // NaN in model_tint.w propagates through every shader
            // output.
            let opacity = if node.material.opacity.is_finite() {
                node.material.opacity
            } else {
                1.0
            };
            let tint = [
                node.material.color[0],
                node.material.color[1],
                node.material.color[2],
                opacity,
            ];
            let uniforms = NodeUniforms { mvp, model: node.transform, prev_mvp, model_tint: tint };
            node.prev_transform = node.transform;

            let off = (slot as usize) * stride;
            self.scratch[off..off + uniform_size].copy_from_slice(bytemuck::bytes_of(&uniforms));
            max_byte_offset = max_byte_offset.max(off + stride);

            if node.gpu_uniform_bg.is_none() {
                node.gpu_uniform_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("scene_node_uniform_bg"),
                    layout: uniform_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: pool_buf,
                            offset: (slot as u64) * NODE_UNIFORM_STRIDE,
                            size: std::num::NonZeroU64::new(uniform_size as u64),
                        }),
                    }],
                }));
            }
        }

        // Phase 4: single write — replaces what used to be N per-node
        // queue.write_buffer calls. On sponza (68 nodes) this cut
        // scene_prepare from ~1.7 ms to ~0.3 ms.
        if visible_count > 0 && max_byte_offset > 0 {
            queue.write_buffer(pool_buf, 0, &self.scratch[..max_byte_offset]);
        }
    }

    /// Build / refresh per-node material bind groups for the scene
    /// pipeline. Must be called every frame after `prepare` and before
    /// `render`. Only rebuilds when a material changed (mat_dirty).
    pub fn prepare_materials(&mut self, renderer: &crate::renderer::Renderer) {
        for (_handle, node) in self.nodes.iter_mut() {
            if !node.visible || node.indices.is_empty() {
                continue;
            }
            if node.mat_dirty || node.gpu_material_bg.is_none() {
                // Allocate or reuse the per-material uniform buffer.
                // (Could be updated in place when factors change, but
                // the current path always rebuilds together with the
                // bind group — cheap and simpler.)
                let uniform = renderer.create_scene_material_uniform(
                    node.material.metalness,
                    node.material.roughness,
                    node.material.emissive,
                    node.material.metallic_roughness_texture_idx != 0,
                    // MASK cutoff from the node material (0 = opaque).
                    // attach_model carries it over from the glTF mesh so
                    // foliage cards keep their cutout + two-sided shading
                    // + wind sway on the scene-graph path.
                    node.material.alpha_cutoff,
                );
                let bg = renderer.create_scene_material_bg(
                    node.material.texture_idx,
                    node.material.normal_texture_idx,
                    node.material.metallic_roughness_texture_idx,
                    node.material.emissive_texture_idx,
                    node.material.occlusion_texture_idx,
                    &uniform,
                );
                node.gpu_material_bg = Some(bg);
                node.gpu_material_uniform_buf = Some(uniform);
                // MASK materials also get an alpha-tested shadow-caster
                // bind group so foliage casts dappled shadows on the
                // scene-graph path (mirrors the cached-model path).
                node.gpu_shadow_cutout_bg = if node.material.alpha_cutoff > 0.0 {
                    Some(renderer.create_shadow_cutout_bg(
                        node.material.texture_idx,
                        node.material.alpha_cutoff,
                    ))
                } else {
                    None
                };
                node.mat_dirty = false;
            }
        }
    }

    /// Render all visible scene nodes into the given render pass.
    /// Must be called after prepare() and after the pipeline/lighting/joints are set.
    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        for (_handle, node) in self.nodes.iter() {
            if !node.visible || node.indices.is_empty() || !node.in_view_frustum || node.occluded {
                continue;
            }
            // Active LOD overrides the base buffers for the camera pass
            // (shadows/picking/BLAS keep using the base geometry).
            let (vb, ib, index_count) = match node
                .lods
                .get(node.active_lod.max(0) as usize)
                .filter(|_| node.active_lod >= 0)
                .and_then(|l| Some((l.gpu_vb.as_ref()?, l.gpu_ib.as_ref()?, l.gpu_index_count)))
            {
                Some(lod) => lod,
                None => {
                    let Some(vb) = &node.gpu_vb else { continue };
                    let Some(ib) = &node.gpu_ib else { continue };
                    (vb, ib, node.gpu_index_count)
                }
            };
            let Some(bg) = &node.gpu_uniform_bg else { continue };
            let Some(mat_bg) = &node.gpu_material_bg else { continue };
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(2, mat_bg, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..index_count, 0, 0..1);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.iter().count()
    }

    /// Draw list for the planar-reflection probe: every visible node with
    /// uploaded geometry as (vb, ib, index_count, material_bg, transform).
    /// Frustum / occlusion flags are intentionally ignored — they were
    /// computed for the MAIN camera and the mirrored probe camera sees a
    /// different set. Base geometry only (no LOD swap): the probe is
    /// half-res and consumed through a perturbed water lookup, where a
    /// LOD pop would be more visible than the detail it saves.
    /// (Treats node.transform as world — flat hierarchies.)
    pub fn reflect_draw_list(&self)
        -> Vec<(&wgpu::Buffer, &wgpu::Buffer, u32, &wgpu::BindGroup, [[f32; 4]; 4])>
    {
        let mut out = Vec::new();
        for (_handle, node) in self.nodes.iter() {
            if !node.visible || node.indices.is_empty() { continue; }
            let Some(vb) = &node.gpu_vb else { continue };
            let Some(ib) = &node.gpu_ib else { continue };
            let Some(mat_bg) = &node.gpu_material_bg else { continue };
            out.push((vb, ib, node.gpu_index_count, mat_bg, node.transform));
        }
        out
    }
}

// ============================================================
// Matrix math (4x4, column-major)
// ============================================================

// ============================================================
// Frustum culling
// ============================================================
// Gribb-Hartmann plane extraction: for a column-major clip matrix M,
// each plane = ±row_i + row_3. We build 6 planes (left/right/bottom/
// top/near/far) in world space directly from the VP matrix, so every
// plane-test below is a world-space dot product.
//
// A node's world-space AABB is outside the frustum if ALL 8 of its
// corners are on the negative side of ANY single plane. The standard
// "positive-vertex-only" optimization is skipped here — testing 8
// corners is still a few dozen multiplies per node, trivial compared
// to the per-node GPU cost we skip on a cull hit.
//
// Plane format: [nx, ny, nz, d] where `nx*x + ny*y + nz*z + d >= 0`
// means the point is inside that plane's half-space. No normalization
// — we only care about the sign.

/// Longest NDC-extent of a world AABB under `vp` — the "screen coverage"
/// that drives LOD selection (1.0 = spans the full viewport). Corners at
/// or behind the near plane return 1.0 (force the finest level).
fn aabb_screen_coverage(vp: &[[f32; 4]; 4], wmin: [f32; 3], wmax: [f32; 3]) -> f32 {
    let mut lo = [f32::MAX, f32::MAX];
    let mut hi = [f32::MIN, f32::MIN];
    for ix in 0..2 {
        for iy in 0..2 {
            for iz in 0..2 {
                let x = if ix == 0 { wmin[0] } else { wmax[0] };
                let y = if iy == 0 { wmin[1] } else { wmax[1] };
                let z = if iz == 0 { wmin[2] } else { wmax[2] };
                let cw = vp[0][3] * x + vp[1][3] * y + vp[2][3] * z + vp[3][3];
                if cw <= 1e-3 {
                    return 1.0;
                }
                let cx = (vp[0][0] * x + vp[1][0] * y + vp[2][0] * z + vp[3][0]) / cw;
                let cy = (vp[0][1] * x + vp[1][1] * y + vp[2][1] * z + vp[3][1]) / cw;
                lo[0] = lo[0].min(cx);
                lo[1] = lo[1].min(cy);
                hi[0] = hi[0].max(cx);
                hi[1] = hi[1].max(cy);
            }
        }
    }
    // NDC spans -1..1, so extent/2 = fraction of the viewport.
    (((hi[0] - lo[0]).max(hi[1] - lo[1])) * 0.5).clamp(0.0, 1.0)
}

pub(crate) fn extract_frustum_planes(vp: &[[f32; 4]; 4]) -> [[f32; 4]; 6] {
    // Row vectors of the column-major matrix: row_i[col] = vp[col][i].
    let row = |i: usize| [vp[0][i], vp[1][i], vp[2][i], vp[3][i]];
    let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
    let add = |a: [f32;4], b: [f32;4]| [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]];
    let sub = |a: [f32;4], b: [f32;4]| [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]];
    [
        add(r3, r0), // left
        sub(r3, r0), // right
        add(r3, r1), // bottom
        sub(r3, r1), // top
        r2,          // near (wgpu uses 0..1 depth → near = row_2)
        sub(r3, r2), // far
    ]
}

pub(crate) fn aabb_outside_frustum(planes: &[[f32; 4]; 6], bmin: [f32; 3], bmax: [f32; 3]) -> bool {
    for p in planes.iter() {
        let mut all_outside = true;
        for ix in 0..2 {
            let x = if ix == 0 { bmin[0] } else { bmax[0] };
            for iy in 0..2 {
                let y = if iy == 0 { bmin[1] } else { bmax[1] };
                for iz in 0..2 {
                    let z = if iz == 0 { bmin[2] } else { bmax[2] };
                    if p[0]*x + p[1]*y + p[2]*z + p[3] >= 0.0 {
                        all_outside = false;
                        break;
                    }
                }
                if !all_outside { break; }
            }
            if !all_outside { break; }
        }
        if all_outside { return true; }
    }
    false
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            result[col][row] = a[0][row] * b[col][0]
                             + a[1][row] * b[col][1]
                             + a[2][row] * b[col][2]
                             + a[3][row] * b[col][3];
        }
    }
    result
}
