//! Retained scene graph for `bloom_scene_*` FFI.
//!
//! Mirrors `bloom-shared/scene.rs` but stripped down to what SceneKit can
//! consume: nodes carry a 4x4 transform, color/PBR material, optional
//! texture handle, and an optional indexed triangle mesh. Swift polls a
//! snapshot each frame via `bloom_watchos_scene_*` FFI and mirrors the
//! graph into SCNNodes.
//!
//! `geometry_version` bumps on every `update_geometry` call so the Swift
//! side can cache `SCNGeometry` objects across frames — rebuilding geometry
//! each frame for static meshes would burn the watch's GPU.

use std::sync::Mutex;

// Dirty-flag bits — Node.dirty_flags is the bitwise-OR of these. Each Node
// mutator sets its corresponding bit; drain_dirty reports them to Swift and
// clears on consumption. Newly created nodes start with DIRTY_ALL so Swift
// sees every field on the first drain.
pub const DIRTY_TRANSFORM: u32  = 1 << 0;
pub const DIRTY_MATERIAL: u32   = 1 << 1;
pub const DIRTY_VISIBILITY: u32 = 1 << 2;
pub const DIRTY_PARENT: u32     = 1 << 3;
pub const DIRTY_GEOMETRY: u32   = 1 << 4;
pub const DIRTY_SKIN: u32       = 1 << 5;
pub const DIRTY_ALL: u32 = DIRTY_TRANSFORM | DIRTY_MATERIAL | DIRTY_VISIBILITY
                         | DIRTY_PARENT | DIRTY_GEOMETRY | DIRTY_SKIN;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SceneNodeInfo {
    pub dirty_flags: u32,
    pub handle: u32,
    pub parent: u32,
    pub visible: u32,
    pub geometry_version: u32,

    pub color: [f32; 4],
    pub roughness: f32,
    pub metalness: f32,
    /// Legacy single-slot texture handle (diffuse / base color). Kept for
    /// bloom_scene_set_material_texture which predates the PBR slot set.
    pub texture: u32,
    pub has_geometry: u32,

    pub transform: [f32; 16],

    /// PBR texture slots — set via bloom_scene_attach_model from a loaded
    /// glTF material. 0 = not set (fall back to color / roughness /
    /// metalness factors instead).
    pub tex_base_color: u32,
    pub tex_normal: u32,
    pub tex_metallic_roughness: u32,
    pub tex_emissive: u32,
    pub tex_occlusion: u32,
    /// Bumps whenever any skin_* data changes on this node. Swift rebuilds
    /// the SCNSkinner when it sees a new version. 0 = no skin.
    pub skin_version: u32,
}

struct Node {
    visible: bool,
    parent: u32,
    transform: [f32; 16],
    color: [f32; 4],
    roughness: f32,
    metalness: f32,
    texture: u32,
    tex_base_color: u32,
    tex_normal: u32,
    tex_metallic_roughness: u32,
    tex_emissive: u32,
    tex_occlusion: u32,
    positions: Vec<f32>,
    normals: Vec<f32>,
    uvs: Vec<f32>,
    indices: Vec<u32>,
    geometry_version: u32,
    /// Bitmask of DIRTY_* flags that need to go out to Swift on the next
    /// drain. Freshly created nodes start with DIRTY_ALL so Swift builds a
    /// full SCNNode on first sight.
    dirty_flags: u32,

    // Skinning data (SCNSkinner). Present only on the mesh-carrying node of
    // a skinned glTF primitive. skin_version bumps when any of the fields
    // below change, signalling Swift to rebuild the skinner.
    skin_joint_handles: Vec<u32>,       // bloom scene handles of the bones
    skin_inverse_bind: Vec<[f32; 16]>,  // one per joint
    skin_vertex_joints: Vec<u32>,       // 4 per vertex (as in glTF JOINTS_0)
    skin_vertex_weights: Vec<f32>,      // 4 per vertex (as in glTF WEIGHTS_0)
    skin_version: u32,

    // Animation tracks attached to this skinned mesh. Each track targets a
    // bone by bloom handle. Swift builds CAKeyframeAnimations from them when
    // attaching the SCNSkinner. anim_version bumps alongside skin_version.
    anim_tracks: Vec<AnimTrack>,
}

pub struct AnimTrack {
    pub bone_handle: u32,
    pub path: u32,       // 0 = translation, 1 = rotation, 2 = scale
    pub times: Vec<f32>,
    pub values: Vec<f32>,
}

impl Node {
    fn empty() -> Self {
        let mut t = [0.0f32; 16];
        t[0] = 1.0; t[5] = 1.0; t[10] = 1.0; t[15] = 1.0;  // identity
        Self {
            visible: true,
            parent: 0,
            transform: t,
            color: [1.0, 1.0, 1.0, 1.0],
            roughness: 0.6,
            metalness: 0.0,
            texture: 0,
            tex_base_color: 0, tex_normal: 0, tex_metallic_roughness: 0,
            tex_emissive: 0, tex_occlusion: 0,
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            geometry_version: 0,
            dirty_flags: DIRTY_ALL,
            skin_joint_handles: Vec::new(),
            skin_inverse_bind: Vec::new(),
            skin_vertex_joints: Vec::new(),
            skin_vertex_weights: Vec::new(),
            skin_version: 0,
            anim_tracks: Vec::new(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Light {
    pub kind: u32,  // 1 = directional, 2 = point
    pub _pad: u32,
    pub pos_or_dir: [f32; 3],
    pub range: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

struct Inner {
    nodes: Vec<Option<Node>>,  // 1-based: nodes[0] is always None (handle 0 = "none")
    lights: Vec<Light>,
    /// Handles of nodes destroyed since the last Swift drain. Swift consumes
    /// via drain_destroyed() and removes the matching SCNNodes from its cache.
    destroyed_queue: Vec<u32>,
}

static INNER: Mutex<Inner> = Mutex::new(Inner {
    nodes: Vec::new(),
    lights: Vec::new(),
    destroyed_queue: Vec::new(),
});

fn with<F, R>(f: F) -> R where F: FnOnce(&mut Inner) -> R {
    let mut g = INNER.lock().unwrap();
    if g.nodes.is_empty() {
        g.nodes.push(None);  // sentinel at index 0
    }
    f(&mut *g)
}

// ============================================================
// Mutations — called from bloom_scene_* FFI in lib.rs
// ============================================================

pub fn create() -> u32 {
    with(|inner| {
        // Reuse a freed slot if one exists.
        for (i, slot) in inner.nodes.iter_mut().enumerate().skip(1) {
            if slot.is_none() {
                *slot = Some(Node::empty());
                return i as u32;
            }
        }
        inner.nodes.push(Some(Node::empty()));
        (inner.nodes.len() - 1) as u32
    })
}

pub fn destroy(handle: u32) {
    with(|inner| {
        let existed = if let Some(slot) = inner.nodes.get_mut(handle as usize) {
            if slot.is_some() { *slot = None; true } else { false }
        } else { false };
        if existed { inner.destroyed_queue.push(handle); }
        // Any node with this as parent becomes a root — flag PARENT dirty
        // so Swift reparents it.
        for n in inner.nodes.iter_mut().flatten() {
            if n.parent == handle {
                n.parent = 0;
                n.dirty_flags |= DIRTY_PARENT;
            }
        }
    });
}

fn mutate<F: FnOnce(&mut Node)>(handle: u32, f: F) {
    with(|inner| {
        if let Some(Some(n)) = inner.nodes.get_mut(handle as usize) {
            f(n);
        }
    });
}

fn mark_dirty<F: FnOnce(&mut Node)>(handle: u32, bits: u32, f: F) {
    mutate(handle, |n| {
        f(n);
        n.dirty_flags |= bits;
    });
}

pub fn set_visible(handle: u32, v: bool) {
    mark_dirty(handle, DIRTY_VISIBILITY, |n| n.visible = v);
}
pub fn set_parent(handle: u32, parent: u32) {
    mark_dirty(handle, DIRTY_PARENT, |n| n.parent = parent);
}
pub fn set_transform(handle: u32, m: [f32; 16]) {
    mark_dirty(handle, DIRTY_TRANSFORM, |n| n.transform = m);
}
pub fn set_color(handle: u32, rgba: [f32; 4]) {
    mark_dirty(handle, DIRTY_MATERIAL, |n| n.color = rgba);
}
pub fn set_pbr(handle: u32, rough: f32, metal: f32) {
    mark_dirty(handle, DIRTY_MATERIAL, |n| { n.roughness = rough; n.metalness = metal; });
}
pub fn set_texture(handle: u32, tex: u32) {
    mark_dirty(handle, DIRTY_MATERIAL, |n| n.texture = tex);
}

pub fn set_pbr_textures(handle: u32,
    base_color: u32, normal: u32, metallic_roughness: u32,
    emissive: u32, occlusion: u32,
) {
    mark_dirty(handle, DIRTY_MATERIAL, |n| {
        n.tex_base_color = base_color;
        n.tex_normal = normal;
        n.tex_metallic_roughness = metallic_roughness;
        n.tex_emissive = emissive;
        n.tex_occlusion = occlusion;
    });
}

pub fn set_skin(handle: u32,
    joint_handles: Vec<u32>, inverse_bind: Vec<[f32; 16]>,
    vertex_joints: Vec<u32>, vertex_weights: Vec<f32>,
) {
    mark_dirty(handle, DIRTY_SKIN, |n| {
        n.skin_joint_handles = joint_handles;
        n.skin_inverse_bind = inverse_bind;
        n.skin_vertex_joints = vertex_joints;
        n.skin_vertex_weights = vertex_weights;
        n.skin_version = n.skin_version.wrapping_add(1);
    });
}

/// Replace the animation tracks on a skinned mesh node. Bumps skin_version
/// so Swift re-picks them up alongside the skinner rebuild.
pub fn set_anim_tracks(handle: u32, tracks: Vec<AnimTrack>) {
    mark_dirty(handle, DIRTY_SKIN, |n| {
        n.anim_tracks = tracks;
        n.skin_version = n.skin_version.wrapping_add(1);
    });
}

/// TS → native FFI layout: 12 f64s per vertex — xyz, nx ny nz, rgba, uv.
/// Color is ignored (material color wins via bloom_scene_set_material_color).
/// Directly install pre-decoded geometry (from a loaded model) into a scene
/// node, bypassing the 12-f64-per-vertex TS FFI layout.
pub fn set_geometry(handle: u32,
    positions: Vec<f32>, normals: Vec<f32>, uvs: Vec<f32>, indices: Vec<u32>,
) {
    mark_dirty(handle, DIRTY_GEOMETRY, |n| {
        n.positions = positions;
        n.normals = normals;
        n.uvs = uvs;
        n.indices = indices;
        n.geometry_version = n.geometry_version.wrapping_add(1);
    });
}

pub fn update_geometry_f64(handle: u32, verts: &[f64], indices: &[f64]) {
    mark_dirty(handle, DIRTY_GEOMETRY, |n| {
        let count = verts.len() / 12;
        n.positions.clear();
        n.normals.clear();
        n.uvs.clear();
        n.positions.reserve(count * 3);
        n.normals.reserve(count * 3);
        n.uvs.reserve(count * 2);
        for i in 0..count {
            let base = i * 12;
            n.positions.push(verts[base] as f32);
            n.positions.push(verts[base+1] as f32);
            n.positions.push(verts[base+2] as f32);
            n.normals.push(verts[base+3] as f32);
            n.normals.push(verts[base+4] as f32);
            n.normals.push(verts[base+5] as f32);
            n.uvs.push(verts[base+10] as f32);
            n.uvs.push(verts[base+11] as f32);
        }
        n.indices.clear();
        n.indices.reserve(indices.len());
        for &i in indices { n.indices.push(i as u32); }
        n.geometry_version = n.geometry_version.wrapping_add(1);
    });
}

pub fn add_directional_light(dx: f32, dy: f32, dz: f32, r: f32, g: f32, b: f32, intensity: f32) {
    with(|inner| {
        inner.lights.push(Light {
            kind: 1, _pad: 0,
            pos_or_dir: [dx, dy, dz], range: 0.0,
            color: [r, g, b], intensity,
        });
    });
}

pub fn add_point_light(x: f32, y: f32, z: f32, range: f32, r: f32, g: f32, b: f32, intensity: f32) {
    with(|inner| {
        inner.lights.push(Light {
            kind: 2, _pad: 0,
            pos_or_dir: [x, y, z], range,
            color: [r, g, b], intensity,
        });
    });
}

pub fn clear_lights() {
    with(|inner| inner.lights.clear());
}

pub fn node_count() -> usize {
    with(|inner| inner.nodes.iter().filter(|n| n.is_some()).count())
}

// ============================================================
// Snapshot API — called from Swift each frame.
// ============================================================

/// Delta-sync: emit only nodes whose dirty_flags are non-zero, and clear
/// flags on the nodes we report. Swift calls this once per frame and
/// applies the reported changes — static scenes get O(0) cost, moving
/// scenes scale with the number of actually-changing nodes.
pub fn drain_dirty(dst: *mut SceneNodeInfo, max: i64) -> i64 {
    if dst.is_null() || max <= 0 { return 0; }
    with(|inner| {
        let mut written = 0i64;
        for (i, slot) in inner.nodes.iter_mut().enumerate() {
            if written >= max { break; }
            let Some(n) = slot else { continue };
            if n.dirty_flags == 0 { continue; }
            let info = SceneNodeInfo {
                dirty_flags: n.dirty_flags,
                handle: i as u32,
                parent: n.parent,
                visible: if n.visible { 1 } else { 0 },
                geometry_version: n.geometry_version,
                color: n.color,
                roughness: n.roughness,
                metalness: n.metalness,
                texture: n.texture,
                has_geometry: if n.indices.is_empty() { 0 } else { 1 },
                transform: n.transform,
                tex_base_color: n.tex_base_color,
                tex_normal: n.tex_normal,
                tex_metallic_roughness: n.tex_metallic_roughness,
                tex_emissive: n.tex_emissive,
                tex_occlusion: n.tex_occlusion,
                skin_version: n.skin_version,
            };
            unsafe { *dst.add(written as usize) = info; }
            n.dirty_flags = 0;
            written += 1;
        }
        written
    })
}

/// Drain the list of handles destroyed since the last call. Swift pulls
/// these and removes the matching SCNNodes from its retainedNodes map.
pub fn drain_destroyed(dst: *mut u32, max: i64) -> i64 {
    if dst.is_null() || max <= 0 { return 0; }
    with(|inner| {
        let n = inner.destroyed_queue.len().min(max as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(inner.destroyed_queue.as_ptr(), dst, n);
        }
        inner.destroyed_queue.drain(..n);
        n as i64
    })
}

pub fn copy_lights(dst: *mut Light, max: i64) -> i64 {
    if dst.is_null() || max <= 0 { return 0; }
    with(|inner| {
        let n = inner.lights.len().min(max as usize);
        unsafe { std::ptr::copy_nonoverlapping(inner.lights.as_ptr(), dst, n); }
        n as i64
    })
}

/// Pointer + counts for a node's geometry arrays. Pointers are valid as long
/// as the node exists and `update_geometry` isn't called on it — Swift reads
/// synchronously in one pass so this is safe for the sync protocol.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GeometryPtrs {
    pub positions: *const f32,
    pub position_count: u32,
    pub normals: *const f32,
    pub normal_count: u32,
    pub uvs: *const f32,
    pub uv_count: u32,
    pub indices: *const u32,
    pub index_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SkinPtrs {
    pub joint_handles: *const u32,
    pub joint_count: u32,
    pub inverse_bind: *const f32,  // flat 16-float matrices
    pub inverse_bind_matrix_count: u32,
    pub vertex_joints: *const u32,  // 4 per vertex
    pub vertex_joint_count: u32,    // total entries (= vertex_count * 4)
    pub vertex_weights: *const f32, // 4 per vertex
    pub vertex_weight_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AnimTrackInfo {
    pub bone_handle: u32,
    pub path: u32,           // 0 = translation, 1 = rotation, 2 = scale
    pub key_count: u32,
    pub _pad: u32,
    pub times: *const f32,   // length = key_count
    pub values: *const f32,  // length = key_count * (3 or 4)
}

pub fn anim_track_count(handle: u32) -> i64 {
    with(|inner| {
        if let Some(Some(n)) = inner.nodes.get(handle as usize) {
            n.anim_tracks.len() as i64
        } else { 0 }
    })
}

pub fn anim_track_info(handle: u32, idx: i64, out: *mut AnimTrackInfo) {
    if out.is_null() { return; }
    with(|inner| {
        if let Some(Some(n)) = inner.nodes.get(handle as usize) {
            if let Some(t) = n.anim_tracks.get(idx as usize) {
                unsafe {
                    *out = AnimTrackInfo {
                        bone_handle: t.bone_handle,
                        path: t.path,
                        key_count: t.times.len() as u32,
                        _pad: 0,
                        times: t.times.as_ptr(),
                        values: t.values.as_ptr(),
                    };
                }
            }
        }
    });
}

pub fn skin_ptrs(handle: u32) -> SkinPtrs {
    with(|inner| {
        let empty = SkinPtrs {
            joint_handles: std::ptr::null(), joint_count: 0,
            inverse_bind: std::ptr::null(), inverse_bind_matrix_count: 0,
            vertex_joints: std::ptr::null(), vertex_joint_count: 0,
            vertex_weights: std::ptr::null(), vertex_weight_count: 0,
        };
        if let Some(Some(n)) = inner.nodes.get(handle as usize) {
            SkinPtrs {
                joint_handles: n.skin_joint_handles.as_ptr(),
                joint_count: n.skin_joint_handles.len() as u32,
                inverse_bind: n.skin_inverse_bind.as_ptr() as *const f32,
                inverse_bind_matrix_count: n.skin_inverse_bind.len() as u32,
                vertex_joints: n.skin_vertex_joints.as_ptr(),
                vertex_joint_count: n.skin_vertex_joints.len() as u32,
                vertex_weights: n.skin_vertex_weights.as_ptr(),
                vertex_weight_count: n.skin_vertex_weights.len() as u32,
            }
        } else {
            empty
        }
    })
}

pub fn geometry_ptrs(handle: u32) -> GeometryPtrs {
    with(|inner| {
        if let Some(Some(n)) = inner.nodes.get(handle as usize) {
            GeometryPtrs {
                positions: n.positions.as_ptr(),
                position_count: n.positions.len() as u32,
                normals: n.normals.as_ptr(),
                normal_count: n.normals.len() as u32,
                uvs: n.uvs.as_ptr(),
                uv_count: n.uvs.len() as u32,
                indices: n.indices.as_ptr(),
                index_count: n.indices.len() as u32,
            }
        } else {
            GeometryPtrs {
                positions: std::ptr::null(), position_count: 0,
                normals: std::ptr::null(), normal_count: 0,
                uvs: std::ptr::null(), uv_count: 0,
                indices: std::ptr::null(), index_count: 0,
            }
        }
    })
}

pub fn get_transform(handle: u32) -> [f32; 16] {
    with(|inner| {
        if let Some(Some(n)) = inner.nodes.get(handle as usize) {
            n.transform
        } else {
            let mut t = [0.0f32; 16]; t[0] = 1.0; t[5] = 1.0; t[10] = 1.0; t[15] = 1.0; t
        }
    })
}
