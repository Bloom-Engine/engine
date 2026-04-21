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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SceneNodeInfo {
    pub handle: u32,
    pub parent: u32,
    pub visible: u32,
    pub geometry_version: u32,

    pub color: [f32; 4],
    pub roughness: f32,
    pub metalness: f32,
    pub texture: u32,
    pub has_geometry: u32,

    pub transform: [f32; 16],
}

struct Node {
    visible: bool,
    parent: u32,
    transform: [f32; 16],
    color: [f32; 4],
    roughness: f32,
    metalness: f32,
    texture: u32,
    positions: Vec<f32>,   // xyz interleaved, length = vcount * 3
    normals: Vec<f32>,
    uvs: Vec<f32>,
    indices: Vec<u32>,
    geometry_version: u32,
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
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            geometry_version: 0,
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
}

static INNER: Mutex<Inner> = Mutex::new(Inner {
    nodes: Vec::new(),
    lights: Vec::new(),
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
        if let Some(slot) = inner.nodes.get_mut(handle as usize) {
            *slot = None;
        }
        // Any node with this as parent becomes a root.
        for n in inner.nodes.iter_mut().flatten() {
            if n.parent == handle { n.parent = 0; }
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

pub fn set_visible(handle: u32, v: bool) { mutate(handle, |n| n.visible = v); }
pub fn set_parent(handle: u32, parent: u32) { mutate(handle, |n| n.parent = parent); }
pub fn set_transform(handle: u32, m: [f32; 16]) { mutate(handle, |n| n.transform = m); }
pub fn set_color(handle: u32, rgba: [f32; 4]) { mutate(handle, |n| n.color = rgba); }
pub fn set_pbr(handle: u32, rough: f32, metal: f32) {
    mutate(handle, |n| { n.roughness = rough; n.metalness = metal; });
}
pub fn set_texture(handle: u32, tex: u32) { mutate(handle, |n| n.texture = tex); }

/// TS → native FFI layout: 12 f64s per vertex — xyz, nx ny nz, rgba, uv.
/// Color is ignored (material color wins via bloom_scene_set_material_color).
/// Directly install pre-decoded geometry (from a loaded model) into a scene
/// node, bypassing the 12-f64-per-vertex TS FFI layout.
pub fn set_geometry(handle: u32,
    positions: Vec<f32>, normals: Vec<f32>, uvs: Vec<f32>, indices: Vec<u32>,
) {
    mutate(handle, |n| {
        n.positions = positions;
        n.normals = normals;
        n.uvs = uvs;
        n.indices = indices;
        n.geometry_version = n.geometry_version.wrapping_add(1);
    });
}

pub fn update_geometry_f64(handle: u32, verts: &[f64], indices: &[f64]) {
    mutate(handle, |n| {
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

/// Copy all alive node infos into the Swift-supplied buffer.
/// Returns the number written (capped at `max`).
pub fn copy_nodes(dst: *mut SceneNodeInfo, max: i64) -> i64 {
    if dst.is_null() || max <= 0 { return 0; }
    with(|inner| {
        let mut written = 0i64;
        for (i, slot) in inner.nodes.iter().enumerate() {
            if written >= max { break; }
            let Some(n) = slot else { continue };
            let info = SceneNodeInfo {
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
            };
            unsafe { *dst.add(written as usize) = info; }
            written += 1;
        }
        written
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
