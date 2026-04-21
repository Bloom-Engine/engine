//! Minimal .glb (glTF 2.0 binary) loader for watchOS.
//!
//! Supports a single mesh's first primitive — POSITION / NORMAL / TEXCOORD_0 /
//! indices, plus PBR baseColorFactor / metallicFactor / roughnessFactor from
//! the referenced material. Multi-primitive meshes, skinning, animations,
//! and external texture files are intentionally out of scope for this first
//! cut; they'll be added as concrete games need them.
//!
//! Uses a hand-rolled JSON value parser to avoid pulling in the full serde /
//! serde_json compile tree — the glTF JSON schema is well-formed output from
//! exporters, so we don't need serde's robustness, just its shape.

use std::sync::{Arc, Mutex};

// ---------- Model registry ----------

/// A Model mirrors glTF's layout: a list of Meshes (each with Primitives),
/// a list of Nodes (transform + optional mesh reference + children), and
/// scene roots (node indices the scene starts from). bloom_scene_attach_model
/// walks the node tree and spawns one bloom scene node per glTF node, with
/// the glTF node's transform baked in.
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub nodes: Vec<Node>,
    pub scene_roots: Vec<usize>,
}

/// One node in the glTF scene graph. Translation/rotation/scale compose into
/// the local transform; if `matrix` is set it wins outright (glTF spec).
pub struct Node {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],  // quaternion xyzw
    pub scale: [f32; 3],
    pub matrix: Option<[f32; 16]>,  // column-major if present
    pub mesh: Option<usize>,
    pub children: Vec<usize>,
}

impl Node {
    pub fn identity() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            matrix: None,
            mesh: None,
            children: Vec::new(),
        }
    }

    /// Compose TRS into a column-major 4x4 matrix, or return `matrix` if set.
    pub fn local_transform(&self) -> [f32; 16] {
        if let Some(m) = self.matrix { return m; }
        let [tx, ty, tz] = self.translation;
        let [qx, qy, qz, qw] = self.rotation;
        let [sx, sy, sz] = self.scale;
        // Rotation matrix from quaternion.
        let xx = qx * qx; let yy = qy * qy; let zz = qz * qz;
        let xy = qx * qy; let xz = qx * qz; let yz = qy * qz;
        let wx = qw * qx; let wy = qw * qy; let wz = qw * qz;
        [
            (1.0 - 2.0 * (yy + zz)) * sx,
            (2.0 * (xy + wz)) * sx,
            (2.0 * (xz - wy)) * sx,
            0.0,

            (2.0 * (xy - wz)) * sy,
            (1.0 - 2.0 * (xx + zz)) * sy,
            (2.0 * (yz + wx)) * sy,
            0.0,

            (2.0 * (xz + wy)) * sz,
            (2.0 * (yz - wx)) * sz,
            (1.0 - 2.0 * (xx + yy)) * sz,
            0.0,

            tx, ty, tz, 1.0,
        ]
    }
}

pub struct Mesh {
    pub primitives: Vec<Primitive>,
}

pub struct Primitive {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub uvs: Vec<f32>,
    pub indices: Vec<u32>,
    pub color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    /// PBR texture slots — texture registry handles (0 = none).
    pub tex_base_color: u32,
    pub tex_normal: u32,
    pub tex_metallic_roughness: u32,
    pub tex_emissive: u32,
    pub tex_occlusion: u32,
}

static MODELS: Mutex<Vec<Option<Arc<Model>>>> = Mutex::new(Vec::new());

pub fn load(path: &str) -> u32 {
    let Ok(bytes) = std::fs::read(path) else { return 0; };
    let Some(model) = parse_glb(&bytes) else { return 0; };

    let mut reg = MODELS.lock().unwrap();
    if reg.is_empty() { reg.push(None); }  // sentinel 0 handle
    reg.push(Some(Arc::new(model)));
    (reg.len() - 1) as u32
}

pub fn get(handle: u32) -> Option<Arc<Model>> {
    let reg = MODELS.lock().unwrap();
    reg.get(handle as usize).and_then(|m| m.as_ref().cloned())
}

// ---------- GLB parsing ----------

const GLB_MAGIC: u32 = 0x46546C67; // "glTF" LE
const CHUNK_JSON: u32 = 0x4E4F534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E4942; // "BIN\0"

fn parse_glb(bytes: &[u8]) -> Option<Model> {
    if bytes.len() < 12 { return None; }
    if u32_le(bytes, 0) != GLB_MAGIC { return None; }
    let _version = u32_le(bytes, 4);
    let _total_len = u32_le(bytes, 8);

    // Walk chunks. Expect JSON first, then BIN.
    let mut off = 12usize;
    let mut json_bytes: &[u8] = &[];
    let mut bin_bytes: &[u8] = &[];
    while off + 8 <= bytes.len() {
        let chunk_len = u32_le(bytes, off) as usize;
        let chunk_type = u32_le(bytes, off + 4);
        let data_start = off + 8;
        let data_end = data_start + chunk_len;
        if data_end > bytes.len() { break; }
        let data = &bytes[data_start..data_end];
        match chunk_type {
            CHUNK_JSON => json_bytes = data,
            CHUNK_BIN => bin_bytes = data,
            _ => {}
        }
        off = data_end;
    }
    if json_bytes.is_empty() { return None; }

    let json = std::str::from_utf8(json_bytes).ok()?;
    let root = json_parse(json)?;

    let meshes_js = root.get("meshes")?.arr()?;
    let accessors = root.get("accessors")?.arr()?;
    let buf_views = root.get("bufferViews")?.arr()?;

    let mut meshes: Vec<Mesh> = Vec::with_capacity(meshes_js.len());
    for mesh_js in meshes_js {
        let prims_js = match mesh_js.get("primitives").and_then(|v| v.arr()) {
            Some(p) => p,
            None => continue,
        };
        let mut primitives: Vec<Primitive> = Vec::with_capacity(prims_js.len());
        for prim in prims_js {
            if let Some(p) = parse_primitive(prim, &root, accessors, buf_views, bin_bytes) {
                primitives.push(p);
            }
        }
        meshes.push(Mesh { primitives });
    }
    if meshes.is_empty() { return None; }

    // Parse nodes + scenes (glTF hierarchy).
    let mut nodes: Vec<Node> = Vec::new();
    if let Some(nodes_js) = root.get("nodes").and_then(|v| v.arr()) {
        nodes.reserve(nodes_js.len());
        for n in nodes_js {
            nodes.push(parse_node(n));
        }
    }
    let mut scene_roots: Vec<usize> = Vec::new();
    if let Some(scenes_js) = root.get("scenes").and_then(|v| v.arr()) {
        // Pick the default scene (glTF's "scene" field) or scene 0.
        let default_idx = root.get("scene").and_then(|v| v.num()).unwrap_or(0.0) as usize;
        if let Some(scene) = scenes_js.get(default_idx).or_else(|| scenes_js.first()) {
            if let Some(arr) = scene.get("nodes").and_then(|v| v.arr()) {
                for n in arr {
                    if let Some(i) = n.num() { scene_roots.push(i as usize); }
                }
            }
        }
    }
    // Fallback: if the file has no scenes/ but has nodes, treat node 0 as root.
    if scene_roots.is_empty() && !nodes.is_empty() {
        scene_roots.push(0);
    }
    // Fallback: if no nodes at all but meshes exist, synthesize a single root
    // node pointing at mesh 0 so older callers still get a valid hierarchy.
    if nodes.is_empty() && !meshes.is_empty() {
        let mut n = Node::identity();
        n.mesh = Some(0);
        nodes.push(n);
        scene_roots.push(0);
    }

    Some(Model { meshes, nodes, scene_roots })
}

fn parse_node(n: &JVal) -> Node {
    let mut node = Node::identity();
    if let Some(t) = n.get("translation").and_then(|v| v.arr()) {
        for (i, v) in t.iter().take(3).enumerate() {
            if let Some(f) = v.num() { node.translation[i] = f as f32; }
        }
    }
    if let Some(r) = n.get("rotation").and_then(|v| v.arr()) {
        for (i, v) in r.iter().take(4).enumerate() {
            if let Some(f) = v.num() { node.rotation[i] = f as f32; }
        }
    }
    if let Some(s) = n.get("scale").and_then(|v| v.arr()) {
        for (i, v) in s.iter().take(3).enumerate() {
            if let Some(f) = v.num() { node.scale[i] = f as f32; }
        }
    }
    if let Some(m) = n.get("matrix").and_then(|v| v.arr()) {
        let mut arr = [0.0f32; 16];
        for (i, v) in m.iter().take(16).enumerate() {
            if let Some(f) = v.num() { arr[i] = f as f32; }
        }
        node.matrix = Some(arr);
    }
    if let Some(mi) = n.get("mesh").and_then(|v| v.num()) {
        node.mesh = Some(mi as usize);
    }
    if let Some(kids) = n.get("children").and_then(|v| v.arr()) {
        for v in kids {
            if let Some(i) = v.num() { node.children.push(i as usize); }
        }
    }
    node
}

fn parse_primitive(
    prim: &JVal, root: &JVal,
    accessors: &[JVal], buf_views: &[JVal], bin_bytes: &[u8],
) -> Option<Primitive> {
    let attrs = prim.get("attributes")?;
    let pos_acc = attrs.get("POSITION")?.num()? as usize;
    let nrm_acc = attrs.get("NORMAL").and_then(|v| v.num().map(|n| n as usize));
    let uv_acc = attrs.get("TEXCOORD_0").and_then(|v| v.num().map(|n| n as usize));
    let idx_acc = prim.get("indices")?.num()? as usize;
    let mat_idx = prim.get("material").and_then(|v| v.num().map(|n| n as usize));

    let positions = read_f32_vec(accessors, buf_views, bin_bytes, pos_acc, 3)?;
    let normals = nrm_acc
        .and_then(|a| read_f32_vec(accessors, buf_views, bin_bytes, a, 3))
        .unwrap_or_default();
    let uvs = uv_acc
        .and_then(|a| read_f32_vec(accessors, buf_views, bin_bytes, a, 2))
        .unwrap_or_default();
    let indices = read_index_vec(accessors, buf_views, bin_bytes, idx_acc)?;

    let mut color = [1.0f32; 4];
    let mut metallic = 1.0f32;
    let mut roughness = 1.0f32;
    let mut tex_base_color = 0u32;
    let mut tex_normal = 0u32;
    let mut tex_metallic_roughness = 0u32;
    let mut tex_emissive = 0u32;
    let mut tex_occlusion = 0u32;

    if let Some(mi) = mat_idx {
        if let Some(mats) = root.get("materials").and_then(|v| v.arr()) {
            if let Some(mat) = mats.get(mi) {
                if let Some(pbr) = mat.get("pbrMetallicRoughness") {
                    if let Some(bc) = pbr.get("baseColorFactor").and_then(|v| v.arr()) {
                        for (i, n) in bc.iter().take(4).enumerate() {
                            if let Some(f) = n.num() { color[i] = f as f32; }
                        }
                    }
                    if let Some(m) = pbr.get("metallicFactor").and_then(|v| v.num()) {
                        metallic = m as f32;
                    }
                    if let Some(r) = pbr.get("roughnessFactor").and_then(|v| v.num()) {
                        roughness = r as f32;
                    }
                    tex_base_color = resolve_texture(root, pbr, "baseColorTexture", bin_bytes);
                    tex_metallic_roughness = resolve_texture(root, pbr, "metallicRoughnessTexture", bin_bytes);
                }
                tex_normal = resolve_texture(root, mat, "normalTexture", bin_bytes);
                tex_emissive = resolve_texture(root, mat, "emissiveTexture", bin_bytes);
                tex_occlusion = resolve_texture(root, mat, "occlusionTexture", bin_bytes);
            }
        }
    }

    Some(Primitive {
        positions, normals, uvs, indices,
        color, metallic, roughness,
        tex_base_color, tex_normal, tex_metallic_roughness, tex_emissive, tex_occlusion,
    })
}

/// Chase a material.<slotName>.index → textures[i].source → images[j]
/// → bufferViews[k] → BIN bytes, register via crate::textures. Returns the
/// texture handle or 0 on any missing link.
fn resolve_texture(root: &JVal, parent: &JVal, slot_name: &str, bin: &[u8]) -> u32 {
    let Some(slot) = parent.get(slot_name) else { return 0 };
    let Some(tex_idx) = slot.get("index").and_then(|v| v.num()) else { return 0 };
    let Some(textures) = root.get("textures").and_then(|v| v.arr()) else { return 0 };
    let Some(tex) = textures.get(tex_idx as usize) else { return 0 };
    let Some(src_idx) = tex.get("source").and_then(|v| v.num()) else { return 0 };
    let Some(images) = root.get("images").and_then(|v| v.arr()) else { return 0 };
    let Some(img) = images.get(src_idx as usize) else { return 0 };
    let Some(bv_idx) = img.get("bufferView").and_then(|v| v.num()) else { return 0 };
    let Some(buf_views) = root.get("bufferViews").and_then(|v| v.arr()) else { return 0 };
    let Some(bv) = buf_views.get(bv_idx as usize) else { return 0 };

    let off = bv.get("byteOffset").and_then(|v| v.num()).unwrap_or(0.0) as usize;
    let len = bv.get("byteLength").and_then(|v| v.num()).unwrap_or(0.0) as usize;
    if off + len > bin.len() { return 0; }
    crate::textures::register_bytes(&bin[off..off + len])
}

fn u32_le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]])
}

/// Resolve accessor → raw float slice from the bin chunk. `components` is
/// the expected vector width (3 for position/normal, 2 for uv). glTF stores
/// accessor.componentType = 5126 (FLOAT) for these.
fn read_f32_vec(
    accessors: &[JVal], buf_views: &[JVal], bin: &[u8],
    acc_idx: usize, components: usize,
) -> Option<Vec<f32>> {
    let acc = accessors.get(acc_idx)?;
    let bv_idx = acc.get("bufferView")?.num()? as usize;
    let count = acc.get("count")?.num()? as usize;
    let byte_off_acc = acc.get("byteOffset").and_then(|v| v.num()).unwrap_or(0.0) as usize;

    let bv = buf_views.get(bv_idx)?;
    let byte_off_bv = bv.get("byteOffset").and_then(|v| v.num()).unwrap_or(0.0) as usize;
    let stride = bv.get("byteStride").and_then(|v| v.num()).map(|n| n as usize)
        .unwrap_or(components * 4);

    let mut out = Vec::with_capacity(count * components);
    for i in 0..count {
        let base = byte_off_bv + byte_off_acc + i * stride;
        if base + components * 4 > bin.len() { return None; }
        for c in 0..components {
            let b = base + c * 4;
            out.push(f32::from_le_bytes([bin[b], bin[b+1], bin[b+2], bin[b+3]]));
        }
    }
    Some(out)
}

/// Indices — componentType 5121 (u8), 5123 (u16), or 5125 (u32). Always
/// returned as u32.
fn read_index_vec(
    accessors: &[JVal], buf_views: &[JVal], bin: &[u8], acc_idx: usize,
) -> Option<Vec<u32>> {
    let acc = accessors.get(acc_idx)?;
    let comp_type = acc.get("componentType")?.num()? as u32;
    let bv_idx = acc.get("bufferView")?.num()? as usize;
    let count = acc.get("count")?.num()? as usize;
    let byte_off_acc = acc.get("byteOffset").and_then(|v| v.num()).unwrap_or(0.0) as usize;

    let bv = buf_views.get(bv_idx)?;
    let byte_off_bv = bv.get("byteOffset").and_then(|v| v.num()).unwrap_or(0.0) as usize;
    let base = byte_off_bv + byte_off_acc;

    let mut out = Vec::with_capacity(count);
    match comp_type {
        5121 => { // u8
            if base + count > bin.len() { return None; }
            for i in 0..count { out.push(bin[base + i] as u32); }
        }
        5123 => { // u16
            if base + count * 2 > bin.len() { return None; }
            for i in 0..count {
                let o = base + i * 2;
                out.push(u16::from_le_bytes([bin[o], bin[o+1]]) as u32);
            }
        }
        5125 => { // u32
            if base + count * 4 > bin.len() { return None; }
            for i in 0..count {
                let o = base + i * 4;
                out.push(u32::from_le_bytes([bin[o], bin[o+1], bin[o+2], bin[o+3]]));
            }
        }
        _ => return None,
    }
    Some(out)
}

// ---------- Tiny JSON value parser ----------
//
// Only what we need for glTF: objects, arrays, numbers, strings. No booleans,
// null, escapes, or unicode. Exporter output tends to be well-formed.

#[derive(Debug, Clone)]
pub enum JVal {
    Obj(Vec<(String, JVal)>),
    Arr(Vec<JVal>),
    Num(f64),
    Str(String),
}

impl JVal {
    pub fn get(&self, key: &str) -> Option<&JVal> {
        match self {
            JVal::Obj(v) => v.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn arr(&self) -> Option<&[JVal]> {
        if let JVal::Arr(a) = self { Some(a.as_slice()) } else { None }
    }
    pub fn num(&self) -> Option<f64> {
        if let JVal::Num(n) = self { Some(*n) } else { None }
    }
}

struct P<'a> { bytes: &'a [u8], pos: usize }

impl<'a> P<'a> {
    fn peek(&self) -> Option<u8> { self.bytes.get(self.pos).copied() }
    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?; self.pos += 1; Some(c)
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' { self.pos += 1; }
            else { break; }
        }
    }
    fn val(&mut self) -> Option<JVal> {
        self.skip_ws();
        match self.peek()? {
            b'{' => self.obj(),
            b'[' => self.arr(),
            b'"' => self.string().map(JVal::Str),
            b'-' | b'0'..=b'9' => self.number().map(JVal::Num),
            b't' => { for c in b"true" { if self.bump() != Some(*c) { return None; } } Some(JVal::Num(1.0)) }
            b'f' => { for c in b"false" { if self.bump() != Some(*c) { return None; } } Some(JVal::Num(0.0)) }
            b'n' => { for c in b"null" { if self.bump() != Some(*c) { return None; } } Some(JVal::Num(0.0)) }
            _ => None,
        }
    }
    fn obj(&mut self) -> Option<JVal> {
        if self.bump() != Some(b'{') { return None; }
        let mut kv = Vec::new();
        loop {
            self.skip_ws();
            match self.peek()? {
                b'}' => { self.bump(); return Some(JVal::Obj(kv)); }
                b',' => { self.bump(); continue; }
                b'"' => {
                    let key = self.string()?;
                    self.skip_ws();
                    if self.bump() != Some(b':') { return None; }
                    let v = self.val()?;
                    kv.push((key, v));
                }
                _ => return None,
            }
        }
    }
    fn arr(&mut self) -> Option<JVal> {
        if self.bump() != Some(b'[') { return None; }
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            match self.peek()? {
                b']' => { self.bump(); return Some(JVal::Arr(items)); }
                b',' => { self.bump(); continue; }
                _ => items.push(self.val()?),
            }
        }
    }
    fn string(&mut self) -> Option<String> {
        if self.bump() != Some(b'"') { return None; }
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == b'"' {
                let s = std::str::from_utf8(&self.bytes[start..self.pos]).ok()?.to_string();
                self.bump();
                return Some(s);
            }
            if c == b'\\' { self.bump(); }  // skip escape + next char
            self.bump();
        }
        None
    }
    fn number(&mut self) -> Option<f64> {
        let start = self.pos;
        if self.peek() == Some(b'-') { self.bump(); }
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => { self.bump(); }
                _ => break,
            }
        }
        std::str::from_utf8(&self.bytes[start..self.pos]).ok()?.parse().ok()
    }
}

pub fn json_parse(s: &str) -> Option<JVal> {
    P { bytes: s.as_bytes(), pos: 0 }.val()
}

// ---------- Procedural cube mesh — for bloom_gen_mesh_cube ----------

pub fn gen_cube_mesh(w: f32, h: f32, d: f32) -> u32 {
    let hx = w * 0.5;
    let hy = h * 0.5;
    let hz = d * 0.5;

    // 6 faces × 4 verts each, positions + normals + uvs per vertex.
    // Face data: normal direction and (a, b) UV-space corners relative.
    let faces: &[[f32; 3]] = &[
        [ 0.0,  0.0,  1.0], [ 0.0,  0.0, -1.0],
        [ 1.0,  0.0,  0.0], [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0], [ 0.0, -1.0,  0.0],
    ];
    let corners: &[[[f32; 3]; 4]] = &[
        [[-hx,-hy, hz],[ hx,-hy, hz],[ hx, hy, hz],[-hx, hy, hz]], // +Z
        [[ hx,-hy,-hz],[-hx,-hy,-hz],[-hx, hy,-hz],[ hx, hy,-hz]], // -Z
        [[ hx,-hy, hz],[ hx,-hy,-hz],[ hx, hy,-hz],[ hx, hy, hz]], // +X
        [[-hx,-hy,-hz],[-hx,-hy, hz],[-hx, hy, hz],[-hx, hy,-hz]], // -X
        [[-hx, hy, hz],[ hx, hy, hz],[ hx, hy,-hz],[-hx, hy,-hz]], // +Y
        [[-hx,-hy,-hz],[ hx,-hy,-hz],[ hx,-hy, hz],[-hx,-hy, hz]], // -Y
    ];
    let uvs_per_corner = [[0.0_f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let mut positions = Vec::with_capacity(6 * 4 * 3);
    let mut normals = Vec::with_capacity(6 * 4 * 3);
    let mut uvs = Vec::with_capacity(6 * 4 * 2);
    let mut indices = Vec::with_capacity(6 * 6);

    for f in 0..6 {
        let n = faces[f];
        for c in 0..4 {
            let p = corners[f][c];
            positions.extend_from_slice(&p);
            normals.extend_from_slice(&n);
            uvs.extend_from_slice(&uvs_per_corner[c]);
        }
        let base = (f * 4) as u32;
        indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    let mut reg = MODELS.lock().unwrap();
    if reg.is_empty() { reg.push(None); }
    let mut root = Node::identity();
    root.mesh = Some(0);
    reg.push(Some(Arc::new(Model {
        meshes: vec![Mesh { primitives: vec![Primitive {
            positions, normals, uvs, indices,
            color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0, roughness: 0.6,
            tex_base_color: 0, tex_normal: 0, tex_metallic_roughness: 0,
            tex_emissive: 0, tex_occlusion: 0,
        }] }],
        nodes: vec![root],
        scene_roots: vec![0],
    })));
    (reg.len() - 1) as u32
}
