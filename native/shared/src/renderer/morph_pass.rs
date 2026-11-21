//! Morph targets on the GPU: the per-instance morph pass, and what the
//! draws read from it. `docs/morph-targets.md` is the overview.
//!
//! A skinned cached model keeps one bind-pose vertex buffer shared by every
//! instance. An instance whose morph weights are not all zero gets its OWN
//! vertex buffer per morphed mesh, which a compute pass rewrites each frame
//! it is drawn: bind pose + sum of weight x delta for position, normal and
//! tangent (normal and tangent renormalised), one thread per vertex reading
//! its own sparse entries (`morph_targets.rs`, re-laid lane-interleaved per
//! block of 64 vertices so a wave reads them coalesced). Untouched vertices are
//! copied once, when the buffer is made, and never again.
//!
//! That buffer then simply replaces the shared one as the draw's vertex
//! buffer in every pass (`morph_vertex()` at each cached-draw site) and as
//! the source of the ray scene's compute skin, so no vertex shader changes
//! for the CURRENT position: skinning, shadows, transmitted shadows, custom
//! passes, the visibility buffer and GI all see the morphed mesh.
//!
//! Motion vectors need the PREVIOUS position too. The pass knows last
//! frame's weights (kept per animation handle, as the joint palettes are),
//! so it also writes `previous - current` displacement of every vertex of
//! the mesh into one frame-wide texture, the "morph arena" (bound at
//! `@group(3) @binding(2)`, next to the joint palettes, in every joint bind
//! group). A morphed draw's uniform `misc.w` is `1 + its first texel`; the
//! scene vertex shader's skinned branch adds the displacement to the
//! position it feeds the previous palette, so TAA and TSR get the real
//! velocity of a moving lip. `misc.w = 0` (every other draw) reads nothing.
//!
//! Costs nothing for models without morphs or instances at zero: no buffer,
//! no job, no texel. The frame an instance's weights return to zero it is
//! morphed once more (its velocity is not zero that frame), then it draws
//! from the shared buffer again; its own buffer is freed after
//! `IDLE_FRAMES` frames unused.
//!
//! Without compute (the WebGL2 baseline tier) the same is done on the CPU:
//! the instance buffer and the arena are uploaded from CPU copies.

use super::*;
use crate::morph_targets::{MorphEntry, MorphTargets};
use std::collections::HashMap;
use std::sync::Arc;

/// Texels per arena row (any texture limit allows 2048).
const ARENA_WIDTH: u32 = 2048;
/// Frames an instance buffer survives unused.
const IDLE_FRAMES: u64 = 240;
/// Uniform stride of a job's parameters.
const PARAMS_STRIDE: u64 = 256;
const WORKGROUP: u32 = 64;
const MAX_GROUPS_X: u32 = 65535;

const MORPH_WGSL: &str = r#"
struct MorphParams {
    vertex_count: u32,
    arena_base: u32,
    weights_offset: u32,
    slot_count: u32,
    full_copy: u32,
    flags: u32,        // bit 0 normals, bit 1 tangents
    pos_scale: f32,
    row_groups: u32,   // workgroups in x
};
@group(0) @binding(0) var<uniform> p: MorphParams;
// Vertex3D as six vec4: [pos.xyz nrm.x] [nrm.yz col.rg] [col.ba uv]
// [joints] [weights] [tangent].
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
// Per block of 64 vertices: (first entry, entries per vertex). A block's
// entries are lane-interleaved: entry k of lane i is at first + k * 64 + i
// (padding slot 0xffffffff), so a wave reads them coalesced.
@group(0) @binding(3) var<storage, read> blocks: array<vec2<u32>>;
// One entry is three vec2<u32>: [slot, pos.xy] [pos.z|nrm.x, nrm.yz] [tan.xy, tan.z|0].
@group(0) @binding(4) var<storage, read> entries: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> weights: array<f32>;
@group(0) @binding(6) var prev_disp: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(64)
fn cs_morph(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let v = gid.x + gid.y * p.row_groups * 64u;
    if (v >= p.vertex_count) {
        return;
    }
    let texel = p.arena_base + v;
    let width = textureDimensions(prev_disp).x;
    let tc = vec2<u32>(texel % width, texel / width);
    let b = v * 6u;
    let block = blocks[v / 64u];
    if (p.full_copy != 0u) {
        for (var k = 0u; k < 6u; k = k + 1u) {
            dst[b + k] = src[b + k];
        }
    }
    if (block.y == 0u) {
        textureStore(prev_disp, tc, vec4<f32>(0.0));
        return;
    }
    var dp = vec3<f32>(0.0);
    var dpp = vec3<f32>(0.0);
    var dn = vec3<f32>(0.0);
    var dt = vec3<f32>(0.0);
    let tangents = (p.flags & 2u) != 0u;
    for (var k = 0u; k < block.y; k = k + 1u) {
        let o = (block.x + k * 64u + lane) * 3u;
        let e0 = entries[o];
        let slot = e0.x;
        if (slot >= p.slot_count) {
            continue;
        }
        let w = weights[p.weights_offset + slot];
        let wp = weights[p.weights_offset + p.slot_count + slot];
        if (w == 0.0 && wp == 0.0) {
            continue;
        }
        let e1 = entries[o + 1u];
        let a = unpack2x16float(e0.y);
        let c = unpack2x16float(e1.x);
        let d = unpack2x16float(e1.y);
        let pos = vec3<f32>(a.x, a.y, c.x) * p.pos_scale;
        dp = dp + w * pos;
        dpp = dpp + wp * pos;
        dn = dn + w * vec3<f32>(c.y, d.x, d.y);
        if (tangents) {
            let e2 = entries[o + 2u];
            let f = unpack2x16float(e2.x);
            let g = unpack2x16float(e2.y);
            dt = dt + w * vec3<f32>(f.x, f.y, g.x);
        }
    }
    let s0 = src[b];
    let s1 = src[b + 1u];
    var o0 = vec4<f32>(s0.xyz + dp, s0.w);
    if ((p.flags & 1u) != 0u) {
        let sn = vec3<f32>(s0.w, s1.x, s1.y);
        let n = sn + dn;
        let l = length(n);
        let nn = select(sn, n / l, l > 1e-12);
        o0.w = nn.x;
        dst[b + 1u] = vec4<f32>(nn.y, nn.z, s1.z, s1.w);
    }
    dst[b] = o0;
    if (tangents) {
        let st = src[b + 5u];
        if (dot(st.xyz, st.xyz) > 0.0) {
            let t = st.xyz + dt;
            let l = length(t);
            dst[b + 5u] = vec4<f32>(select(st.xyz, t / l, l > 1e-12), st.w);
        }
    }
    textureStore(prev_disp, tc, vec4<f32>(dpp - dp, 0.0));
}
"#;

/// One mesh's deltas on the GPU (shared by every instance).
pub(super) struct MorphMeshGpu {
    targets: Arc<MorphTargets>,
    /// `None` on the CPU path. Blocks and lane-interleaved entries
    /// (`interleave_blocks`).
    blocks: Option<wgpu::Buffer>,
    entries: Option<wgpu::Buffer>,
}

struct MorphInstance {
    vertex: wgpu::Buffer,
    initialized: bool,
    last_frame: u64,
    /// CPU path: the instance's current vertices.
    cpu: Option<Vec<Vertex3D>>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct StagedMorph {
    /// The animation handle (keys the instance's own buffers).
    key: u64,
    weights_offset: u32,
    slots: u32,
}

struct Job {
    key: (u64, u64, usize),
    arena_base: u32,
    staged: StagedMorph,
}

/// What the morph pass did in the last frame (tests, stats, benches).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MorphFrameStats {
    pub jobs: u32,
    pub vertices: u32,
    pub instances: u32,
    pub arena_texels: u32,
    pub gpu: bool,
    /// GPU time of the morph pass in milliseconds, measured only while
    /// `set_morph_gpu_timing(true)` (blocking; for benches), else 0.
    pub gpu_ms: f64,
}

pub(super) struct MorphPass {
    meshes: HashMap<u64, Vec<Option<MorphMeshGpu>>>,
    instances: HashMap<(u64, u64, usize), MorphInstance>,
    /// In lockstep with `pending_skin_groups`: the morph state staged with
    /// each pose (`None` for a pose without morphs).
    pending: Vec<Option<StagedMorph>>,
    /// The last staged pose has not had its weights yet.
    pose_open: bool,
    /// The morph state of the pose `take_staged_skin_offset` just popped.
    current: Option<StagedMorph>,
    /// Last weights per animation handle, with the frame they were staged.
    prev: HashMap<u64, (u64, Vec<f32>)>,
    /// This frame's weights: per staged instance, current then previous.
    weights: Vec<f32>,
    jobs: Vec<Job>,
    arena_cursor: u32,
    /// Morphed vertex buffers by (absolute joint offset, mesh index): those
    /// recorded for the coming frame, and those of the frame being drawn.
    pending_overrides: HashMap<(u32, usize), wgpu::Buffer>,
    overrides: HashMap<(u32, usize), wgpu::Buffer>,
    arena: wgpu::Texture,
    arena_view: wgpu::TextureView,
    arena_rows: u32,
    compute: bool,
    pipeline: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    weights_buffer: Option<(wgpu::Buffer, u64)>,
    params_buffer: Option<(wgpu::Buffer, u64)>,
    cpu_arena: Vec<[f32; 4]>,
    /// Timestamp queries around the pass (benches): set, resolve, readback.
    timing: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer)>,
    frame: u64,
    stats: MorphFrameStats,
}

fn create_arena(
    device: &wgpu::Device,
    rows: u32,
    compute: bool,
) -> (wgpu::Texture, wgpu::TextureView) {
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    if compute {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("morph_prev_displacement"),
        size: wgpu::Extent3d {
            width: ARENA_WIDTH,
            height: rows.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Whether this device can run the morph compute pass.
fn device_has_compute(device: &wgpu::Device) -> bool {
    let l = device.limits();
    l.max_compute_invocations_per_workgroup >= WORKGROUP
        && l.max_compute_workgroup_size_x >= WORKGROUP
        && l.max_storage_buffers_per_shader_stage >= 5
        && l.max_storage_textures_per_shader_stage >= 1
}

/// The GPU layout of a mesh's entries: per block of `WORKGROUP` vertices
/// (one workgroup), `(first entry, entries per vertex)`, and the block's
/// entries lane-interleaved and padded to its busiest vertex. Neighbouring
/// threads then read neighbouring entries. A block of untouched vertices
/// has none. Never empty (a storage binding may not be).
pub(super) fn interleave_blocks(targets: &MorphTargets) -> (Vec<[u32; 2]>, Vec<MorphEntry>) {
    let n = targets.vertex_count as usize;
    let lanes = WORKGROUP as usize;
    let pad = MorphEntry {
        slot: u32::MAX,
        packed: [0; 5],
    };
    let mut blocks = Vec::with_capacity(n.div_ceil(lanes).max(1));
    let mut entries = Vec::with_capacity(targets.entries.len());
    for first in (0..n).step_by(lanes) {
        let last = (first + lanes).min(n);
        let most = (first..last)
            .map(|v| targets.vertex_entries(v).len())
            .max()
            .unwrap_or(0);
        blocks.push([entries.len() as u32, most as u32]);
        for k in 0..most {
            for lane in 0..lanes {
                let v = first + lane;
                let e = if v < last {
                    targets.vertex_entries(v).get(k).copied()
                } else {
                    None
                };
                entries.push(e.unwrap_or(pad));
            }
        }
    }
    if blocks.is_empty() {
        blocks.push([0, 0]);
    }
    if entries.is_empty() {
        entries.push(pad);
    }
    (blocks, entries)
}

impl MorphPass {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let compute = device_has_compute(device) && std::env::var_os("BLOOM_MORPH_CPU").is_none();
        let (arena, arena_view) = create_arena(device, 1, device_has_compute(device));
        Self {
            meshes: HashMap::new(),
            instances: HashMap::new(),
            pending: Vec::new(),
            pose_open: false,
            current: None,
            prev: HashMap::new(),
            weights: Vec::new(),
            jobs: Vec::new(),
            arena_cursor: 0,
            pending_overrides: HashMap::new(),
            overrides: HashMap::new(),
            arena,
            arena_view,
            arena_rows: 1,
            compute,
            pipeline: None,
            weights_buffer: None,
            params_buffer: None,
            cpu_arena: Vec::new(),
            timing: None,
            frame: 1,
            stats: MorphFrameStats::default(),
        }
    }

    /// The previous-position displacement texture, bound at group 3
    /// binding 2 of every joint bind group.
    pub(super) fn arena_view(&self) -> &wgpu::TextureView {
        &self.arena_view
    }

    /// A pose was staged (`pending_skin_groups.push`): keep step with it.
    #[inline]
    pub(super) fn note_group(&mut self) {
        self.pending.push(None);
        self.pose_open = true;
    }

    /// A pose was popped (`pending_skin_groups.remove(0)`): its morph state
    /// becomes the current one, for the draw that popped it.
    #[inline]
    pub(super) fn pop_group(&mut self) {
        self.current = if self.pending.is_empty() {
            None
        } else {
            self.pending.remove(0)
        };
    }

    pub(super) fn clear_pending(&mut self) {
        self.pending.clear();
        self.pose_open = false;
        self.current = None;
    }

    pub(super) fn reset_history(&mut self) {
        self.prev.clear();
    }

    /// The draw's vertex buffer: the morphed one of this instance's mesh if
    /// it was morphed this frame, else `draw` as it is.
    #[inline]
    pub(super) fn morph_vertex<'a>(
        &'a self,
        mut draw: MeshDrawRef<'a>,
        skinned: bool,
        joint_offset: f32,
        mesh_idx: usize,
    ) -> MeshDrawRef<'a> {
        if skinned && !self.overrides.is_empty() {
            if let Some(vb) = self.vertex_buffer(joint_offset, mesh_idx) {
                draw.vertex = vb;
            }
        }
        draw
    }

    /// The morphed vertex buffer of the draw at this joint offset and mesh.
    #[inline]
    pub(super) fn vertex_buffer(
        &self,
        joint_offset: f32,
        mesh_idx: usize,
    ) -> Option<&wgpu::Buffer> {
        if self.overrides.is_empty() {
            return None;
        }
        self.overrides
            .get(&(joint_offset.max(0.0) as u32, mesh_idx))
    }

    pub(super) fn forget_model(&mut self, cache_handle: u64) {
        self.meshes.remove(&cache_handle);
        self.instances.retain(|k, _| k.1 != cache_handle);
    }

    pub(super) fn stats(&self) -> MorphFrameStats {
        self.stats
    }
}

impl Renderer {
    /// Upload a just-cached model's morph deltas and grow its meshes' rest
    /// bounds by what the targets can move (so an open jaw is never culled).
    /// Only skinned models morph: weights travel with the pose.
    #[cfg(feature = "models3d")]
    pub(super) fn register_model_morphs(
        &mut self,
        handle: u64,
        meshes: &[&crate::models::MeshData],
    ) {
        if meshes.iter().all(|m| m.morph.is_none()) {
            return;
        }
        if !self.model_skinned.contains(&handle) {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!("bloom: a model with morph targets has no skin; morphs are applied to skinned models only (drawn through an animation instance)");
            }
            return;
        }
        let compute = self.morph.compute;
        let mut gpu = Vec::with_capacity(meshes.len());
        for (i, mesh) in meshes.iter().enumerate() {
            let Some(targets) = mesh
                .morph
                .as_ref()
                .filter(|t| t.vertex_count as usize == mesh.vertices.len())
            else {
                gpu.push(None);
                continue;
            };
            if let Some(Some(cached)) = self.model_gpu_cache.get_mut(&handle) {
                if let Some(c) = cached.get_mut(i) {
                    targets.grow_bounds(&mut c.local_min, &mut c.local_max);
                }
            }
            let upload = |label, contents: &[u8]| {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents,
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let (blocks, entries) = if compute {
                let (blocks, entries) = interleave_blocks(targets);
                (
                    Some(upload("morph_blocks", bytemuck::cast_slice(&blocks))),
                    Some(upload("morph_entries", bytemuck::cast_slice(&entries))),
                )
            } else {
                (None, None)
            };
            gpu.push(Some(MorphMeshGpu {
                targets: targets.clone(),
                blocks,
                entries,
            }));
        }
        self.morph.meshes.insert(handle, gpu);
    }

    /// Stage an animation instance's morph weights with the pose just staged
    /// by `set_joint_matrices_*` (call right after it; once). `key` is the
    /// same animation handle that keys the joint palettes. Ignored when no
    /// pose was just staged.
    pub fn stage_morph_weights(&mut self, key: u64, weights: &[f32]) {
        let m = &mut self.morph;
        if !std::mem::replace(&mut m.pose_open, false) {
            return;
        }
        if weights.is_empty() && !m.prev.contains_key(&key) {
            return;
        }
        let frame = m.frame;
        let nonzero = weights.iter().any(|w| *w != 0.0);
        let prev = m
            .prev
            .get(&key)
            .filter(|(f, _)| f.wrapping_add(1) == frame)
            .map(|(_, w)| w.as_slice());
        let prev_nonzero = prev.is_some_and(|p| p.iter().any(|w| *w != 0.0));
        if nonzero || prev_nonzero {
            let n = weights.len().max(prev.map_or(0, |p| p.len()));
            let offset = m.weights.len();
            m.weights.extend_from_slice(weights);
            m.weights.resize(offset + n, 0.0);
            match prev {
                Some(p) => m.weights.extend_from_slice(p),
                // First frame seen: no motion of its own.
                None => m.weights.extend_from_slice(weights),
            }
            m.weights.resize(offset + 2 * n, 0.0);
            if let Some(last) = m.pending.last_mut() {
                *last = Some(StagedMorph {
                    key,
                    weights_offset: offset as u32,
                    slots: n as u32,
                });
            }
        }
        let entry = m.prev.entry(key).or_default();
        entry.0 = frame;
        entry.1.clear();
        entry.1.extend_from_slice(weights);
    }

    /// The draw of a skinned cached model with the pose just popped: if that
    /// pose carries morph weights, give each morphed mesh its instance buffer
    /// and arena range for this frame. Returns `misc.w` per mesh (0 = not
    /// morphed).
    pub(super) fn morph_draw(
        &mut self,
        handle: u64,
        joint_offset: f32,
        mesh_count: usize,
    ) -> Vec<f32> {
        let Some(staged) = self.morph.current.take() else {
            return Vec::new();
        };
        let MorphPass {
            meshes,
            instances,
            jobs,
            arena_cursor,
            pending_overrides,
            frame,
            ..
        } = &mut self.morph;
        let Some(gpu) = meshes.get(&handle) else {
            return Vec::new();
        };
        let mut out = vec![0.0f32; mesh_count];
        for (i, mesh) in gpu.iter().enumerate().take(mesh_count) {
            let Some(mesh) = mesh else {
                continue;
            };
            let vc = mesh.targets.vertex_count;
            if (*arena_cursor as u64 + vc as u64) >= (1u64 << 24) {
                // misc.w is an f32: past 16M texels the base is not exact.
                break;
            }
            let key = (staged.key, handle, i);
            let device = &self.device;
            let instance = instances.entry(key).or_insert_with(|| MorphInstance {
                vertex: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("morph_instance_vb"),
                    size: (vc as u64 * std::mem::size_of::<Vertex3D>() as u64).max(4),
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                initialized: false,
                last_frame: 0,
                cpu: None,
            });
            instance.last_frame = *frame;
            let base = *arena_cursor;
            *arena_cursor += vc;
            pending_overrides.insert((joint_offset.max(0.0) as u32, i), instance.vertex.clone());
            jobs.push(Job {
                key,
                arena_base: base,
                staged,
            });
            out[i] = (base + 1) as f32;
        }
        out
    }

    /// Run this frame's morph jobs (called where the joint palettes are
    /// uploaded, before any pass that draws): GPU compute in its own submit,
    /// or the CPU path. Hands the morphed buffers to the coming passes.
    pub(super) fn encode_morph_pass(&mut self) {
        let jobs = std::mem::take(&mut self.morph.jobs);
        self.morph.overrides = std::mem::take(&mut self.morph.pending_overrides);
        let frame = self.morph.frame;
        self.morph.frame += 1;
        if frame.is_multiple_of(64) {
            self.morph
                .instances
                .retain(|_, inst| frame.saturating_sub(inst.last_frame) <= IDLE_FRAMES);
            self.morph
                .prev
                .retain(|_, (f, _)| frame.saturating_sub(*f) <= 2);
        }
        let used = self.morph.arena_cursor;
        self.morph.stats = MorphFrameStats {
            jobs: jobs.len() as u32,
            vertices: used,
            instances: self.morph.instances.len() as u32,
            arena_texels: self.morph.arena_rows * ARENA_WIDTH,
            gpu: self.morph.compute,
            gpu_ms: 0.0,
        };
        if jobs.is_empty() {
            self.morph.weights.clear();
            self.morph.arena_cursor = 0;
            return;
        }
        let rows = used.div_ceil(ARENA_WIDTH).max(1);
        if rows > self.morph.arena_rows {
            let rows = rows.next_power_of_two();
            let (arena, view) = create_arena(&self.device, rows, device_has_compute(&self.device));
            self.morph.arena = arena;
            self.morph.arena_view = view;
            self.morph.arena_rows = rows;
            self.rebuild_joint_bind_groups();
            self.morph.stats.arena_texels = rows * ARENA_WIDTH;
        }
        if self.morph.compute {
            self.encode_morph_gpu(&jobs);
        } else {
            self.morph_cpu(&jobs, used);
        }
        self.morph.weights.clear();
        self.morph.arena_cursor = 0;
    }

    fn ensure_morph_buffer(
        device: &wgpu::Device,
        slot: &mut Option<(wgpu::Buffer, u64)>,
        bytes: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) {
        if slot.as_ref().is_none_or(|(_, size)| *size < bytes) {
            let size = bytes.next_power_of_two().max(256);
            *slot = Some((
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: usage | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                size,
            ));
        }
    }

    fn morph_pipeline(&mut self) {
        if self.morph.pipeline.is_some() {
            return;
        }
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("morph_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    storage(1, true),
                    storage(2, false),
                    storage(3, true),
                    storage(4, true),
                    storage(5, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("morph_shader"),
                source: wgpu::ShaderSource::Wgsl(MORPH_WGSL.into()),
            });
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("morph_pl"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("morph_pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("cs_morph"),
                compilation_options: Default::default(),
                cache: None,
            });
        self.created_pipelines(1);
        self.morph.pipeline = Some((pipeline, layout));
    }

    fn encode_morph_gpu(&mut self, jobs: &[Job]) {
        self.morph_pipeline();
        let weights_bytes = (self.morph.weights.len().max(1) * 4) as u64;
        Self::ensure_morph_buffer(
            &self.device,
            &mut self.morph.weights_buffer,
            weights_bytes,
            wgpu::BufferUsages::STORAGE,
            "morph_weights",
        );
        Self::ensure_morph_buffer(
            &self.device,
            &mut self.morph.params_buffer,
            jobs.len() as u64 * PARAMS_STRIDE,
            wgpu::BufferUsages::UNIFORM,
            "morph_params",
        );
        if !self.morph.weights.is_empty() {
            self.queue.write_buffer(
                &self.morph.weights_buffer.as_ref().unwrap().0,
                0,
                bytemuck::cast_slice(&self.morph.weights),
            );
        }
        let mut params = vec![0u8; jobs.len() * PARAMS_STRIDE as usize];
        let mut dispatches: Vec<(wgpu::BindGroup, u32, u32)> = Vec::with_capacity(jobs.len());
        for (j, job) in jobs.iter().enumerate() {
            let (_, cache, mesh_idx) = job.key;
            let Some(mesh) = self
                .morph
                .meshes
                .get(&cache)
                .and_then(|m| m.get(mesh_idx))
                .and_then(|m| m.as_ref())
            else {
                continue;
            };
            let Some(src) = self
                .model_gpu_cache
                .get(&cache)
                .and_then(|m| m.as_ref())
                .and_then(|m| m.get(mesh_idx))
                .and_then(|m| match &m.geometry {
                    gpu_driven::MeshGeometry::Dedicated { vertex, .. } => Some(vertex),
                    gpu_driven::MeshGeometry::Shared(_) => None,
                })
            else {
                continue;
            };
            let Some(instance) = self.morph.instances.get_mut(&job.key) else {
                continue;
            };
            let t = &mesh.targets;
            let groups = t.vertex_count.div_ceil(WORKGROUP).max(1);
            let gx = groups.min(MAX_GROUPS_X);
            let gy = groups.div_ceil(gx);
            let flags = (t.has_normals as u32) | ((t.has_tangents as u32) << 1);
            let words: [u32; 8] = [
                t.vertex_count,
                job.arena_base,
                job.staged.weights_offset,
                job.staged.slots,
                (!instance.initialized) as u32,
                flags,
                t.pos_scale.to_bits(),
                gx,
            ];
            instance.initialized = true;
            let at = j * PARAMS_STRIDE as usize;
            params[at..at + 32].copy_from_slice(bytemuck::cast_slice(&words));
            let (layout, weights, params_buffer) = (
                &self.morph.pipeline.as_ref().unwrap().1,
                &self.morph.weights_buffer.as_ref().unwrap().0,
                &self.morph.params_buffer.as_ref().unwrap().0,
            );
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("morph_bg"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: params_buffer,
                            offset: at as u64,
                            size: std::num::NonZeroU64::new(32),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: instance.vertex.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: mesh.blocks.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: mesh.entries.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&self.morph.arena_view),
                    },
                ],
            });
            dispatches.push((bg, gx, gy));
        }
        self.queue
            .write_buffer(&self.morph.params_buffer.as_ref().unwrap().0, 0, &params);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("morph_encoder"),
            });
        {
            let timestamp_writes =
                self.morph
                    .timing
                    .as_ref()
                    .map(|(set, _, _)| wgpu::ComputePassTimestampWrites {
                        query_set: set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morph"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.morph.pipeline.as_ref().unwrap().0);
            for (bg, gx, gy) in &dispatches {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
        }
        if let Some((set, resolve, readback)) = self.morph.timing.as_ref() {
            encoder.resolve_query_set(set, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 16);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        if let Some((_, _, readback)) = self.morph.timing.as_ref() {
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            if let Ok(Ok(())) = rx.recv() {
                let ticks: [u64; 2] = {
                    let data = slice.get_mapped_range();
                    bytemuck::pod_read_unaligned(&data[..16])
                };
                readback.unmap();
                let period = self.queue.get_timestamp_period() as f64;
                self.morph.stats.gpu_ms = ticks[1].saturating_sub(ticks[0]) as f64 * period / 1e6;
            }
        }
    }

    /// The same as the compute pass, on the CPU (no compute on the device).
    fn morph_cpu(&mut self, jobs: &[Job], used: u32) {
        let mut arena = std::mem::take(&mut self.morph.cpu_arena);
        arena.clear();
        arena.resize(used as usize, [0.0; 4]);
        for job in jobs {
            let (_, cache, mesh_idx) = job.key;
            let Some(mesh) = self
                .morph
                .meshes
                .get(&cache)
                .and_then(|m| m.get(mesh_idx))
                .and_then(|m| m.as_ref())
            else {
                continue;
            };
            let Some(base) = self
                .model_gpu_cache
                .get(&cache)
                .and_then(|m| m.as_ref())
                .and_then(|m| m.get(mesh_idx))
                .and_then(|m| m.cpu_vertices.as_ref())
            else {
                continue;
            };
            let Some(instance) = self.morph.instances.get_mut(&job.key) else {
                continue;
            };
            let t = &mesh.targets;
            if base.len() != t.vertex_count as usize {
                continue;
            }
            let at = job.staged.weights_offset as usize;
            let n = job.staged.slots as usize;
            let cur = &self.morph.weights[at..at + n];
            let prev = &self.morph.weights[at + n..at + 2 * n];
            let out = instance.cpu.get_or_insert_with(|| base.clone());
            for v in 0..base.len() {
                if t.ranges[v] == t.ranges[v + 1] {
                    continue;
                }
                out[v] = t.morph_vertex(&base[v], v, cur);
                let pc = t.position_offset(v, prev);
                let cc = t.position_offset(v, cur);
                arena[job.arena_base as usize + v] =
                    [pc[0] - cc[0], pc[1] - cc[1], pc[2] - cc[2], 0.0];
            }
            instance.initialized = true;
            self.queue
                .write_buffer(&instance.vertex, 0, bytemuck::cast_slice(out));
        }
        let rows = used.div_ceil(ARENA_WIDTH);
        arena.resize((rows * ARENA_WIDTH) as usize, [0.0; 4]);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.morph.arena,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&arena),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ARENA_WIDTH * 16),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: ARENA_WIDTH,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        self.morph.cpu_arena = arena;
    }

    /// What the morph pass did last frame.
    pub fn morph_frame_stats(&self) -> MorphFrameStats {
        self.morph.stats()
    }

    /// Measure the morph pass's GPU time into `morph_frame_stats().gpu_ms`
    /// (timestamp queries, read back with a blocking wait each frame: for
    /// benches only). Returns false when the device has no timestamps.
    pub fn set_morph_gpu_timing(&mut self, on: bool) -> bool {
        if !on
            || !self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            self.morph.timing = None;
            return false;
        }
        if self.morph.timing.is_none() {
            let set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("morph_timing"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let buffer = |usage| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("morph_timing"),
                    size: 16,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let resolve = buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
            let readback = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            self.morph.timing = Some((set, resolve, readback));
        }
        true
    }

    /// Force the CPU path (the one used without compute) on a device that
    /// has compute, or go back to the GPU pass. For tests and comparison;
    /// a device without compute stays on the CPU. Returns whether the GPU
    /// pass is now used.
    pub fn set_morph_cpu_fallback(&mut self, cpu: bool) -> bool {
        self.morph.compute = !cpu && device_has_compute(&self.device);
        self.morph.compute
    }
}
