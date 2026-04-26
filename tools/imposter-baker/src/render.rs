//! Headless wgpu renderer: bake N×N octahedral views of a model into
//! one or more atlas textures, then read back to PNGs.
//!
//! V1 shading: Lambert with a fixed sun. Optional normal-encoded and
//! linear-depth atlases share the same camera setup and grid layout.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::gltf_load::MeshData;
use crate::octahedral::cell_direction;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub nrm: [f32; 3],
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj:  [[f32; 4]; 4],
    view:       [[f32; 4]; 4],
    sun_dir:    [f32; 4],     // xyz dir, w unused
    radius_inv: [f32; 4],     // x = 1 / model_radius (depth normalize), yzw unused
}

pub struct BakeOptions<'a> {
    pub grid: u32,
    pub cell_px: u32,
    pub bake_color: bool,
    pub bake_normal: bool,
    pub bake_depth: bool,
    pub mesh: &'a MeshData,
}

pub struct BakedAtlas {
    pub width: u32,
    pub height: u32,
    pub color: Option<Vec<u8>>,   // RGBA8 (sRGB)
    pub normal: Option<Vec<u8>>,  // RGBA8 (linear, oct-encoded normal in xy)
    pub depth: Option<Vec<u8>>,   // R8 (linear normalized depth)
}

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_VIS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub fn bake(opts: BakeOptions<'_>) -> Result<BakedAtlas, String> {
    pollster::block_on(bake_async(opts))
}

async fn bake_async(opts: BakeOptions<'_>) -> Result<BakedAtlas, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("no adapter: {e}"))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("imposter-baker"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .map_err(|e| format!("device: {e}"))?;

    let n = opts.grid;
    let cell = opts.cell_px;
    let atlas_w = n * cell;
    let atlas_h = n * cell;

    // ── Geometry buffers ────────────────────────────────────────────
    let verts: Vec<Vertex> = opts
        .mesh
        .positions
        .iter()
        .zip(opts.mesh.normals.iter())
        .map(|(p, n)| Vertex { pos: *p, _pad0: 0.0, nrm: *n, _pad1: 0.0 })
        .collect();
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("imposter-vb"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("imposter-ib"),
        contents: bytemuck::cast_slice(&opts.mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let index_count = opts.mesh.indices.len() as u32;

    // ── Uniform buffer (re-uploaded per view) ───────────────────────
    let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imposter-ubo"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("imposter-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("imposter-bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("imposter-pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    // Three pipelines (one per output kind), all sharing the same
    // vertex stage. Built lazily (only what we need).
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("imposter-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let make_pipeline = |fs_entry: &str, target_fmt: wgpu::TextureFormat| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(fs_entry),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 16, shader_location: 1 },
                    ],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,   // many GLBs have inconsistent winding; bake both sides.
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_fmt,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    };

    let color_pipeline = opts.bake_color.then(|| make_pipeline("fs_lambert", COLOR_FORMAT));
    let normal_pipeline = opts.bake_normal.then(|| make_pipeline("fs_normal", NORMAL_FORMAT));
    let depth_pipeline = opts.bake_depth.then(|| make_pipeline("fs_depth_vis", DEPTH_VIS_FORMAT));

    // ── Atlases (one per output) ────────────────────────────────────
    let mk_atlas = |label: &str, fmt: wgpu::TextureFormat| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: atlas_w, height: atlas_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    };
    let color_atlas = color_pipeline.as_ref().map(|_| mk_atlas("color-atlas", COLOR_FORMAT));
    let normal_atlas = normal_pipeline.as_ref().map(|_| mk_atlas("normal-atlas", NORMAL_FORMAT));
    let depth_atlas = depth_pipeline.as_ref().map(|_| mk_atlas("depth-atlas", DEPTH_VIS_FORMAT));

    // ── Per-cell render targets (reused across views) ───────────────
    let mk_cell = |label: &str, fmt: wgpu::TextureFormat, attach_only: bool| {
        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if !attach_only {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: cell, height: cell, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        })
    };
    let color_cell = color_pipeline.as_ref().map(|_| mk_cell("color-cell", COLOR_FORMAT, false));
    let normal_cell = normal_pipeline.as_ref().map(|_| mk_cell("normal-cell", NORMAL_FORMAT, false));
    let depth_vis_cell = depth_pipeline.as_ref().map(|_| mk_cell("depth-vis-cell", DEPTH_VIS_FORMAT, false));
    let depth_cell = mk_cell("depth-cell", DEPTH_FORMAT, true);

    let depth_view = depth_cell.create_view(&Default::default());
    let color_view = color_cell.as_ref().map(|t| t.create_view(&Default::default()));
    let normal_view = normal_cell.as_ref().map(|t| t.create_view(&Default::default()));
    let depth_vis_view = depth_vis_cell.as_ref().map(|t| t.create_view(&Default::default()));

    // ── Camera basis ────────────────────────────────────────────────
    let center = opts.mesh.center();
    let radius = opts.mesh.radius().max(1e-4);
    let cam_dist = radius * 1.5;
    let radius_inv = 1.0 / (cam_dist + radius); // normalize view-Z to [0, 1]

    // ── Per-view loop ───────────────────────────────────────────────
    for j in 0..n {
        for i in 0..n {
            let dir = cell_direction(i, j, n);
            // Camera position = center + dir * cam_dist
            let eye = [
                center[0] + dir[0] * cam_dist,
                center[1] + dir[1] * cam_dist,
                center[2] + dir[2] * cam_dist,
            ];
            // Up vector: avoid degenerate look-at when dir ≈ ±Y.
            let up = if dir[1].abs() > 0.999 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
            let view = look_at_rh(eye, center, up);
            // Orthographic sized to the bounding sphere with a little
            // margin. Near/far span the view-aligned bounds.
            let half = radius * 1.05;
            let near = (cam_dist - radius) - 0.05 * radius;
            let far = (cam_dist + radius) + 0.05 * radius;
            let proj = ortho_rh(-half, half, -half, half, near.max(0.01), far);
            let vp = mat_mul(proj, view);

            let sun = normalize3([0.4, 0.85, 0.35]);
            let u = Uniforms {
                view_proj: vp,
                view,
                sun_dir: [sun[0], sun[1], sun[2], 0.0],
                radius_inv: [radius_inv, 0.0, 0.0, 0.0],
            };
            queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&u));

            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("imposter-cell"),
            });

            // Render each enabled output into its per-cell texture.
            if let (Some(pipe), Some(view_t), Some(atlas)) =
                (color_pipeline.as_ref(), color_view.as_ref(), color_atlas.as_ref())
            {
                render_one(&mut enc, pipe, view_t, &depth_view, &bg, &vbuf, &ibuf, index_count, true);
                blit_into_atlas(&mut enc, color_cell.as_ref().unwrap(), atlas, i, j, cell);
            }
            if let (Some(pipe), Some(view_t), Some(atlas)) =
                (normal_pipeline.as_ref(), normal_view.as_ref(), normal_atlas.as_ref())
            {
                render_one(&mut enc, pipe, view_t, &depth_view, &bg, &vbuf, &ibuf, index_count, true);
                blit_into_atlas(&mut enc, normal_cell.as_ref().unwrap(), atlas, i, j, cell);
            }
            if let (Some(pipe), Some(view_t), Some(atlas)) =
                (depth_pipeline.as_ref(), depth_vis_view.as_ref(), depth_atlas.as_ref())
            {
                render_one(&mut enc, pipe, view_t, &depth_view, &bg, &vbuf, &ibuf, index_count, true);
                blit_into_atlas(&mut enc, depth_vis_cell.as_ref().unwrap(), atlas, i, j, cell);
            }

            queue.submit(Some(enc.finish()));
        }
    }

    // ── Readback ────────────────────────────────────────────────────
    let color = if let Some(t) = &color_atlas {
        Some(read_atlas(&device, &queue, t, atlas_w, atlas_h, 4).await?)
    } else { None };
    let normal = if let Some(t) = &normal_atlas {
        Some(read_atlas(&device, &queue, t, atlas_w, atlas_h, 4).await?)
    } else { None };
    let depth = if let Some(t) = &depth_atlas {
        Some(read_atlas(&device, &queue, t, atlas_w, atlas_h, 1).await?)
    } else { None };

    Ok(BakedAtlas { width: atlas_w, height: atlas_h, color, normal, depth })
}

fn render_one(
    enc: &mut wgpu::CommandEncoder,
    pipe: &wgpu::RenderPipeline,
    color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    bg: &wgpu::BindGroup,
    vbuf: &wgpu::Buffer,
    ibuf: &wgpu::Buffer,
    index_count: u32,
    clear: bool,
) {
    let load = if clear {
        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
    } else {
        wgpu::LoadOp::Load
    };
    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("imposter-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target: None,
            ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    rp.set_pipeline(pipe);
    rp.set_bind_group(0, bg, &[]);
    rp.set_vertex_buffer(0, vbuf.slice(..));
    rp.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
    rp.draw_indexed(0..index_count, 0, 0..1);
}

fn blit_into_atlas(
    enc: &mut wgpu::CommandEncoder,
    cell: &wgpu::Texture,
    atlas: &wgpu::Texture,
    i: u32,
    j: u32,
    cell_px: u32,
) {
    enc.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: cell,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: atlas,
            mip_level: 0,
            origin: wgpu::Origin3d { x: i * cell_px, y: j * cell_px, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: cell_px, height: cell_px, depth_or_array_layers: 1 },
    );
}

/// Copy texture to a CPU-mappable buffer, handling wgpu's 256-byte
/// row-pitch alignment, and return the unpadded pixel bytes.
async fn read_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> Result<Vec<u8>, String> {
    let unpadded_bpr = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr = ((unpadded_bpr + align - 1) / align) * align;
    let buf_size = (padded_bpr * height) as u64;

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback-enc") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit(Some(enc.finish()));

    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    rx.recv()
        .map_err(|e| format!("map_async send: {e}"))?
        .map_err(|e| format!("map_async: {e}"))?;

    let view = slice.get_mapped_range();
    // Strip per-row padding.
    let mut out = Vec::with_capacity((unpadded_bpr * height) as usize);
    for row in 0..height {
        let src = (row * padded_bpr) as usize;
        out.extend_from_slice(&view[src..src + unpadded_bpr as usize]);
    }
    drop(view);
    buf.unmap();
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────
// Tiny matrix helpers (column-major, row-vector × matrix multiplication
// matches WGSL `proj * view * model * vec4(pos, 1.0)` when uploaded
// as-is).
// ────────────────────────────────────────────────────────────────────

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / l, v[1] / l, v[2] / l]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let s = normalize3(cross(f, up));
    let u = cross(s, f);
    [
        [ s[0],  u[0], -f[0], 0.0],
        [ s[1],  u[1], -f[1], 0.0],
        [ s[2],  u[2], -f[2], 0.0],
        [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

fn ortho_rh(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> [[f32; 4]; 4] {
    // wgpu clip-space Z = [0, 1].
    let rcp_w = 1.0 / (r - l);
    let rcp_h = 1.0 / (t - b);
    let rcp_d = 1.0 / (f - n);
    [
        [2.0 * rcp_w, 0.0, 0.0, 0.0],
        [0.0, 2.0 * rcp_h, 0.0, 0.0],
        [0.0, 0.0, -rcp_d, 0.0],
        [-(r + l) * rcp_w, -(t + b) * rcp_h, -n * rcp_d, 1.0],
    ]
}

fn mat_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Column-major: result[col][row] = sum_k a[k][row] * b[col][k]
    let mut out = [[0.0; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k][row] * b[col][k];
            }
            out[col][row] = s;
        }
    }
    out
}

// ────────────────────────────────────────────────────────────────────
// WGSL: vertex stage shared, three fragment outputs (lambert / normal /
// linear-depth-visualization).
// ────────────────────────────────────────────────────────────────────

const SHADER_SRC: &str = r#"
struct U {
  view_proj:  mat4x4<f32>,
  view:       mat4x4<f32>,
  sun_dir:    vec4<f32>,
  radius_inv: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct VsOut {
  @builtin(position) clip_pos: vec4<f32>,
  @location(0) world_nrm: vec3<f32>,
  @location(1) view_z:    f32,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>) -> VsOut {
  var o: VsOut;
  o.clip_pos  = u.view_proj * vec4<f32>(pos, 1.0);
  o.world_nrm = normalize(nrm);
  let view_pos = u.view * vec4<f32>(pos, 1.0);
  // Distance from camera along view -Z (RH look-at points -Z forward).
  o.view_z = -view_pos.z;
  return o;
}

@fragment
fn fs_lambert(i: VsOut) -> @location(0) vec4<f32> {
  let n   = normalize(i.world_nrm);
  let sun = normalize(u.sun_dir.xyz);
  let ndl = max(dot(n, sun), 0.0);
  let ambient = 0.18;
  let lit = ambient + (1.0 - ambient) * ndl;
  // Neutral grey base — V1 ignores GLB textures; V2 follow-up will
  // sample the base-color texture and modulate.
  let base = vec3<f32>(0.78, 0.78, 0.78);
  return vec4<f32>(base * lit, 1.0);
}

// View-space normal, octahedral-encoded into xy. z = 0, w = 1.
fn oct_encode(d: vec3<f32>) -> vec2<f32> {
  let n = d / (abs(d.x) + abs(d.y) + abs(d.z));
  if (n.y >= 0.0) { return vec2<f32>(n.x, n.z); }
  let sx = select(-1.0, 1.0, n.x >= 0.0);
  let sz = select(-1.0, 1.0, n.z >= 0.0);
  return vec2<f32>((1.0 - abs(n.z)) * sx, (1.0 - abs(n.x)) * sz);
}

@fragment
fn fs_normal(i: VsOut) -> @location(0) vec4<f32> {
  let nrm_view = (u.view * vec4<f32>(normalize(i.world_nrm), 0.0)).xyz;
  let oct = oct_encode(normalize(nrm_view)) * 0.5 + 0.5;
  return vec4<f32>(oct, 0.0, 1.0);
}

@fragment
fn fs_depth_vis(i: VsOut) -> @location(0) vec4<f32> {
  let d = clamp(i.view_z * u.radius_inv.x, 0.0, 1.0);
  return vec4<f32>(d, d, d, 1.0);
}
"#;
