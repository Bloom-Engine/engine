//! Hi-Z occlusion culling — coarse max-depth grid with async readback.
//!
//! The engine already builds a linear-depth Hi-Z pyramid for SSAO/SSR,
//! but that chain is min-reduced (nearest depth — what ray marching
//! wants). Occlusion needs the opposite bound: a node is provably hidden
//! only if its nearest point is farther than the FARTHEST depth across
//! its whole screen footprint. So this module adds one small compute
//! reduce: Hi-Z mip 0 → a 64×64 max-depth grid, copied to a mappable
//! buffer and read back asynchronously.
//!
//! The CPU test runs one frame late (against last frame's grid and last
//! frame's view-projection) — the standard latency trade that avoids any
//! GPU stall. Every uncertain case resolves to "visible": no grid yet,
//! corner behind the near plane, footprint off the captured screen,
//! depth within the safety margin. Disocclusion artifacts from camera
//! cuts last exactly one frame.
//!
//! `bloom_set_occlusion_culling(0/1)` exposes the kill switch to games;
//! default on.

use wgpu::util::DeviceExt;

pub(super) const GRID_W: u32 = 64;
pub(super) const GRID_H: u32 = 64;
// 64 texels * 4 bytes = 256 bytes per row — exactly wgpu's
// COPY_BYTES_PER_ROW_ALIGNMENT, so the readback needs no padding.
const ROW_BYTES: u32 = GRID_W * 4;

const REDUCE_SHADER: &str = "
struct Params {
    // xy = source (hiz mip0) size, zw = tile size in source texels
    size: vec4<u32>,
};
@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var dst_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 64u || gid.y >= 64u) { return; }
    let base = vec2<u32>(gid.x * u.size.z, gid.y * u.size.w);
    var m: f32 = 0.0;
    for (var ty: u32 = 0u; ty < u.size.w; ty = ty + 1u) {
        for (var tx: u32 = 0u; tx < u.size.z; tx = tx + 1u) {
            let p = base + vec2<u32>(tx, ty);
            if (p.x < u.size.x && p.y < u.size.y) {
                m = max(m, textureLoad(src_tex, vec2<i32>(p), 0).r);
            }
        }
    }
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(m, 0.0, 0.0, 0.0));
}
";

struct Readback {
    buffer: wgpu::Buffer,
    /// capture VP for the data this buffer holds
    vp: [[f32; 4]; 4],
    /// copy submitted, map_async issued, result not yet collected
    in_flight: bool,
    map_done: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub struct OcclusionCuller {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    grid_tex: wgpu::Texture,
    grid_view: wgpu::TextureView,
    bg_cache: Option<wgpu::BindGroup>,
    readbacks: [Readback; 2],
    parity: usize,
    /// most recent completed grid
    grid: Vec<f32>,
    grid_valid: bool,
    grid_vp: [[f32; 4]; 4],
    pub enabled: bool,
    /// EN-057 — false when no rasterized scene node exists to consume the
    /// culling verdicts (e.g. a scene whose only nodes are gi_only proxies:
    /// they never draw, so the reduce + readback benefited zero draws every
    /// frame). Set per frame by the engine from the scene graph; defaults to
    /// true so hosts that never call the setter keep today's behaviour.
    has_consumers: bool,
    /// set when record() ran this frame so after_submit() knows to map
    recorded_this_frame: bool,
}

impl OcclusionCuller {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("occlusion_reduce_shader"),
            source: wgpu::ShaderSource::Wgsl(REDUCE_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("occlusion_reduce_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("occlusion_reduce_pl"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("occlusion_reduce_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("occlusion_reduce_uniform"),
            contents: &[0u8; 16],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let grid_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("occlusion_grid"),
            size: wgpu::Extent3d { width: GRID_W, height: GRID_H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let grid_view = grid_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mk_readback = || Readback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("occlusion_readback"),
                size: (ROW_BYTES * GRID_H) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            vp: [[0.0; 4]; 4],
            in_flight: false,
            map_done: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        Self {
            pipeline,
            layout,
            uniform,
            grid_tex,
            grid_view,
            bg_cache: None,
            readbacks: [mk_readback(), mk_readback()],
            parity: 0,
            grid: vec![0.0; (GRID_W * GRID_H) as usize],
            grid_valid: false,
            grid_vp: [[0.0; 4]; 4],
            enabled: true,
            has_consumers: true,
            recorded_this_frame: false,
        }
    }

    /// EN-057 — tell the culler whether any rasterized consumer exists this
    /// frame. Going consumer-less also invalidates the grid, so if a
    /// consumer appears later the interim frames read the conservative
    /// "potentially visible" answer (test_aabb on an invalid grid) instead
    /// of a stale capture — the gate cannot cost a wrongly-culled draw by
    /// construction.
    pub fn set_has_consumers(&mut self, has: bool) {
        if !has {
            self.grid_valid = false;
        }
        self.has_consumers = has;
    }

    /// Drop cached bind group (call when the Hi-Z chain is reallocated,
    /// e.g. on resize / render-scale change).
    pub fn invalidate_bindings(&mut self) {
        self.bg_cache = None;
    }

    /// Collect any finished readback. Call once per frame before culling.
    pub fn poll(&mut self, device: &wgpu::Device) {
        use std::sync::atomic::Ordering;
        // Non-blocking pump so map_async callbacks make progress even on
        // frames where nothing else polls the device.
        let _ = device.poll(wgpu::PollType::Poll);
        for rb in &mut self.readbacks {
            if rb.in_flight && rb.map_done.load(Ordering::Acquire) {
                {
                    let n = self.grid.len();
                    let view = rb.buffer.slice(..).get_mapped_range();
                    let floats: &[f32] = bytemuck::cast_slice(&view);
                    self.grid.copy_from_slice(&floats[..n]);
                }
                rb.buffer.unmap();
                rb.in_flight = false;
                rb.map_done.store(false, Ordering::Release);
                self.grid_vp = rb.vp;
                self.grid_valid = true;
            }
        }
    }

    /// Record the reduce + copy for this frame's grid capture.
    /// `src` is Hi-Z mip 0 (linear |view_z|, sky = 10000) of `src_size`.
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        src_size: (u32, u32),
        vp: [[f32; 4]; 4],
    ) {
        self.recorded_this_frame = false;
        if !self.enabled || !self.has_consumers {
            return;
        }
        let rb = &mut self.readbacks[self.parity];
        if rb.in_flight {
            // Previous capture still in flight (GPU more than a frame
            // behind) — skip; the grid just stays one frame staler.
            return;
        }
        let tile_w = src_size.0.div_ceil(GRID_W).max(1);
        let tile_h = src_size.1.div_ceil(GRID_H).max(1);
        let params: [u32; 4] = [src_size.0, src_size.1, tile_w, tile_h];
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&params));

        if self.bg_cache.is_none() {
            self.bg_cache = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("occlusion_reduce_bg"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.grid_view) },
                ],
            }));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("occlusion_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.bg_cache.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(GRID_W / 8, GRID_H / 8, 1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.grid_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &rb.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW_BYTES),
                    rows_per_image: Some(GRID_H),
                },
            },
            wgpu::Extent3d { width: GRID_W, height: GRID_H, depth_or_array_layers: 1 },
        );
        rb.vp = vp;
        self.recorded_this_frame = true;
    }

    /// Issue the async map for the capture recorded this frame. Call
    /// after queue.submit() of the encoder passed to record().
    pub fn after_submit(&mut self) {
        if !self.recorded_this_frame {
            return;
        }
        let rb = &mut self.readbacks[self.parity];
        let done = rb.map_done.clone();
        rb.in_flight = true;
        rb.buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                done.store(true, std::sync::atomic::Ordering::Release);
            }
            // On error the buffer stays flagged in-flight until the next
            // successful cycle on the other parity; culling simply keeps
            // using the older grid.
        });
        self.parity = 1 - self.parity;
        self.recorded_this_frame = false;
    }

    /// Conservative visibility test for a world-space AABB against the
    /// last completed grid. `true` = potentially visible (draw it).
    pub fn test_aabb(&self, wmin: [f32; 3], wmax: [f32; 3]) -> bool {
        if !self.enabled || !self.grid_valid {
            return true;
        }
        let vp = &self.grid_vp;
        let mut uv_min = [f32::MAX, f32::MAX];
        let mut uv_max = [f32::MIN, f32::MIN];
        let mut nearest = f32::MAX;
        for ix in 0..2 {
            for iy in 0..2 {
                for iz in 0..2 {
                    let x = if ix == 0 { wmin[0] } else { wmax[0] };
                    let y = if iy == 0 { wmin[1] } else { wmax[1] };
                    let z = if iz == 0 { wmin[2] } else { wmax[2] };
                    let cw = vp[0][3] * x + vp[1][3] * y + vp[2][3] * z + vp[3][3];
                    if cw <= 0.05 {
                        // corner at/behind the captured near plane —
                        // can't bound the footprint; play safe
                        return true;
                    }
                    let cx = vp[0][0] * x + vp[1][0] * y + vp[2][0] * z + vp[3][0];
                    let cy = vp[0][1] * x + vp[1][1] * y + vp[2][1] * z + vp[3][1];
                    let u = (cx / cw) * 0.5 + 0.5;
                    let v = 1.0 - ((cy / cw) * 0.5 + 0.5);
                    uv_min[0] = uv_min[0].min(u);
                    uv_min[1] = uv_min[1].min(v);
                    uv_max[0] = uv_max[0].max(u);
                    uv_max[1] = uv_max[1].max(v);
                    nearest = nearest.min(cw);
                }
            }
        }
        // Fully outside the captured view → last frame's depth says
        // nothing about it. (Current-frame frustum culling handles
        // actual offscreen-ness.)
        if uv_max[0] <= 0.0 || uv_min[0] >= 1.0 || uv_max[1] <= 0.0 || uv_min[1] >= 1.0 {
            return true;
        }
        // Expand by one texel for footprint conservatism, clamp to grid.
        let tx0 = ((uv_min[0] * GRID_W as f32) as i32 - 1).clamp(0, GRID_W as i32 - 1) as usize;
        let tx1 = ((uv_max[0] * GRID_W as f32) as i32 + 1).clamp(0, GRID_W as i32 - 1) as usize;
        let ty0 = ((uv_min[1] * GRID_H as f32) as i32 - 1).clamp(0, GRID_H as i32 - 1) as usize;
        let ty1 = ((uv_max[1] * GRID_H as f32) as i32 + 1).clamp(0, GRID_H as i32 - 1) as usize;
        let mut grid_max = 0.0f32;
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                grid_max = grid_max.max(self.grid[ty * GRID_W as usize + tx]);
            }
        }
        // Margin: 2% relative + 0.1 absolute absorbs linearization and
        // one frame of camera motion for typical scenes.
        nearest <= grid_max * 1.02 + 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// VP looking down -Z from the origin in the engine's column-major
    /// convention (vp[col][row]): clip.w = -z, x/y pass through.
    fn look_down_neg_z() -> [[f32; 4]; 4] {
        let mut m = [[0.0f32; 4]; 4];
        m[0][0] = 1.0;
        m[1][1] = 1.0;
        m[2][2] = -1.0;
        m[2][3] = -1.0; // w = -z
        m
    }

    #[test]
    fn occluded_and_visible_cases() {
        let Some((device, _queue)) = try_device() else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut c = OcclusionCuller::new(&device);
        // Grid: farthest visible surface everywhere is at depth 10.
        c.grid.fill(10.0);
        c.grid_valid = true;
        c.grid_vp = look_down_neg_z();

        // Box fully behind that wall (depth 20..21, small footprint).
        assert!(
            !c.test_aabb([-0.1, -0.1, -21.0], [0.1, 0.1, -20.0]),
            "box behind a full-screen depth-10 wall should be culled"
        );
        // Box in front of the wall (depth 5).
        assert!(c.test_aabb([-0.1, -0.1, -5.2], [0.1, 0.1, -5.0]));
        // Box straddling the near plane — must play safe.
        assert!(c.test_aabb([-0.1, -0.1, -20.0], [0.1, 0.1, 1.0]));
        // Without a valid grid, never cull.
        c.grid_valid = false;
        assert!(c.test_aabb([-0.1, -0.1, -21.0], [0.1, 0.1, -20.0]));
        // Disabled culler never culls.
        c.grid_valid = true;
        c.enabled = false;
        assert!(c.test_aabb([-0.1, -0.1, -21.0], [0.1, 0.1, -20.0]));
    }
}
