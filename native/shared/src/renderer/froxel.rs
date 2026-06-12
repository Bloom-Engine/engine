//! Froxel light clustering — task #23 of the architecture audit.
//!
//! The 8+256 light-cap raise removed the capability ceiling but left
//! the scene shader paying O(live point lights) per fragment. This
//! module restores O(cluster lights): a compute pass assigns the point
//! lights (read from the same lighting UBO the shaders already use) to
//! a 16×9×24 view-frustum froxel grid each frame, and a clustered
//! variant of the scene shader loops only its froxel's index list.
//!
//! Backend split, by capability rather than cfg: storage buffers in
//! fragment shaders don't exist on WebGL2, so [`FroxelPass::supported`]
//! gates on the device limits. Unsupported backends keep the plain
//! count-driven loop (the semantic reference — the clustered path must
//! match it exactly, which the many_point_lights golden enforces).
//!
//! Memory: counts 3456×4 B ≈ 14 KB; index list 3456×256×4 B ≈ 3.5 MB
//! (256 = worst-case every light in one froxel — exact parity with the
//! reference loop, no truncation).

use wgpu::util::DeviceExt;

pub(super) const GRID_X: u32 = 16;
pub(super) const GRID_Y: u32 = 9;
pub(super) const GRID_Z: u32 = 24;
pub(super) const CLUSTER_COUNT: u32 = GRID_X * GRID_Y * GRID_Z;
pub(super) const MAX_LIGHTS_PER_CLUSTER: u32 = 256;

/// Uniform parameters shared by the assignment compute pass and the
/// clustered fragment loop. Layout mirrored in WGSL below and in the
/// fragment include.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FroxelParams {
    /// View matrix (world → view) for light-position transform.
    pub view: [[f32; 4]; 4],
    /// x = grid_x, y = grid_y, z = grid_z, w = live point-light count.
    pub grid: [u32; 4],
    /// x = znear, y = zfar, z = log(zfar/znear), w = unused.
    pub depth_range: [f32; 4],
    /// x = 1/tile_w_px, y = 1/tile_h_px (fragment tile lookup),
    /// z = p22, w = p32 (depth linearization, same convention as Hi-Z).
    pub screen: [f32; 4],
    /// Inverse projection — froxel corner reconstruction in the
    /// assignment pass.
    pub inv_proj: [[f32; 4]; 4],
}

const ASSIGN_SHADER: &str = "
struct FroxelParams {
    view: mat4x4<f32>,
    grid: vec4<u32>,
    depth_range: vec4<f32>,
    screen: vec4<f32>,
    inv_proj: mat4x4<f32>,
};
struct PointLight { position: vec4<f32>, color: vec4<f32> };
struct Lights {
    // Mirrors the tail of the Lighting UBO relevant here. The host
    // binds a dedicated compact UBO (positions+ranges only would do,
    // but reusing PointLight keeps one struct).
    count: vec4<f32>,
    lights: array<PointLight, 256>,
};

@group(0) @binding(0) var<uniform> p: FroxelParams;
@group(0) @binding(1) var<uniform> l: Lights;
@group(0) @binding(2) var<storage, read_write> cluster_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> cluster_indices: array<u32>;

// View-space Z of slice boundary k (logarithmic distribution).
fn slice_z(k: u32) -> f32 {
    let t = f32(k) / f32(p.grid.z);
    return p.depth_range.x * exp(t * p.depth_range.z);
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.grid.x || gid.y >= p.grid.y || gid.z >= p.grid.z) { return; }
    let cluster = gid.x + gid.y * p.grid.x + gid.z * p.grid.x * p.grid.y;

    // Froxel AABB in view space: reconstruct the tile's corner rays on
    // the near plane of the projection and scale to the slice depths.
    // NDC tile extents:
    let x0 = (f32(gid.x)       / f32(p.grid.x)) * 2.0 - 1.0;
    let x1 = (f32(gid.x + 1u)  / f32(p.grid.x)) * 2.0 - 1.0;
    // NDC y is up; tile row 0 is the TOP of the screen.
    let y1 = 1.0 - (f32(gid.y)      / f32(p.grid.y)) * 2.0;
    let y0 = 1.0 - (f32(gid.y + 1u) / f32(p.grid.y)) * 2.0;

    // Unproject the four corners at an arbitrary depth and normalize to
    // rays through the camera (view space, looking down -Z).
    var mn = vec3<f32>( 1e30,  1e30,  1e30);
    var mx = vec3<f32>(-1e30, -1e30, -1e30);
    let z_near_s = slice_z(gid.z);
    let z_far_s  = slice_z(gid.z + 1u);
    for (var cx = 0u; cx < 2u; cx++) {
        for (var cy = 0u; cy < 2u; cy++) {
            let nx = select(x0, x1, cx == 1u);
            let ny = select(y0, y1, cy == 1u);
            let h = p.inv_proj * vec4<f32>(nx, ny, 0.5, 1.0);
            let dir = h.xyz / h.w;          // a point on the ray (view space)
            let ray = dir / max(-dir.z, 1e-6); // scale so z == -1
            // corner at both slice depths (view z is negative forward)
            let a = ray * z_near_s;
            let b = ray * z_far_s;
            mn = min(mn, min(vec3<f32>(a.xy, -z_near_s), vec3<f32>(b.xy, -z_far_s)));
            mx = max(mx, max(vec3<f32>(a.xy, -z_near_s), vec3<f32>(b.xy, -z_far_s)));
        }
    }

    // Sphere/AABB tests against every live light.
    var count = 0u;
    let n = u32(l.count.x);
    let base = cluster * 256u;
    for (var i = 0u; i < n; i++) {
        let pos_w = l.lights[i].position;
        let pos_v = (p.view * vec4<f32>(pos_w.xyz, 1.0)).xyz;
        let r = pos_w.w;
        let closest = clamp(pos_v, mn, mx);
        let d = pos_v - closest;
        if (dot(d, d) <= r * r) {
            cluster_indices[base + count] = i;
            count++;
        }
    }
    cluster_counts[cluster] = count;
}
";

/// The fragment-side replacement for the plain point-light loop, plus
/// the bindings it needs. Spliced into SCENE_SHADER between the
/// BEGIN/END-POINT-LIGHT-LOOP markers by [`clustered_scene_shader`].
const CLUSTERED_BINDINGS: &str = "
struct FroxelParams {
    view: mat4x4<f32>,
    grid: vec4<u32>,
    depth_range: vec4<f32>,
    screen: vec4<f32>,
    inv_proj: mat4x4<f32>,
};
@group(1) @binding(10) var<uniform> froxel: FroxelParams;
@group(1) @binding(11) var<storage, read> cluster_counts: array<u32>;
@group(1) @binding(12) var<storage, read> cluster_indices: array<u32>;
";

const CLUSTERED_LOOP: &str = "
    // Froxel-clustered point lights: identical shading math to the
    // reference loop, restricted to this fragment's cluster list.
    let view_z = -froxel.screen.w / (in.clip_position.z + froxel.screen.z);
    let slice = clamp(
        u32(log(max(view_z, froxel.depth_range.x) / froxel.depth_range.x)
            / froxel.depth_range.z * f32(froxel.grid.z)),
        0u, froxel.grid.z - 1u);
    let tile_x = min(u32(in.clip_position.x * froxel.screen.x), froxel.grid.x - 1u);
    let tile_y = min(u32(in.clip_position.y * froxel.screen.y), froxel.grid.y - 1u);
    let cluster = tile_x + tile_y * froxel.grid.x + slice * froxel.grid.x * froxel.grid.y;
    let cl_count = cluster_counts[cluster];
    let cl_base = cluster * 256u;
    for (var ci = 0u; ci < cl_count; ci++) {
        let pl = lighting.point_lights[cluster_indices[cl_base + ci]];
        let to_light = pl.position.xyz - in.world_pos;
        let dist = length(to_light);
        let range = pl.position.w;
        if (dist < range && dist > 0.0) {
            let l = to_light / dist;
            let atten = 1.0 - (dist / range);
            let atten2 = atten * atten;
            lit += shade_pbr(n, v, l, pl.color.rgb, pl.color.w * atten2,
                             base_color, metallic, roughness);
        }
    }
";

/// Build the clustered SCENE_SHADER variant from the canonical source.
pub(super) fn clustered_scene_shader(source: &str) -> String {
    let begin = source
        .find("// BEGIN-POINT-LIGHT-LOOP")
        .expect("scene shader missing BEGIN-POINT-LIGHT-LOOP marker");
    let end_marker = "// END-POINT-LIGHT-LOOP";
    let end = source.find(end_marker).expect("scene shader missing END marker") + end_marker.len();
    format!(
        "{}{}{}{}",
        CLUSTERED_BINDINGS,
        &source[..begin],
        CLUSTERED_LOOP,
        &source[end..]
    )
}

/// The three entries appended to `lighting_layout` (group 1) when the
/// device supports the clustered path. Pipelines whose shaders don't
/// reference them (SHADER_3D's pipeline_3d) are unaffected — extra
/// layout entries are legal as long as the bind group provides them.
pub(super) fn extra_lighting_layout_entries() -> [wgpu::BindGroupLayoutEntry; 3] {
    let storage_ro = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    [
        wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry { binding: 11, visibility: wgpu::ShaderStages::FRAGMENT, ty: storage_ro, count: None },
        wgpu::BindGroupLayoutEntry { binding: 12, visibility: wgpu::ShaderStages::FRAGMENT, ty: storage_ro, count: None },
    ]
}

pub struct FroxelPass {
    pub assign_pipeline: wgpu::ComputePipeline,
    pub assign_layout: wgpu::BindGroupLayout,
    pub params_buffer: wgpu::Buffer,
    /// Compact point-light UBO for the compute pass (count + 256 lights).
    pub lights_buffer: wgpu::Buffer,
    pub counts_buffer: wgpu::Buffer,
    pub indices_buffer: wgpu::Buffer,
    assign_bg: wgpu::BindGroup,
}

impl FroxelPass {
    /// Storage buffers must be available in BOTH compute and fragment
    /// stages (WebGL2 has neither). `BLOOM_DISABLE_FROXEL=1` forces the
    /// reference loop — used to (re)generate the clustered-parity
    /// golden and to bisect suspected clustering bugs in the field.
    pub fn supported(device: &wgpu::Device) -> bool {
        if std::env::var_os("BLOOM_DISABLE_FROXEL").is_some_and(|v| v == "1") {
            return false;
        }
        let l = device.limits();
        l.max_storage_buffers_per_shader_stage >= 2
            && l.max_storage_buffer_binding_size as u64
                >= (CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER * 4) as u64
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("froxel_assign_shader"),
            source: wgpu::ShaderSource::Wgsl(ASSIGN_SHADER.into()),
        });
        let assign_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("froxel_assign_layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("froxel_assign_pl"),
            bind_group_layouts: &[Some(&assign_layout)],
            ..Default::default()
        });
        let assign_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("froxel_assign_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("froxel_params"),
            contents: &[0u8; std::mem::size_of::<FroxelParams>()],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // count vec4 + 256 lights × 2 vec4
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("froxel_lights"),
            size: 16 + 256 * 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("froxel_counts"),
            size: (CLUSTER_COUNT * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("froxel_indices"),
            size: (CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let assign_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("froxel_assign_bg"),
            layout: &assign_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lights_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: indices_buffer.as_entire_binding() },
            ],
        });
        Self {
            assign_pipeline,
            assign_layout,
            params_buffer,
            lights_buffer,
            counts_buffer,
            indices_buffer,
            assign_bg,
        }
    }

    /// The bind-group entries matching [`extra_lighting_layout_entries`],
    /// appended to every lighting bind group the renderer builds.
    pub(super) fn extra_lighting_bind_entries(&self) -> [wgpu::BindGroupEntry<'_>; 3] {
        [
            wgpu::BindGroupEntry { binding: 10, resource: self.params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: self.counts_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: self.indices_buffer.as_entire_binding() },
        ]
    }

    /// Record the per-frame assignment dispatch. The caller uploads
    /// params + lights first (see Renderer::record_froxel_assign).
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("froxel_assign_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.assign_pipeline);
        pass.set_bind_group(0, &self.assign_bg, &[]);
        pass.dispatch_workgroups(GRID_X / 4, GRID_Y.div_ceil(4), GRID_Z / 4);
    }
}

impl super::Renderer {
    /// Upload froxel params + the compact light list and dispatch the
    /// assignment pass. Runs every 3D frame on supported devices —
    /// even with zero lights, so `cluster_counts` never carries stale
    /// data from a previous frame's camera.
    pub(super) fn record_froxel_assign(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(froxel) = &self.froxel else { return };

        let proj = self.current_proj_matrix;
        let p22 = proj[2][2];
        let p32 = proj[3][2];
        // Same linearization as Hi-Z: view_z(depth) = -p32/(depth + p22),
        // positive forward. Evaluate at depth 0 and 1; min/max makes
        // this hold for reversed-Z too, and the clamps keep an
        // infinite-far projection (division by ~0) finite.
        let z_at = |d: f32| -p32 / (d + p22);
        let (z0, z1) = (z_at(0.0), z_at(1.0));
        let znear = z0.min(z1).max(1e-3);
        let zfar = z0.max(z1).clamp(znear * 1.001, 1e9);

        // clip_position.xy is in render-target pixels — the HDR scene
        // pass runs at render_extent (render_scale-aware), not surface
        // size.
        let (rw, rh) = self.render_extent();
        let n = (self.lighting_uniforms.point_light_count[0] as u32)
            .min(MAX_LIGHTS_PER_CLUSTER);
        let params = FroxelParams {
            view: self.current_view_matrix,
            grid: [GRID_X, GRID_Y, GRID_Z, n],
            depth_range: [znear, zfar, (zfar / znear).ln(), 0.0],
            screen: [
                GRID_X as f32 / rw.max(1) as f32,
                GRID_Y as f32 / rh.max(1) as f32,
                p22,
                p32,
            ],
            inv_proj: self.current_inv_proj_matrix,
        };
        self.queue.write_buffer(&froxel.params_buffer, 0, bytemuck::bytes_of(&params));
        let count = [n as f32, 0.0, 0.0, 0.0_f32];
        self.queue.write_buffer(&froxel.lights_buffer, 0, bytemuck::bytes_of(&count));
        self.queue.write_buffer(
            &froxel.lights_buffer,
            16,
            bytemuck::cast_slice(&self.lighting_uniforms.point_lights),
        );
        froxel.record(encoder);
    }
}
