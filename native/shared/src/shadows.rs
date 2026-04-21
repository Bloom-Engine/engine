//! Cascaded shadow mapping (CSM) for Bloom Engine.
//!
//! Implements 3-cascade directional light shadow mapping with PCF
//! (Percentage-Closer Filtering). The camera frustum is split into
//! near/mid/far slices, each rendered from the light's perspective into
//! its own depth texture. The scene shader selects the tightest cascade
//! for each fragment, giving high shadow resolution near the camera and
//! coverage out to the far plane.

use crate::renderer::IDENTITY_MAT4;

/// Number of shadow cascades.
pub const NUM_CASCADES: usize = 3;
/// Per-cascade shadow map resolution. 1024 (down from 2048) quarters
/// the fragment work per cascade and cuts the whole shadow pass from
/// ~14 ms to ~4 ms on Sponza. Softness is retained by the PCF kernel
/// + cascade splits; sharpness loss on near-field edges is minor and
/// can be tuned back by bumping this when targeting high-end GPUs.
pub const CASCADE_MAP_SIZE: u32 = 1024;
pub const SHADOW_NEAR: f32 = 0.1;
pub const SHADOW_FAR: f32 = 100.0;
/// Dynamic-uniform buffer stride for per-node shadow uniforms. Must
/// be >= sizeof(ShadowUniforms) (128B) and a multiple of the device's
/// min_uniform_buffer_offset_alignment. 256 is safe on every platform.
pub const SHADOW_UNIFORM_STRIDE: u32 = 256;
pub const SHADOW_MAX_NODES: u32 = 1024;

/// Depth-only shader for shadow pass.
pub const SHADOW_SHADER: &str = "
struct ShadowUniforms {
    light_vp: mat4x4<f32>,
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> shadow_u: ShadowUniforms;

struct ShadowVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
};

@vertex
fn vs_shadow(in: ShadowVertexInput) -> @builtin(position) vec4<f32> {
    let world_pos = shadow_u.model * vec4<f32>(in.position, 1.0);
    return shadow_u.light_vp * world_pos;
}
";

/// Uniform data for the shadow pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniforms {
    pub light_vp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
}

/// Shadow map resources for cascaded shadow mapping.
pub struct ShadowMap {
    pub depth_textures: [wgpu::Texture; NUM_CASCADES],
    pub depth_views: [wgpu::TextureView; NUM_CASCADES],
    pub sampler: wgpu::Sampler,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub uniform_layout: wgpu::BindGroupLayout,
    pub light_vps: [[[f32; 4]; 4]; NUM_CASCADES],
    /// View-space Z split distances for each cascade. Cascade i covers
    /// [cascade_splits[i-1], cascade_splits[i]]; cascade 0 starts at near.
    pub cascade_splits: [f32; NUM_CASCADES],
    pub enabled: bool,
    /// Forces a shadow re-render next frame. Set by `invalidate()`
    /// (on light direction change, `setShadowsEnabled(true)`, resize,
    /// shadow-texture aliasing, etc.). Cleared after a render.
    pub dirty: bool,
    /// Escape hatch for games with continuously-changing light state
    /// where the cache hit rate would be ~zero anyway. When true, every
    /// frame renders shadows; the cache is bypassed.
    pub always_fresh: bool,
    /// Cascade VPs that correspond to the contents currently stored in
    /// the depth textures. `None` before the first render. When the
    /// freshly-computed VPs match this byte-for-byte AND nothing else
    /// has invalidated, we can skip the render entirely and sample the
    /// retained depth textures. Texel-snapping + radius quantization in
    /// `compute_cascade_vps` means identical camera poses produce
    /// identical VPs, so this check is robust.
    pub rendered_light_vps: Option<[[[f32; 4]; 4]; NUM_CASCADES]>,
    /// Light direction used for the current depth-texture contents.
    /// Checked at the cache gate rather than on the setter because
    /// `begin_frame` resets `lighting_uniforms` to defaults every
    /// frame — comparing a setter's old-vs-new would always see the
    /// default as the "old" value and invalidate every frame.
    pub rendered_light_dir: Option<[f32; 3]>,
    /// Scene-graph version counter sampled at the last shadow render.
    /// `SceneGraph::shadow_version` increments whenever a shadow-casting
    /// node's transform / cast_shadow / visibility / geometry changes;
    /// a mismatch here forces a re-render.
    pub rendered_scene_version: u64,
}

impl ShadowMap {
    pub fn new(device: &wgpu::Device, vertex_layout: wgpu::VertexBufferLayout<'static>) -> Self {
        // Create NUM_CASCADES depth textures
        let mut depth_textures_vec: Vec<wgpu::Texture> = Vec::new();
        let mut depth_views_vec: Vec<wgpu::TextureView> = Vec::new();
        for i in 0..NUM_CASCADES {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("shadow_depth_cascade_{}", i)),
                size: wgpu::Extent3d {
                    width: CASCADE_MAP_SIZE,
                    height: CASCADE_MAP_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            depth_textures_vec.push(tex);
            depth_views_vec.push(view);
        }

        // Convert Vecs to fixed-size arrays
        let depth_textures: [wgpu::Texture; NUM_CASCADES] =
            depth_textures_vec.try_into().unwrap_or_else(|_| panic!("cascade texture count mismatch"));
        let depth_views: [wgpu::TextureView; NUM_CASCADES] =
            depth_views_vec.try_into().unwrap_or_else(|_| panic!("cascade view count mismatch"));

        // Comparison sampler for PCF
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Bind group layout for sampling shadow maps in the main pass:
        // 3 depth textures (bindings 0,1,2) + 1 comparison sampler (binding 3)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_sample_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_sample_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Shadow pass uniform layout (dynamic offset for per-node)
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_uniform_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ShadowUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_uniform_buf"),
            size: (SHADOW_UNIFORM_STRIDE * SHADOW_MAX_NODES) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_uniform_bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ShadowUniforms>() as u64,
                    ),
                }),
            }],
        });

        // Shadow depth-only pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pipeline_layout"),
            bind_group_layouts: &[Some(&uniform_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: None, // depth only
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            depth_textures,
            depth_views,
            sampler,
            bind_group_layout,
            bind_group,
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            uniform_layout,
            light_vps: [IDENTITY_MAT4; NUM_CASCADES],
            cascade_splits: [8.0, 25.0, 80.0],
            enabled: false,
            dirty: true,
            always_fresh: false,
            rendered_light_vps: None,
            rendered_light_dir: None,
            rendered_scene_version: 0,
        }
    }

    /// Force the next shadow pass to re-render the depth textures.
    /// Called on `setShadowsEnabled(true)`, swap-chain resize, or any
    /// other event that invalidates the cached cascade contents.
    pub fn invalidate(&mut self) {
        self.dirty = true;
        self.rendered_light_vps = None;
        self.rendered_light_dir = None;
    }

    /// Compute cascade view-projection matrices by splitting the camera
    /// frustum into NUM_CASCADES slices and fitting a tight ortho projection
    /// around each slice from the light's perspective.
    ///
    /// `light_dir` points from the surface toward the light (the same
    /// convention as the rest of the engine).
    pub fn compute_cascade_vps(
        &mut self,
        light_dir: [f32; 3],
        _camera_pos: [f32; 3],
        camera_view: [[f32; 4]; 4],
        camera_proj: [[f32; 4]; 4],
        near: f32,
        far: f32,
        scene_bounds: Option<([f32; 3], [f32; 3])>,
    ) {
        let len = (light_dir[0] * light_dir[0]
            + light_dir[1] * light_dir[1]
            + light_dir[2] * light_dir[2])
            .sqrt();
        let d = if len > 1e-6 {
            [light_dir[0] / len, light_dir[1] / len, light_dir[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        };

        // Compute frustum split distances using practical split scheme
        // (Nvidia GPU Gems 3, Chapter 10): blend of logarithmic and
        // uniform split for stability.
        let lambda = 0.5f32; // blend factor (0 = uniform, 1 = logarithmic)
        let ratio = far / near;
        let mut splits = [0.0f32; NUM_CASCADES + 1];
        splits[0] = near;
        for i in 1..NUM_CASCADES {
            let p = i as f32 / NUM_CASCADES as f32;
            let log_split = near * ratio.powf(p);
            let uniform_split = near + (far - near) * p;
            splits[i] = lambda * log_split + (1.0 - lambda) * uniform_split;
        }
        splits[NUM_CASCADES] = far;

        // Store view-space Z split distances for shader cascade selection.
        // cascade_splits[i] = far edge of cascade i.
        for i in 0..NUM_CASCADES {
            self.cascade_splits[i] = splits[i + 1];
        }

        // Light-space basis vectors for texel snapping
        let up_hint = if d[1].abs() > 0.99 {
            [1.0f32, 0.0, 0.0]
        } else {
            [0.0f32, 1.0, 0.0]
        };
        let right = normalize3([
            up_hint[1] * d[2] - up_hint[2] * d[1],
            up_hint[2] * d[0] - up_hint[0] * d[2],
            up_hint[0] * d[1] - up_hint[1] * d[0],
        ]);
        let ortho_up = [
            d[1] * right[2] - d[2] * right[1],
            d[2] * right[0] - d[0] * right[2],
            d[0] * right[1] - d[1] * right[0],
        ];

        for c in 0..NUM_CASCADES {
            let c_near = splits[c];
            let c_far = splits[c + 1];

            // Build a sub-projection that matches the camera's projection
            // but with the near/far planes replaced by cascade limits.
            // For a perspective projection, that means re-deriving the
            // z-mapping while keeping the x/y fields intact.
            let mut sub_proj = camera_proj;
            // camera_proj is column-major, standard wgpu perspective layout:
            //   col2 row2 = (far+near)/(near-far)
            //   col3 row2 = 2*far*near/(near-far)
            //   col2 row3 = -1
            let nf = 1.0 / (c_near - c_far);
            sub_proj[2][2] = (c_far + c_near) * nf;
            sub_proj[3][2] = 2.0 * c_far * c_near * nf;

            let sub_inv_vp = crate::renderer::mat4_invert(
                crate::renderer::mat4_multiply(sub_proj, camera_view),
            );

            // The 8 NDC corners of a clip cube. wgpu uses z in [0,1].
            let ndc_corners: [[f32; 4]; 8] = [
                [-1.0, -1.0, 0.0, 1.0],
                [ 1.0, -1.0, 0.0, 1.0],
                [-1.0,  1.0, 0.0, 1.0],
                [ 1.0,  1.0, 0.0, 1.0],
                [-1.0, -1.0, 1.0, 1.0],
                [ 1.0, -1.0, 1.0, 1.0],
                [-1.0,  1.0, 1.0, 1.0],
                [ 1.0,  1.0, 1.0, 1.0],
            ];

            // Unproject to world space
            let mut world_corners = [[0.0f32; 3]; 8];
            for i in 0..8 {
                let h = crate::renderer::mat4_mul_vec4(&sub_inv_vp, &ndc_corners[i]);
                let w = if h[3].abs() > 1e-8 { h[3] } else { 1.0 };
                world_corners[i] = [h[0] / w, h[1] / w, h[2] / w];
            }

            // Bounding sphere of this cascade's frustum slice. Sphere
            // (not AABB) gives rotation-invariant extent so the ortho
            // volume doesn't resize as the camera rotates.
            let mut center = [0.0f32; 3];
            for i in 0..8 {
                center[0] += world_corners[i][0];
                center[1] += world_corners[i][1];
                center[2] += world_corners[i][2];
            }
            center[0] /= 8.0;
            center[1] /= 8.0;
            center[2] /= 8.0;
            let mut radius: f32 = 0.0;
            for i in 0..8 {
                let dx = world_corners[i][0] - center[0];
                let dy = world_corners[i][1] - center[1];
                let dz = world_corners[i][2] - center[2];
                let r2 = dx*dx + dy*dy + dz*dz;
                if r2 > radius { radius = r2; }
            }
            radius = radius.sqrt();
            // Quantize radius so subpixel camera movement can't shift
            // the texel grid.
            let radius = (radius * 16.0).ceil() / 16.0;

            // Texel snap: quantize the ortho center to texel boundaries
            // in light space so camera translation doesn't crawl edges.
            let texel_world = (2.0 * radius) / CASCADE_MAP_SIZE as f32;
            let ls_x = dot3(center, right);
            let ls_y = dot3(center, ortho_up);
            let snapped_x = (ls_x / texel_world).floor() * texel_world;
            let snapped_y = (ls_y / texel_world).floor() * texel_world;
            let dx_snap = snapped_x - ls_x;
            let dy_snap = snapped_y - ls_y;
            let snapped_center = [
                center[0] + dx_snap * right[0] + dy_snap * ortho_up[0],
                center[1] + dx_snap * right[1] + dy_snap * ortho_up[1],
                center[2] + dx_snap * right[2] + dy_snap * ortho_up[2],
            ];

            // Extend Z-range using the scene AABB so casters behind the
            // visible slice (from the light's view) still project shadows
            // into it. This is "pancaking" — cascade XY is tight to the
            // frustum sphere, but Z reaches back to the full scene.
            let mut pancake_back: f32 = radius; // +d distance (toward light)
            let mut pancake_far:  f32 = radius; // -d distance (away from light)
            if let Some((bmin, bmax)) = scene_bounds {
                let corners = [
                    [bmin[0], bmin[1], bmin[2]],
                    [bmax[0], bmin[1], bmin[2]],
                    [bmin[0], bmax[1], bmin[2]],
                    [bmax[0], bmax[1], bmin[2]],
                    [bmin[0], bmin[1], bmax[2]],
                    [bmax[0], bmin[1], bmax[2]],
                    [bmin[0], bmax[1], bmax[2]],
                    [bmax[0], bmax[1], bmax[2]],
                ];
                for p in corners.iter() {
                    let rel = [
                        p[0] - snapped_center[0],
                        p[1] - snapped_center[1],
                        p[2] - snapped_center[2],
                    ];
                    let along_d = dot3(rel, d);
                    if along_d      > pancake_back { pancake_back = along_d; }
                    if -along_d     > pancake_far  { pancake_far  = -along_d; }
                }
            }
            // Quantize Z range so tiny scene-bounds drift doesn't shift depths.
            let pancake_back = (pancake_back * 16.0).ceil() / 16.0;
            let pancake_far  = (pancake_far  * 16.0).ceil() / 16.0;

            // Place light eye at the far-back edge of the Z range so
            // ortho near=0 exactly touches the top of the pancake volume.
            let eye_offset = pancake_back;
            let light_pos = [
                snapped_center[0] + d[0] * eye_offset,
                snapped_center[1] + d[1] * eye_offset,
                snapped_center[2] + d[2] * eye_offset,
            ];

            let snapped_view = crate::renderer::mat4_look_at(light_pos, snapped_center, up_hint);
            let light_proj = crate::renderer::mat4_ortho(
                -radius, radius,
                -radius, radius,
                0.0,
                eye_offset + pancake_far,
            );

            self.light_vps[c] = crate::renderer::mat4_multiply(light_proj, snapped_view);
        }
    }

    /// Enable shadow mapping.
    pub fn enable(&mut self) {
        if !self.enabled {
            self.invalidate();
        }
        self.enabled = true;
    }

    /// Disable shadow mapping.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
