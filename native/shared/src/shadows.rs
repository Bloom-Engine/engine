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
/// Per-cascade shadow map resolution. Back to 2048 for desktop targets:
/// the 1024 cut was made chasing 60 fps on the Sponza benchmark machine
/// (integrated GPU); discrete desktop GPUs have shadow-pass headroom, and
/// at fullscreen native resolutions 1024 maps read visibly soft on
/// near-field edges. The normal-offset receiver bias and PCF radius are
/// texel-proportional, so both adapt to the size automatically.
pub const CASCADE_MAP_SIZE: u32 = 2048;
pub const SHADOW_NEAR: f32 = 0.1;
pub const SHADOW_FAR: f32 = 100.0;
/// Dynamic-uniform buffer stride for per-node shadow uniforms. Must
/// be >= sizeof(ShadowUniforms) (144B) and a multiple of the device's
/// min_uniform_buffer_offset_alignment. 256 is safe on every platform.
pub const SHADOW_UNIFORM_STRIDE: u32 = 256;
pub const SHADOW_MAX_NODES: u32 = 1024;
/// Slots at the TAIL of each cascade's uniform region reserved for
/// dynamic casters, which re-render every frame while static casters keep their
/// cached depth. Disjoint slot ranges keep the every-frame dynamic writes from
/// clobbering the uniforms the static render was encoded against (all
/// `write_buffer`s land at submit, before any pass executes).
///
/// EN-042 — raised from 64. Sixty-four was fine while "dynamic" meant a handful of
/// characters, and became a trap the moment a *forest* could go dynamic: 88 trees x
/// 4 primitives is 352 casters, the overflow was dropped in queue order, and what
/// disappeared was whatever happened to be last — twice this session, that was the
/// player's own shadow from under their feet. 256 covers a moving crowd; the drop is
/// also RANKED now (see shadow_pass.rs) so an overflow costs a canopy shadow rather
/// than a character's.
pub const SHADOW_MAX_DYNAMIC: u32 = 256;

/// Depth-only shader for shadow pass.
pub const SHADOW_SHADER: &str = concat!(
    include_str!("../shaders/common/foliage_wind.wgsl"),
    r#"
struct ShadowUniforms {
    light_vp: mat4x4<f32>,
    model: mat4x4<f32>,
    misc: vec4<f32>,   // x = joint offset (skinned variant), z = foliage wind amount
    wind: vec4<f32>,   // xy = dir, z = amplitude, w = time
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
    // is_leaf = 0: this pipeline draws the opaque casters (trunks, branches).
    let p = foliage_wind_local(in.position, shadow_u.model, shadow_u.wind, shadow_u.misc.z, 0.0);
    let world_pos = shadow_u.model * vec4<f32>(p, 1.0);
    return shadow_u.light_vp * world_pos;
}
"#);

/// Alpha-tested shadow shader for cutout foliage (trees, grass, leaves). Same
/// depth-only output as SHADOW_SHADER but samples the caster's base-colour
/// alpha and discards below the material cutoff, so cutout cards cast their
/// real shape (dappled light) instead of an opaque billboard blob. Used by a
/// dedicated pipeline; the opaque shadow path stays untouched.
pub const SHADOW_SHADER_CUTOUT: &str = concat!(
    include_str!("../shaders/common/foliage_wind.wgsl"),
    r#"
struct ShadowUniforms {
    light_vp: mat4x4<f32>,
    model: mat4x4<f32>,
    misc: vec4<f32>,   // x = joint offset (skinned variant), z = foliage wind amount
    wind: vec4<f32>,   // xy = dir, z = amplitude, w = time
};
@group(0) @binding(0) var<uniform> shadow_u: ShadowUniforms;

struct CutoutUniforms { cutoff: vec4<f32> };   // x = alpha cutoff
@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var base_samp: sampler;
@group(1) @binding(2) var<uniform> cut: CutoutUniforms;

struct ShadowVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
};
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};


@vertex
fn vs_shadow_cutout(in: ShadowVertexInput) -> VsOut {
    var o: VsOut;
    // is_leaf = 1: this pipeline draws the cutout cards, so they get the fast
    // flutter layer -- and their shadows now flutter with them.
    let p = foliage_wind_local(in.position, shadow_u.model, shadow_u.wind, shadow_u.misc.z, 1.0);
    let world_pos = shadow_u.model * vec4<f32>(p, 1.0);
    o.pos = shadow_u.light_vp * world_pos;
    o.uv = in.uv;
    return o;
}

@fragment
fn fs_shadow_cutout(in: VsOut) {
    let a = textureSample(base_tex, base_samp, in.uv).a;
    if (a < cut.cutoff.x) { discard; }
}
"#);

/// Skinned shadow shader for animated characters (player, enemies). Their
/// vertices are *rest-pose* (cached model VBs with raw joint indices, or the
/// immediate-mode batch with pre-offset ones), with world placement living
/// entirely in the per-frame joint matrices. The plain `vs_shadow` doesn't
/// skin, so it would render those characters as a rest pose at the world
/// origin — i.e. no shadow under their feet. This variant skins per-vertex
/// (same math as the main scene shader) so characters cast a real, posed
/// shadow. The branch on total weight means the non-skinned verts sharing a
/// batch (ground cube, rigid accessories) still transform by the model
/// matrix as before.
pub const SHADOW_SHADER_SKINNED: &str = "
struct ShadowUniforms {
    light_vp: mat4x4<f32>,
    model: mat4x4<f32>,
    misc: vec4<f32>,   // x = joint offset for cached skinned casters
};
@group(0) @binding(0) var<uniform> shadow_u: ShadowUniforms;

struct JointMatrices { matrices: array<mat4x4<f32>, 1024> };
@group(1) @binding(0) var<uniform> joints: JointMatrices;

struct ShadowVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<f32>,
    @location(5) weights: vec4<f32>,
};

@vertex
fn vs_shadow_skinned(in: ShadowVertexInput) -> @builtin(position) vec4<f32> {
    let total_weight = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
    var pos = vec4<f32>(in.position, 1.0);
    if (total_weight > 0.01) {
        // misc.x = joint-buffer base offset for cached skinned casters
        // (raw VB indices); 0 for the immediate batch (pre-offset).
        let j0 = u32(in.joints.x + shadow_u.misc.x); let j1 = u32(in.joints.y + shadow_u.misc.x);
        let j2 = u32(in.joints.z + shadow_u.misc.x); let j3 = u32(in.joints.w + shadow_u.misc.x);
        // Joint matrices already bake scale + world position + rotation, so the
        // skinned result is world-space; the model matrix is identity for the
        // immediate-mode batch and is intentionally not re-applied here.
        pos = joints.matrices[j0] * pos * in.weights.x
            + joints.matrices[j1] * pos * in.weights.y
            + joints.matrices[j2] * pos * in.weights.z
            + joints.matrices[j3] * pos * in.weights.w;
        return shadow_u.light_vp * pos;
    }
    let world_pos = shadow_u.model * pos;
    return shadow_u.light_vp * world_pos;
}
";

/// Uniform data for the shadow pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniforms {
    pub light_vp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    /// x = joint-buffer base offset for skinned CACHED casters (their
    /// VBs keep raw joint indices; `vs_shadow_skinned` adds this before
    /// indexing). 0 for everything else, including the immediate batch
    /// whose vertex joints are pre-offset CPU-side.
    /// z = foliage wind amount for this caster (0 = rigid). A tree that bends in
    /// the scene pass but not in the shadow pass detaches from its own shadow.
    /// yw unused.
    pub misc: [f32; 4],
    /// Global wind: xy = direction in XZ, z = amplitude, w = elapsed seconds.
    /// The shadow pass needs it for the same reason the scene pass does.
    pub wind: [f32; 4],
}

/// Shadow map resources for cascaded shadow mapping.
pub struct ShadowMap {
    pub depth_textures: [wgpu::Texture; NUM_CASCADES],
    pub depth_views: [wgpu::TextureView; NUM_CASCADES],
    /// Cached static-caster depth per cascade ("cached whole-scene
    /// shadows"): scene nodes / cached models / material draws render in
    /// here only when the cascade's VP or static content changes. Every
    /// frame the live cascade texture starts as a copy of this and only
    /// dynamic casters (the immediate batch: animated characters,
    /// moving primitives) are drawn on top — so one animated model no
    /// longer forces the whole forest to re-render three times.
    pub static_depth_textures: [wgpu::Texture; NUM_CASCADES],
    pub static_depth_views: [wgpu::TextureView; NUM_CASCADES],
    pub sampler: wgpu::Sampler,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    /// Alpha-tested variant for cutout foliage casters. Opaque casters keep
    /// using `pipeline` (byte-identical to before this was added).
    pub pipeline_cutout: wgpu::RenderPipeline,
    /// Skinning-aware variant for the immediate-mode batch (animated
    /// characters). Binds the joint-matrix buffer at group 1 and skins
    /// per-vertex so player/enemies cast a real posed shadow.
    pub pipeline_skinned: wgpu::RenderPipeline,
    /// Group-1 layout for `pipeline_cutout`: base-colour tex + sampler + a
    /// cutoff uniform. Per-mesh bind groups are built against this in
    /// `cache_model_if_static`.
    pub cutout_tex_layout: wgpu::BindGroupLayout,
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
    /// Shadow-flicker fix — per-cascade accepted pancake extents
    /// (back, far). Animated caster bounds drift a few centimetres per
    /// frame, and the raw 1/16 m ceil() quantization made the fitted VP
    /// toggle between two matrices whenever that drift straddled a
    /// step. The accepted extent grows immediately (casters must stay
    /// inside the volume) but shrinks only when the raw need has
    /// dropped well below it — so idle animations stop re-fitting the
    /// cascades every few frames.
    pancake_hysteresis: [[f32; 2]; NUM_CASCADES],
    /// Per-cascade STATIC caster-content signature at the last render
    /// of that cascade's static depth texture (hash of every
    /// non-immediate caster that passed its frustum filter: identity +
    /// transform). 0 = never rendered. Together with a per-cascade VP
    /// compare this decides when the static cache must re-render — the
    /// whole-pass cache above only ever hit for fully-retained scenes.
    pub rendered_cascade_sig: [u64; NUM_CASCADES],
    /// Whether the live cascade texture currently contains dynamic
    /// casters. A cascade whose dynamics all left still needs one
    /// refresh copy to clear their stale shadows.
    pub had_dynamic: [bool; NUM_CASCADES],
    /// Monotonic counter folded into animated casters' signatures so
    /// any cascade containing one re-renders every frame.
    pub frame_nonce: u64,
    /// Cascade re-fit slack — the accepted ortho fit per cascade
    /// (cascades ≥ 1). While the freshly-required bounding sphere and
    /// pancake extents still fit inside the accepted volume, the
    /// cascade keeps its previous VP byte-for-byte, so camera travel
    /// of a few metres no longer invalidates the far cascades' cached
    /// depth. Accepted fits are inflated by `REFIT_SLACK` (the
    /// resolution cost of that inflation is bounded and uniform; the
    /// near cascade is exempt and stays exact-fit).
    accepted_fit: [Option<AcceptedFit>; NUM_CASCADES],
    accepted_light_dir: Option<[f32; 3]>,
}

/// Accepted (slack-inflated) ortho fit for one cascade. `ls_x`/`ls_y`
/// are the snapped center in the light-plane basis; `radius` is the
/// final ortho half-extent; `back`/`far` the accepted pancake extents
/// along the light axis relative to `center`.
#[derive(Copy, Clone)]
struct AcceptedFit {
    ls_x: f32,
    ls_y: f32,
    center: [f32; 3],
    radius: f32,
    back: f32,
    far: f32,
}

/// Re-fit slack factor for cascades ≥ 1. 15% larger ortho extent buys
/// ~0.15 × radius of camera travel between re-fits (≈ 4-5 m on a far
/// cascade) at the cost of ~15% coarser texels on the mid/far cascades
/// only. PCF radius and the normal-offset receiver bias are
/// texel-proportional, so the softening stays coherent (no acne).
const REFIT_SLACK: f32 = 1.15;

impl ShadowMap {
    pub fn new(
        device: &wgpu::Device,
        vertex_layout: wgpu::VertexBufferLayout<'static>,
        joint_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create NUM_CASCADES depth textures (live + static cache).
        let mut depth_textures_vec: Vec<wgpu::Texture> = Vec::new();
        let mut depth_views_vec: Vec<wgpu::TextureView> = Vec::new();
        let mut static_textures_vec: Vec<wgpu::Texture> = Vec::new();
        let mut static_views_vec: Vec<wgpu::TextureView> = Vec::new();
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
                    | wgpu::TextureUsages::COPY_SRC
                    // Refreshed from the static cache before dynamic
                    // casters draw on top.
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            depth_textures_vec.push(tex);
            depth_views_vec.push(view);
            let stex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("shadow_static_depth_cascade_{}", i)),
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
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let sview = stex.create_view(&wgpu::TextureViewDescriptor::default());
            static_textures_vec.push(stex);
            static_views_vec.push(sview);
        }

        // Convert Vecs to fixed-size arrays
        let depth_textures: [wgpu::Texture; NUM_CASCADES] =
            depth_textures_vec.try_into().unwrap_or_else(|_| panic!("cascade texture count mismatch"));
        let depth_views: [wgpu::TextureView; NUM_CASCADES] =
            depth_views_vec.try_into().unwrap_or_else(|_| panic!("cascade view count mismatch"));
        let static_depth_textures: [wgpu::Texture; NUM_CASCADES] =
            static_textures_vec.try_into().unwrap_or_else(|_| panic!("static cascade texture count mismatch"));
        let static_depth_views: [wgpu::TextureView; NUM_CASCADES] =
            static_views_vec.try_into().unwrap_or_else(|_| panic!("static cascade view count mismatch"));

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

        // One region PER CASCADE. The pass encodes all three cascades before
        // the queue submits, and `queue.write_buffer` executes at submit —
        // sharing one region meant every cascade rendered with the LAST
        // cascade's light_vp + models, killing shadows for everything that
        // sampled cascades 0/1 (i.e. all near-camera receivers: the player
        // and enemies never had shadows; distant trees kept theirs because
        // cascade 2's write happened to be the surviving one).
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_uniform_buf"),
            size: (SHADOW_UNIFORM_STRIDE * SHADOW_MAX_NODES * NUM_CASCADES as u32) as u64,
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
                buffers: &[vertex_layout.clone()],
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

        // Cutout (alpha-tested) shadow pipeline. Separate so the opaque path
        // above is untouched. Adds a group-1 texture/sampler/cutoff layout and
        // a fragment stage that discards below the alpha cutoff. Still depth-
        // only (no colour targets).
        let cutout_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader_cutout"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER_CUTOUT.into()),
        });
        let cutout_tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_cutout_tex_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let cutout_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_cutout_pipeline_layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&cutout_tex_layout)],
            immediate_size: 0,
        });
        let pipeline_cutout = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline_cutout"),
            layout: Some(&cutout_pl_layout),
            vertex: wgpu::VertexState {
                module: &cutout_shader,
                entry_point: Some("vs_shadow_cutout"),
                buffers: &[vertex_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cutout_shader,
                entry_point: Some("fs_shadow_cutout"),
                targets: &[],   // depth only
                compilation_options: Default::default(),
            }),
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
                bias: wgpu::DepthBiasState { constant: 1, slope_scale: 1.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Skinned (animated-character) shadow pipeline. Group 0 = the shared
        // shadow uniforms (light_vp + model, dynamic offset); group 1 = the
        // joint-matrix buffer. Depth-only, same bias as the opaque path.
        let skinned_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader_skinned"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER_SKINNED.into()),
        });
        let skinned_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_skinned_pipeline_layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(joint_layout)],
            immediate_size: 0,
        });
        let pipeline_skinned = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline_skinned"),
            layout: Some(&skinned_pl_layout),
            vertex: wgpu::VertexState {
                module: &skinned_shader,
                entry_point: Some("vs_shadow_skinned"),
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
                bias: wgpu::DepthBiasState { constant: 1, slope_scale: 1.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            depth_textures,
            depth_views,
            static_depth_textures,
            static_depth_views,
            sampler,
            bind_group_layout,
            bind_group,
            pipeline,
            pipeline_cutout,
            pipeline_skinned,
            cutout_tex_layout,
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
            pancake_hysteresis: [[0.0; 2]; NUM_CASCADES],
            rendered_cascade_sig: [0; NUM_CASCADES],
            had_dynamic: [false; NUM_CASCADES],
            frame_nonce: 0,
            accepted_fit: [None; NUM_CASCADES],
            accepted_light_dir: None,
        }
    }

    /// Force the next shadow pass to re-render the depth textures.
    /// Called on `setShadowsEnabled(true)`, swap-chain resize, or any
    /// other event that invalidates the cached cascade contents.
    pub fn invalidate(&mut self) {
        self.dirty = true;
        self.rendered_light_vps = None;
        self.rendered_light_dir = None;
        self.rendered_cascade_sig = [0; NUM_CASCADES];
        self.had_dynamic = [false; NUM_CASCADES];
        self.accepted_fit = [None; NUM_CASCADES];
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

        // A light-direction change invalidates every accepted fit (the
        // light-plane basis itself moves).
        if self.accepted_light_dir != Some(d) {
            self.accepted_fit = [None; NUM_CASCADES];
            self.accepted_light_dir = Some(d);
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

            // Frustum-slice corners for [c_near, c_far], computed DIRECTLY in
            // view space from the camera FOV, then transformed to world by the
            // (affine, well-conditioned) view inverse.
            //
            // We deliberately do NOT invert the perspective projection here.
            // `mat4_perspective` uses the OpenGL [-1,1] NDC-z convention, so
            // unprojecting its NDC clip corners handed the near plane a NEGATIVE
            // homogeneous w — every corner then divided down onto the z=-1 plane
            // and the frustum bounding sphere collapsed to ~0.12 m. The cascade
            // ortho covered almost nothing, so terrain a few metres from the
            // camera projected outside the shadow frustum and self-shadowed:
            // the reported "moving dark patch" that scaled disproportionately
            // with the camera. Half-extents per unit view depth come straight
            // from the projection (no inversion, no convention hazard):
            //   tan(fovy/2)          = 1 / proj[1][1]
            //   tan(fovy/2) * aspect = 1 / proj[0][0]
            let inv_view = crate::renderer::mat4_invert(camera_view);
            let half_w_per_d = 1.0 / camera_proj[0][0];
            let half_h_per_d = 1.0 / camera_proj[1][1];
            let mut world_corners = [[0.0f32; 3]; 8];
            let mut ci = 0usize;
            for &d in [c_near, c_far].iter() {
                let hw = d * half_w_per_d;
                let hh = d * half_h_per_d;
                for &(sx, sy) in [(-1.0f32, -1.0f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].iter() {
                    // View space looks down -Z, so this slice sits at z = -d.
                    let vp = [sx * hw, sy * hh, -d, 1.0];
                    let wc = crate::renderer::mat4_mul_vec4(&inv_view, &vp);
                    world_corners[ci] = [wc[0], wc[1], wc[2]];
                    ci += 1;
                }
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

            // Re-fit slack (cascades ≥ 1): if the required sphere and
            // pancake extents still fit inside the previously accepted
            // (slack-inflated) ortho volume, keep the previous VP
            // byte-identical. The per-cascade shadow cache compares VPs
            // exactly, so a kept VP means the cascade's cached depth
            // stays valid while the camera travels within the slack.
            // EN-045 — cascade 0 gets the slack too.
            //
            // It was excluded, and that quietly made the whole static-shadow cache a
            // title-screen feature. Cascade 0 is the NEAR cascade: it holds the
            // player and everything they are standing next to. Re-fitting it every
            // frame means its VP changes every frame the camera moves — which is all
            // of gameplay — so its cached depth was thrown away and every static
            // caster in it re-rendered, every frame. Measured: shadow_pass 0.12 ms on
            // the stationary title screen, 3.2 ms in a moving fight.
            //
            // The slack costs ~15% of near-field shadow resolution and buys a cache
            // that survives ~15 frames of walking instead of zero.
            {
                if let Some(acc) = self.accepted_fit[c] {
                    let ls_x = dot3(center, right);
                    let ls_y = dot3(center, ortho_up);
                    let fits_xy = (ls_x - acc.ls_x).abs() + radius <= acc.radius
                        && (ls_y - acc.ls_y).abs() + radius <= acc.radius;
                    // Required extents along the light axis, relative to
                    // the ACCEPTED center: the slice sphere plus the
                    // scene AABB corners (same needs the pancake fit
                    // below covers, measured against the old volume).
                    let rel = [
                        center[0] - acc.center[0],
                        center[1] - acc.center[1],
                        center[2] - acc.center[2],
                    ];
                    let along = dot3(rel, d);
                    let mut req_back = along + radius;
                    let mut req_far = radius - along;
                    if let Some((bmin, bmax)) = scene_bounds {
                        for i in 0..8 {
                            let p = [
                                if i & 1 == 0 { bmin[0] } else { bmax[0] },
                                if i & 2 == 0 { bmin[1] } else { bmax[1] },
                                if i & 4 == 0 { bmin[2] } else { bmax[2] },
                            ];
                            let a = dot3(
                                [
                                    p[0] - acc.center[0],
                                    p[1] - acc.center[1],
                                    p[2] - acc.center[2],
                                ],
                                d,
                            );
                            if a > req_back { req_back = a; }
                            if -a > req_far { req_far = -a; }
                        }
                    }
                    if fits_xy && req_back <= acc.back && req_far <= acc.far {
                        // Keep light_vps[c] from the accepted fit.
                        continue;
                    }
                }
            }
            let radius = radius * REFIT_SLACK;
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
            // Quantize Z range so scene-bounds drift doesn't shift depths.
            // Flicker fix: animated casters (idle anims, wind-swayed
            // proxies) drift the raw pancake need by centimetres-to-
            // decimetres every cycle, and every resulting VP change
            // re-rolls the acne pattern on grazing receivers — visible
            // as periodic banding bursts. Quantize UP to whole 2 m steps
            // and only shrink after a full 2-step (4 m) drop, so the
            // fitted VP is byte-stable against anything short of a
            // structural scene change. The cost is a few metres of extra
            // ortho depth range on a Depth32Float target — irrelevant —
            // and coverage stays correct: the accepted extent is never
            // below the raw need.
            const PANCAKE_STEP: f32 = 2.0;
            let quantize = |v: f32| (v / PANCAKE_STEP).ceil() * PANCAKE_STEP;
            let prev = self.pancake_hysteresis[c];
            let pancake_back = if pancake_back > prev[0]
                || pancake_back < prev[0] - 2.0 * PANCAKE_STEP
            {
                quantize(pancake_back)
            } else {
                prev[0]
            };
            let pancake_far = if pancake_far > prev[1]
                || pancake_far < prev[1] - 2.0 * PANCAKE_STEP
            {
                quantize(pancake_far)
            } else {
                prev[1]
            };
            self.pancake_hysteresis[c] = [pancake_back, pancake_far];

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

            // Record the accepted fit so subsequent frames can keep this VP while
            // their requirements stay inside it. EN-045 — cascade 0 included now;
            // excluding it was what made the static-shadow cache a title-screen
            // feature, because cascade 0's VP changed on every frame the camera moved.
            {
                self.accepted_fit[c] = Some(AcceptedFit {
                    ls_x: dot3(snapped_center, right),
                    ls_y: dot3(snapped_center, ortho_up),
                    center: snapped_center,
                    radius,
                    back: pancake_back,
                    far: pancake_far,
                });
            }
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
