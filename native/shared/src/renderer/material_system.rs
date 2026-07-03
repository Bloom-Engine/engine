// Material system — the runtime state behind Phase 1c.
//
// Owns the compiled `MaterialPipeline`s, the per-frame / per-view /
// per-material / per-draw uniform buffers, the bind groups wiring
// them to the ABI layouts, and the per-frame draw command list.
//
// The Renderer owns one `MaterialSystem` instance. Games interact via
// three methods: `compile_material`, `submit_draw`, and (internally)
// `dispatch` which runs inside the main HDR pass.

use wgpu::util::DeviceExt;

use super::material_pipeline::{
    MaterialAbiLayouts, MaterialPipeline, MaterialCompileDesc, FragmentProfile,
    Bucket, compile_material, MaterialCompileError,
};
use super::types::Vertex3D;

// =====================================================================
// Uniform structs — repr(C), bytemuck-Pod, mirror the WGSL in
// material_abi.wgsl. Kept local to this module so changes to ABI
// struct layouts happen in exactly one place per language.
// =====================================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerFrameUniforms {
    pub time:              f32,
    pub delta_time:        f32,
    pub frame_index:       u32,
    pub _pad0:             u32,
    pub screen_resolution: [f32; 2],
    pub render_resolution: [f32; 2],
    pub taa_jitter:        [f32; 2],
    pub _pad1:             [f32; 2],
    /// Global wind: x=dir_x, y=dir_z, z=amplitude, w=frequency.
    pub wind:              [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerViewDirLight {
    pub direction: [f32; 4],  // xyz + intensity
    pub color:     [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerViewPointLight {
    pub position: [f32; 4],
    pub color:    [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerViewUniforms {
    pub view:           [[f32; 4]; 4],
    pub proj:           [[f32; 4]; 4],
    pub view_proj:      [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub inv_proj:       [[f32; 4]; 4],
    pub camera_pos:     [f32; 4],
    pub camera_dir:     [f32; 4],
    pub ambient:        [f32; 4],
    pub fog:            [f32; 4],
    pub sun_dir:        [f32; 4],
    pub sun_color:      [f32; 4],
    pub dir_light_count:   [f32; 4],
    pub dir_lights:        [PerViewDirLight; 8],
    pub point_light_count: [f32; 4],
    pub point_lights:      [PerViewPointLight; 256],
    pub shadow_splits:   [f32; 4],
    pub shadow_view:     [[f32; 4]; 4],
    pub shadow_cascades: [[[f32; 4]; 4]; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialFactorsUniforms {
    pub metal_rough:    [f32; 4],
    pub emissive:       [f32; 4],
    pub base_color:     [f32; 4],
    /// EN-012 — shading-model selector + foliage transmission tint.
    /// x = shading_model enum (0 = default lit, 1 = foliage,
    ///                         2 = subsurface — V2 stub),
    /// yzw = transmission_color (rgb tint for back-lit foliage; ignored
    ///       when shading_model == 0).
    pub shading_model:  [f32; 4],
    /// EN-012 — foliage shading parameters. Only consumed when
    /// `shading_model.x == 1.0`.
    /// x = transmission_amount (0..1),
    /// y = wrap_factor          (0..1),
    /// zw = reserved.
    pub foliage_params: [f32; 4],
}

impl Default for MaterialFactorsUniforms {
    fn default() -> Self {
        Self {
            metal_rough:    [0.0, 1.0, 0.0, 0.0],   // non-metal, rough, no MR tex, no cutoff
            emissive:       [0.0, 0.0, 0.0, 0.0],
            base_color:     [1.0, 1.0, 1.0, 1.0],
            // EN-012 — default lit, white transmission tint. Materials
            // that never call `setMaterialShadingModel` get standard PBR
            // (shading_model.x == 0.0).
            shading_model:  [0.0, 1.0, 1.0, 1.0],
            // EN-012 — moderate defaults so a freshly-flagged foliage
            // material (shading_model = 1) looks reasonable before any
            // tuning. Wrap=0.5 + transmission=0.5 gives soft back-face
            // shading + a noticeable halo against the sun.
            foliage_params: [0.5, 0.5, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerDrawUniforms {
    pub mvp:        [[f32; 4]; 4],
    pub model:      [[f32; 4]; 4],
    pub prev_mvp:   [[f32; 4]; 4],
    pub model_tint: [f32; 4],
    pub skin_info:  [u32; 4],
}

// =====================================================================
// Handles + draw commands
// =====================================================================

pub type MaterialHandle = u32;

pub struct MaterialDrawCommand {
    pub material:    MaterialHandle,
    pub mesh_handle: u64,       // matches model_gpu_cache keys
    pub mesh_idx:    usize,     // sub-mesh index within that cached model
    pub draw_slot:   usize,     // which slot in per_draw_buffers to bind
    /// Clip-space w of the object pivot at submit time (= view-space
    /// depth for a standard projection). Drives the back-to-front sort
    /// of the translucent bucket; opaque draws ignore it.
    pub view_depth:  f32,
    /// EN-001 — when set, the engine binds vertex slot 1 to this
    /// instance buffer and emits draw_indexed with `0..count` instances.
    /// `None` means a single-instance draw (the legacy path).
    pub instance:    Option<InstanceDrawInfo>,
    /// CPU-side copy of the model matrix (the GPU one lives in the
    /// per-draw UBO slot). The shadow pass needs it to re-render these
    /// draws into the sun cascades without reading buffers back.
    pub model:       [[f32; 4]; 4],
}

/// Reference to an instance buffer for an instanced draw command.
/// `buffer_handle` is 1-based into `MaterialSystem::instance_buffers`;
/// `count` is the number of GPU instances to emit. Created via
/// `MaterialSystem::create_instance_buffer`, consumed by
/// `submit_draw_instanced`.
#[derive(Copy, Clone, Debug)]
pub struct InstanceDrawInfo {
    pub buffer_handle: u32,
    pub count:         u32,
}

/// EN-001 — owned wgpu::Buffer + element count for an instance buffer.
/// Lives in the `MaterialSystem::instance_buffers` registry, indexed
/// by 1-based handle (0 = invalid).
pub struct InstanceBuffer {
    pub buffer: wgpu::Buffer,
    pub count:  u32,
}

/// EN-014 — owned wgpu::Texture + view + layer-count for a texture
/// array. Lives in the `MaterialSystem::texture_arrays` registry,
/// indexed by 1-based handle (0 = invalid). The view is a D2Array
/// view over all `layer_count` layers (so `textureSample(arr, samp,
/// uv, layer_idx)` resolves layers 0..layer_count-1).
pub struct TextureArray {
    pub texture:     wgpu::Texture,
    pub view:        wgpu::TextureView,
    pub layer_count: u32,
}

/// EN-014 — V1 cap on layers per texture array. The ticket calls this
/// "enough for any landscape shader" (UE5's typical landscape uses
/// 4–8 layers; 16 leaves headroom for foliage/debris splats).
pub const MAX_TEXTURE_ARRAY_LAYERS: u32 = 16;

/// EN-014 V2 — texture-array format codes, as exposed to the TS API
/// via `TEX_ARRAY_FORMAT_*`. Anything unrecognised falls back to sRGB
/// since albedo is the most common splat layer.
pub const TEX_ARRAY_FORMAT_SRGB:   u32 = 0;
pub const TEX_ARRAY_FORMAT_LINEAR: u32 = 1;

/// Map a TS-side format code to a wgpu format. sRGB suits albedo; linear
/// is mandatory for normal/MR data textures (sRGB decoding would corrupt
/// the encoded normals or rough/metal channels).
pub fn map_texture_array_format(code: u32) -> wgpu::TextureFormat {
    match code {
        TEX_ARRAY_FORMAT_SRGB   => wgpu::TextureFormat::Rgba8UnormSrgb,
        TEX_ARRAY_FORMAT_LINEAR => wgpu::TextureFormat::Rgba8Unorm,
        _                       => wgpu::TextureFormat::Rgba8UnormSrgb,
    }
}

// =====================================================================
// The system
// =====================================================================

pub struct MaterialSystem {
    pub layouts: MaterialAbiLayouts,

    // Compiled pipelines, indexed by MaterialHandle (1-based; 0 = invalid).
    pub pipelines: Vec<Option<MaterialPipeline>>,

    // Per-frame UBO + bind group (rewritten at the start of every frame).
    pub per_frame_buffer: wgpu::Buffer,
    pub per_frame_bg:     wgpu::BindGroup,

    // Per-view UBO — one for now (single camera). Phase 2 may add more
    // for split-screen / shadow cascades.
    pub per_view_buffer: wgpu::Buffer,
    pub per_view_bg:     wgpu::BindGroup,

    // Default per-material bind group: all white 1×1 textures, default
    // factors, zero-initialised user-params. Materials that don't
    // provide their own share this.
    pub default_per_material_bg: wgpu::BindGroup,
    /// Kept alive so the BG it backs doesn't dangle.
    _default_material_factors_buffer: wgpu::Buffer,
    _default_user_params_buffer:      wgpu::Buffer,
    _default_white_tex:               wgpu::Texture,
    _default_white_view:              wgpu::TextureView,
    _default_sampler:                 wgpu::Sampler,
    /// EN-011 — 1×1 black texture bound at @group(2) @binding(12) for
    /// materials that don't have a planar reflection probe linked.
    /// Lets shaders unconditionally `textureSample(planar_reflection_tex,
    /// …)` without branching on probe presence.
    _default_black_tex:               wgpu::Texture,
    pub default_black_view:           wgpu::TextureView,
    /// EN-014 — 1×1×1 transparent-black texture-array stub bound to
    /// bindings 14/15/16 (@group(2)) for materials that don't declare
    /// their own array. Has to be a real D2Array view (not a 2D
    /// texture cast) so the layout's `view_dimension: D2Array`
    /// matches at bind time.
    _default_array_tex:               wgpu::Texture,
    pub default_array_view:           wgpu::TextureView,

    /// Phase 5 — per-material `user_params` UBOs. Indexed by
    /// `MaterialHandle - 1`; `None` means the material uses the default
    /// (zero-initialised) bind group. Created lazily on first
    /// `set_user_params` call.
    pub material_params_buffers: Vec<Option<wgpu::Buffer>>,
    pub material_per_material_bgs: Vec<Option<wgpu::BindGroup>>,

    /// EN-011 — link from material → planar reflection probe handle
    /// (1-based; `None` → default 1×1 black texture at binding 12).
    /// Indexed by `MaterialHandle - 1`. Set via
    /// `Renderer::set_material_reflection_probe`; the per-material
    /// bind group is rebuilt at that call so subsequent draws see
    /// the probe's texture even though the engine repaints the
    /// texture each frame (the wgpu Texture identity stays stable).
    pub material_reflection_probe: Vec<Option<u32>>,

    // Per-draw UBO pool. Buffers grow 1-by-1 as draws pile up.
    // Each entry is `(PerDraw UBO, bind group binding it + the global
    // joint buffer at binding 1)`.
    pub per_draw_buffers: Vec<wgpu::Buffer>,
    pub per_draw_bgs:     Vec<wgpu::BindGroup>,

    // Phase 4b — group 4 (SceneInputs) bind group. Rebuilt per-frame
    // when any Refractive material is submitted and a scene-colour
    // snapshot is available. `None` means "no material needs this
    // group this frame" — translucent dispatch skips group 4
    // binding entirely.
    pub scene_inputs_bg: Option<wgpu::BindGroup>,
    /// Linear sampler for scene-colour sampling (group 4 binding 1).
    _scene_color_sampler: wgpu::Sampler,
    /// Non-filtering sampler for scene-depth sampling (binding 3).
    _scene_depth_sampler: wgpu::Sampler,
    /// 1×1 default texture for impulse / motion-vectors slots when
    /// no Phase 7 impulse system is wired up yet.
    _scene_stub_tex:      wgpu::Texture,
    _scene_stub_view:     wgpu::TextureView,
    /// 1×1 stub depth texture — bound to scene_depth_tex in Phase 4b
    /// because the live depth buffer can't be simultaneously sampled
    /// and used as a depth-stencil attachment. Phase 4c will add a
    /// copy-to-sample depth snapshot for shoreline-fade materials.
    _scene_stub_depth:    wgpu::Texture,
    _scene_stub_depth_view: wgpu::TextureView,

    // Frame state — commands split by bucket so the graph can
    // schedule them into the right pass. Phase 4a keeps them in
    // parallel lists; Phase 4b dispatches the translucent lists in
    // their own sub-pass.
    pub commands:              Vec<MaterialDrawCommand>,  // Bucket::Opaque + Bucket::Cutout
    pub translucent_commands:  Vec<MaterialDrawCommand>,  // Transparent + Refractive + Additive
    next_draw_slot: usize,

    /// EN-001 — instance buffers, indexed by InstanceBufferHandle
    /// (1-based; 0 = invalid). Each entry owns a wgpu Buffer + element
    /// count. Created via `create_instance_buffer`, consumed by
    /// `submit_draw_instanced` commands. Slots remain `None` after
    /// `destroy_instance_buffer` so existing handles never collide
    /// with re-issued ones.
    pub instance_buffers: Vec<Option<InstanceBuffer>>,

    /// EN-014 — texture arrays, indexed by 1-based handle (0 = invalid).
    /// Each entry owns a wgpu D2Array texture + view + layer-count.
    /// Created via `create_texture_array`, linked to a material's
    /// per-material BG via `set_material_texture_array`.
    pub texture_arrays: Vec<Option<TextureArray>>,

    /// EN-014 — per-material → array-handle link, one slot for each
    /// of the 3 array bindings (0 = albedo, 1 = normal, 2 = MR).
    /// `None` means "use the default 1×1×1 stub array". Indexed by
    /// `MaterialHandle - 1`, parallel to `material_per_material_bgs`
    /// / `material_reflection_probe`.
    pub material_texture_arrays: Vec<[Option<u32>; 3]>,

    /// EN-012 — per-material `MaterialFactorsUniforms` UBO. `None` means
    /// the material uses the shared `_default_material_factors_buffer`
    /// (uniform default) — lazily allocated on the first
    /// `set_material_shading_model` / `set_material_foliage` call.
    /// Indexed by `MaterialHandle - 1`. The CPU-side mirror in
    /// `material_factors_data` lets us partial-update one field at a
    /// time without losing the others.
    pub material_factors_buffers: Vec<Option<wgpu::Buffer>>,
    pub material_factors_data:    Vec<MaterialFactorsUniforms>,
}

impl MaterialSystem {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        // Joint-buffer plumbing is per-draw (used by ensure_draw_slot,
        // not at construction time). Kept on the constructor's
        // signature for symmetry with the per-draw path.
        _joint_buffer: &wgpu::Buffer,
    ) -> Self {
        let layouts = MaterialAbiLayouts::create(device);

        // Per-frame UBO.
        let per_frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_per_frame"),
            size: std::mem::size_of::<PerFrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let per_frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_per_frame_bg"),
            layout: &layouts.per_frame,
            entries: &[wgpu::BindGroupEntry {
                binding: 0, resource: per_frame_buffer.as_entire_binding(),
            }],
        });

        // Default white 1×1 texture + sampler shared across optional PBR bindings.
        let default_white_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("material_default_white"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            Default::default(),
            &[255, 255, 255, 255],
        );
        let default_white_view = default_white_tex.create_view(&Default::default());

        // EN-011 — Default 1×1 black HDR texture for the planar
        // reflection slot (binding 12). Format is Rgba16Float to match
        // the actual probe RT so the bind-group layout is identical
        // whether a probe is bound or the default.
        let black_pixel: [u16; 4] = [0, 0, 0, 0];
        let default_black_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("material_default_black_reflection"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: super::formats::HDR_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            Default::default(),
            bytemuck::cast_slice(&black_pixel),
        );
        let default_black_view = default_black_tex.create_view(&Default::default());

        // EN-014 — Default 1×1×1 transparent-black texture-array stub
        // for bindings 14/15/16. Has to be a real D2Array (depth_or_
        // _array_layers = 1, viewed as D2Array) so the bind-group
        // layout's view_dimension matches at bind time. Format is
        // Rgba8Unorm — same as a typical albedo source so a game
        // dropping in albedo data without thinking gets the expected
        // sRGB-vs-linear behaviour. Materials that need linear
        // (normal/MR) typically encode that into the layer they upload.
        let default_array_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("material_default_array_stub"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            Default::default(),
            &[0, 0, 0, 0],
        );
        let default_array_view = default_array_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("material_default_array_stub_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material_default_samp"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Per-view UBO — zero init; write the real data each frame in `begin_frame`.
        let per_view_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_per_view"),
            size: std::mem::size_of::<PerViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // We don't have the env / BRDF LUT / shadow textures plumbed through
        // the material-system layout yet — Phase 2 wires them from the
        // existing renderer state. For now the PerView bind group uses the
        // default white texture + sampler for the env / BRDF / shadow slots
        // as placeholders so draws validate.
        let white_samp = &default_sampler;
        let white_view = &default_white_view;
        let cmp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material_default_cmp_samp"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        // PerView layout wants a depth texture for shadow bindings 6-8.
        // Use a 1×1 depth as a stub.
        let stub_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material_stub_depth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let stub_depth_view = stub_depth.create_view(&Default::default());
        let per_view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_per_view_bg"),
            layout: &layouts.per_view,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: per_view_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(white_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(white_samp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(white_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(white_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(white_samp) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&stub_depth_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&stub_depth_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&stub_depth_view) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&cmp_sampler) },
            ],
        });
        // The stub_depth texture and cmp_sampler outlive the bind group via
        // wgpu internal Arc; we don't need to hold them in the struct.
        std::mem::forget(stub_depth);
        std::mem::forget(stub_depth_view);
        std::mem::forget(cmp_sampler);

        // Default MaterialFactors UBO.
        let default_mf = MaterialFactorsUniforms::default();
        let default_material_factors_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("material_default_factors"),
                contents: bytemuck::bytes_of(&default_mf),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        // Default user-params UBO — 256 bytes of zeros (ABI §1.4).
        let default_user_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_default_user_params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Need a SECOND sampler here because wgpu requires distinct sampler
        // handles when binding the same logical sampler to multiple slots
        // within one BG — actually it doesn't, the same handle works fine.
        // We reuse default_sampler for every material-tex sampler slot.
        let default_per_material_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_default_per_material_bg"),
            layout: &layouts.per_material,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 1,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 2,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 3,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 4,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 5,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 6,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 7,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 8,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 9,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 10, resource: default_material_factors_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: default_user_params_buffer.as_entire_binding() },
                // EN-011 — default black 1×1 reflection texture +
                // shared linear sampler. Replaced per-material when a
                // game calls `set_material_reflection_probe`.
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&default_black_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(&default_sampler) },
                // EN-014 — default 1×1×1 stub array bound to all 3
                // texture-array slots, with the shared default
                // sampler at binding 17. Replaced per-slot when a
                // game calls `set_material_texture_array`.
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&default_array_view) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&default_array_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&default_array_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&default_sampler) },
            ],
        });

        // Phase 4b — scene-inputs scratch resources. Samplers are
        // stable across frames; the bind group itself is rebuilt
        // when a scene-colour snapshot becomes available.
        let scene_color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene_color_samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let scene_depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene_depth_samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        // 1×1 black texture as the stub impulse / motion-vector slot
        // until Phase 7 wires real sources.
        let scene_stub_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("scene_inputs_stub"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            Default::default(),
            &[0, 0, 0, 0],
        );
        let scene_stub_view = scene_stub_tex.create_view(&Default::default());

        // Stub depth texture — Depth32Float 1×1, cleared to 1.0 (far).
        let scene_stub_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene_depth_stub"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let scene_stub_depth_view = scene_stub_depth.create_view(&Default::default());

        Self {
            layouts,
            pipelines: Vec::new(),
            per_frame_buffer,
            per_frame_bg,
            per_view_buffer,
            per_view_bg,
            default_per_material_bg,
            _default_material_factors_buffer: default_material_factors_buffer,
            _default_user_params_buffer: default_user_params_buffer,
            _default_white_tex: default_white_tex,
            _default_white_view: default_white_view,
            _default_sampler: default_sampler,
            _default_black_tex: default_black_tex,
            default_black_view,
            _default_array_tex: default_array_tex,
            default_array_view,
            material_params_buffers: Vec::new(),
            material_per_material_bgs: Vec::new(),
            material_reflection_probe: Vec::new(),
            per_draw_buffers: Vec::new(),
            per_draw_bgs: Vec::new(),
            scene_inputs_bg: None,
            _scene_color_sampler: scene_color_sampler,
            _scene_depth_sampler: scene_depth_sampler,
            _scene_stub_tex: scene_stub_tex,
            _scene_stub_view: scene_stub_view,
            _scene_stub_depth: scene_stub_depth,
            _scene_stub_depth_view: scene_stub_depth_view,
            commands: Vec::new(),
            translucent_commands: Vec::new(),
            next_draw_slot: 0,
            instance_buffers: Vec::new(),
            texture_arrays: Vec::new(),
            material_texture_arrays: Vec::new(),
            // EN-012 — per-material MaterialFactors UBOs are lazy.
            // Until a material calls `set_shading_model` /
            // `set_foliage`, it shares the default factors buffer.
            material_factors_buffers: Vec::new(),
            material_factors_data:    Vec::new(),
        }
    }

    // --- Material registry --------------------------------------------

    /// Compile a material and return its handle. Handles are 1-based;
    /// 0 is reserved for "invalid material".
    pub fn compile(
        &mut self,
        device: &wgpu::Device,
        wgsl_source: &str,
        profile: FragmentProfile,
        bucket: Bucket,
        reads_scene: bool,
        wants_instancing: bool,
        hdr_format: wgpu::TextureFormat,
        material_format: wgpu::TextureFormat,
        velocity_format: wgpu::TextureFormat,
        albedo_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Result<MaterialHandle, MaterialCompileError> {
        // Inject the user's WGSL under a synthetic path so the
        // preprocessor can resolve `#include "material_abi.wgsl"` etc.
        let entry_path = "__user_material.wgsl";
        let desc = MaterialCompileDesc {
            label: "user_material",
            entry_path,
            extra_sources: &[(entry_path, wgsl_source)],
            profile,
            bucket,
            reads_scene,
            hdr_format,
            material_format,
            velocity_format,
            albedo_format,
            depth_format,
            vertex_buffers: &[Vertex3D::desc()],
            wants_instancing,
        };
        let pipeline = compile_material(device, &self.layouts, &desc)?;
        self.pipelines.push(Some(pipeline));
        Ok(self.pipelines.len() as MaterialHandle)
    }

    // --- Frame lifecycle ----------------------------------------------

    /// Write the per-frame + per-view UBOs. Safe to call any time
    /// before `dispatch`; callers should call exactly once per frame
    /// to keep `PerFrame.time` accurate. Does NOT clear the commands
    /// list — that's `reset_frame`'s job (called at begin_frame, not
    /// end_frame, so draws submitted during the frame survive).
    pub fn update_frame_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        per_frame: &PerFrameUniforms,
        per_view:  &PerViewUniforms,
    ) {
        queue.write_buffer(&self.per_frame_buffer, 0, bytemuck::bytes_of(per_frame));
        queue.write_buffer(&self.per_view_buffer,  0, bytemuck::bytes_of(per_view));
    }

    /// Reset the per-draw slot cursor. Commands lists are cleared by
    /// the Renderer from its own `begin_frame` so the order of reset
    /// vs. submit is deterministic.
    pub fn reset_draw_slot(&mut self) {
        self.next_draw_slot = 0;
    }

    /// Phase 5 — set/replace `user_params` for a specific material. The
    /// next dispatch of this handle binds a per-material BindGroup with
    /// the given bytes uploaded to `@group(2) @binding(11)`. Materials
    /// that never receive a `set_user_params` call keep using the
    /// default zero-initialised UBO.
    ///
    /// `params.len()` must be ≤ 256 bytes (ABI §1.4 cap). The buffer
    /// is allocated lazily on first call per handle and reused on
    /// subsequent updates. Pass an empty slice to revert to the default.
    pub fn set_user_params(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: MaterialHandle,
        params: &[u8],
    ) -> Result<(), &'static str> {
        if handle == 0 { return Err("invalid material handle"); }
        let idx = (handle - 1) as usize;
        if idx >= self.pipelines.len() || self.pipelines[idx].is_none() {
            return Err("material handle not registered");
        }
        if params.len() > 256 {
            return Err("user_params exceeds 256-byte cap");
        }

        // Grow parallel vectors so handle index is valid.
        while self.material_params_buffers.len() <= idx {
            self.material_params_buffers.push(None);
            self.material_per_material_bgs.push(None);
        }

        // Reverting to default — drop the per-material entries.
        if params.is_empty() {
            self.material_params_buffers[idx] = None;
            self.material_per_material_bgs[idx] = None;
            return Ok(());
        }

        // Allocate the per-material UBO + BG on first set. Padded to 256 B
        // so the ABI cap is reflected in the buffer size and write_buffer
        // never partially fills. EN-011 bindings 12/13 default to the
        // shared 1×1 black reflection texture; if a probe was previously
        // linked via `set_reflection_probe`, the reflection BG will be
        // rebuilt by that path the next time the link changes.
        if self.material_params_buffers[idx].is_none() {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("material_user_params"),
                size: 256,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // EN-012 — resolve the per-material factors UBO so a
            // material that previously called `set_shading_model` /
            // `set_foliage` keeps its custom factors at binding 10
            // when user_params is later set on it.
            let factors_buf: &wgpu::Buffer = self.material_factors_buffers
                .get(idx)
                .and_then(|b| b.as_ref())
                .unwrap_or(&self._default_material_factors_buffer);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material_per_material_bg_user"),
                layout: &self.layouts.per_material,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                    wgpu::BindGroupEntry { binding: 1,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                    wgpu::BindGroupEntry { binding: 2,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                    wgpu::BindGroupEntry { binding: 3,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                    wgpu::BindGroupEntry { binding: 4,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                    wgpu::BindGroupEntry { binding: 5,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                    wgpu::BindGroupEntry { binding: 6,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                    wgpu::BindGroupEntry { binding: 7,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                    wgpu::BindGroupEntry { binding: 8,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                    wgpu::BindGroupEntry { binding: 9,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                    wgpu::BindGroupEntry { binding: 10, resource: factors_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.default_black_view) },
                    wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                    // EN-014 — default stub array on all 3 slots.
                    wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&self.default_array_view) },
                    wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&self.default_array_view) },
                    wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(&self.default_array_view) },
                    wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                ],
            });
            self.material_params_buffers[idx] = Some(buf);
            self.material_per_material_bgs[idx] = Some(bg);
        }

        // Pad short writes to 256 B so write_buffer doesn't read past `params`.
        let mut padded = [0u8; 256];
        padded[..params.len()].copy_from_slice(params);
        let buf = self.material_params_buffers[idx].as_ref().unwrap();
        queue.write_buffer(buf, 0, &padded);
        Ok(())
    }

    /// Per-material BG when set, otherwise the shared default.
    fn per_material_bg_for(&self, handle: MaterialHandle) -> &wgpu::BindGroup {
        let idx = (handle as usize).wrapping_sub(1);
        self.material_per_material_bgs.get(idx)
            .and_then(|b| b.as_ref())
            .unwrap_or(&self.default_per_material_bg)
    }

    /// EN-011 — link a material to a planar reflection probe's RT
    /// view. The per-material bind group is rebuilt so binding 12
    /// resolves to the probe's `color_view` instead of the default
    /// 1×1 black texture. Subsequent draws of `handle` see the
    /// probe's contents.
    ///
    /// Pass `probe_view = None` (or revert to the default by passing
    /// `&self.default_black_view` from the caller side) to unlink.
    /// The `probe_handle` is opaque to this method — it's stored
    /// purely so `material_reflection_probe[idx]` reflects the link
    /// for diagnostics + the renderer's per-frame dispatch.
    ///
    /// The wgpu Texture identity is stable for the probe's lifetime,
    /// so we DON'T need to rebuild this bind group every frame; the
    /// engine repaints the probe's texture but the same view keeps
    /// pointing at it.
    ///
    /// Why we route through here vs direct field access: the BG
    /// also needs to bind a user-params UBO (binding 11) — either
    /// the per-material UBO when one was allocated by
    /// `set_user_params`, or the shared default zero-UBO. This
    /// helper picks the right one.
    pub fn set_reflection_probe(
        &mut self,
        device: &wgpu::Device,
        handle:       MaterialHandle,
        probe_handle: u32,
        probe_view:   &wgpu::TextureView,
    ) -> Result<(), &'static str> {
        if handle == 0 { return Err("invalid material handle"); }
        let idx = (handle - 1) as usize;
        if idx >= self.pipelines.len() || self.pipelines[idx].is_none() {
            return Err("material handle not registered");
        }

        // Grow parallel vectors so the index is in bounds. We grow
        // BOTH params + reflection registries together — they share
        // an index domain (MaterialHandle - 1).
        while self.material_params_buffers.len()  <= idx {
            self.material_params_buffers.push(None);
            self.material_per_material_bgs.push(None);
        }
        while self.material_reflection_probe.len() <= idx {
            self.material_reflection_probe.push(None);
        }
        self.material_reflection_probe[idx] = Some(probe_handle);

        // Resolve binding 11 — per-material UBO if one's been
        // allocated, else the shared zero-init default.
        let user_params_buf: &wgpu::Buffer = self.material_params_buffers
            .get(idx)
            .and_then(|b| b.as_ref())
            .unwrap_or(&self._default_user_params_buffer);

        // EN-012 — resolve binding 10 to the per-material factors UBO
        // when one's been allocated by `set_shading_model` /
        // `set_foliage`. Otherwise the reflection-probe rebind would
        // silently revert a foliage-flagged material back to the
        // default factors at binding 10.
        let factors_buf: &wgpu::Buffer = self.material_factors_buffers
            .get(idx)
            .and_then(|b| b.as_ref())
            .unwrap_or(&self._default_material_factors_buffer);

        // EN-014 — resolve binding 14/15/16 from any per-material
        // texture-array links so a probe-bound material doesn't lose
        // its splat layers when the BG gets rebuilt.
        let [albedo_view, normal_view, mr_view] = self.resolve_array_views(idx);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_per_material_bg_reflection"),
            layout: &self.layouts.per_material,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 1,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 2,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 3,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 4,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 5,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 6,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 7,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 8,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 9,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 10, resource: factors_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: user_params_buf.as_entire_binding() },
                // EN-011 — probe's color view + a filtering sampler.
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(probe_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                // EN-014 — resolved texture arrays + shared sampler.
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(albedo_view) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(mr_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
            ],
        });
        self.material_per_material_bgs[idx] = Some(bg);
        Ok(())
    }

    /// EN-014 — resolve the 3 D2Array texture views for a material
    /// (albedo / normal / MR), falling back to the shared 1×1×1 stub
    /// when the slot has no link. Returns borrowed views that share
    /// the lifetime of `self`. Callers use these to populate bindings
    /// 14/15/16 when (re)building a per-material BG.
    fn resolve_array_views(&self, idx: usize) -> [&wgpu::TextureView; 3] {
        let mut out: [&wgpu::TextureView; 3] = [
            &self.default_array_view,
            &self.default_array_view,
            &self.default_array_view,
        ];
        if let Some(slots) = self.material_texture_arrays.get(idx) {
            for (slot, link) in slots.iter().enumerate() {
                if let Some(handle) = link {
                    let h = *handle as usize;
                    if h > 0 {
                        if let Some(Some(arr)) = self.texture_arrays.get(h - 1) {
                            out[slot] = &arr.view;
                        }
                    }
                }
            }
        }
        out
    }

    /// EN-012 — resolve the per-material `MaterialFactors` UBO for
    /// `idx`, falling back to the shared default buffer when no
    /// per-material factors UBO has been allocated yet. Used by every
    /// per-material BG-build path so EN-011 / EN-014 rebinds don't
    /// silently revert a material's foliage / shading-model state to
    /// the default factors.
    fn resolve_factors_buffer(&self, idx: usize) -> &wgpu::Buffer {
        self.material_factors_buffers
            .get(idx)
            .and_then(|b| b.as_ref())
            .unwrap_or(&self._default_material_factors_buffer)
    }

    /// EN-012 — ensure a per-material `MaterialFactorsUniforms` UBO
    /// exists for `idx`, allocating + initialising from the default
    /// values on first use. Returns mutable access to the CPU mirror
    /// so the caller can mutate one field then write back via
    /// `flush_material_factors`. Grows `material_factors_buffers` /
    /// `material_factors_data` so `idx` is in bounds.
    fn ensure_material_factors(
        &mut self, device: &wgpu::Device, idx: usize,
    ) -> &mut MaterialFactorsUniforms {
        while self.material_factors_buffers.len() <= idx {
            self.material_factors_buffers.push(None);
            self.material_factors_data.push(MaterialFactorsUniforms::default());
        }
        if self.material_factors_buffers[idx].is_none() {
            // Allocate a fresh per-material factors UBO seeded with the
            // current CPU-side data (defaults on first call).
            let init = self.material_factors_data[idx];
            let buf = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("material_factors_per_material"),
                    contents: bytemuck::bytes_of(&init),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                },
            );
            self.material_factors_buffers[idx] = Some(buf);
        }
        &mut self.material_factors_data[idx]
    }

    /// EN-012 — write the current CPU-side `MaterialFactorsUniforms`
    /// for `idx` to the per-material UBO. Caller must have previously
    /// called `ensure_material_factors` so the buffer exists.
    fn flush_material_factors(&self, queue: &wgpu::Queue, idx: usize) {
        if let Some(Some(buf)) = self.material_factors_buffers.get(idx) {
            let data = &self.material_factors_data[idx];
            queue.write_buffer(buf, 0, bytemuck::bytes_of(data));
        }
    }

    /// EN-012 — rebuild the per-material BG for `idx` after a
    /// MaterialFactors mutation. Resolves the per-material UBO,
    /// user_params UBO, reflection-probe view, and texture-array
    /// views from the current state so we don't clobber any of the
    /// EN-005/EN-011/EN-014 wiring. Caller is responsible for
    /// passing `probe_view` (the renderer wrapper looks the linked
    /// probe up via `material_reflection_probe_handle` and supplies
    /// either the probe's RT view or the default 1×1 black view).
    fn rebuild_per_material_bg(
        &mut self,
        device:     &wgpu::Device,
        idx:        usize,
        probe_view: &wgpu::TextureView,
    ) {
        // Grow parallel BG vector so `idx` is in bounds.
        while self.material_per_material_bgs.len() <= idx {
            self.material_params_buffers.push(None);
            self.material_per_material_bgs.push(None);
        }

        let factors_buf: &wgpu::Buffer = self.resolve_factors_buffer(idx);
        let user_params_buf: &wgpu::Buffer = self.material_params_buffers
            .get(idx)
            .and_then(|b| b.as_ref())
            .unwrap_or(&self._default_user_params_buffer);
        let [albedo_view, normal_view, mr_view] = self.resolve_array_views(idx);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_per_material_bg_factors"),
            layout: &self.layouts.per_material,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 1,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 2,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 3,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 4,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 5,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 6,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 7,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 8,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 9,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 10, resource: factors_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: user_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(probe_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(albedo_view) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(mr_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
            ],
        });
        self.material_per_material_bgs[idx] = Some(bg);
    }

    /// EN-012 — set the shading model for a material (`0` = default
    /// lit, `1` = foliage, `2` = subsurface — V2 stub). Lazily
    /// allocates a per-material `MaterialFactors` UBO on first call,
    /// then rebuilds the per-material BG so subsequent draws sample
    /// the new shading model. Caller passes `probe_view` so EN-011
    /// reflection links survive — see `Renderer::set_material_shading_model`.
    pub fn set_material_shading_model(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        material:   MaterialHandle,
        model:      u32,
        probe_view: &wgpu::TextureView,
    ) -> Result<(), &'static str> {
        if material == 0 { return Err("invalid material handle"); }
        let idx = (material - 1) as usize;
        if idx >= self.pipelines.len() || self.pipelines[idx].is_none() {
            return Err("material handle not registered");
        }
        {
            let factors = self.ensure_material_factors(device, idx);
            factors.shading_model[0] = model as f32;
        }
        self.flush_material_factors(queue, idx);
        self.rebuild_per_material_bg(device, idx, probe_view);
        Ok(())
    }

    /// EN-012 — set the foliage shading parameters for a material.
    /// Only takes effect when `shading_model == 1` (foliage). Updates
    /// `MaterialFactors.shading_model.yzw` (transmission_color) and
    /// `foliage_params.xy` (transmission_amount, wrap_factor). Lazily
    /// allocates a per-material UBO on first call.
    pub fn set_material_foliage(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        material:    MaterialHandle,
        trans_color: [f32; 3],
        trans_amount: f32,
        wrap_factor:  f32,
        probe_view:  &wgpu::TextureView,
    ) -> Result<(), &'static str> {
        if material == 0 { return Err("invalid material handle"); }
        let idx = (material - 1) as usize;
        if idx >= self.pipelines.len() || self.pipelines[idx].is_none() {
            return Err("material handle not registered");
        }
        {
            let factors = self.ensure_material_factors(device, idx);
            factors.shading_model[1] = trans_color[0];
            factors.shading_model[2] = trans_color[1];
            factors.shading_model[3] = trans_color[2];
            factors.foliage_params[0] = trans_amount;
            factors.foliage_params[1] = wrap_factor;
        }
        self.flush_material_factors(queue, idx);
        self.rebuild_per_material_bg(device, idx, probe_view);
        Ok(())
    }

    /// EN-014 — create a 2D texture array from a slice of layer data.
    /// Each `(bytes, w, h)` describes one layer's RGBA8 source. All
    /// layers must share `w × h` (wgpu requires a uniform extent for
    /// D2Array). V1 panics on mismatch — V2 may resize. Layer count
    /// is capped at `MAX_TEXTURE_ARRAY_LAYERS`; extra layers are
    /// dropped. Returns a 1-based handle (0 on failure: empty layers
    /// or zero extent).
    ///
    /// Defaults: `format = 0` (Rgba8UnormSrgb, suitable for albedo),
    /// `mip_levels = 1` (no mips). For data textures (normal / MR)
    /// or auto-mip generation, see `create_texture_array_ex`.
    pub fn create_texture_array(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        layers: &[(&[u8], u32, u32)],
    ) -> u32 {
        // V1 default: sRGB albedo, no mips. V2 callers use the _ex
        // variant directly.
        self.create_texture_array_ex(device, queue, layers, 0, 1)
    }

    /// EN-014 V2 — create a 2D texture array with explicit pixel format
    /// and mip-level control. Layer extent / count rules match V1.
    ///
    /// `format`:
    ///   0 → `Rgba8UnormSrgb` (albedo / colour textures; default)
    ///   1 → `Rgba8Unorm`     (normal / MR / data textures — linear)
    ///   _ → falls back to `Rgba8UnormSrgb`
    ///
    /// `mip_levels`:
    ///   1     → no mips (matches V1 behaviour)
    ///   0     → auto-generate `floor(log2(max(w,h))) + 1` mips, filled
    ///           by point-downsample (`copy_texture_to_texture` halving
    ///           the previous mip). Cheap, correct sized, but aliased
    ///           — a render-pass box filter is the V2.5 follow-up.
    ///   N > 1 → not yet supported (game-supplied per-mip bytes); V2
    ///           treats this as auto-generate.
    pub fn create_texture_array_ex(
        &mut self,
        device:     &wgpu::Device,
        queue:      &wgpu::Queue,
        layers:     &[(&[u8], u32, u32)],
        format:     u32,
        mip_levels: u32,
    ) -> u32 {
        let layer_count = (layers.len() as u32).min(MAX_TEXTURE_ARRAY_LAYERS);
        if layer_count == 0 { return 0; }
        let (_first_bytes, w, h) = layers[0];
        if w == 0 || h == 0 { return 0; }
        // Uniform extent check — V1 hard-fail surfaces obvious bugs at
        // creation rather than during silent GPU truncation later.
        for (i, (_, lw, lh)) in layers.iter().enumerate().take(layer_count as usize) {
            if *lw != w || *lh != h {
                eprintln!(
                    "[texture_array] layer {} extent {}×{} does not match layer 0 ({}×{}); aborting create",
                    i, lw, lh, w, h,
                );
                return 0;
            }
        }
        let wgpu_format = map_texture_array_format(format);
        // Resolve mip count. mip_levels = 1 → single mip. mip_levels = 0
        // → engine-generated max. Anything else (game-supplied per-mip)
        // is V2.5; for now treat N > 1 as auto so games opting into mips
        // don't silently regress to no-mips.
        let max_mips = (w.max(h) as f32).log2().floor() as u32 + 1;
        let auto_generate = mip_levels == 0 || mip_levels > 1;
        let mip_level_count = if mip_levels == 1 { 1 } else { max_mips.max(1) };
        // Auto-gen needs COPY_SRC on the texture so we can ping-pong each
        // mip into the next via copy_texture_to_texture.
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
                      | wgpu::TextureUsages::COPY_DST;
        if auto_generate && mip_level_count > 1 {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }
        let bytes_per_layer = (w as usize) * (h as usize) * 4;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material_texture_array"),
            size: wgpu::Extent3d {
                width:                  w,
                height:                 h,
                depth_or_array_layers:  layer_count,
            },
            mip_level_count,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu_format,
            usage,
            view_formats: &[],
        });
        for (i, (bytes, _, _)) in layers.iter().enumerate().take(layer_count as usize) {
            // Defensive — a short layer slice would panic in
            // write_texture; skip and emit a diagnostic instead.
            if bytes.len() < bytes_per_layer {
                eprintln!(
                    "[texture_array] layer {} short: {} B < {} B (skipping)",
                    i, bytes.len(), bytes_per_layer,
                );
                continue;
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes[..bytes_per_layer],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        // EN-014 V2 — auto-generate the mip chain via point-sample copies.
        // For each mip > 0, copy a half-size region of mip-1 into mip,
        // for every layer. wgpu's copy_texture_to_texture covers a single
        // mip level + array layer per call. This is point-filtered (not
        // box-filtered): correct sizes, sampleable at distance, but
        // aliased. V2.5 follow-up upgrades this to a render-pass box
        // filter (one fullscreen draw per (mip, layer)).
        if auto_generate && mip_level_count > 1 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("material_texture_array_mipgen"),
            });
            for mip in 1..mip_level_count {
                let src_w = (w >> (mip - 1)).max(1);
                let src_h = (h >> (mip - 1)).max(1);
                let dst_w = (w >> mip).max(1);
                let dst_h = (h >> mip).max(1);
                // Copy region is dst-sized so we read the top-left
                // 2x2 reduction implicitly. Truly we'd want a filter,
                // but copy is the cheapest "mips exist" path.
                let copy_w = dst_w.min(src_w);
                let copy_h = dst_h.min(src_h);
                for layer in 0..layer_count {
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: mip - 1,
                            origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: mip,
                            origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: copy_w,
                            height: copy_h,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            queue.submit(std::iter::once(encoder.finish()));
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("material_texture_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.texture_arrays.push(Some(TextureArray {
            texture, view, layer_count,
        }));
        self.texture_arrays.len() as u32
    }

    /// EN-014 — link a texture array to a material's per-material
    /// bind-group at one of three slots (0 = albedo, 1 = normal,
    /// 2 = MR). Pass `array = 0` to revert the slot to the default
    /// 1×1×1 stub. The per-material BG is rebuilt so subsequent draws
    /// see the new array view at @binding 14/15/16.
    ///
    /// `probe_view` is the binding-12 source — the caller (typically
    /// `Renderer::set_material_texture_array`) resolves the
    /// material's currently-linked reflection probe and passes its
    /// `color_view` here, or `&self.default_black_view` to default.
    /// This indirection keeps EN-011 reflection links live across
    /// EN-014 rebinds without the MaterialSystem having to reach
    /// into the Renderer's probe registry. Out-of-range slots /
    /// unknown handles are no-ops with a diagnostic.
    pub fn set_material_texture_array(
        &mut self,
        device:     &wgpu::Device,
        material:   MaterialHandle,
        slot:       u32,
        array:      u32,
        probe_view: &wgpu::TextureView,
    ) {
        if material == 0 { return; }
        let idx = (material - 1) as usize;
        if idx >= self.pipelines.len() || self.pipelines[idx].is_none() {
            eprintln!("[texture_array] unknown material handle {material}");
            return;
        }
        if slot >= 3 {
            eprintln!("[texture_array] slot {slot} out of range (0..=2)");
            return;
        }
        if array != 0 {
            let h = array as usize;
            if h == 0 || h > self.texture_arrays.len()
                || self.texture_arrays[h - 1].is_none()
            {
                eprintln!("[texture_array] unknown array handle {array}");
                return;
            }
        }

        // Grow the link table so `idx` is in bounds. Parallel to
        // the params + reflection registries but not strictly tied
        // to either — texture-array links are independent.
        while self.material_texture_arrays.len() <= idx {
            self.material_texture_arrays.push([None, None, None]);
        }
        let link = if array == 0 { None } else { Some(array) };
        self.material_texture_arrays[idx][slot as usize] = link;

        // Rebuild the per-material BG. Resolve user_params from
        // existing state so we don't clobber EN-005 links.
        while self.material_params_buffers.len() <= idx {
            self.material_params_buffers.push(None);
            self.material_per_material_bgs.push(None);
        }
        let user_params_buf: &wgpu::Buffer = self.material_params_buffers
            .get(idx)
            .and_then(|b| b.as_ref())
            .unwrap_or(&self._default_user_params_buffer);
        // EN-012 — preserve any per-material MaterialFactors UBO
        // across an EN-014 array rebind.
        let factors_buf: &wgpu::Buffer = self.material_factors_buffers
            .get(idx)
            .and_then(|b| b.as_ref())
            .unwrap_or(&self._default_material_factors_buffer);

        let [albedo_view, normal_view, mr_view] = self.resolve_array_views(idx);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_per_material_bg_array"),
            layout: &self.layouts.per_material,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 1,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 2,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 3,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 4,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 5,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 6,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 7,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 8,  resource: wgpu::BindingResource::TextureView(&self._default_white_view) },
                wgpu::BindGroupEntry { binding: 9,  resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 10, resource: factors_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: user_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(probe_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(albedo_view) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(mr_view) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self._default_sampler) },
            ],
        });
        self.material_per_material_bgs[idx] = Some(bg);
    }

    /// EN-014 — accessor for `material_reflection_probe[idx]`. The
    /// Renderer wrapper uses this to resolve the currently-linked
    /// probe handle when rebuilding the BG via
    /// `set_material_texture_array`. Returns `None` for unset / out-
    /// of-range / unlinked materials.
    pub fn material_reflection_probe_handle(&self, material: MaterialHandle) -> Option<u32> {
        if material == 0 { return None; }
        let idx = (material as usize).checked_sub(1)?;
        self.material_reflection_probe.get(idx).copied().flatten()
    }

    /// Convenience — true if either bucket has queued work this frame.
    pub fn any_commands(&self) -> bool {
        !self.commands.is_empty() || !self.translucent_commands.is_empty()
    }

    /// Submit a draw against a compiled material. Allocates (or reuses)
    /// a per-draw UBO slot, writes the MVP / model / tint / skin info,
    /// and queues the command for dispatch.
    pub fn submit_draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        joint_buffer: &wgpu::Buffer,
        material: MaterialHandle,
        mesh_handle: u64,
        mesh_idx: usize,
        mvp: [[f32; 4]; 4],
        model: [[f32; 4]; 4],
        prev_mvp: [[f32; 4]; 4],
        tint: [f32; 4],
        skin_info: [u32; 4],
    ) {
        let idx = material as usize;
        if material == 0 || idx > self.pipelines.len() { return; }
        let bucket = match self.pipelines[idx - 1].as_ref() {
            Some(p) => p.bucket,
            None => return,
        };

        let slot = self.next_draw_slot;
        self.next_draw_slot += 1;
        self.ensure_draw_slot(device, joint_buffer, slot);

        let per_draw = PerDrawUniforms { mvp, model, prev_mvp, model_tint: tint, skin_info };
        queue.write_buffer(&self.per_draw_buffers[slot], 0, bytemuck::bytes_of(&per_draw));

        let cmd = MaterialDrawCommand {
            material, mesh_handle, mesh_idx, draw_slot: slot,
            view_depth: mvp[3][3],
            instance: None,
            model,
        };
        if bucket.is_translucent() {
            self.translucent_commands.push(cmd);
        } else {
            self.commands.push(cmd);
        }
    }

    /// EN-001 — submit an instanced draw. Identical to `submit_draw`
    /// except the engine binds vertex slot 1 to the registered
    /// instance buffer and emits `draw_indexed(.., 0..count)`. The
    /// pipeline must have been compiled with `wants_instancing=true`
    /// (use `compile_material_instanced` on the renderer).
    ///
    /// `model` / `mvp` here are the instance-local→world fallback
    /// transform — the per-instance buffer's `instance_pos`/`rot_y`/
    /// `scale` typically dominate, so callers usually pass identity
    /// for `model` and the camera VP for `mvp`. `tint` is multiplied
    /// per-draw (in addition to the per-instance tint).
    pub fn submit_draw_instanced(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        joint_buffer: &wgpu::Buffer,
        material: MaterialHandle,
        mesh_handle: u64,
        mesh_idx: usize,
        instance_buffer: u32,
        instance_count: u32,
        mvp: [[f32; 4]; 4],
        model: [[f32; 4]; 4],
        prev_mvp: [[f32; 4]; 4],
        tint: [f32; 4],
        skin_info: [u32; 4],
    ) {
        let idx = material as usize;
        if material == 0 || idx > self.pipelines.len() { return; }
        let bucket = match self.pipelines[idx - 1].as_ref() {
            Some(p) => p.bucket,
            None => return,
        };

        let slot = self.next_draw_slot;
        self.next_draw_slot += 1;
        self.ensure_draw_slot(device, joint_buffer, slot);

        let per_draw = PerDrawUniforms { mvp, model, prev_mvp, model_tint: tint, skin_info };
        queue.write_buffer(&self.per_draw_buffers[slot], 0, bytemuck::bytes_of(&per_draw));

        let cmd = MaterialDrawCommand {
            material, mesh_handle, mesh_idx, draw_slot: slot,
            // Instanced draws sort as a group by their fallback-transform
            // pivot — per-instance ordering inside one buffer is the
            // standard engine limitation.
            view_depth: mvp[3][3],
            instance: Some(InstanceDrawInfo {
                buffer_handle: instance_buffer,
                count:         instance_count,
            }),
            model,
        };
        if bucket.is_translucent() {
            self.translucent_commands.push(cmd);
        } else {
            self.commands.push(cmd);
        }
    }

    /// EN-001 — create a persistent instance buffer from CPU-side
    /// floats. The data layout matches `InstanceData3D` (9 floats per
    /// instance: pos.xyz, rot_y, scale, tint.rgba); this method pads
    /// each instance to 12 floats at upload time so the GPU side gets
    /// the correct 48-byte stride. Returns a 1-based handle to use
    /// with `submit_draw_instanced`. Pair with `destroy_instance_buffer`
    /// when the buffer's no longer needed.
    pub fn create_instance_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raw: &[f32],
        instance_count: u32,
    ) -> u32 {
        let count = instance_count as usize;
        let mut packed: Vec<f32> = Vec::with_capacity(count * 12);
        for i in 0..count {
            let off = i * 9;
            if off + 9 > raw.len() { break; }
            packed.extend_from_slice(&raw[off..off + 3]);     // pos.xyz
            packed.push(raw[off + 3]);                        // rot_y
            packed.push(raw[off + 4]);                        // scale
            packed.extend_from_slice(&raw[off + 5..off + 9]); // tint.rgba
            packed.extend_from_slice(&[0.0, 0.0, 0.0]);       // pad to 48 bytes
        }
        let size = (packed.len() * std::mem::size_of::<f32>()) as u64;
        // Empty buffers can't be created (size 0 is invalid in wgpu).
        // Reserve at least one stride so the BG/binding remains valid.
        let buffer_size = size.max(48);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_instance_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !packed.is_empty() {
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&packed));
        }
        self.instance_buffers.push(Some(InstanceBuffer { buffer, count: instance_count }));
        self.instance_buffers.len() as u32
    }

    /// EN-001 — drop an instance buffer slot. The slot is left as
    /// `None` so previously-issued handles never alias a future
    /// allocation. No-op for `handle == 0` or out-of-range handles.
    pub fn destroy_instance_buffer(&mut self, handle: u32) {
        if handle == 0 { return; }
        let idx = handle as usize - 1;
        if idx < self.instance_buffers.len() {
            self.instance_buffers[idx] = None;
        }
    }

    fn ensure_draw_slot(
        &mut self,
        device: &wgpu::Device,
        joint_buffer: &wgpu::Buffer,
        slot: usize,
    ) {
        while self.per_draw_buffers.len() <= slot {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("material_per_draw"),
                size: std::mem::size_of::<PerDrawUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material_per_draw_bg"),
                layout: &self.layouts.per_draw,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: joint_buffer.as_entire_binding() },
                ],
            });
            self.per_draw_buffers.push(buf);
            self.per_draw_bgs.push(bg);
        }
    }

    /// Dispatch all queued material draws. Caller owns the render pass;
    /// this method binds the pipelines + groups + meshes and issues
    /// indexed draws. `mesh_fetch` is a closure that returns
    /// `(vertex_buffer, index_buffer, index_count)` for a given
    /// (mesh_handle, mesh_idx) — lets the renderer hand over its
    /// `model_gpu_cache` without this module taking a dependency on it.
    pub fn dispatch<'pass, F>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        mesh_fetch: F,
    )
    where F: FnMut(u64, usize) -> Option<(&'pass wgpu::Buffer, &'pass wgpu::Buffer, u32)>
    {
        self.dispatch_with_view(pass, &self.per_view_bg, |_| true, false, mesh_fetch);
    }

    /// EN-011 — like `dispatch`, but uses a caller-supplied PerView
    /// bind group (so the planar-reflection pass can render the same
    /// draws against a mirrored camera UBO) and skips any command
    /// whose material handle the `accept` predicate rejects (so the
    /// reflection pass can drop hardcoded-excluded materials —
    /// foliage, particles, grass — without touching the original
    /// command list).
    ///
    /// `accept` is consulted once per command; the engine passes a
    /// closure that closes over a small `HashSet<MaterialHandle>` of
    /// excluded material handles.
    ///
    /// EN-011 V2 — `use_reflection_pipeline` swaps in the material's
    /// sibling pipeline (cull_mode flipped) when one was compiled.
    /// Falls back to the main pipeline if no reflection variant
    /// exists (translucent / cutout materials, where the original
    /// pipeline already cull-mode = None and no flip is needed).
    pub fn dispatch_with_view<'pass, F, A>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        per_view_bg: &'pass wgpu::BindGroup,
        mut accept:  A,
        use_reflection_pipeline: bool,
        mut mesh_fetch: F,
    )
    where
        F: FnMut(u64, usize) -> Option<(&'pass wgpu::Buffer, &'pass wgpu::Buffer, u32)>,
        A: FnMut(MaterialHandle) -> bool,
    {
        if self.commands.is_empty() { return; }

        let mut last_material: MaterialHandle = 0;
        for cmd in &self.commands {
            if !accept(cmd.material) { continue; }
            if cmd.material != last_material {
                let mat = match self.pipelines.get(cmd.material as usize - 1) {
                    Some(Some(m)) => m,
                    _ => continue,
                };
                let pipeline = if use_reflection_pipeline {
                    mat.reflection_pipeline.as_ref().unwrap_or(&mat.pipeline)
                } else {
                    &mat.pipeline
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.per_frame_bg, &[]);
                pass.set_bind_group(1, per_view_bg, &[]);
                pass.set_bind_group(2, self.per_material_bg_for(cmd.material), &[]);
                last_material = cmd.material;
            }
            if let Some((vb, ib, icount)) = mesh_fetch(cmd.mesh_handle, cmd.mesh_idx) {
                pass.set_bind_group(3, &self.per_draw_bgs[cmd.draw_slot], &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                let instance_range = self.bind_instance_buffer(pass, &cmd.instance);
                if instance_range.end > instance_range.start {
                    pass.draw_indexed(0..icount, 0, instance_range);
                }
            }
        }
    }

    /// EN-001 — resolve an instanced draw command's vertex slot 1 binding
    /// and return the instance range. For non-instanced draws (`info` is
    /// None) this is a no-op and returns `0..1`. For instanced draws
    /// with a missing/destroyed buffer slot we return an empty range
    /// so the caller skips the draw rather than crashing on a stale
    /// handle.
    fn bind_instance_buffer<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        info: &Option<InstanceDrawInfo>,
    ) -> std::ops::Range<u32> {
        match info {
            None => 0..1,
            Some(inst) => {
                if inst.buffer_handle == 0 { return 0..1; }
                let slot_idx = inst.buffer_handle as usize - 1;
                match self.instance_buffers.get(slot_idx).and_then(|s| s.as_ref()) {
                    Some(ib_slot) => {
                        pass.set_vertex_buffer(1, ib_slot.buffer.slice(..));
                        0..inst.count
                    }
                    None => 0..0,
                }
            }
        }
    }

    /// Phase 4b — rebuild the SceneInputs (group 4) bind group with
    /// the current frame's snapshot textures. Called by the Renderer
    /// once per frame when translucent draws exist and a SceneColor
    /// transient has been allocated. `scene_color_view` is the
    /// copy-to-sample snapshot from `hdr_rt`; `scene_depth_view` is
    /// the live depth buffer the opaque pass wrote. Other slots
    /// (impulse, motion vectors) bind to internal stub textures
    /// until Phase 7 wires them.
    pub fn update_scene_inputs(
        &mut self,
        device: &wgpu::Device,
        scene_color_view: &wgpu::TextureView,
        scene_depth_view: Option<&wgpu::TextureView>,
        impulse_view: Option<(&wgpu::TextureView, &wgpu::Sampler)>,
    ) {
        let depth_view = scene_depth_view.unwrap_or(&self._scene_stub_depth_view);
        // Layout entry 5 is NonFiltering — fallback uses the depth
        // sampler (which is also NonFiltering) rather than the
        // filtering color sampler so the layout matches either way.
        let (imp_view, imp_samp): (&wgpu::TextureView, &wgpu::Sampler) = match impulse_view {
            Some((v, s)) => (v, s),
            None         => (&self._scene_stub_view, &self._scene_depth_sampler),
        };
        // Phase 4c — group 4 binding 2 receives a COPY_DST snapshot of
        // the opaque depth buffer, rather than the live depth-stencil
        // attachment (wgpu rejects read+write aliasing in the same
        // pass). Callers that don't need depth pass the stub view
        // already held here; `Renderer::end_frame_with_scene` acquires
        // a transient depth texture when any translucent material
        // declares a depth read.
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_inputs_bg"),
            layout: &self.layouts.scene_inputs,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self._scene_color_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self._scene_depth_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(imp_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(imp_samp) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self._scene_stub_view) },
            ],
        });
        self.scene_inputs_bg = Some(bg);
    }

    /// Sort the translucent bucket back-to-front by view depth. Call
    /// once per frame after all submits, before dispatch_translucent —
    /// alpha blending only composites correctly when farther surfaces
    /// land first. The sort is stable: equal-depth draws keep submission
    /// order, so existing games that hand-ordered their transparency
    /// keep their result. (Sorting costs pipeline-switch batching; with
    /// translucent counts in the hundreds the correctness is worth it.)
    pub fn sort_translucent(&mut self) {
        self.translucent_commands
            .sort_by(|a, b| b.view_depth.total_cmp(&a.view_depth));
    }

    /// Dispatch translucent-bucket draws (Transparent, Refractive,
    /// Additive). Caller owns a render pass set up with a single HDR
    /// attachment (LoadOp::Load) + depth read-only. Refractive
    /// materials additionally receive the SceneInputs bind group at
    /// group 4 — `update_scene_inputs` must have been called this
    /// frame for that to be non-None.
    pub fn dispatch_translucent<'pass, F>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        mut mesh_fetch: F,
    )
    where F: FnMut(u64, usize) -> Option<(&'pass wgpu::Buffer, &'pass wgpu::Buffer, u32)>
    {
        if self.translucent_commands.is_empty() { return; }

        let mut last_material: MaterialHandle = 0;
        let mut last_reads_scene: bool = false;
        for cmd in &self.translucent_commands {
            if cmd.material != last_material {
                let mat = match self.pipelines.get(cmd.material as usize - 1) {
                    Some(Some(m)) => m,
                    _ => continue,
                };
                pass.set_pipeline(&mat.pipeline);
                pass.set_bind_group(0, &self.per_frame_bg, &[]);
                pass.set_bind_group(1, &self.per_view_bg, &[]);
                pass.set_bind_group(2, self.per_material_bg_for(cmd.material), &[]);
                if mat.reads_scene {
                    if let Some(bg) = self.scene_inputs_bg.as_ref() {
                        pass.set_bind_group(4, bg, &[]);
                    }
                }
                last_material = cmd.material;
                last_reads_scene = mat.reads_scene;
            }
            // Re-bind group 4 if the material switches its reads_scene
            // between subsequent draws — rarely happens with a
            // stable bucket but keeps the state machine honest.
            let _ = last_reads_scene;

            if let Some((vb, ib, icount)) = mesh_fetch(cmd.mesh_handle, cmd.mesh_idx) {
                pass.set_bind_group(3, &self.per_draw_bgs[cmd.draw_slot], &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                let instance_range = self.bind_instance_buffer(pass, &cmd.instance);
                if instance_range.end > instance_range.start {
                    pass.draw_indexed(0..icount, 0, instance_range);
                }
            }
        }
    }
}

// =====================================================================
// Tests
// =====================================================================


// GPU-backed tests live in material_system_tests.rs (2000-line file
// policy); the #[path] keeps them a private child module of this one.
#[cfg(test)]
#[path = "material_system_tests.rs"]
mod tests_file;
