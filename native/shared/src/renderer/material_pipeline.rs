// Material pipeline — the entry point for custom WGSL materials against
// the ABI described in docs/rfc/0001-material-render-graph.md.
//
// This module is deliberately self-contained. It owns the bind-group
// layouts, pipeline-layout composition, and the preprocessor driver.
// The rest of the renderer doesn't know or care how a material is
// compiled — it just receives a `MaterialPipeline` and uses the
// layouts for binding groups that the shader declares it consumes.
//
// Phase 1b scope: everything needed to *compile* a material pipeline
// from user WGSL against the ABI. Draw dispatch, per-draw uniform
// writes, and FFI glue all land in follow-up phases.

use super::shader_include::{process, BakedSource, IncludeError};

// =====================================================================
// Bind-group layouts — one struct, five layouts, matching RFC §1
// =====================================================================

/// The five bind-group layouts every ABI-compliant pipeline binds.
/// Owned by Renderer once per process (not per pipeline). Cheap to clone
/// references since `wgpu::BindGroupLayout` is Arc'd internally.
pub struct MaterialAbiLayouts {
    pub per_frame: wgpu::BindGroupLayout,
    pub per_view: wgpu::BindGroupLayout,
    pub per_material: wgpu::BindGroupLayout,
    pub per_draw: wgpu::BindGroupLayout,
    pub scene_inputs: wgpu::BindGroupLayout,
}

impl MaterialAbiLayouts {
    pub fn create(device: &wgpu::Device) -> Self {
        Self {
            per_frame: create_per_frame_layout(device),
            per_view: create_per_view_layout(device),
            per_material: create_per_material_layout(device),
            per_draw: create_per_draw_layout(device),
            scene_inputs: create_scene_inputs_layout(device),
        }
    }
}

/// EN-063 — wasm32 ABI contract. WebGPU in the browser caps
/// `maxBindGroups` at 4, so the SceneInputs group (group 4 on native)
/// cannot exist on the web: one reads_scene pipeline would fail
/// creation and poison the whole frame's command buffer. On wasm32
/// the seven scene-input bindings fold into the per_frame group
/// (group 0) at bindings `WASM_SCENE_INPUTS_BASE .. +6`, in the same
/// order and with the same types as `create_scene_inputs_layout`.
/// per_frame's only native binding is 0 (the PerFrame UBO), so the
/// folded block starts at 1.
///
/// Three places must agree on this contract and all read this
/// constant:
///   1. the shader-source rewrite in `compile_material`
///      (`rewrite_scene_inputs_for_wasm`),
///   2. the wasm32 `create_per_frame_layout` below,
///   3. `MaterialSystem`'s per_frame bind-group builder
///      (`build_per_frame_bg_wasm` in material_system.rs).
///
/// Native targets keep the five-group layout bit-identically.
#[cfg(fold_scene_inputs)]
pub const WASM_SCENE_INPUTS_BASE: u32 = 1;

#[cfg(not(fold_scene_inputs))]
fn create_per_frame_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_per_frame"),
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
    })
}

/// EN-063 — wasm32 variant: the PerFrame UBO at binding 0 plus the
/// seven folded SceneInputs entries (same types, same order as
/// `create_scene_inputs_layout`) at `WASM_SCENE_INPUTS_BASE..+6`.
/// Every bind group created against this layout must supply all
/// eight entries — see `build_per_frame_bg_wasm` in
/// material_system.rs, the single creation site.
#[cfg(fold_scene_inputs)]
fn create_per_frame_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    const B: u32 = WASM_SCENE_INPUTS_BASE;
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_per_frame"),
        entries: &[
            entry_ubo(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            entry_tex_f(B, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                B + 1,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            entry_tex_depth(B + 2, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                B + 3,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::NonFiltering,
            ),
            entry_tex_f_nonfilt(B + 4, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                B + 5,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::NonFiltering,
            ),
            entry_tex_f(B + 6, wgpu::ShaderStages::FRAGMENT),
        ],
    })
}

fn create_per_view_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    // Mirror of the ABI header: UBO at 0, env colour + sampler at 1+2,
    // env diffuse at 3, BRDF LUT + sampler at 4+5, three cascades at
    // 6..8, comparison sampler at 9.
    let frag = wgpu::ShaderStages::FRAGMENT;
    let mut entries = vec![
        entry_ubo(0, wgpu::ShaderStages::VERTEX | frag),
        entry_tex_f(1, frag),
        entry_samp(2, frag, wgpu::SamplerBindingType::Filtering),
        entry_tex_f(3, frag),
        entry_tex_f(4, frag),
        entry_samp(5, frag, wgpu::SamplerBindingType::Filtering),
        entry_tex_depth(6, frag),
        entry_tex_depth(7, frag),
        entry_tex_depth(8, frag),
        entry_samp(9, frag, wgpu::SamplerBindingType::Comparison),
    ];
    if crate::virtual_shadows::virtual_shadows_requested() {
        entries.extend([
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: frag,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: frag,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: frag,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        crate::virtual_shadows::VSM_SAMPLING_PARAMS_BYTES,
                    ),
                },
                count: None,
            },
        ]);
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_per_view"),
        entries: &entries,
    })
}

fn create_per_material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    // PBR texture + sampler pairs: base, normal, mr, emissive, occlusion.
    // Bindings 0..9 are texture/sampler pairs. Binding 10 is
    // MaterialFactors UBO. Binding 11 is the user_params UBO (shader-
    // defined type; 256-byte cap enforced by the pipeline-creation
    // helper, not by the layout itself). Bindings 12 (texture) + 13
    // (sampler) are the EN-011 planar reflection RT — bound to the
    // engine's 1×1 black default for materials without a probe so
    // unconditional sampling is always safe. Bindings 14/15/16/17 are
    // the EN-014 texture-array slots (albedo / normal / MR + a shared
    // sampler) for splat-mapped terrain materials — bound to a 1×1×1
    // black stub array when a material doesn't declare its own.
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_per_material"),
        entries: &[
            entry_tex_f(0, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                1,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            entry_tex_f(2, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                3,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            entry_tex_f(4, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                5,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            entry_tex_f(6, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                7,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            entry_tex_f(8, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                9,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            entry_ubo(
                10,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ),
            entry_ubo(
                11,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ),
            // EN-011 — planar reflection RT (texture + sampler).
            entry_tex_f(12, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                13,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            // EN-014 — texture-array slots for splat-mapped terrain.
            entry_tex_f_array(14, wgpu::ShaderStages::FRAGMENT),
            entry_tex_f_array(15, wgpu::ShaderStages::FRAGMENT),
            entry_tex_f_array(16, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                17,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
        ],
    })
}

fn create_per_draw_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_per_draw"),
        entries: &[
            entry_ubo(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            entry_ubo(1, wgpu::ShaderStages::VERTEX), // JointMatrices (1024 × mat4)
        ],
    })
}

fn create_scene_inputs_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_scene_inputs"),
        entries: &[
            entry_tex_f(0, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                1,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            entry_tex_depth(2, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                3,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::NonFiltering,
            ),
            // Phase 7 — impulse_tex is R32Float which is non-filterable
            // in wgpu 29 without a feature flag, so the binding is
            // declared NonFiltering. Materials sample via textureLoad
            // (no filtering; 0.5 m / texel is already coarse).
            entry_tex_f_nonfilt(4, wgpu::ShaderStages::FRAGMENT),
            entry_samp(
                5,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::NonFiltering,
            ),
            entry_tex_f(6, wgpu::ShaderStages::FRAGMENT),
        ],
    })
}

fn entry_tex_f_nonfilt(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

// Small helpers for binding entry construction.
fn entry_ubo(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn entry_tex_f(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
/// EN-014 — `texture_2d_array<f32>` binding entry. Used for the
/// splat-mapped terrain slots (albedo / normal / MR arrays) at
/// bindings 14/15/16. Filterable so games can use linear filtering
/// per-layer.
fn entry_tex_f_array(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}
fn entry_tex_depth(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
fn entry_samp(
    binding: u32,
    vis: wgpu::ShaderStages,
    ty: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

// =====================================================================
// Fragment output profile — opaque or translucent
// =====================================================================

/// Fragment output profile a material declares. Decides the pipeline's
/// colour attachment layout and blend state. See ABI §1.8.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FragmentProfile {
    /// Four MRT attachments: HDR, material, velocity, albedo.
    Opaque,
    /// Single HDR attachment, alpha-blended. Does not write depth.
    Translucent,
}

/// Scheduling bucket — tells the render graph which pass a draw
/// belongs in and what sort order to apply. Distinct from
/// `FragmentProfile` (which describes pipeline outputs): a material
/// with profile `Translucent` could be in either `Transparent` or
/// `Refractive`, and `Additive` is its own bucket even though it
/// shares attachment layout with `Transparent`.
///
/// See RFC 0001 §3.2.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Bucket {
    /// Opaque draws. Front-to-back sort for early-z efficiency. Runs
    /// in the main HDR pass; writes the full 4-MRT G-buffer.
    Opaque,
    /// Alpha-cutout draws (foliage cards, chain-link fences, leaf
    /// silhouettes). Runs in the opaque pass with full G-buffer write
    /// + sun shadow + SSAO, but the fragment shader is expected to
    /// `discard` against `MaterialFactors.alpha_cutoff`. Rendered
    /// double-sided so foliage is visible from both faces.
    Cutout,
    /// Translucent draws. Back-to-front sort for correct blending.
    /// Single HDR attachment, alpha-blended, depth-test without
    /// depth-write. Runs in the translucent sub-pass (Phase 4b).
    Transparent,
    /// Translucent + reads the scene colour snapshot (for refraction,
    /// Fresnel, shoreline effects). Same pass + sort as Transparent
    /// but the graph inserts a SceneColor snapshot before this
    /// bucket runs.
    Refractive,
    /// Additive-blend draws — particle flares, sparks, weapon
    /// glows. Order-independent, so no sort needed. Runs in the
    /// translucent sub-pass.
    Additive,
}

impl Bucket {
    /// True if this bucket dispatches in the translucent sub-pass
    /// (single HDR attachment, alpha/additive blending) rather than
    /// the main HDR pass.
    pub fn is_translucent(self) -> bool {
        matches!(
            self,
            Bucket::Transparent | Bucket::Refractive | Bucket::Additive
        )
    }
    /// True if this bucket requires a SceneColor snapshot before it
    /// runs. Only Refractive does today; future buckets (e.g.
    /// VolumetricFog) may join.
    pub fn needs_scene_color(self) -> bool {
        matches!(self, Bucket::Refractive)
    }
}

// =====================================================================
// Material pipeline — the compiled artefact
// =====================================================================

struct OwnedVertexBufferLayout {
    array_stride: wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode,
    attributes: Vec<wgpu::VertexAttribute>,
}

/// CPU-only recipe retained until a custom translucent material first shares
/// a TAA-reactive sorted pass with imported BLEND geometry. The sibling keeps
/// the material's exact shader/blend/depth contract while declaring the
/// reactive attachment with an empty write mask. Keeping source instead of an
/// eagerly compiled GPU pipeline avoids startup work for the ordinary and
/// unmixed paths; the recipe is dropped immediately after specialization.
struct TranslucentReactiveRecipe {
    source: TranslucentReactiveSource,
    pipeline_layout: wgpu::PipelineLayout,
    vertex_buffers: Vec<OwnedVertexBufferLayout>,
    hdr_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    bucket: Bucket,
    label: String,
    writes_reactive: bool,
}

enum TranslucentReactiveSource {
    /// General `compile_material` callers may supply an arbitrary include
    /// overlay; retain its already validated expansion as the safe fallback.
    Expanded(String),
    /// The public custom-material API has one synthetic user source plus the
    /// baked library. Retaining only the authored source avoids keeping a full
    /// expanded shader per never-mixed translucent material.
    User(String),
}

/// A pipeline ready to receive draws. Owns only the `RenderPipeline`;
/// the layouts are borrowed from the shared `MaterialAbiLayouts`.
pub struct MaterialPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub(crate) reactive_pipeline: Option<wgpu::RenderPipeline>,
    reactive_recipe: Option<TranslucentReactiveRecipe>,
    pub profile: FragmentProfile,
    pub bucket: Bucket,
    pub reads_scene: bool,
    /// EN-001 — true when the pipeline was compiled with the
    /// per-instance vertex layout (slot 1, step_mode = Instance). The
    /// dispatch site uses this as a sanity flag — it doesn't change
    /// what bind groups get bound, only which `set_vertex_buffer(1, …)`
    /// path runs for a given draw command.
    pub wants_instancing: bool,
    /// The authored source exposes `fs_reactive`, so submitted translucent
    /// draws can activate and write the lazy temporal-reactive attachment.
    pub(crate) writes_reactive: bool,
    /// Label carried through for debug output.
    pub label: String,
    /// EN-011 V2 — sibling pipeline with front-face culling for use in
    /// planar reflection passes. Reflection inverts triangle winding,
    /// so an opaque material that would normally cull back-faces needs
    /// to cull front-faces in the mirrored pass — otherwise single-
    /// sided geometry renders inside-out. Lazily compiled the first
    /// time the material gets linked to a probe via
    /// `MaterialSystem::set_reflection_probe`. Reuses the main
    /// pipeline's shader module — the only difference is `cull_mode`.
    ///
    /// `None` for materials whose original cull mode is already
    /// `None` (translucent, cutout) — flipping a non-culled pipeline
    /// is a no-op so we don't bother compiling a duplicate.
    pub reflection_pipeline: Option<wgpu::RenderPipeline>,
}

impl MaterialPipeline {
    /// Lazily create the attachment-compatible sibling used only when this
    /// custom material is globally interleaved with imported reactive BLEND.
    ///
    /// Ordinary custom shaders use `fs_main` with an empty location-1 write
    /// mask. Opt-in responsive shaders use their authored `fs_reactive` entry
    /// and union its coverage into the R8 attachment.
    pub(crate) fn ensure_reactive_pipeline(&mut self, device: &wgpu::Device) -> bool {
        if self.reactive_pipeline.is_some() {
            return false;
        }
        let Some(recipe) = self.reactive_recipe.take() else {
            return false;
        };
        let source = match recipe.source {
            TranslucentReactiveSource::Expanded(source) => source,
            TranslucentReactiveSource::User(source) => {
                expand_material_source("__user_material.wgsl", &[("__user_material.wgsl", &source)])
                    .expect("custom material source was validated by its ordinary pipeline")
            }
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&recipe.label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let vertex_buffers = recipe
            .vertex_buffers
            .iter()
            .map(|layout| wgpu::VertexBufferLayout {
                array_stride: layout.array_stride,
                step_mode: layout.step_mode,
                attributes: &layout.attributes,
            })
            .collect::<Vec<_>>();
        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };
        let color_blend = if recipe.bucket == Bucket::Additive {
            additive_blend
        } else {
            wgpu::BlendState::ALPHA_BLENDING
        };
        let reactive_target = if recipe.writes_reactive {
            wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: Some(super::temporal_reactive::reactive_union_blend()),
                write_mask: wgpu::ColorWrites::RED,
            }
        } else {
            wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            }
        };
        let targets = [
            Some(wgpu::ColorTargetState {
                format: recipe.hdr_format,
                blend: Some(color_blend),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(reactive_target),
        ];
        self.reactive_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{}_reactive_compatible", recipe.label)),
                layout: Some(&recipe.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if recipe.writes_reactive {
                        "fs_reactive"
                    } else {
                        "fs_main"
                    }),
                    targets: &targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: recipe.depth_format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));
        true
    }
}

/// Options passed to compile a material. Matches the material-descriptor
/// shape in RFC §3.1, minus the textures / parameters (those are game
/// data, set per draw).
pub struct MaterialCompileDesc<'a> {
    pub label: &'a str,
    pub entry_path: &'a str, // e.g. "materials/water.wgsl"
    /// Additional (path, source) entries layered over the baked library.
    /// Game-supplied shaders live here.
    pub extra_sources: &'a [(&'a str, &'a str)],
    /// Compact source that can be re-expanded if a TAA-reactive mixed sorted
    /// frame later needs an attachment-compatible sibling. `None` retains the
    /// validated expanded source for general custom include overlays.
    pub lazy_reactive_source: Option<&'a str>,
    pub profile: FragmentProfile,
    pub bucket: Bucket,
    pub reads_scene: bool,
    pub hdr_format: wgpu::TextureFormat,
    pub material_format: wgpu::TextureFormat,
    pub velocity_format: wgpu::TextureFormat,
    pub albedo_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
    pub vertex_buffers: &'a [wgpu::VertexBufferLayout<'a>],
    /// EN-001 — when true, `compile_material` appends
    /// `InstanceData3D::desc()` to `vertex_buffers` so the pipeline
    /// expects a second VB at slot 1 with step_mode = Instance.
    pub wants_instancing: bool,
}

#[derive(Debug)]
pub enum MaterialCompileError {
    Include(IncludeError),
    Naga(String),
    Wgpu(String),
}
impl From<IncludeError> for MaterialCompileError {
    fn from(e: IncludeError) -> Self {
        MaterialCompileError::Include(e)
    }
}

fn expand_material_source(
    entry_path: &str,
    extra_sources: &[(&str, &str)],
) -> Result<String, IncludeError> {
    let mut entries: Vec<(&str, &str)> = BAKED_ENTRIES_SNAPSHOT.to_vec();
    entries.extend(extra_sources.iter().copied());
    let source = BakedSource { entries: &entries };
    let expanded = process(&source, entry_path)?;
    let expanded = if crate::virtual_shadows::virtual_shadows_requested() {
        crate::virtual_shadows::directional_material_shader(expanded)
    } else {
        expanded
    };
    // Browser WebGPU and folded mobile tiers keep SceneInputs in group 0.
    #[cfg(fold_scene_inputs)]
    let expanded = rewrite_scene_inputs_for_wasm(expanded);
    Ok(expanded)
}

fn wgsl_tokens(source: &str) -> Vec<&str> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
        } else if bytes[index..].starts_with(b"//") {
            index += bytes[index..]
                .iter()
                .position(|byte| *byte == b'\n')
                .unwrap_or(bytes.len() - index);
        } else if bytes[index..].starts_with(b"/*") {
            let mut depth = 1usize;
            index += 2;
            while index < bytes.len() && depth > 0 {
                if bytes[index..].starts_with(b"/*") {
                    depth += 1;
                    index += 2;
                } else if bytes[index..].starts_with(b"*/") {
                    depth -= 1;
                    index += 2;
                } else {
                    index += 1;
                }
            }
        } else if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            tokens.push(&source[start..index]);
        } else if bytes[index..].starts_with(b"->") {
            tokens.push(&source[index..index + 2]);
            index += 2;
        } else if !bytes[index].is_ascii() {
            index += source[index..]
                .chars()
                .next()
                .expect("index is inside source")
                .len_utf8();
        } else {
            tokens.push(&source[index..index + 1]);
            index += 1;
        }
    }
    tokens
}

fn declares_reactive_fragment(source: &str) -> Result<bool, MaterialCompileError> {
    if !source.contains("fs_reactive") {
        return Ok(false);
    }
    let tokens = wgsl_tokens(source);
    for function in tokens
        .windows(2)
        .enumerate()
        .filter_map(|(index, pair)| (pair == ["fn", "fs_reactive"]).then_some(index))
    {
        let declaration_start = tokens[..function]
            .iter()
            .rposition(|token| matches!(*token, "}" | ";"))
            .map_or(0, |index| index + 1);
        let is_fragment = tokens[declaration_start..function]
            .windows(2)
            .any(|pair| pair == ["@", "fragment"]);
        if !is_fragment {
            continue;
        }
        let Some(open) = tokens[function + 2..]
            .iter()
            .position(|token| *token == "(")
            .map(|index| function + 2 + index)
        else {
            return Err(MaterialCompileError::Naga(
                "fs_reactive has no parameter list".to_owned(),
            ));
        };
        let mut depth = 0usize;
        let close = tokens[open..]
            .iter()
            .enumerate()
            .find_map(|(offset, token)| {
                match *token {
                    "(" => depth += 1,
                    ")" => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(open + offset);
                        }
                    }
                    _ => {}
                }
                None
            });
        let valid_result = close.is_some_and(|close| {
            tokens.get(close + 1) == Some(&"->")
                && tokens.get(close + 2) == Some(&"ReactiveTranslucentOut")
        });
        if !valid_result {
            return Err(MaterialCompileError::Naga(
                "fs_reactive must return ReactiveTranslucentOut with @location(0) HDR and \
                 @location(1) f32 reactive coverage"
                    .to_owned(),
            ));
        }
        return Ok(true);
    }
    Ok(false)
}

/// Compile a material pipeline. This is the happy-path you call at
/// `loadMaterial()` time.
pub fn compile_material(
    device: &wgpu::Device,
    layouts: &MaterialAbiLayouts,
    desc: &MaterialCompileDesc<'_>,
) -> Result<MaterialPipeline, MaterialCompileError> {
    // 1. Resolve #include chain against the baked library +
    //    game-supplied overlay.
    let expanded = expand_material_source(desc.entry_path, desc.extra_sources)?;

    // 2. Create shader module. wgpu's WGSL parser surfaces errors as
    //    panics through the default handler; we catch them by pushing
    //    the scope and popping on failure.
    let _ = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(desc.label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(expanded.as_str())),
    });
    // Note: we don't poll the error scope here because wgpu 29 returns
    // validation errors synchronously via the device's uncaptured-error
    // handler; callers should install their own handler for hot-reload.

    // 3. Pipeline layout — always binds groups 0..3; only includes
    //    scene_inputs when the material declares it.
    let mut bg_layouts: Vec<Option<&wgpu::BindGroupLayout>> = vec![
        Some(&layouts.per_frame),
        Some(&layouts.per_view),
        Some(&layouts.per_material),
        Some(&layouts.per_draw),
    ];
    // EN-063 — on wasm32 the scene inputs live inside per_frame
    // (group 0, bindings WASM_SCENE_INPUTS_BASE..+6), so the pipeline
    // layout stays at 4 groups: the browser caps maxBindGroups at 4.
    if desc.reads_scene && cfg!(not(fold_scene_inputs)) {
        bg_layouts.push(Some(&layouts.scene_inputs));
    }
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(desc.label),
        bind_group_layouts: &bg_layouts,
        immediate_size: 0,
    });

    // 4. Colour targets based on profile.
    // SH-055 — `lean_mrt` (Android): drop the `material` and `albedo` targets
    // to `None`. Both are unread on Android (material only feeds SSR/PT,
    // both off/unavailable there; albedo only feeds SSGI-modulation and
    // SSAO-alpha-weighting in scene-compose, also both off), and dropping
    // them from main_hdr_pass's simultaneous render-target set is what
    // actually addresses the Adreno GMEM-overflow cost (see build.rs). The
    // material's WGSL is unchanged — `out.material`/`out.albedo` writes are
    // silently discarded with no backing attachment (wgpu-core requires the
    // pipeline target and the render-pass attachment to agree index-for-index
    // on `None`; scene_pass.rs's main_hdr_pass color_attachments mirrors this).
    #[cfg(lean_mrt)]
    let opaque_targets = [
        Some(wgpu::ColorTargetState {
            format: desc.hdr_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        None,
        Some(wgpu::ColorTargetState {
            format: desc.velocity_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        None,
    ];
    #[cfg(not(lean_mrt))]
    let opaque_targets = [
        Some(wgpu::ColorTargetState {
            format: desc.hdr_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: desc.material_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: desc.velocity_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: desc.albedo_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ];
    // Additive bucket uses src+dst on color; alpha-blending buckets
    // (Transparent, Refractive) use SrcAlpha/OneMinusSrcAlpha. Both
    // share the single HDR attachment.
    let additive_blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent::OVER,
    };
    let translucent_blend = match desc.bucket {
        Bucket::Additive => additive_blend,
        _ => wgpu::BlendState::ALPHA_BLENDING,
    };
    let translucent_targets = [Some(wgpu::ColorTargetState {
        format: desc.hdr_format,
        blend: Some(translucent_blend),
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let targets: &[Option<wgpu::ColorTargetState>] = match desc.profile {
        FragmentProfile::Opaque => &opaque_targets,
        FragmentProfile::Translucent => &translucent_targets,
    };

    // 5. Depth-stencil — translucent reads depth, doesn't write.
    let depth_stencil = Some(wgpu::DepthStencilState {
        format: desc.depth_format,
        depth_write_enabled: Some(matches!(desc.profile, FragmentProfile::Opaque)),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });

    // 6. Vertex + fragment entry points. Convention: `vs_main` / `fs_main`.
    //    Materials can override by prefixing their shader with
    //    `// @entry vs:foo fs:bar` but the first version is fixed names.
    //
    //    EN-001 — when `wants_instancing` is set, append the standard
    //    InstanceData3D layout so the pipeline expects a second VB at
    //    slot 1 (step_mode = Instance). The owned Vec only lives long
    //    enough to be referenced by the RenderPipelineDescriptor.
    let writes_reactive = matches!(desc.profile, FragmentProfile::Translucent)
        && declares_reactive_fragment(&expanded)?;
    let vertex_buffers_owned: Vec<wgpu::VertexBufferLayout<'_>>;
    let vertex_buffers: &[wgpu::VertexBufferLayout<'_>] = if desc.wants_instancing {
        vertex_buffers_owned = desc
            .vertex_buffers
            .iter()
            .cloned()
            .chain(std::iter::once(
                crate::renderer::types::InstanceData3D::desc(),
            ))
            .collect();
        &vertex_buffers_owned
    } else {
        desc.vertex_buffers
    };
    let reactive_vertex_buffers = matches!(desc.profile, FragmentProfile::Translucent).then(|| {
        vertex_buffers
            .iter()
            .map(|layout| OwnedVertexBufferLayout {
                array_stride: layout.array_stride,
                step_mode: layout.step_mode,
                attributes: layout.attributes.to_vec(),
            })
            .collect::<Vec<_>>()
    });
    // Translucent materials (water, glass, particles) are commonly
    // viewed from both sides, so they render double-sided. Cutout
    // materials (foliage cards, chain-link fences) likewise need to
    // be visible from both faces. Plain Opaque materials cull backfaces.
    let main_cull = if matches!(desc.profile, FragmentProfile::Translucent)
        || matches!(desc.bucket, Bucket::Cutout)
    {
        None
    } else {
        Some(wgpu::Face::Back)
    };

    let make_pipeline = |cull: Option<wgpu::Face>, label_suffix: &str| {
        let label_owned = format!("{}{}", desc.label, label_suffix);
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&label_owned),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: cull,
                ..Default::default()
            },
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };

    let pipeline = make_pipeline(main_cull, "");

    // EN-011 V2 — eagerly compile a reflection sibling with cull_mode
    // flipped (Back → Front). Reflection mirrors world-space, which
    // inverts triangle winding; without this swap, single-sided opaque
    // geometry renders inside-out in the planar-reflection pass.
    //
    // Only build the variant when the original cull mode is `Back` —
    // for `None` (translucent, cutout) and any future `Front`-default
    // case, the reflection variant would be identical to the main
    // pipeline (or already-flipped), so we save the compile cost.
    //
    // The cost when we DO build is a single extra pipeline per
    // material (typically 5-30 in a scene). Compiled here rather
    // than lazily at `set_reflection_probe` time so we never need to
    // stash the WGSL source for a later recompile.
    let reflection_pipeline = match main_cull {
        Some(wgpu::Face::Back) => Some(make_pipeline(Some(wgpu::Face::Front), "_reflection")),
        Some(wgpu::Face::Front) => Some(make_pipeline(Some(wgpu::Face::Back), "_reflection")),
        None => None,
    };
    let reactive_recipe = reactive_vertex_buffers.map(|vertex_buffers| {
        let source = match desc.lazy_reactive_source {
            Some(source) => TranslucentReactiveSource::User(source.to_string()),
            None => TranslucentReactiveSource::Expanded(expanded),
        };
        TranslucentReactiveRecipe {
            source,
            pipeline_layout,
            vertex_buffers,
            hdr_format: desc.hdr_format,
            depth_format: desc.depth_format,
            bucket: desc.bucket,
            label: desc.label.to_string(),
            writes_reactive,
        }
    });

    Ok(MaterialPipeline {
        pipeline,
        reactive_pipeline: None,
        reactive_recipe,
        profile: desc.profile,
        bucket: desc.bucket,
        reads_scene: desc.reads_scene,
        wants_instancing: desc.wants_instancing,
        writes_reactive,
        label: desc.label.to_string(),
        reflection_pipeline,
    })
}

/// EN-063 — fold the SceneInputs declarations into group 0 for wasm32.
///
/// Rewrites each of the seven exact strings `@group(4) @binding(N)`
/// (N = 0..6) that material_abi.wgsl declares to
/// `@group(0) @binding(WASM_SCENE_INPUTS_BASE + N)`. The rewrite is
/// unconditional (not gated on reads_scene): every material includes
/// material_abi.wgsl, and a browser shader module must never declare
/// group 4 at all. Materials that don't use the scene inputs simply
/// leave the group-0 bindings statically unused, which wgpu ignores
/// at pipeline-layout validation exactly as it did for group 4.
#[cfg(fold_scene_inputs)]
fn rewrite_scene_inputs_for_wasm(expanded: String) -> String {
    let had_group4 = expanded.contains("@group(4)");
    let mut out = expanded;
    let mut replaced: u32 = 0;
    for n in 0..7u32 {
        let from = format!("@group(4) @binding({n})");
        let to = format!("@group(0) @binding({})", WASM_SCENE_INPUTS_BASE + n);
        replaced += out.matches(from.as_str()).count() as u32;
        out = out.replace(from.as_str(), to.as_str());
    }
    if had_group4 {
        debug_assert_eq!(
            replaced, 7,
            "material_abi.wgsl scene-input declarations changed; \
             update the EN-063 wasm32 group-4 fold to match"
        );
        debug_assert!(
            !out.contains("@group(4)"),
            "a @group(4) binding survived the EN-063 wasm32 fold — \
             group 4 exceeds the browser's maxBindGroups"
        );
    }
    out
}

// =====================================================================
// Baked library snapshot
// =====================================================================
//
// `shader_library::library()` returns an `impl ShaderSource`; for the
// compile path above we need a `&[(&str, &str)]` slice so the
// BakedSource we build can layer user overrides on top. Mirror the
// library contents here; kept in sync by a test.

const BAKED_ENTRIES_SNAPSHOT: &[(&str, &str)] = &[
    (
        "material_abi.wgsl",
        include_str!("../../../shared/shaders/material_abi.wgsl"),
    ),
    (
        "common/pbr.wgsl",
        include_str!("../../../shared/shaders/common/pbr.wgsl"),
    ),
    (
        "common/shadows.wgsl",
        include_str!("../../../shared/shaders/common/shadows.wgsl"),
    ),
    (
        "common/fog.wgsl",
        include_str!("../../../shared/shaders/common/fog.wgsl"),
    ),
    (
        "common/tonemap.wgsl",
        include_str!("../../../shared/shaders/common/tonemap.wgsl"),
    ),
    (
        "common/sky.wgsl",
        include_str!("../../../shared/shaders/common/sky.wgsl"),
    ),
    (
        "common/clouds.wgsl",
        include_str!("../../../shared/shaders/common/clouds.wgsl"),
    ),
    (
        "materials/test_minimal.wgsl",
        include_str!("../../../shared/shaders/materials/test_minimal.wgsl"),
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensure the baked snapshot used by compile_material matches the
    /// shader library. If this fails, an entry was added to the library
    /// but not the snapshot (or vice-versa) — fix either.
    #[test]
    fn snapshot_matches_library() {
        use super::super::shader_include::ShaderSource;
        use super::super::shader_library;
        let lib = shader_library::library();
        for (path, body) in BAKED_ENTRIES_SNAPSHOT {
            let from_lib = lib
                .fetch(path)
                .unwrap_or_else(|| panic!("snapshot includes '{}' not in library", path));
            assert_eq!(*body, from_lib, "mismatch for {}", path);
        }
    }

    #[test]
    fn material_shadow_cascades_match_the_fitted_view_frustum_depth() {
        let source = include_str!("../../../shared/shaders/common/shadows.wgsl");
        assert!(source.contains("let view_pos = view.shadow_view * vec4<f32>(world_pos, 1.0);"));
        assert!(source.contains("return max(-view_pos.z, 0.0);"));
        assert!(source.contains("split_far - view_depth"));
        assert!(!source.contains("length(world_pos - cam)"));
        assert!(source.contains("cascade_idx: u32, world_pos: vec3<f32>, outside_value: f32"));
        assert!(source.contains("sample_shadow_cascade(cascade, world_pos, -1.0)"));
        assert!(source.contains("shadow_val < 0.0"));
        assert!(source.contains("sample_shadow_cascade(cascade + 1u, world_pos, shadow_val)"));
    }

    /// End-to-end validation: resolve the minimal test material's
    /// includes and parse the result through naga (wgpu's WGSL
    /// front-end). If the ABI header has a syntax error, or if a
    /// struct reference is missing, naga fails this test.
    ///
    /// This test does not create a wgpu device — it's pure front-end
    /// parsing — so it runs in any CI or dev environment without a
    /// GPU. The downside is it doesn't verify the full pipeline
    /// descriptor (blend state, vertex buffer layout, etc.); those are
    /// exercised when `compile_material` runs at application startup.
    #[test]
    fn test_minimal_parses_through_naga() {
        let source = BakedSource {
            entries: BAKED_ENTRIES_SNAPSHOT,
        };
        let expanded = process(&source, "materials/test_minimal.wgsl")
            .expect("preprocessor resolves test_minimal.wgsl");
        let result = wgpu::naga::front::wgsl::parse_str(&expanded);
        if let Err(ref e) = result {
            eprintln!("naga parse error:\n{}", e.emit_to_string(&expanded));
        }
        assert!(
            result.is_ok(),
            "test_minimal.wgsl should parse via naga after include expansion"
        );
    }

    #[test]
    fn reactive_fragment_detection_requires_the_named_fragment_entry() {
        let ordinary = "
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0);
            }
        ";
        assert!(!declares_reactive_fragment(ordinary).unwrap());

        let helper_only = "
            // @fragment fn fs_reactive() -> ReactiveTranslucentOut {}
            fn fs_reactive() -> f32 {
                return 1.0;
            }
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0);
            }
        ";
        assert!(!declares_reactive_fragment(helper_only).unwrap());

        let responsive = "
            struct ReactiveTranslucentOut {
                @location(0) hdr: vec4<f32>,
                @location(1) reactive: f32,
            };
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0);
            }
            @fragment
            fn fs_reactive() -> ReactiveTranslucentOut {
                return ReactiveTranslucentOut(vec4<f32>(1.0), 1.0);
            }
        ";
        assert!(declares_reactive_fragment(responsive).unwrap());

        let malformed = "
            struct Out {
                @location(0) hdr: vec4<f32>,
                @location(1) reactive: vec4<f32>,
            };
            @fragment
            fn fs_reactive() -> Out {
                return Out(vec4<f32>(1.0), vec4<f32>(1.0));
            }
        ";
        assert!(matches!(
            declares_reactive_fragment(malformed),
            Err(MaterialCompileError::Naga(message))
                if message.contains("must return ReactiveTranslucentOut")
        ));
    }
}
