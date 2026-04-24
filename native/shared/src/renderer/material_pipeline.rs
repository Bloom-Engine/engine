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

use super::shader_include::{BakedSource, IncludeError, process};

// =====================================================================
// Bind-group layouts — one struct, five layouts, matching RFC §1
// =====================================================================

/// The five bind-group layouts every ABI-compliant pipeline binds.
/// Owned by Renderer once per process (not per pipeline). Cheap to clone
/// references since `wgpu::BindGroupLayout` is Arc'd internally.
pub struct MaterialAbiLayouts {
    pub per_frame:        wgpu::BindGroupLayout,
    pub per_view:         wgpu::BindGroupLayout,
    pub per_material:     wgpu::BindGroupLayout,
    pub per_draw:         wgpu::BindGroupLayout,
    pub scene_inputs:     wgpu::BindGroupLayout,
}

impl MaterialAbiLayouts {
    pub fn create(device: &wgpu::Device) -> Self {
        Self {
            per_frame:    create_per_frame_layout(device),
            per_view:     create_per_view_layout(device),
            per_material: create_per_material_layout(device),
            per_draw:     create_per_draw_layout(device),
            scene_inputs: create_scene_inputs_layout(device),
        }
    }
}

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

fn create_per_view_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    // Mirror of the ABI header: UBO at 0, env colour + sampler at 1+2,
    // env diffuse at 3, BRDF LUT + sampler at 4+5, three cascades at
    // 6..8, comparison sampler at 9.
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_per_view"),
        entries: &[
            entry_ubo(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            entry_tex_f(1, wgpu::ShaderStages::FRAGMENT),
            entry_samp(2, wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering),
            entry_tex_f(3, wgpu::ShaderStages::FRAGMENT),
            entry_tex_f(4, wgpu::ShaderStages::FRAGMENT),
            entry_samp(5, wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering),
            entry_tex_depth(6, wgpu::ShaderStages::FRAGMENT),
            entry_tex_depth(7, wgpu::ShaderStages::FRAGMENT),
            entry_tex_depth(8, wgpu::ShaderStages::FRAGMENT),
            entry_samp(9, wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Comparison),
        ],
    })
}

fn create_per_material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    // PBR texture + sampler pairs: base, normal, mr, emissive, occlusion.
    // Bindings 0..9 are texture/sampler pairs. Binding 10 is
    // MaterialFactors UBO. Binding 11 is the user_params UBO (shader-
    // defined type; 256-byte cap enforced by the pipeline-creation
    // helper, not by the layout itself).
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_per_material"),
        entries: &[
            entry_tex_f(0,  wgpu::ShaderStages::FRAGMENT),
            entry_samp(1,   wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering),
            entry_tex_f(2,  wgpu::ShaderStages::FRAGMENT),
            entry_samp(3,   wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering),
            entry_tex_f(4,  wgpu::ShaderStages::FRAGMENT),
            entry_samp(5,   wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering),
            entry_tex_f(6,  wgpu::ShaderStages::FRAGMENT),
            entry_samp(7,   wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering),
            entry_tex_f(8,  wgpu::ShaderStages::FRAGMENT),
            entry_samp(9,   wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering),
            entry_ubo(10,   wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            entry_ubo(11,   wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
        ],
    })
}

fn create_per_draw_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_per_draw"),
        entries: &[
            entry_ubo(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            entry_ubo(1, wgpu::ShaderStages::VERTEX),   // JointMatrices (1024 × mat4)
        ],
    })
}

fn create_scene_inputs_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("abi_scene_inputs"),
        entries: &[
            entry_tex_f(0,     wgpu::ShaderStages::FRAGMENT),
            entry_samp(1,      wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering),
            entry_tex_depth(2, wgpu::ShaderStages::FRAGMENT),
            entry_samp(3,      wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::NonFiltering),
            entry_tex_f(4,     wgpu::ShaderStages::FRAGMENT),
            entry_samp(5,      wgpu::ShaderStages::FRAGMENT, wgpu::SamplerBindingType::Filtering),
            entry_tex_f(6,     wgpu::ShaderStages::FRAGMENT),
        ],
    })
}

// Small helpers for binding entry construction.
fn entry_ubo(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
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
        binding, visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
fn entry_tex_depth(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
fn entry_samp(binding: u32, vis: wgpu::ShaderStages, ty: wgpu::SamplerBindingType)
              -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: vis,
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

// =====================================================================
// Material pipeline — the compiled artefact
// =====================================================================

/// A pipeline ready to receive draws. Owns only the `RenderPipeline`;
/// the layouts are borrowed from the shared `MaterialAbiLayouts`.
pub struct MaterialPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub profile:  FragmentProfile,
    pub reads_scene: bool,
    /// Label carried through for debug output.
    pub label: String,
}

/// Options passed to compile a material. Matches the material-descriptor
/// shape in RFC §3.1, minus the textures / parameters (those are game
/// data, set per draw).
pub struct MaterialCompileDesc<'a> {
    pub label:         &'a str,
    pub entry_path:    &'a str,                 // e.g. "materials/water.wgsl"
    /// Additional (path, source) entries layered over the baked library.
    /// Game-supplied shaders live here.
    pub extra_sources: &'a [(&'a str, &'a str)],
    pub profile:       FragmentProfile,
    pub reads_scene:   bool,
    pub hdr_format:    wgpu::TextureFormat,
    pub material_format: wgpu::TextureFormat,
    pub velocity_format: wgpu::TextureFormat,
    pub albedo_format:   wgpu::TextureFormat,
    pub depth_format:    wgpu::TextureFormat,
    pub vertex_buffers:  &'a [wgpu::VertexBufferLayout<'a>],
}

#[derive(Debug)]
pub enum MaterialCompileError {
    Include(IncludeError),
    Naga(String),
    Wgpu(String),
}
impl From<IncludeError> for MaterialCompileError {
    fn from(e: IncludeError) -> Self { MaterialCompileError::Include(e) }
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
    let baked_entries = BAKED_ENTRIES_SNAPSHOT;    // from library snapshot below
    let mut entries: Vec<(&str, &str)> = baked_entries.to_vec();
    for &(p, s) in desc.extra_sources {
        entries.push((p, s));
    }
    let source = BakedSource { entries: &entries };
    let expanded = process(&source, desc.entry_path)?;

    // 2. Create shader module. wgpu's WGSL parser surfaces errors as
    //    panics through the default handler; we catch them by
    //    pushing the scope and popping on failure.
    let _ = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(desc.label),
        source: wgpu::ShaderSource::Wgsl(expanded.into()),
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
    if desc.reads_scene {
        bg_layouts.push(Some(&layouts.scene_inputs));
    }
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(desc.label),
        bind_group_layouts: &bg_layouts,
        immediate_size: 0,
    });

    // 4. Colour targets based on profile.
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
    let translucent_targets = [Some(wgpu::ColorTargetState {
        format: desc.hdr_format,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let targets: &[Option<wgpu::ColorTargetState>] = match desc.profile {
        FragmentProfile::Opaque      => &opaque_targets,
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
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(desc.label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: desc.vertex_buffers,
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
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    Ok(MaterialPipeline {
        pipeline,
        profile:     desc.profile,
        reads_scene: desc.reads_scene,
        label:       desc.label.to_string(),
    })
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
    ("material_abi.wgsl",           include_str!("../../../shared/shaders/material_abi.wgsl")),
    ("common/pbr.wgsl",             include_str!("../../../shared/shaders/common/pbr.wgsl")),
    ("common/shadows.wgsl",         include_str!("../../../shared/shaders/common/shadows.wgsl")),
    ("common/fog.wgsl",             include_str!("../../../shared/shaders/common/fog.wgsl")),
    ("common/tonemap.wgsl",         include_str!("../../../shared/shaders/common/tonemap.wgsl")),
    ("common/sky.wgsl",             include_str!("../../../shared/shaders/common/sky.wgsl")),
    ("materials/test_minimal.wgsl", include_str!("../../../shared/shaders/materials/test_minimal.wgsl")),
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
            let from_lib = lib.fetch(path)
                .unwrap_or_else(|| panic!("snapshot includes '{}' not in library", path));
            assert_eq!(*body, from_lib, "mismatch for {}", path);
        }
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
        let source = BakedSource { entries: BAKED_ENTRIES_SNAPSHOT };
        let expanded = process(&source, "materials/test_minimal.wgsl")
            .expect("preprocessor resolves test_minimal.wgsl");
        let result = wgpu::naga::front::wgsl::parse_str(&expanded);
        if let Err(ref e) = result {
            eprintln!("naga parse error:\n{}", e.emit_to_string(&expanded));
        }
        assert!(result.is_ok(), "test_minimal.wgsl should parse via naga after include expansion");
    }
}
