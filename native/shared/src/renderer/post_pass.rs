//! EN-017 — Post-pass slot for game-supplied fullscreen WGSL FX.
//!
//! A single slot the game can install via `set_post_pass(wgsl)` and
//! remove via `clear_post_pass()`. When set, the engine's composite
//! output is redirected to an LDR intermediate render target, then
//! the user's fragment shader runs as a fullscreen draw — sampling
//! `scene_color_tex` (LDR, post-tonemap) and `scene_depth_tex` —
//! and writes the final image to the swapchain. The 2D overlay
//! still draws on top of the post-pass output, so the HUD stays
//! crisp.
//!
//! V1 ships with one slot (no stack), depth access on, and a
//! dedicated bind group layout — see EN-017 for rationale.

use wgpu;

/// Describes the format the post-pass fragment shader writes to
/// (swapchain) plus the format of the intermediate LDR RT it
/// samples. Both equal `surface_config.format` so the post-pass
/// composes the engine's normal LDR output 1:1.
pub struct PostPassPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(Debug)]
pub enum PostPassCompileError {
    Naga(String),
    Wgpu(String),
}

/// Auto-prepended ABI for every post-pass shader. Declares the
/// engine-provided bindings + a fullscreen-triangle vertex shader.
/// User WGSL is appended verbatim and must declare:
///   `@fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32>`
pub const POST_PASS_PRELUDE: &str = r#"
// Auto-prepended by the engine for post-pass shaders (EN-017).
@group(0) @binding(0) var scene_color_tex:  texture_2d<f32>;
@group(0) @binding(1) var scene_color_samp: sampler;
@group(0) @binding(2) var scene_depth_tex:  texture_depth_2d;
@group(0) @binding(3) var scene_depth_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    // Standard fullscreen triangle.
    var out: VertexOutput;
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vid & 2u) * 2.0 - 1.0;
    out.position = vec4<f32>(x, -y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (y + 1.0) * 0.5);
    return out;
}

// User WGSL follows. Must declare `fs_main` with the signature:
//   @fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32>
"#;

/// Build the bind-group layout the post-pass uses. One layout per
/// pipeline (recreated on each `set_post_pass`) — cheap and avoids
/// having to thread a shared layout through the Renderer struct.
pub fn create_post_pass_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("post_pass_layout"),
        entries: &[
            // 0: scene color (LDR composite output).
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 1: filtering sampler for scene color.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // 2: scene depth (Depth32Float — texture_depth_2d in WGSL).
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
            // 3: non-filtering sampler for scene depth.
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    })
}

/// Compile a post-pass pipeline from user WGSL.
///
/// `swapchain_format` is the colour-target format the post-pass
/// writes to (the engine's surface format) — the LDR intermediate
/// the post-pass samples is allocated in the same format so the
/// user shader sees identical bits regardless of whether the
/// post-pass is active.
pub fn compile_post_pass(
    device: &wgpu::Device,
    user_wgsl: &str,
    swapchain_format: wgpu::TextureFormat,
) -> Result<PostPassPipeline, PostPassCompileError> {
    let full_source = format!("{}\n{}", POST_PASS_PRELUDE, user_wgsl);

    // wgpu 29 surfaces validation errors via the device's
    // uncaptured-error handler — same pattern material_pipeline.rs
    // uses. Pushing the scope keeps a syntax error from killing the
    // renderer; the real error text reaches the platform's eprintln
    // / console.error glue layer.
    let _ = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("post_pass_shader"),
        source: wgpu::ShaderSource::Wgsl(full_source.into()),
    });

    let bind_group_layout = create_post_pass_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("post_pass_pl_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("post_pass_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: swapchain_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    Ok(PostPassPipeline { pipeline, bind_group_layout })
}

/// Allocate the LDR intermediate render target the composite pass
/// writes into when a post-pass is installed. Format mirrors the
/// swapchain so the user shader sees identical bits to a normal
/// composite output.
pub fn create_composite_ldr_rt(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("composite_ldr_rt"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
             | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Headless wgpu device for tests. Mirrors the pattern in
    /// `transient.rs` so tests run on machines without a real GPU.
    fn try_create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
        })).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("post-pass-test-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        )).ok()?;
        Some((device, queue))
    }

    /// EN-017 — the underwater-tint example from the API docs must
    /// compile against the post-pass prelude. Skipped on adapters
    /// where `try_create_device` returns None (CI without a GPU).
    #[test]
    fn underwater_tint_compiles() {
        let Some((device, _queue)) = try_create_device() else { return; };
        let wgsl = r#"
            @fragment
            fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                let scene = textureSample(scene_color_tex, scene_color_samp, uv);
                return vec4<f32>(scene.rgb * vec3<f32>(0.4, 0.7, 0.9), 1.0);
            }
        "#;
        let result = compile_post_pass(
            &device, wgsl, wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        assert!(result.is_ok(), "underwater tint should compile: {:?}", result.err());
    }
}
