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
    /// Both possible LDR ping-pong inputs. A pass normally uses only its
    /// index parity, but retaining both identities keeps the cache complete
    /// if stack editing grows beyond append/clear.
    pub bind_group_cache: [Option<wgpu::BindGroup>; 2],
    /// `post_frame` (binding 4): this pass's copy of the frame clock,
    /// written before the pass is drawn.
    pub frame_buffer: wgpu::Buffer,
}

/// The `post_frame` uniform every post-pass shader can read (binding 4).
/// `time` and `delta_time` are the clock material shaders see in
/// `frame.time` / `frame.delta_time`; `frame_index` counts the frames the
/// post stack has drawn (never reset, unlike the TAA counter);
/// `resolution` is the output size in pixels and `inv_resolution` one
/// pixel in uv.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostFrameUniforms {
    pub time: f32,
    pub delta_time: f32,
    pub frame_index: u32,
    pub _pad0: u32,
    pub resolution: [f32; 2],
    pub inv_resolution: [f32; 2],
}

impl PostFrameUniforms {
    pub fn new(time: f32, delta_time: f32, frame_index: u32, width: u32, height: u32) -> Self {
        let w = width.max(1) as f32;
        let h = height.max(1) as f32;
        Self {
            time,
            delta_time,
            frame_index,
            _pad0: 0,
            resolution: [w, h],
            inv_resolution: [1.0 / w, 1.0 / h],
        }
    }
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

// The frame clock, for animated effects (heat haze, grain, pulses).
struct PostFrame {
    time:           f32,        // seconds, the clock of material shaders' frame.time
    delta_time:     f32,        // seconds since the previous frame
    frame_index:    u32,        // frames the post stack has drawn; never reset
    _pad0:          u32,
    resolution:     vec2<f32>,  // output size in pixels
    inv_resolution: vec2<f32>,  // one pixel in uv
};
@group(0) @binding(4) var<uniform> post_frame: PostFrame;

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
            // 4: the frame clock (`post_frame`).
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<PostFrameUniforms>() as u64,
                    ),
                },
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

    let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("post_pass_frame"),
        size: std::mem::size_of::<PostFrameUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Ok(PostPassPipeline {
        pipeline,
        bind_group_layout,
        bind_group_cache: [None, None],
        frame_buffer,
    })
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
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("post-pass-test-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .ok()?;
        Some((device, queue))
    }

    /// EN-017 — the underwater-tint example from the API docs must
    /// compile against the post-pass prelude. Skipped on adapters
    /// where `try_create_device` returns None (CI without a GPU).
    #[test]
    fn underwater_tint_compiles() {
        let Some((device, _queue)) = try_create_device() else {
            return;
        };
        let wgsl = r#"
            @fragment
            fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                let scene = textureSample(scene_color_tex, scene_color_samp, uv);
                return vec4<f32>(scene.rgb * vec3<f32>(0.4, 0.7, 0.9), 1.0);
            }
        "#;
        let result = compile_post_pass(&device, wgsl, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(
            result.is_ok(),
            "underwater tint should compile: {:?}",
            result.err()
        );
    }

    /// The heat-haze example from the API docs reads the frame clock.
    #[test]
    fn heat_haze_with_frame_clock_compiles() {
        let wgsl = r#"
            @fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
              let t = post_frame.time;
              let rise = smoothstep(0.35, 0.95, uv.y);
              let wob = vec2<f32>(sin(uv.y * 90.0 - t * 5.0) + sin(uv.y * 37.0 + uv.x * 23.0 - t * 3.1),
                                  cos(uv.x * 70.0 + t * 4.0)) * 0.5;
              let d = wob * rise * 2.0 * post_frame.inv_resolution;
              return vec4<f32>(textureSample(scene_color_tex, scene_color_samp, uv + d).rgb, 1.0);
            }
        "#;
        let module = wgpu::naga::front::wgsl::parse_str(&format!("{POST_PASS_PRELUDE}\n{wgsl}"))
            .unwrap_or_else(|error| panic!("heat haze failed to parse: {error:?}"));
        // The WGSL struct and the Rust upload agree on size.
        let (_, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("PostFrame"))
            .expect("prelude declares PostFrame");
        assert_eq!(
            ty.inner.size(module.to_ctx()) as usize,
            std::mem::size_of::<PostFrameUniforms>()
        );
        let Some((device, _queue)) = try_create_device() else {
            return;
        };
        let result = compile_post_pass(&device, wgsl, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(
            result.is_ok(),
            "heat haze should compile: {:?}",
            result.err()
        );
    }
}
