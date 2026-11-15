//! Volumetric fog: a mist that is lit.
//!
//! The height fog in the scene-compose pass is one colour everywhere. This is
//! a second, optional medium whose brightness depends on what reaches it: the
//! sun through the cascaded shadow maps (shafts between trees, a dark volume
//! behind a house), the nearest point lights (a halo round a lamp) and a flat
//! ambient colour. Off by default; a frame with it off records nothing and
//! allocates nothing.
//!
//! Three small passes, recorded right after the scene compose:
//!
//! 1. **scatter** (compute): a view-frustum grid of 160 x 90 x 64 cells, the
//!    depth slices spaced exponentially from 0.3 m to the range. One jittered
//!    sample per cell per frame: density at that point, one hardware-filtered
//!    shadow lookup, up to eight point lights. The result is blended with last
//!    frame's grid looked up at the same WORLD position (0.9 history), which is
//!    what hides the single sample.
//! 2. **integrate** (compute): walks each column front to back and stores, per
//!    cell, the light scattered toward the camera up to there and how much of
//!    what lies behind survives.
//! 3. **apply** (fullscreen, blended into the composed HDR image): one depth
//!    read and one trilinear lookup per pixel; `colour = colour * transmittance
//!    + in-scatter`.
//!
//! Why a grid and not a per-pixel ray march: the grid's cost does not depend
//! on the output resolution, the temporal filter lives in world space (so a
//! moving character cannot leave a trail: the per-pixel step uses this frame's
//! depth only), and nothing has to be upsampled across depth edges.
//!
//! What it does not do: point lights are not shadowed (the engine has no
//! shadow maps for them), so their halo is kept tight (inverse square within
//! the light's own range) rather than filling the range. Transparent surfaces
//! get the fog of the opaque surface behind them, exactly like the height fog.
//! Perspective cameras only. Desktop only: the pass is not created on mobile
//! or web.

use super::lazy_pipeline::{LazyComputePipeline, LazyRenderPipeline};
use super::{Renderer, HDR_FORMAT};

const GRID_X: u32 = 160;
const GRID_Y: u32 = 90;
const GRID_Z: u32 = 64;
/// Nearest slice boundary, metres.
const NEAR: f32 = 0.3;
/// Share of last frame's value kept per frame.
const HISTORY_WEIGHT: f32 = 0.9;
const MAX_LIGHTS: usize = 8;
const VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// What the game sets. See `Renderer::set_volumetric_fog`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct Settings {
    pub enabled: bool,
    /// Extinction per metre at and below `height_ref`.
    pub density: f32,
    pub height_ref: f32,
    /// Per metre above `height_ref`; 0 = the same density at every height.
    pub height_falloff: f32,
    /// How far from the camera the medium reaches, metres.
    pub range: f32,
    /// Radiance of the medium where no light reaches it.
    pub ambient: [f32; 3],
    /// 1 = a physically isotropic medium lit by the sun; an art control.
    pub sun_strength: f32,
    /// The same for point lights.
    pub light_strength: f32,
    /// Henyey-Greenstein g: 0 = the same from every side, toward 1 = bright
    /// only when looking toward the sun.
    pub anisotropy: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: false,
            density: 0.02,
            height_ref: 0.0,
            height_falloff: 0.1,
            range: 80.0,
            ambient: [0.5, 0.55, 0.6],
            sun_strength: 1.0,
            light_strength: 1.0,
            anisotropy: 0.4,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FogLight {
    pos_range: [f32; 4],
    color: [f32; 4],
}

/// Mirrors `FogParams` in `COMMON_WGSL`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FogParams {
    inv_view: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    cascade_vps: [[[f32; 4]; 4]; 3],
    /// xyz = split distances, w = 1 when the sun casts shadows.
    cascade_splits: [f32; 4],
    /// p00, p11, p20, p21 of the unjittered projection.
    proj: [f32; 4],
    /// near, far, ln(far / near), history weight.
    depth: [f32; 4],
    /// proj[2][2], proj[3][2], frame number, point-light count.
    lin: [f32; 4],
    grid: [f32; 4],
    /// density, height_ref, height_falloff, anisotropy.
    medium: [f32; 4],
    ambient: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    lights: [FogLight; MAX_LIGHTS],
}

const COMMON_WGSL: &str = "
struct FogLight { pos_range: vec4<f32>, color: vec4<f32> };
struct FogParams {
    inv_view: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    cascade_vps: array<mat4x4<f32>, 3>,
    cascade_splits: vec4<f32>,
    proj: vec4<f32>,
    depth: vec4<f32>,
    lin: vec4<f32>,
    grid: vec4<f32>,
    medium: vec4<f32>,
    ambient: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    lights: array<FogLight, 8>,
};
@group(0) @binding(0) var<uniform> p: FogParams;

// Interleaved gradient noise (Jimenez 2014).
fn ign(q: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(q, vec2<f32>(0.06711056, 0.00583715))));
}
// View depth (metres in front of the camera) of slice coordinate w in 0..1.
fn slice_depth(w: f32) -> f32 {
    return p.depth.x * exp(w * p.depth.z);
}
// View-space position per metre of view depth for a screen uv.
fn view_ray(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3<f32>((ndc.x + p.proj.z) / p.proj.x, (ndc.y + p.proj.w) / p.proj.y, -1.0);
}
";

const SCATTER_WGSL: &str = "
@group(0) @binding(1) var shadow_tex: texture_depth_2d_array;
@group(0) @binding(2) var shadow_samp: sampler_comparison;
@group(0) @binding(3) var history_tex: texture_3d<f32>;
@group(0) @binding(4) var lin_samp: sampler;
@group(0) @binding(5) var out_tex: texture_storage_3d<rgba16float, write>;

fn sun_visibility(world: vec3<f32>, view_depth: f32) -> f32 {
    if (p.cascade_splits.w < 0.5) {
        return 1.0;
    }
    var cascade = 2;
    if (view_depth <= p.cascade_splits.x) {
        cascade = 0;
    } else if (view_depth <= p.cascade_splits.y) {
        cascade = 1;
    }
    // A cell can sit outside the cascade its depth selects (the cascades are
    // fitted to the camera frustum, the cell may be off to the side): take the
    // next wider one, and call it lit past the last.
    for (var i = cascade; i < 3; i = i + 1) {
        let clip = p.cascade_vps[i] * vec4<f32>(world, 1.0);
        let ndc = clip.xyz / clip.w;
        if (all(abs(ndc.xy) <= vec2<f32>(1.0)) && ndc.z >= 0.0 && ndc.z <= 1.0) {
            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            return textureSampleCompareLevel(shadow_tex, shadow_samp, uv, i, ndc.z);
        }
    }
    return 1.0;
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid = vec3<u32>(p.grid.xyz);
    if (gid.x >= grid.x || gid.y >= grid.y || gid.z >= grid.z) {
        return;
    }
    // One sample per cell per frame, at a different place in the cell each
    // frame: a per-cell offset advanced by an additive recurrence, so the
    // history blend converges on the cell's average.
    let frame = p.lin.z;
    let seed = vec2<f32>(gid.xy) + vec2<f32>(f32(gid.z) * 17.0, f32(gid.z) * 31.0);
    let jitter = vec3<f32>(
        fract(ign(seed + vec2<f32>(47.0, 113.0)) + frame * 0.75487766),
        fract(ign(seed + vec2<f32>(131.0, 19.0)) + frame * 0.56984029),
        fract(ign(seed) + frame * 0.61803399),
    );
    let cell = (vec3<f32>(gid) + jitter) / p.grid.xyz;
    let view_depth = slice_depth(cell.z);
    let ray = view_ray(cell.xy);
    let world = (p.inv_view * vec4<f32>(ray * view_depth, 1.0)).xyz;
    let cam = p.inv_view[3].xyz;
    let view_dir = normalize(world - cam);

    let height_fade = exp(-p.medium.z * max(world.y - p.medium.y, 0.0));
    let sigma = p.medium.x * height_fade;

    // Radiance scattered toward the camera per unit of optical depth.
    var radiance = p.ambient.rgb;
    let g = p.medium.w;
    let cos_t = dot(view_dir, p.sun_dir.xyz);
    // Henyey-Greenstein, scaled so that g = 0 gives 1.
    let phase = (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * cos_t, 1e-4), 1.5);
    if (p.sun_color.w > 0.0) {
        radiance += p.sun_color.rgb * (phase * sun_visibility(world, view_depth));
    }
    let n_lights = u32(p.lin.w);
    for (var i = 0u; i < n_lights; i = i + 1u) {
        let l = p.lights[i];
        let to_light = l.pos_range.xyz - world;
        let d2 = dot(to_light, to_light);
        let range = l.pos_range.w;
        if (d2 < range * range) {
            // The engine's own window to zero at the range, times an inverse
            // square softened at 0.4 m (about what a cell can resolve): a
            // bright core and a halo that is mostly gone by three metres.
            let window = 1.0 - sqrt(d2) / range;
            radiance += l.color.rgb * (window * window / (0.15 + d2));
        }
    }
    var value = vec4<f32>(radiance, sigma);

    // Last frame's value at this cell centre's world position.
    let centre = (vec3<f32>(gid) + vec3<f32>(0.5)) / p.grid.xyz;
    let centre_world = (p.inv_view * vec4<f32>(view_ray(centre.xy) * slice_depth(centre.z), 1.0)).xyz;
    let prev_clip = p.prev_view_proj * vec4<f32>(centre_world, 1.0);
    if (p.depth.w > 0.0 && prev_clip.w > p.depth.x) {
        let prev_ndc = prev_clip.xy / prev_clip.w;
        let prev = vec3<f32>(
            prev_ndc.x * 0.5 + 0.5,
            0.5 - prev_ndc.y * 0.5,
            log(prev_clip.w / p.depth.x) / p.depth.z,
        );
        if (all(prev >= vec3<f32>(0.0)) && all(prev <= vec3<f32>(1.0))) {
            let history = textureSampleLevel(history_tex, lin_samp, prev, 0.0);
            value = mix(value, history, p.depth.w);
        }
    }
    textureStore(out_tex, vec3<i32>(gid), value);
}
";

const INTEGRATE_WGSL: &str = "
@group(0) @binding(1) var scatter_tex: texture_3d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid = vec3<u32>(p.grid.xyz);
    if (gid.x >= grid.x || gid.y >= grid.y) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / p.grid.xy;
    // Metres travelled along this column's ray per metre of view depth.
    let stretch = length(view_ray(uv));
    var in_scatter = vec3<f32>(0.0);
    var transmittance = 1.0;
    var z0 = p.depth.x;
    for (var k = 0u; k < grid.z; k = k + 1u) {
        let z1 = slice_depth(f32(k + 1u) / p.grid.z);
        let s = textureLoad(scatter_tex, vec3<i32>(vec2<i32>(gid.xy), i32(k)), 0);
        let slice_t = exp(-s.a * (z1 - z0) * stretch);
        // Exact for a slice of constant radiance and density.
        in_scatter += s.rgb * (transmittance * (1.0 - slice_t));
        transmittance *= slice_t;
        textureStore(out_tex, vec3<i32>(vec2<i32>(gid.xy), i32(k)), vec4<f32>(in_scatter, transmittance));
        z0 = z1;
    }
}
";

const APPLY_WGSL: &str = "
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var volume_tex: texture_3d<f32>;
@group(0) @binding(3) var lin_samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vid & 1u) * 4u) - 1.0;
    let y = f32((vid >> 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let depth = textureLoad(depth_tex, vec2<i32>(frag.xy), 0);
    // The sky and anything past the range get the whole column.
    var view_depth = p.depth.y;
    if (depth < 0.9999) {
        view_depth = clamp(p.lin.y / (depth + p.lin.x), p.depth.x, p.depth.y);
    }
    // Slices as cells: texel k holds the column up to the FAR side of slice k.
    let cells = log(view_depth / p.depth.x) / p.depth.z * p.grid.z;
    // Half a slice of per-pixel, per-frame noise turns the slice steps into
    // grain the temporal pass averages away.
    let noise = ign(frag.xy + vec2<f32>(p.lin.z * 5.588238, p.lin.z * 3.14159)) - 0.5;
    let at = max(cells + noise * 0.75, 0.0);
    let v = textureSampleLevel(volume_tex, lin_samp, vec3<f32>(frag.xy / dims, (at - 0.5) / p.grid.z), 0.0);
    // Inside the first slice, fade from nothing at the camera.
    let first = clamp(at, 0.0, 1.0);
    // Blended: rgb = in-scatter + dst * transmittance; dst alpha is kept.
    return vec4<f32>(v.rgb * first, mix(1.0, v.a, first));
}
";

struct Gpu {
    params: wgpu::Buffer,
    scatter: LazyComputePipeline,
    integrate: LazyComputePipeline,
    apply: LazyRenderPipeline,
    apply_layout: wgpu::BindGroupLayout,
    /// `scatter_bgs[i]` writes volume `i` and reads volume `1 - i`.
    scatter_bgs: [wgpu::BindGroup; 2],
    /// `integrate_bgs[i]` reads volume `i`.
    integrate_bgs: [wgpu::BindGroup; 2],
    integrated_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// Holds the scene depth view: dropped on resize.
    apply_bg: Option<wgpu::BindGroup>,
}

pub(super) struct VolumetricFog {
    pub(super) settings: Settings,
    gpu: Option<Gpu>,
    /// Unjittered projection times view of the last frame the grid was lit.
    prev_view_proj: [[f32; 4]; 4],
    /// The range the history was built with; a change invalidates it.
    history_range: f32,
    history_valid: bool,
    frame: u32,
}

impl VolumetricFog {
    pub(super) fn new() -> Self {
        Self {
            settings: Settings::default(),
            gpu: None,
            prev_view_proj: [[0.0; 4]; 4],
            history_range: 0.0,
            history_valid: false,
            frame: 0,
        }
    }

    /// The render targets were recreated.
    pub(super) fn drop_target_bindings(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.apply_bg = None;
        }
    }
}

/// Desktop only. The grid needs compute and a writable 16-bit float volume;
/// neither is in doubt there, and no phone or browser has been measured.
fn platform_supported() -> bool {
    cfg!(not(any(
        target_arch = "wasm32",
        target_os = "android",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos"
    )))
}

fn uniform_entry(visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn volume_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: VOLUME_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D3,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

impl Gpu {
    fn new(device: &wgpu::Device, shadow_map: &crate::shadows::ShadowMap) -> Self {
        let volume = |label: &str| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: GRID_X,
                        height: GRID_Y,
                        depth_or_array_layers: GRID_Z,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: VOLUME_FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let scatter_views = [volume("vfog_scatter_0"), volume("vfog_scatter_1")];
        let integrated_view = volume("vfog_integrated");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vfog_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vfog_params"),
            size: std::mem::size_of::<FogParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = |label: &str, body: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}{body}").into()),
            })
        };
        let compute = wgpu::ShaderStages::COMPUTE;

        // --- scatter ---
        let scatter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfog_scatter_layout"),
            entries: &[
                uniform_entry(compute),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: compute,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                volume_entry(3, compute),
                sampler_entry(4, compute),
                storage_entry(5),
            ],
        });
        let scatter_module = module("vfog_scatter_shader", SCATTER_WGSL);
        let scatter_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vfog_scatter_pl"),
            bind_group_layouts: &[Some(&scatter_layout)],
            ..Default::default()
        });
        let scatter = LazyComputePipeline::new(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("vfog_scatter_pipeline"),
                layout: Some(&scatter_pl),
                module: &scatter_module,
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        let scatter_bg = |write: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vfog_scatter_bg"),
                layout: &scatter_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&shadow_map.depth_array_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&shadow_map.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&scatter_views[1 - write]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&scatter_views[write]),
                    },
                ],
            })
        };
        let scatter_bgs = [scatter_bg(0), scatter_bg(1)];

        // --- integrate ---
        let integrate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfog_integrate_layout"),
            entries: &[
                uniform_entry(compute),
                volume_entry(1, compute),
                storage_entry(2),
            ],
        });
        let integrate_module = module("vfog_integrate_shader", INTEGRATE_WGSL);
        let integrate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vfog_integrate_pl"),
            bind_group_layouts: &[Some(&integrate_layout)],
            ..Default::default()
        });
        let integrate = LazyComputePipeline::new(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("vfog_integrate_pipeline"),
                layout: Some(&integrate_pl),
                module: &integrate_module,
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        let integrate_bg = |read: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vfog_integrate_bg"),
                layout: &integrate_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&scatter_views[read]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&integrated_view),
                    },
                ],
            })
        };
        let integrate_bgs = [integrate_bg(0), integrate_bg(1)];

        // --- apply ---
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let apply_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfog_apply_layout"),
            entries: &[
                uniform_entry(fragment),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: fragment,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                volume_entry(2, fragment),
                sampler_entry(3, fragment),
            ],
        });
        let apply_module = module("vfog_apply_shader", APPLY_WGSL);
        let apply_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vfog_apply_pl"),
            bind_group_layouts: &[Some(&apply_layout)],
            ..Default::default()
        });
        let apply = LazyRenderPipeline::new(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("vfog_apply_pipeline"),
                layout: Some(&apply_pl),
                vertex: wgpu::VertexState {
                    module: &apply_module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &apply_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        // colour = in-scatter + colour * transmittance. The
                        // target's alpha carries the scene's indirect-light
                        // share for the final composite: leave it alone.
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::SrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        Self {
            params,
            scatter,
            integrate,
            apply,
            apply_layout,
            scatter_bgs,
            integrate_bgs,
            integrated_view,
            sampler,
            apply_bg: None,
        }
    }
}

/// Inverse of a rigid (rotation + translation) column-major matrix.
fn rigid_inverse(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let t = [m[3][0], m[3][1], m[3][2]];
    let mut out = [[0.0f32; 4]; 4];
    for c in 0..3 {
        for r in 0..3 {
            out[c][r] = m[r][c];
        }
    }
    for r in 0..3 {
        out[3][r] = -(m[r][0] * t[0] + m[r][1] * t[1] + m[r][2] * t[2]);
    }
    out[3][3] = 1.0;
    out
}

/// The point lights worth lighting the grid with: those whose reach comes
/// within `range` of the camera, nearest first, at most `MAX_LIGHTS`.
/// `lights` are (position + range, colour + intensity) as the lighting
/// uniforms hold them.
fn pick_lights(
    lights: &[([f32; 4], [f32; 4])],
    cam: [f32; 3],
    range: f32,
    scale: f32,
) -> ([FogLight; MAX_LIGHTS], usize) {
    let mut near: Vec<(f32, FogLight)> = lights
        .iter()
        .filter_map(|(pos, color)| {
            let d =
                ((pos[0] - cam[0]).powi(2) + (pos[1] - cam[1]).powi(2) + (pos[2] - cam[2]).powi(2))
                    .sqrt();
            let k = color[3] * scale;
            (pos[3] > 0.0 && k > 0.0 && d - pos[3] < range).then_some((
                d,
                FogLight {
                    pos_range: *pos,
                    color: [color[0] * k, color[1] * k, color[2] * k, 0.0],
                },
            ))
        })
        .collect();
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = [FogLight {
        pos_range: [0.0; 4],
        color: [0.0; 4],
    }; MAX_LIGHTS];
    let n = near.len().min(MAX_LIGHTS);
    for (slot, (_, light)) in out.iter_mut().zip(near) {
        *slot = light;
    }
    (out, n)
}

impl Renderer {
    /// Switch the lit fog on or off and describe the medium: `density` is the
    /// extinction per metre at and below `height_ref`, thinning by
    /// `height_falloff` per metre above it (0 = uniform), out to `range`
    /// metres from the camera. Off by default. Ignored on mobile and web.
    pub fn set_volumetric_fog(
        &mut self,
        enabled: bool,
        density: f32,
        height_ref: f32,
        height_falloff: f32,
        range: f32,
    ) {
        let s = &mut self.volumetric_fog.settings;
        s.enabled = enabled;
        s.density = if density.is_finite() {
            density.max(0.0)
        } else {
            0.0
        };
        s.height_ref = if height_ref.is_finite() {
            height_ref
        } else {
            0.0
        };
        s.height_falloff = if height_falloff.is_finite() {
            height_falloff.max(0.0)
        } else {
            0.0
        };
        s.range = if range.is_finite() {
            range.clamp(8.0, 400.0)
        } else {
            80.0
        };
    }

    /// How the lit fog is lit: its colour where no light reaches (linear
    /// radiance, the same units as the height fog's colour), how strongly the
    /// sun and the point lights scatter in it (1 = an isotropic medium), and
    /// how much of the sun's share is thrown forward (0 to 0.9).
    pub fn set_volumetric_fog_light(
        &mut self,
        ambient: [f32; 3],
        sun_strength: f32,
        light_strength: f32,
        anisotropy: f32,
    ) {
        let clean = |v: f32, hi: f32| if v.is_finite() { v.clamp(0.0, hi) } else { 0.0 };
        let s = &mut self.volumetric_fog.settings;
        s.ambient = [
            clean(ambient[0], 100.0),
            clean(ambient[1], 100.0),
            clean(ambient[2], 100.0),
        ];
        s.sun_strength = clean(sun_strength, 100.0);
        s.light_strength = clean(light_strength, 100.0);
        s.anisotropy = clean(anisotropy, 0.9);
    }

    /// Light the grid and blend it into the composed HDR image. Called right
    /// after the scene compose, so the shadow cascades of this frame are
    /// fitted and everything (transparent surfaces included) is in the image.
    pub(super) fn record_volumetric_fog(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
    ) {
        let s = self.volumetric_fog.settings;
        let perspective = self.current_proj_matrix[2][3] < -0.5;
        if !s.enabled
            || s.density <= 0.0
            || !perspective
            || !platform_supported()
            || self.pt_owns_frame()
            || self.dbg_skip("volumetric_fog")
        {
            self.volumetric_fog.history_valid = false;
            return;
        }
        profiler.begin("volumetric_fog");
        if self.volumetric_fog.gpu.is_none() {
            self.volumetric_fog.gpu = Some(Gpu::new(&self.device, &self.shadow_map));
        }
        let view = self.current_view_matrix;
        let proj = self.current_proj_matrix_unjittered;
        let inv_view = rigid_inverse(&view);
        let cam = [inv_view[3][0], inv_view[3][1], inv_view[3][2]];
        let far = s.range.max(NEAR * 4.0);

        let fog = &mut self.volumetric_fog;
        let history_ok = fog.history_valid && fog.history_range == far;
        fog.frame = (fog.frame + 1) % 1024;
        let write = (fog.frame & 1) as usize;

        let lu = &self.lighting_uniforms;
        let n_point = (lu.point_light_count[0].max(0.0) as usize).min(lu.point_lights.len());
        let all: Vec<([f32; 4], [f32; 4])> = lu.point_lights[..n_point]
            .iter()
            .map(|l| (l.position, l.color))
            .collect();
        let per_sr = 1.0 / (4.0 * std::f32::consts::PI);
        let (lights, n_lights) = pick_lights(&all, cam, far, s.light_strength * per_sr);
        let dir = [lu.light_dir[0], lu.light_dir[1], lu.light_dir[2]];
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let sun_k = if len > 1e-6 {
            (lu.light_dir[3] * s.sun_strength * per_sr).max(0.0)
        } else {
            0.0
        };
        let inv_len = if len > 1e-6 { 1.0 / len } else { 0.0 };
        let splits = self.shadow_map.cascade_splits;
        let params = FogParams {
            inv_view,
            prev_view_proj: fog.prev_view_proj,
            cascade_vps: self.shadow_map.light_vps,
            cascade_splits: [
                splits[0],
                splits[1],
                splits[2],
                if self.shadow_map.enabled { 1.0 } else { 0.0 },
            ],
            proj: [proj[0][0], proj[1][1], proj[2][0], proj[2][1]],
            depth: [
                NEAR,
                far,
                (far / NEAR).ln(),
                if history_ok { HISTORY_WEIGHT } else { 0.0 },
            ],
            lin: [
                self.current_proj_matrix[2][2],
                self.current_proj_matrix[3][2],
                fog.frame as f32,
                n_lights as f32,
            ],
            grid: [GRID_X as f32, GRID_Y as f32, GRID_Z as f32, 0.0],
            medium: [s.density, s.height_ref, s.height_falloff, s.anisotropy],
            ambient: [s.ambient[0], s.ambient[1], s.ambient[2], 0.0],
            sun_dir: [dir[0] * inv_len, dir[1] * inv_len, dir[2] * inv_len, 0.0],
            sun_color: [
                lu.light_color[0] * sun_k,
                lu.light_color[1] * sun_k,
                lu.light_color[2] * sun_k,
                sun_k,
            ],
            lights,
        };
        fog.prev_view_proj = super::util::mat4_multiply(proj, view);
        fog.history_range = far;
        fog.history_valid = true;

        let gpu = fog.gpu.as_mut().expect("created above");
        self.queue
            .write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        if gpu.apply_bg.is_none() {
            gpu.apply_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vfog_apply_bg"),
                layout: &gpu.apply_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gpu.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&gpu.integrated_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&gpu.sampler),
                    },
                ],
            }));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vfog_scatter"),
                timestamp_writes: profiler.compute_pass_timestamp_writes("vfog_scatter"),
            });
            pass.set_pipeline(&gpu.scatter);
            pass.set_bind_group(0, &gpu.scatter_bgs[write], &[]);
            pass.dispatch_workgroups(GRID_X.div_ceil(4), GRID_Y.div_ceil(4), GRID_Z.div_ceil(4));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vfog_integrate"),
                timestamp_writes: profiler.compute_pass_timestamp_writes("vfog_integrate"),
            });
            pass.set_pipeline(&gpu.integrate);
            pass.set_bind_group(0, &gpu.integrate_bgs[write], &[]);
            pass.dispatch_workgroups(GRID_X.div_ceil(8), GRID_Y.div_ceil(8), 1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vfog_apply"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.composed_rt_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: profiler.pass_timestamp_writes("vfog_apply"),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&gpu.apply);
            pass.set_bind_group(0, gpu.apply_bg.as_ref().expect("built above"), &[]);
            pass.draw(0..3, 0..1);
        }
        profiler.end("volumetric_fog");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_match_the_wgsl_layout() {
        // 5 mat4 + 9 vec4 + 8 lights of 2 vec4.
        assert_eq!(std::mem::size_of::<FogParams>(), 5 * 64 + 9 * 16 + 8 * 32);
    }

    #[test]
    fn shaders_parse_and_validate() {
        for (name, body) in [
            ("scatter", SCATTER_WGSL),
            ("integrate", INTEGRATE_WGSL),
            ("apply", APPLY_WGSL),
        ] {
            let src = format!("{COMMON_WGSL}{body}");
            let module = wgpu::naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    #[test]
    fn rigid_inverse_undoes_a_look_at() {
        let view =
            crate::renderer::util::mat4_look_at([3.0, 2.0, -5.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);
        let inv = rigid_inverse(&view);
        let id = crate::renderer::util::mat4_multiply(view, inv);
        for c in 0..4 {
            for r in 0..4 {
                let want = if c == r { 1.0 } else { 0.0 };
                assert!((id[c][r] - want).abs() < 1e-4, "[{c}][{r}] = {}", id[c][r]);
            }
        }
        // The camera position is the inverse's translation.
        assert!((inv[3][0] - 3.0).abs() < 1e-4 && (inv[3][2] + 5.0).abs() < 1e-4);
    }

    #[test]
    fn lights_are_culled_by_reach_and_sorted_nearest_first() {
        let l = |x: f32, range: f32, i: f32| ([x, 0.0, 0.0, range], [1.0, 1.0, 1.0, i]);
        let lights = [
            l(50.0, 10.0, 1.0),
            l(5.0, 10.0, 2.0),
            l(500.0, 10.0, 1.0),
            l(20.0, 10.0, 0.0),
        ];
        let (picked, n) = pick_lights(&lights, [0.0; 3], 80.0, 0.5);
        assert_eq!(n, 2);
        assert_eq!(picked[0].pos_range[0], 5.0);
        assert_eq!(picked[0].color[0], 1.0);
        assert_eq!(picked[1].pos_range[0], 50.0);
        let many: Vec<_> = (0..20).map(|i| l(i as f32, 5.0, 1.0)).collect();
        assert_eq!(pick_lights(&many, [0.0; 3], 80.0, 1.0).1, MAX_LIGHTS);
    }
}
