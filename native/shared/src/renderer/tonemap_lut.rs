//! The AgX display curve as a 3D table, so the composite pass reads it with
//! one filtered lookup instead of evaluating it per pixel (and up to four times
//! per pixel: centre, sharpen average, chromatic aberration R and B).
//!
//! Baked once on the GPU from the same WGSL curve (`agx_curve()` in
//! `shaders/tonemap.wgsl`); its indexing is in `shaders/tonemap_lut.wgsl`,
//! shared by the bake and the composite. The ACES curves are cheap and stay
//! analytic.

use super::shaders::TONEMAP_LUT_BAKE_WGSL;

/// Texels per axis; `TONEMAP_LUT_SIZE` in `shaders/tonemap_lut.wgsl`.
pub(super) const TONEMAP_LUT_SIZE: u32 = 64;
const TONEMAP_LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub(super) struct TonemapLut {
    _texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
}

impl TonemapLut {
    /// Create and bake the table (submitted on `queue`).
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tonemap_lut"),
            size: wgpu::Extent3d {
                width: TONEMAP_LUT_SIZE,
                height: TONEMAP_LUT_SIZE,
                depth_or_array_layers: TONEMAP_LUT_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: TONEMAP_LUT_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap_lut_bake"),
            source: wgpu::ShaderSource::Wgsl(TONEMAP_LUT_BAKE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap_lut_bake_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: TONEMAP_LUT_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap_lut_bake_pl_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tonemap_lut_bake"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("bake"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap_lut_bake_bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tonemap_lut_bake"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tonemap_lut_bake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = TONEMAP_LUT_SIZE.div_ceil(4);
            pass.dispatch_workgroups(groups, groups, groups);
        }
        queue.submit([encoder.finish()]);
        Self {
            _texture: texture,
            view,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_table_size_matches() {
        let wgsl = include_str!("shaders/tonemap_lut.wgsl");
        assert!(wgsl.contains(&format!(
            "const TONEMAP_LUT_SIZE: f32 = {TONEMAP_LUT_SIZE}.0;"
        )));
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tonemap-lut-test-device"),
            ..Default::default()
        }))
        .ok()
    }

    /// Display-encoded (sRGB) difference between the lookup the composite pass
    /// does and agx_curve(), over HDR colours from 2^-14 to 2^10 in every
    /// combination of channels. Skipped without a GPU.
    #[test]
    fn table_matches_agx_curve() {
        let Some((device, queue)) = device() else {
            return;
        };
        const N: u32 = 64;
        let check = format!(
            "{}{}
@group(0) @binding(0) var lut_tex: texture_3d<f32>;
@group(0) @binding(1) var lut_samp: sampler;
@group(0) @binding(2) var<storage, read_write> worst: array<atomic<u32>, 2>;

fn srgb(c: vec3<f32>) -> vec3<f32> {{
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}}

@compute @workgroup_size(4, 4, 4)
fn check(@builtin(global_invocation_id) id: vec3<u32>) {{
    let hdr = exp2(mix(vec3<f32>(-14.0), vec3<f32>(10.0), (vec3<f32>(id) + 0.37) / {N}.0));
    let exact = srgb(agx_curve(hdr));
    let table = srgb(textureSampleLevel(lut_tex, lut_samp, tonemap_lut_uvw(hdr), 0.0).rgb);
    let d = abs(exact - table);
    let e = u32(max(d.x, max(d.y, d.z)) * 1000000.0);
    atomicMax(&worst[0], e);
    if (all(hdr > vec3<f32>(0.01)) && all(hdr < vec3<f32>(4.0))) {{
        atomicMax(&worst[1], e);
    }}
}}
",
            include_str!("shaders/tonemap_lut.wgsl"),
            include_str!("shaders/tonemap.wgsl"),
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap_lut_check"),
            source: wgpu::ShaderSource::Wgsl(check.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tonemap_lut_check"),
            layout: None,
            module: &module,
            entry_point: Some("check"),
            compilation_options: Default::default(),
            cache: None,
        });
        // The composite's sampler: linear, clamped.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let lut = TonemapLut::new(&device, &queue);
        let worst = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&lut.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: worst.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(N / 4, N / 4, N / 4);
        }
        encoder.copy_buffer_to_buffer(&worst, 0, &readback, 0, 8);
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let v: Vec<u32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec();
        let (all, mid) = (v[0] as f32 / 1e6, v[1] as f32 / 1e6);
        println!(
            "AgX table: worst {:.2}/255, mid-tones {:.2}/255",
            all * 255.0,
            mid * 255.0
        );
        assert!(all < 2.0 / 255.0, "worst {all}");
        assert!(mid < 1.0 / 255.0, "mid-tones {mid}");
    }
}
