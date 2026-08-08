//! Capture-only SSR temporal diagnostics.
//!
//! The diagnostic shader is derived from the production temporal shader and
//! reuses its bind group after the real SSR history has been written. Normal
//! frames compile no pipeline, allocate no target, and execute no pass.

use super::*;

pub(super) const SSR_TEMPORAL_DIAGNOSTIC_NAMES: [&str; 2] =
    ["ssr-rejection-reason", "ssr-temporal-confidence"];
const DIAGNOSTIC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub(super) struct SsrTemporalDiagnosticResources {
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    pipeline: wgpu::RenderPipeline,
    width: u32,
    height: u32,
}

fn diagnostic_shader_source() -> String {
    let signature = "@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {";
    let diagnostic_signature = r#"struct SsrTemporalDiagnosticOut {
    @location(0) rejection_reason: vec4<f32>,
    @location(1) temporal_confidence: vec4<f32>,
};

@fragment
fn fs_diagnostics(in: VsOut) -> SsrTemporalDiagnosticOut {"#;
    let mut source = SSR_TEMPORAL_SHADER_WGSL.replacen(signature, diagnostic_signature, 1);
    assert!(
        source.contains("fn fs_diagnostics"),
        "SSR temporal entry point changed; diagnostics must follow it"
    );
    source = source.replacen(
        "    if (off_screen) { return current; }",
        "    // Diagnostics continue with a clamped lookup so they can classify\n\
         // the off-screen rejection without sampling outside the history.",
        1,
    );
    source = source.replacen(
        "textureSampleLevel(history_tex, history_samp, prev_uv, 0.0)",
        "textureSampleLevel(\n\
             history_tex,\n\
             history_samp,\n\
             clamp(prev_uv, vec2<f32>(0.0), vec2<f32>(1.0)),\n\
             0.0,\n\
         )",
        1,
    );
    let final_return = "    return select(current, blended, blended == blended);";
    let diagnostic_return = r#"    let history_finite = all(history_raw == history_raw);
    let clamp_delta = length(history_raw.rgb - clamped_history.rgb);
    let local_luma_range = abs(dot(
        nmax.rgb - nmin.rgb,
        vec3<f32>(0.2126, 0.7152, 0.0722),
    ));
    let variation_heat = 1.0 - exp(-local_luma_range * 4.0);
    let clamp_heat = select(
        1.0,
        1.0 - exp(-clamp_delta * 4.0),
        history_finite,
    );
    let history_in_bounds = !off_screen;
    let history_confidence = select(
        0.0,
        clamp(1.0 - u.params.x, 0.0, 1.0),
        history_in_bounds && history_finite,
    );

    // Shared temporal palette: gray seed, red off-screen, magenta invalid
    // history, yellow neighborhood clamp, and green accepted history.
    var reason = vec3<f32>(0.05, 0.65, 0.10);
    if (u.params.x >= 0.999) {
        reason = vec3<f32>(0.25);
    } else if (!history_in_bounds) {
        reason = vec3<f32>(1.0, 0.05, 0.02);
    } else if (!history_finite) {
        reason = vec3<f32>(1.0, 0.0, 0.8);
    } else if (clamp_delta > 0.0001) {
        reason = vec3<f32>(1.0, 0.75, 0.0);
    }
    return SsrTemporalDiagnosticOut(
        vec4<f32>(reason, 1.0),
        vec4<f32>(
            variation_heat,
            clamp_heat,
            history_confidence,
            1.0,
        ),
    );"#;
    source = source.replacen(final_return, diagnostic_return, 1);
    assert!(
        source.contains("return SsrTemporalDiagnosticOut"),
        "SSR temporal final output changed; diagnostics must follow it"
    );
    source
}

impl SsrTemporalDiagnosticResources {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, width: u32, height: u32) -> Self {
        let textures = SSR_TEMPORAL_DIAGNOSTIC_NAMES
            .iter()
            .map(|name| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(name),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: DIAGNOSTIC_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
            })
            .collect::<Vec<_>>();
        let views = textures
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect::<Vec<_>>();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr_temporal_diagnostic_shader"),
            source: wgpu::ShaderSource::Wgsl(diagnostic_shader_source().into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssr_temporal_diagnostic_pipeline_layout"),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
        let targets = SSR_TEMPORAL_DIAGNOSTIC_NAMES
            .iter()
            .map(|_| {
                Some(wgpu::ColorTargetState {
                    format: DIAGNOSTIC_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect::<Vec<_>>();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssr_temporal_diagnostic_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_diagnostics"),
                targets: &targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            textures,
            views,
            pipeline,
            width,
            height,
        }
    }
}

impl Renderer {
    pub(super) fn record_ssr_temporal_diagnostics(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
    ) {
        let size = self.ssr_rt_texture.size();
        let resize = self
            .ssr_temporal_diagnostics
            .as_ref()
            .is_some_and(|resources| {
                resources.width != size.width || resources.height != size.height
            });
        if resize {
            self.ssr_temporal_diagnostics = None;
        }
        if self.ssr_temporal_diagnostics.is_none() {
            self.ssr_temporal_diagnostics = Some(SsrTemporalDiagnosticResources::new(
                &self.device,
                &self.ssr_temporal_layout,
                size.width,
                size.height,
            ));
        }
        let resources = self.ssr_temporal_diagnostics.as_ref().unwrap();
        let attachments = resources
            .views
            .iter()
            .map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect::<Vec<_>>();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssr_temporal_diagnostic_pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&resources.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub(super) fn ssr_temporal_diagnostic_textures(&self) -> Option<&[wgpu::Texture]> {
        self.ssr_temporal_diagnostics
            .as_ref()
            .map(|resources| resources.textures.as_slice())
    }

    pub(super) fn release_ssr_temporal_diagnostics(&mut self) {
        self.ssr_temporal_diagnostics = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_shader_parses_without_modifying_production_ssr() {
        wgpu::naga::front::wgsl::parse_str(&diagnostic_shader_source())
            .unwrap_or_else(|error| panic!("SSR temporal diagnostics WGSL failed: {error}"));
        assert!(!SSR_TEMPORAL_SHADER_WGSL.contains("SsrTemporalDiagnosticOut"));
        assert!(!SSR_TEMPORAL_SHADER_WGSL.contains("fs_diagnostics"));
    }
}
