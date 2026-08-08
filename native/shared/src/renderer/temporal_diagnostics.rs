//! Capture-only TAA/TSR diagnostics.
//!
//! Resources exist only for a requested qualification capture. The production
//! shader and frame path remain unchanged, and all diagnostic GPU objects are
//! released after readback.

use super::*;

pub(super) const TAA_DIAGNOSTIC_NAMES: [&str; 4] = [
    "taa-rejection-reason",
    "taa-motion",
    "taa-reprojected-uv",
    "taa-temporal-confidence",
];
pub(super) const TAA_DIAGNOSTIC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub(super) struct TaaDiagnosticResources {
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    pipeline: wgpu::RenderPipeline,
    reactive_pipeline: Option<wgpu::RenderPipeline>,
    width: u32,
    height: u32,
}

fn diagnostic_shader_source(reactive: bool) -> String {
    let source = if reactive {
        super::temporal_reactive::taa_reactive_shader_source()
    } else {
        TAA_SHADER_WGSL.to_owned()
    };
    let signature = "@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {";
    let diagnostic_signature = r#"struct TaaDiagnosticOut {
    @location(0) rejection_reason: vec4<f32>,
    @location(1) motion: vec4<f32>,
    @location(2) reprojected_uv: vec4<f32>,
    @location(3) temporal_confidence: vec4<f32>,
};

@fragment
fn fs_diagnostics(in: VsOut) -> TaaDiagnosticOut {"#;
    let source = source.replacen(signature, diagnostic_signature, 1);
    assert!(
        source.contains("fn fs_diagnostics"),
        "TAA entry point changed; diagnostics must follow it"
    );

    let final_return = "    return vec4<f32>(blended, blended_w);";
    let reactive_value = if reactive { "reactive" } else { "0.0" };
    let diagnostic_return = format!(
        r#"    let history_in_bounds =
        prev_uv.x >= 0.0 && prev_uv.x <= 1.0 &&
        prev_uv.y >= 0.0 && prev_uv.y <= 1.0;
    let clamped_ycocg = vec3<f32>(history_y_clamped, co_clamped, cg_clamped);
    let clamp_delta = length(history_ycocg - clamped_ycocg);
    let variance_heat = 1.0 - exp(-stddev.x * 4.0);
    let clamp_heat = 1.0 - exp(-clamp_delta * 4.0);
    let history_confidence = select(
        0.0, clamp(1.0 - alpha, 0.0, 1.0), history_in_bounds);
    let reactive_weight = {reactive_value};

    // Categorical dominant reason: gray seed, red off-screen, cyan reactive,
    // magenta disocclusion, yellow neighborhood clamp, blue motion, green keep.
    var reason = vec3<f32>(0.05, 0.65, 0.10);
    if (u.params.x >= 0.999) {{
        reason = vec3<f32>(0.25);
    }} else if (!history_in_bounds) {{
        reason = vec3<f32>(1.0, 0.05, 0.02);
    }} else if (reactive_weight > 0.01 &&
               reactive_weight >= max(disocclusion, motion_ramped)) {{
        reason = vec3<f32>(0.0, 0.9, 1.0);
    }} else if (disocclusion > 0.01 && disocclusion >= motion_ramped) {{
        reason = vec3<f32>(1.0, 0.0, 0.8);
    }} else if (clamp_delta > 0.0001) {{
        reason = vec3<f32>(1.0, 0.75, 0.0);
    }} else if (motion_alpha > 0.01) {{
        reason = vec3<f32>(0.05, 0.25, 1.0);
    }}

    // Motion RG is signed around 0.5; B is magnitude. Reprojection RG stores
    // previous-frame UV and B is its validity. Confidence RGB stores local
    // luma variance, clamp magnitude, and retained-history contribution.
    let motion_debug = vec3<f32>(
        clamp(0.5 + vel.x * 32.0, 0.0, 1.0),
        clamp(0.5 - vel.y * 32.0, 0.0, 1.0),
        clamp(vel_len * 64.0, 0.0, 1.0),
    );
    return TaaDiagnosticOut(
        vec4<f32>(reason, 1.0),
        vec4<f32>(motion_debug, 1.0),
        vec4<f32>(clamp(prev_uv, vec2<f32>(0.0), vec2<f32>(1.0)),
                  select(0.0, 1.0, history_in_bounds), 1.0),
        vec4<f32>(variance_heat, clamp_heat, history_confidence, 1.0),
    );"#
    );
    let source = source.replacen(final_return, &diagnostic_return, 1);
    assert!(
        source.contains("return TaaDiagnosticOut"),
        "TAA final output changed; diagnostics must follow it"
    );
    source
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    reactive: bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("taa_diagnostic_shader"),
        source: wgpu::ShaderSource::Wgsl(diagnostic_shader_source(reactive).into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("taa_diagnostic_pipeline_layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let targets = (0..TAA_DIAGNOSTIC_NAMES.len())
        .map(|_| {
            Some(wgpu::ColorTargetState {
                format: TAA_DIAGNOSTIC_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect::<Vec<_>>();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("taa_diagnostic_pipeline"),
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
    })
}

fn create_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>) {
    let textures = TAA_DIAGNOSTIC_NAMES
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
                format: TAA_DIAGNOSTIC_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        })
        .collect::<Vec<_>>();
    let views = textures
        .iter()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
        .collect();
    (textures, views)
}

impl TaaDiagnosticResources {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, width: u32, height: u32) -> Self {
        let (textures, views) = create_targets(device, width, height);
        Self {
            textures,
            views,
            pipeline: create_pipeline(device, layout, false),
            reactive_pipeline: None,
            width,
            height,
        }
    }
}

impl Renderer {
    pub(super) fn record_taa_diagnostics(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        reactive: bool,
    ) {
        let (width, height) = (self.surface_config.width, self.surface_config.height);
        let resize = self
            .temporal_diagnostics
            .as_ref()
            .is_some_and(|resources| resources.width != width || resources.height != height);
        if resize {
            self.temporal_diagnostics = None;
        }
        if self.temporal_diagnostics.is_none() {
            self.temporal_diagnostics = Some(TaaDiagnosticResources::new(
                &self.device,
                &self.taa_layout,
                width,
                height,
            ));
        }
        if reactive
            && self
                .temporal_diagnostics
                .as_ref()
                .is_some_and(|resources| resources.reactive_pipeline.is_none())
        {
            let layout = self
                .taa_reactive_layout
                .as_ref()
                .expect("reactive diagnostics require the reactive TAA layout");
            let pipeline = create_pipeline(&self.device, layout, true);
            self.temporal_diagnostics
                .as_mut()
                .unwrap()
                .reactive_pipeline = Some(pipeline);
        }

        let resources = self.temporal_diagnostics.as_ref().unwrap();
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
            label: Some("taa_diagnostic_pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(if reactive {
            resources.reactive_pipeline.as_ref().unwrap()
        } else {
            &resources.pipeline
        });
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub(super) fn taa_diagnostic_textures(&self) -> Option<&[wgpu::Texture]> {
        self.temporal_diagnostics
            .as_ref()
            .map(|resources| resources.textures.as_slice())
    }

    pub(super) fn release_temporal_diagnostics(&mut self) {
        self.temporal_diagnostics = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_variants_parse_without_modifying_production_taa() {
        for reactive in [false, true] {
            wgpu::naga::front::wgsl::parse_str(&diagnostic_shader_source(reactive))
                .unwrap_or_else(|error| panic!("TAA diagnostics WGSL failed: {error}"));
        }
        assert!(!TAA_SHADER_WGSL.contains("TaaDiagnosticOut"));
        assert!(!TAA_SHADER_WGSL.contains("fs_diagnostics"));
    }
}
