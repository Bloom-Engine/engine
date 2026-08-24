//! Capture-only TAA/TSR diagnostics.
//!
//! Resources exist only for a requested qualification capture. The production
//! shader and frame path remain unchanged, and all diagnostic GPU objects are
//! released after readback.

use super::*;

pub(super) const TAA_DIAGNOSTIC_NAMES: [&str; 7] = [
    "taa-rejection-reason",
    "taa-motion",
    "taa-reprojected-uv",
    "taa-temporal-confidence",
    "taa-reactive-history",
    "taa-history-policy",
    "taa-reconstruction-footprint",
];
pub(super) const TAA_DIAGNOSTIC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const TAA_DIAGNOSTIC_BATCHES: [(usize, usize); 2] = [(0, 4), (4, 7)];

pub(super) struct TaaDiagnosticResources {
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    pipelines: Vec<wgpu::RenderPipeline>,
    reactive_pipelines: Option<Vec<wgpu::RenderPipeline>>,
    width: u32,
    height: u32,
}

fn diagnostic_shader_source(reactive: bool, batch: usize) -> String {
    let source = if reactive {
        super::temporal_reactive::taa_reactive_shader_source()
    } else {
        TAA_SHADER_WGSL.to_owned()
    };
    let signature = "@fragment\nfn fs_main(in: VsOut) -> TaaOut {";
    let diagnostic_signature = match batch {
        0 => {
            r#"struct TaaDiagnosticOut {
    @location(0) rejection_reason: vec4<f32>,
    @location(1) motion: vec4<f32>,
    @location(2) reprojected_uv: vec4<f32>,
    @location(3) temporal_confidence: vec4<f32>,
};

@fragment
fn fs_diagnostics(in: VsOut) -> TaaDiagnosticOut {"#
        }
        1 => {
            r#"struct TaaDiagnosticOut {
    @location(0) reactive_history: vec4<f32>,
    @location(1) history_policy: vec4<f32>,
    @location(2) reconstruction_footprint: vec4<f32>,
};

@fragment
fn fs_diagnostics(in: VsOut) -> TaaDiagnosticOut {"#
        }
        _ => panic!("invalid TAA diagnostic batch {batch}"),
    };
    let source = source.replacen(signature, diagnostic_signature, 1);
    assert!(
        source.contains("fn fs_diagnostics"),
        "TAA entry point changed; diagnostics must follow it"
    );

    let diagnostic_body = r#"    let clamped_ycocg = vec3<f32>(history_y_clamped, co_clamped, cg_clamped);
    let clamp_delta = length(history_ycocg - clamped_ycocg);
    let reactive_weight = reactive;

    // Categorical dominant reason: gray seed, red off-screen, cyan reactive,
    // magenta depth/color disocclusion, yellow neighborhood clamp, blue motion,
    // green keep.
    var reason = vec3<f32>(0.05, 0.65, 0.10);
    if (abs(u.params.x) >= 0.999) {
        reason = vec3<f32>(0.25);
    } else if (!history_in_bounds) {
        reason = vec3<f32>(1.0, 0.05, 0.02);
    } else if (reactive_weight > 0.01 &&
               reactive_weight >= max(disocclusion, motion_ramped)) {
        reason = vec3<f32>(0.0, 0.9, 1.0);
    } else if (max(disocclusion, depth_disocclusion) > 0.01 &&
               max(disocclusion, depth_disocclusion) >= motion_ramped) {
        reason = vec3<f32>(1.0, 0.0, 0.8);
    } else if (clamp_delta > 0.0001) {
        reason = vec3<f32>(1.0, 0.75, 0.0);
    } else if (motion_alpha > 0.01) {
        reason = vec3<f32>(0.05, 0.25, 1.0);
    }

    // Motion RG is signed around 0.5; B is magnitude. Reprojection RG stores
    // previous-frame UV and B is its validity. Confidence RGB stores the
    // persistent incoming lock, outgoing lock, and retained-history weight.
    let motion_debug = vec3<f32>(
        clamp(0.5 + vel.x * 32.0, 0.0, 1.0),
        clamp(0.5 - vel.y * 32.0, 0.0, 1.0),
        clamp(vel_len * 64.0, 0.0, 1.0),
    );
"#;
    let diagnostic_return = match batch {
        0 => {
            r#"    return TaaDiagnosticOut(
        vec4<f32>(reason, 1.0),
        vec4<f32>(motion_debug, 1.0),
        vec4<f32>(clamp(prev_uv, vec2<f32>(0.0), vec2<f32>(1.0)),
                  select(0.0, 1.0, history_in_bounds), 1.0),
        vec4<f32>(
            clamp(history_confidence, 0.0, 1.0),
            clamp(next_history_confidence, 0.0, 1.0),
            clamp(1.0 - alpha, 0.0, 1.0),
            1.0,
        ),
    );"#
        }
        1 => {
            r#"    return TaaDiagnosticOut(
        vec4<f32>(
            clamp(current_reactive, 0.0, 1.0),
            clamp(history_reactive, 0.0, 1.0),
            clamp(reactive, 0.0, 1.0),
            1.0,
        ),
        // R = variance-clamp displacement, G = current-frame blend weight,
        // B = persistent confidence rejection. The displacement scale keeps
        // ordinary sub-percent HDR corrections visible in an 8-bit capture.
        vec4<f32>(
            clamp(clamp_delta * 8.0, 0.0, 1.0),
            clamp(alpha, 0.0, 1.0),
            clamp(temporal_rejection, 0.0, 1.0),
            1.0,
        ),
        // R = current reconstruction residual against the linear center,
        // G = relative local luma sigma, B = rectification displacement.
        // Together these separate authored/source detail from broad variance
        // and the history detail actually removed by the current policy.
        vec4<f32>(
            clamp(length(current - center_rgb) * 8.0, 0.0, 1.0),
            clamp(stddev.x / max(abs(mean.x), 0.05), 0.0, 1.0),
            clamp(clamp_delta * 8.0, 0.0, 1.0),
            1.0,
        ),
    );"#
        }
        _ => unreachable!(),
    };
    let diagnostic_return = format!("{diagnostic_body}{diagnostic_return}");
    let final_start_signature = "    return TaaOut(\n        vec4<f32>(blended, blended_w),";
    let final_start = source.find(final_start_signature).unwrap_or_else(|| {
        panic!("TAA final output changed; diagnostics must follow it (reactive={reactive})")
    });
    let final_end_signature = "\n    );";
    let final_end = source[final_start..]
        .find(final_end_signature)
        .map(|offset| final_start + offset + final_end_signature.len())
        .unwrap_or_else(|| {
            panic!("TAA final output changed; diagnostics must follow it (reactive={reactive})")
        });
    let mut source = source;
    source.replace_range(final_start..final_end, &diagnostic_return);
    source
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    reactive: bool,
    batch: usize,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("taa_diagnostic_shader"),
        source: wgpu::ShaderSource::Wgsl(diagnostic_shader_source(reactive, batch).into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("taa_diagnostic_pipeline_layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let target_count = TAA_DIAGNOSTIC_BATCHES[batch].1 - TAA_DIAGNOSTIC_BATCHES[batch].0;
    let targets = (0..target_count)
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
            pipelines: (0..TAA_DIAGNOSTIC_BATCHES.len())
                .map(|batch| create_pipeline(device, layout, false, batch))
                .collect(),
            reactive_pipelines: None,
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
                .is_some_and(|resources| resources.reactive_pipelines.is_none())
        {
            let layout = self
                .taa_reactive_layout
                .as_ref()
                .expect("reactive diagnostics require the reactive TAA layout");
            let pipelines = (0..TAA_DIAGNOSTIC_BATCHES.len())
                .map(|batch| create_pipeline(&self.device, layout, true, batch))
                .collect();
            self.temporal_diagnostics
                .as_mut()
                .unwrap()
                .reactive_pipelines = Some(pipelines);
        }

        let resources = self.temporal_diagnostics.as_ref().unwrap();
        for (batch, &(start, end)) in TAA_DIAGNOSTIC_BATCHES.iter().enumerate() {
            let attachments = resources.views[start..end]
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
            let pipelines = if reactive {
                resources.reactive_pipelines.as_ref().unwrap()
            } else {
                &resources.pipelines
            };
            pass.set_pipeline(&pipelines[batch]);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
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
            for batch in 0..TAA_DIAGNOSTIC_BATCHES.len() {
                wgpu::naga::front::wgsl::parse_str(&diagnostic_shader_source(reactive, batch))
                    .unwrap_or_else(|error| {
                        panic!("TAA diagnostics WGSL batch {batch} failed: {error}")
                    });
            }
        }
        assert!(TAA_DIAGNOSTIC_BATCHES
            .iter()
            .all(|(start, end)| end - start <= 4));
        assert!(!TAA_SHADER_WGSL.contains("TaaDiagnosticOut"));
        assert!(!TAA_SHADER_WGSL.contains("fs_diagnostics"));
    }

    #[test]
    fn static_jitter_depth_flips_keep_only_color_compatible_history() {
        assert!(TAA_SHADER_WGSL.contains("let raw_depth_disocclusion = select("));
        assert!(TAA_SHADER_WGSL.contains("let static_zero_velocity = !camera_moving"));
        assert!(TAA_SHADER_WGSL.contains("velocity_divergence < 0.0000001"));
        assert!(TAA_SHADER_WGSL
            .contains("let jitter_coverage_compatible = gross_color_dist <= reject_hi;"));
        assert!(TAA_SHADER_WGSL.contains("static_zero_velocity && jitter_coverage_compatible,"));
        assert!(TAA_SHADER_WGSL.contains("let temporal_rejection = max(depth_disocclusion,"));
        assert!(!TAA_SHADER_WGSL.contains("let temporal_rejection = max(raw_depth_disocclusion,"));
        assert!(TAA_SHADER_WGSL.contains("let settled_coherent_lock = select("));
        assert!(TAA_SHADER_WGSL.contains("(!camera_moving || reconstruction_scale > 0.95)"));
        assert!(TAA_SHADER_WGSL.contains("reprojection_motion < 0.00025"));
        assert!(TAA_SHADER_WGSL.contains("current_weight >= 0.095"));
        assert!(TAA_SHADER_WGSL.contains("let settled_static_phase_candidate = select("));
        assert!(TAA_SHADER_WGSL.contains("depth < 0.9999"));
        assert!(TAA_SHADER_WGSL.contains("let settled_static_phase_lock ="));
        assert!(TAA_SHADER_WGSL.contains(
            "let static_current_cap = mix(0.041666667, 0.015625, settled_static_phase_lock);"
        ));
        assert!(
            TAA_SHADER_WGSL.contains("let color_motion_ramped = max(motion_ramped, disocclusion);")
        );
        assert!(TAA_SHADER_WGSL.contains("abs(dpdx(vec2<f32>(expected_prev_depth, disocclusion)))"));
        assert!(TAA_SHADER_WGSL.contains("temporal_gradients.y"));
        assert!(!TAA_SHADER_WGSL.contains("dpdx(disocclusion)"));
        assert!(TAA_SHADER_WGSL.contains("min(color_motion_ramped, settled_current_cap)"));
        assert!(TAA_SHADER_WGSL.contains("min(bootstrap_alpha, settled_current_cap)"));
        assert!(TAA_SHADER_WGSL.contains(
            "let stable_history = mix(clamped_history, history, settled_static_phase_lock);"
        ));
        assert!(!TAA_SHADER_WGSL.contains("disocclusion <= 0.01"));
    }
}
