//! Lazy, bounded physical-transmission representation for probe GI.
//!
//! Opaque scenes keep the established first-hit shaders and TLAS masks. When
//! SSGI and imported transmission are both live, specialized kernels continue
//! through at most one nearest glass instance. The specialization reuses spare
//! lanes in `InstanceGiDataCpu`; it allocates no texture or storage buffer.

use super::*;

pub(super) const TRANSPARENT_GI_SHADER_SWITCH: &str = "const BLOOM_TRANSPARENT_GI: bool = false;";
const TRANSPARENT_GI_SHADER_ENABLED: &str = "const BLOOM_TRANSPARENT_GI: bool = true;";

pub(super) fn transparent_gi_enabled() -> bool {
    std::env::var("BLOOM_TRANSPARENT_GI")
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "disabled"
            )
        })
        .unwrap_or(true)
}

#[derive(Copy, Clone, Debug)]
pub(super) struct InstanceTransport {
    pub absorption: [f32; 3],
    pub coverage: f32,
    pub transmission_weight: f32,
    pub fresnel_pass: f32,
}

impl InstanceTransport {
    pub const OPAQUE: Self = Self {
        absorption: [1.0; 3],
        coverage: 1.0,
        transmission_weight: 0.0,
        fresnel_pass: 0.0,
    };

    pub fn active(self) -> bool {
        self.transmission_weight > 0.0
    }
}

pub(super) fn instance_transport(
    material: &crate::scene::PbrMaterial,
    transform: &[[f32; 4]; 4],
    enabled: bool,
) -> InstanceTransport {
    let transmission = material.transmission;
    if !enabled || !material.has_gi_transmission() {
        return InstanceTransport::OPAQUE;
    }

    let metallic = material.metalness.clamp(0.0, 1.0);
    let transmission_weight = transmission.factor.clamp(0.0, 1.0) * (1.0 - metallic);
    if transmission_weight <= 0.0 {
        return InstanceTransport::OPAQUE;
    }

    let ior = transmission.effective_ior();
    let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
    let model_scale =
        ((transform[0][0].powi(2) + transform[0][1].powi(2) + transform[0][2].powi(2)).sqrt()
            + (transform[1][0].powi(2) + transform[1][1].powi(2) + transform[1][2].powi(2)).sqrt()
            + (transform[2][0].powi(2) + transform[2][1].powi(2) + transform[2][2].powi(2)).sqrt())
            / 3.0;
    let thickness = transmission.thickness_factor.max(0.0) * model_scale;
    let absorption = if transmission.attenuation_distance.is_finite()
        && transmission.attenuation_distance > 0.0
        && thickness > 0.0
    {
        std::array::from_fn(|channel| {
            transmission.attenuation_color[channel]
                .clamp(1.0e-6, 1.0)
                .powf(thickness / transmission.attenuation_distance)
        })
    } else {
        [1.0; 3]
    };
    let coverage = if material.alpha_mode == crate::models::MaterialAlphaMode::Blend {
        material.opacity.clamp(0.0, 1.0)
    } else {
        1.0
    };

    InstanceTransport {
        absorption,
        coverage,
        transmission_weight,
        fresnel_pass: 1.0 - f0,
    }
}

fn transmission_source(source: &str) -> String {
    let replaced = source.replacen(
        TRANSPARENT_GI_SHADER_SWITCH,
        TRANSPARENT_GI_SHADER_ENABLED,
        1,
    );
    debug_assert_ne!(replaced, source, "transparent-GI shader switch is missing");
    replaced
}

impl Renderer {
    pub(super) fn select_transparent_gi_route(&mut self, scene: &crate::scene::SceneGraph) {
        let was_active = self.transparent_gi_active;
        let instance_count = scene.transparent_gi_instance_count();
        self.transparent_gi_instance_count = instance_count;
        self.transparent_gi_active = instance_count > 0
            && self.ssgi_enabled
            && self.imported_refraction_enabled
            && transparent_gi_enabled();

        if self.transparent_gi_active {
            self.ensure_transparent_gi_pipelines();
        }
        if self.transparent_gi_active != was_active {
            // Probe history and the long-lived WSRC encode the old visibility
            // representation. Refresh them on the exact frame the route flips.
            self.transparent_gi_force_probe_refresh = true;
            self.probe_history_idx = 0;
            self.probe_history_valid = false;
            self.wsrc_built = [false; WSRC_CASCADE_COUNT as usize];
            self.scene_sdf_clipmap_rebake_needed = true;
        }
    }

    fn ensure_transparent_gi_pipelines(&mut self) {
        if self.probe_trace_sdf_transparent_pipeline.is_some()
            && (!self.hw_rt_enabled
                || (self.probe_trace_hw_transparent_pipeline.is_some()
                    && self.wsrc_bake_hw_transparent_pipeline.is_some()))
        {
            return;
        }
        if self.probe_trace_sdf_transparent_pipeline.is_none() {
            let source = format!(
                "{}{}",
                PROBE_HELPERS_WGSL,
                transmission_source(SSGI_PROBE_TRACE_SDF_WGSL)
            );
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("probe_trace_sdf_transparent_shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("probe_trace_sdf_transparent_pl"),
                    bind_group_layouts: &[Some(&self.probe_trace_sdf_layout)],
                    immediate_size: 0,
                });
            self.probe_trace_sdf_transparent_pipeline = Some(self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("probe_trace_sdf_transparent_pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("cs_main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.created_pipelines(1);
        }

        if self.hw_rt_enabled && self.probe_trace_hw_transparent_pipeline.is_none() {
            let source = format!(
                "enable wgpu_ray_query;\n{}{}{}",
                ray_query_backend_variant(&self.device),
                PROBE_HELPERS_WGSL,
                transmission_source(SSGI_PROBE_TRACE_HW_WGSL),
            );
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("probe_trace_hw_transparent_shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let bind_layout = self
                .probe_trace_hw_layout
                .as_ref()
                .expect("HW transparent GI requires the established HW trace layout");
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("probe_trace_hw_transparent_pl"),
                    bind_group_layouts: &[Some(bind_layout)],
                    immediate_size: 0,
                });
            self.probe_trace_hw_transparent_pipeline = Some(self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("probe_trace_hw_transparent_pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("cs_main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.created_pipelines(1);
        }

        if self.hw_rt_enabled && self.wsrc_bake_hw_transparent_pipeline.is_none() {
            let source = format!(
                "enable wgpu_ray_query;\n{}{}{}",
                ray_query_backend_variant(&self.device),
                PROBE_HELPERS_WGSL,
                transmission_source(WSRC_BAKE_HW_WGSL),
            );
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("wsrc_bake_hw_transparent_shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let bind_layout = self
                .wsrc_bake_hw_layout
                .as_ref()
                .expect("HW transparent GI requires the established WSRC layout");
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("wsrc_bake_hw_transparent_pl"),
                    bind_group_layouts: &[Some(bind_layout)],
                    immediate_size: 0,
                });
            self.wsrc_bake_hw_transparent_pipeline = Some(self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("wsrc_bake_hw_transparent_pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("cs_main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.created_pipelines(1);
        }

        log::info!(
            "bloom GI: physical transmission enabled \
             (one-layer colored continuation, lazy, zero additional textures/buffers)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_specializations_replace_exactly_one_compile_time_switch() {
        for source in [
            SSGI_PROBE_TRACE_HW_WGSL,
            SSGI_PROBE_TRACE_SDF_WGSL,
            WSRC_BAKE_HW_WGSL,
        ] {
            assert_eq!(source.matches(TRANSPARENT_GI_SHADER_SWITCH).count(), 1);
            let specialized = transmission_source(source);
            assert!(!specialized.contains(TRANSPARENT_GI_SHADER_SWITCH));
            assert_eq!(
                specialized.matches(TRANSPARENT_GI_SHADER_ENABLED).count(),
                1
            );
        }
    }

    #[test]
    fn ordinary_and_transparent_gi_shader_variants_parse() {
        let probe_prefix = format!(
            "enable wgpu_ray_query;\nconst BLOOM_RAY_QUERY_NEEDS_PROCEED: bool = false;\n{}",
            PROBE_HELPERS_WGSL
        );
        for source in [
            format!("{probe_prefix}{SSGI_PROBE_TRACE_HW_WGSL}"),
            format!(
                "{probe_prefix}{}",
                transmission_source(SSGI_PROBE_TRACE_HW_WGSL)
            ),
            format!("{PROBE_HELPERS_WGSL}{SSGI_PROBE_TRACE_SDF_WGSL}"),
            format!(
                "{PROBE_HELPERS_WGSL}{}",
                transmission_source(SSGI_PROBE_TRACE_SDF_WGSL)
            ),
            format!("{probe_prefix}{WSRC_BAKE_HW_WGSL}"),
            format!("{probe_prefix}{}", transmission_source(WSRC_BAKE_HW_WGSL)),
        ] {
            wgpu::naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("transparent-GI WGSL failed: {error:?}"));
        }
    }

    #[test]
    fn instance_transport_is_bounded_and_preserves_opaque_sentinel() {
        assert_eq!(
            std::mem::size_of::<InstanceGiDataCpu>(),
            144,
            "CPU records must retain the WGSL array stride"
        );
        let material = crate::scene::PbrMaterial::default();
        assert!(!instance_transport(&material, &IDENTITY_MAT4, true).active());

        let mut glass = material;
        glass.transmission.authored = true;
        glass.transmission.factor = 1.0;
        glass.transmission.ior = 1.5;
        glass.transmission.thickness_factor = 1.0;
        glass.transmission.attenuation_distance = 2.0;
        glass.transmission.attenuation_color = [0.25, 0.5, 1.0];
        let transport = instance_transport(&glass, &IDENTITY_MAT4, true);
        assert!(transport.active());
        assert_eq!(transport.coverage, 1.0);
        assert!(transport.absorption[0] < transport.absorption[1]);
        assert!(transport.absorption[1] < transport.absorption[2]);
        assert!((transport.fresnel_pass - 0.96).abs() < 1.0e-5);

        glass.metalness = 1.0;
        assert!(
            !instance_transport(&glass, &IDENTITY_MAT4, true).active(),
            "metallic suppression must keep the instance in the opaque mask"
        );
    }
}
