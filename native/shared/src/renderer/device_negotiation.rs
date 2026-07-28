//! Tier-aware native device requests and documented fallback (#138).

use super::capabilities::{forced_renderer_tier, RendererCapabilities, RendererCapabilityTier};
use super::material_indirection::request_tier_a_if_supported;

const CORE_BIND_GROUPS: u32 = 5;
const CORE_COLOR_ATTACHMENTS: u32 = 4;
const CORE_SAMPLED_TEXTURES: u32 = 19;
const CORE_SAMPLERS: u32 = 16;
const CORE_STORAGE_BUFFERS: u32 = 8;
const CORE_UNIFORM_BUFFER_BINDING_SIZE: u64 = 64 * 1024;
const PATH_TRACING_STORAGE_BUFFERS: u32 = 9;
const FOLDED_MOBILE_BIND_GROUPS: u32 = 4;
const FOLDED_MOBILE_STORAGE_BUFFERS: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceRequestProfile {
    NativeFull,
    FoldedMobile,
}

impl DeviceRequestProfile {
    const fn name(self) -> &'static str {
        match self {
            Self::NativeFull => "native-full",
            Self::FoldedMobile => "folded-mobile",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DeviceRequestOptions {
    pub allow_ray_query: bool,
    pub profile: DeviceRequestProfile,
}

impl Default for DeviceRequestOptions {
    fn default() -> Self {
        Self {
            allow_ray_query: true,
            profile: DeviceRequestProfile::NativeFull,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceRequestPlan {
    pub label: &'static str,
    pub tier: RendererCapabilityTier,
    pub required_features: wgpu::Features,
    pub required_limits: wgpu::Limits,
    pub experimental_features: wgpu::ExperimentalFeatures,
}

impl DeviceRequestPlan {
    fn descriptor(&self, trace: wgpu::Trace) -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: Some(self.label),
            required_features: self.required_features,
            required_limits: self.required_limits.clone(),
            experimental_features: self.experimental_features,
            trace,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceNegotiationReport {
    pub profile: DeviceRequestProfile,
    pub preferred_tier: RendererCapabilityTier,
    pub selected_tier: RendererCapabilityTier,
    pub selected_label: &'static str,
    pub fallback_cause: Option<String>,
    pub required_features: wgpu::Features,
    pub required_limits: wgpu::Limits,
}

impl DeviceNegotiationReport {
    pub fn report_json(&self) -> String {
        let fallback_cause = self
            .fallback_cause
            .as_deref()
            .map(json_string)
            .unwrap_or_else(|| "null".to_owned());
        format!(
            concat!(
                "{{\"preferred_tier\":\"{}\",\"selected_tier\":\"{}\",",
                "\"profile\":\"{}\",\"selected_request\":\"{}\",\"fallback_cause\":{},",
                "\"required_features\":{},\"required_limits\":{{",
                "\"max_bind_groups\":{},\"max_color_attachments\":{},",
                "\"max_sampled_textures_per_shader_stage\":{},",
                "\"max_samplers_per_shader_stage\":{},",
                "\"max_storage_buffers_per_shader_stage\":{},",
                "\"max_uniform_buffer_binding_size\":{},",
                "\"max_binding_array_elements_per_shader_stage\":{},",
                "\"max_binding_array_sampler_elements_per_shader_stage\":{}}}}}"
            ),
            self.preferred_tier.name(),
            self.selected_tier.name(),
            self.profile.name(),
            self.selected_label,
            fallback_cause,
            json_string(&format!("{:?}", self.required_features)),
            self.required_limits.max_bind_groups,
            self.required_limits.max_color_attachments,
            self.required_limits.max_sampled_textures_per_shader_stage,
            self.required_limits.max_samplers_per_shader_stage,
            self.required_limits.max_storage_buffers_per_shader_stage,
            self.required_limits.max_uniform_buffer_binding_size,
            self.required_limits
                .max_binding_array_elements_per_shader_stage,
            self.required_limits
                .max_binding_array_sampler_elements_per_shader_stage,
        )
    }
}

pub struct NegotiatedDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub report: DeviceNegotiationReport,
}

pub async fn request_device_with_fallback(
    adapter: &wgpu::Adapter,
    options: DeviceRequestOptions,
) -> Result<NegotiatedDevice, String> {
    request_device_with_fallback_and_trace(adapter, options, wgpu::Trace::Off).await
}

/// Production device negotiation with an optional wgpu API trace.
///
/// Normal engine startup always calls [`request_device_with_fallback`] and
/// keeps tracing disabled. Qualification tools use this entry point so their
/// traced device has the exact same bounded feature/limit request and fallback
/// report as the engine being measured.
pub async fn request_device_with_fallback_and_trace(
    adapter: &wgpu::Adapter,
    options: DeviceRequestOptions,
    trace: wgpu::Trace,
) -> Result<NegotiatedDevice, String> {
    let plans = build_device_request_plans(adapter.features(), &adapter.limits(), options)?;
    let preferred = &plans[0];
    match adapter
        .request_device(&preferred.descriptor(trace.clone()))
        .await
    {
        Ok((device, queue)) => Ok(NegotiatedDevice {
            device,
            queue,
            report: report_for(preferred, options.profile, preferred.tier, None),
        }),
        Err(preferred_error) => {
            let Some(fallback) = plans.get(1) else {
                return Err(format!(
                    "{} device request failed with no distinct fallback: {preferred_error}",
                    preferred.label
                ));
            };
            match adapter.request_device(&fallback.descriptor(trace)).await {
                Ok((device, queue)) => Ok(NegotiatedDevice {
                    device,
                    queue,
                    report: report_for(
                        fallback,
                        options.profile,
                        preferred.tier,
                        Some(preferred_error.to_string()),
                    ),
                }),
                Err(fallback_error) => Err(format!(
                    "{} device request failed: {preferred_error}; {} failed: {fallback_error}",
                    preferred.label, fallback.label
                )),
            }
        }
    }
}

pub fn build_device_request_plans(
    supported: wgpu::Features,
    adapter_limits: &wgpu::Limits,
    options: DeviceRequestOptions,
) -> Result<Vec<DeviceRequestPlan>, String> {
    build_device_request_plans_with_override(
        supported,
        adapter_limits,
        options,
        forced_renderer_tier(),
    )
}

fn build_device_request_plans_with_override(
    supported: wgpu::Features,
    adapter_limits: &wgpu::Limits,
    options: DeviceRequestOptions,
    forced_tier: Option<RendererCapabilityTier>,
) -> Result<Vec<DeviceRequestPlan>, String> {
    validate_core_contract(adapter_limits, options.profile)?;
    let capability =
        RendererCapabilities::detect_with_override(supported, adapter_limits, forced_tier);
    let high_paths_allowed = forced_tier.is_none_or(|tier| tier >= RendererCapabilityTier::HighEnd);
    let mut preferred_features = wgpu::Features::empty();
    for optional in [
        wgpu::Features::TIMESTAMP_QUERY,
        wgpu::Features::TEXTURE_COMPRESSION_BC,
    ] {
        if supported.contains(optional) {
            preferred_features |= optional;
        }
    }

    let mut preferred_limits = core_required_limits(adapter_limits, options.profile);
    if high_paths_allowed {
        request_tier_a_if_supported(
            supported,
            adapter_limits,
            &mut preferred_features,
            &mut preferred_limits,
        );
        super::gpu_driven::request_features_if_supported(supported, &mut preferred_features);
    }

    let ray_query = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    if options.allow_ray_query && high_paths_allowed && supported.contains(ray_query) {
        preferred_features |= ray_query;
        preferred_limits.max_storage_buffers_per_shader_stage = PATH_TRACING_STORAGE_BUFFERS;
        preferred_limits = preferred_limits.using_minimum_supported_acceleration_structure_values();
    }
    let preferred_experimental = if preferred_features.contains(ray_query) {
        // The renderer guards every query path and uses the wgpu v29 contract.
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    };
    let preferred = DeviceRequestPlan {
        label: "bloom_device_preferred",
        tier: capability.selected_tier,
        required_features: preferred_features,
        required_limits: preferred_limits,
        experimental_features: preferred_experimental,
    };

    let fallback_limits = core_required_limits(adapter_limits, options.profile);
    let fallback_capability = RendererCapabilities::detect_with_override(
        wgpu::Features::empty(),
        &fallback_limits,
        forced_tier,
    );
    let fallback = DeviceRequestPlan {
        label: "bloom_device_fallback",
        tier: fallback_capability.selected_tier,
        required_features: wgpu::Features::empty(),
        required_limits: fallback_limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    };
    if preferred.required_features.is_empty()
        && preferred.required_limits == fallback.required_limits
    {
        Ok(vec![preferred])
    } else {
        Ok(vec![preferred, fallback])
    }
}

fn core_required_limits(
    adapter_limits: &wgpu::Limits,
    profile: DeviceRequestProfile,
) -> wgpu::Limits {
    let mut limits = wgpu::Limits::downlevel_defaults()
        .using_resolution(adapter_limits.clone())
        .using_alignment(adapter_limits.clone());
    limits.max_bind_groups = match profile {
        DeviceRequestProfile::NativeFull => CORE_BIND_GROUPS,
        DeviceRequestProfile::FoldedMobile => FOLDED_MOBILE_BIND_GROUPS,
    };
    limits.max_color_attachments = CORE_COLOR_ATTACHMENTS;
    limits.max_sampled_textures_per_shader_stage = CORE_SAMPLED_TEXTURES;
    limits.max_samplers_per_shader_stage = CORE_SAMPLERS;
    limits.max_storage_buffers_per_shader_stage = match profile {
        DeviceRequestProfile::NativeFull => CORE_STORAGE_BUFFERS,
        DeviceRequestProfile::FoldedMobile => FOLDED_MOBILE_STORAGE_BUFFERS,
    };
    limits.max_uniform_buffer_binding_size = CORE_UNIFORM_BUFFER_BINDING_SIZE;
    limits
}

fn validate_core_contract(
    limits: &wgpu::Limits,
    profile: DeviceRequestProfile,
) -> Result<(), String> {
    let required_bind_groups = match profile {
        DeviceRequestProfile::NativeFull => CORE_BIND_GROUPS,
        DeviceRequestProfile::FoldedMobile => FOLDED_MOBILE_BIND_GROUPS,
    };
    let required_storage_buffers = match profile {
        DeviceRequestProfile::NativeFull => CORE_STORAGE_BUFFERS,
        DeviceRequestProfile::FoldedMobile => FOLDED_MOBILE_STORAGE_BUFFERS,
    };
    let requirements = [
        (
            "max_bind_groups",
            required_bind_groups,
            limits.max_bind_groups,
        ),
        (
            "max_color_attachments",
            CORE_COLOR_ATTACHMENTS,
            limits.max_color_attachments,
        ),
        (
            "max_sampled_textures_per_shader_stage",
            CORE_SAMPLED_TEXTURES,
            limits.max_sampled_textures_per_shader_stage,
        ),
        (
            "max_samplers_per_shader_stage",
            CORE_SAMPLERS,
            limits.max_samplers_per_shader_stage,
        ),
        (
            "max_storage_buffers_per_shader_stage",
            required_storage_buffers,
            limits.max_storage_buffers_per_shader_stage,
        ),
    ];
    let mut missing = requirements
        .into_iter()
        .filter(|(_, required, available)| available < required)
        .map(|(name, required, available)| {
            format!("{name} requires {required}, adapter has {available}")
        })
        .collect::<Vec<_>>();
    if limits.max_uniform_buffer_binding_size < CORE_UNIFORM_BUFFER_BINDING_SIZE {
        missing.push(format!(
            "max_uniform_buffer_binding_size requires {}, adapter has {}",
            CORE_UNIFORM_BUFFER_BINDING_SIZE, limits.max_uniform_buffer_binding_size
        ));
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "adapter is below Bloom's active platform renderer contract: {}",
            missing.join("; ")
        ))
    }
}

fn report_for(
    selected: &DeviceRequestPlan,
    profile: DeviceRequestProfile,
    preferred_tier: RendererCapabilityTier,
    fallback_cause: Option<String>,
) -> DeviceNegotiationReport {
    DeviceNegotiationReport {
        profile,
        preferred_tier,
        selected_tier: selected.tier,
        selected_label: selected.label,
        fallback_cause,
        required_features: selected.required_features,
        required_limits: selected.required_limits.clone(),
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c <= '\u{1f}' => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linux_asahi_limits() -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        limits.max_color_attachments = 5;
        limits.max_bind_groups = 7;
        limits.max_sampled_textures_per_shader_stage = 128;
        limits.max_samplers_per_shader_stage = 32;
        limits.max_storage_buffers_per_shader_stage = 16;
        limits
    }

    #[test]
    fn native_plan_requests_actual_four_mrt_contract_not_webgpu_default_eight() {
        let plans = build_device_request_plans_with_override(
            wgpu::Features::empty(),
            &linux_asahi_limits(),
            DeviceRequestOptions::default(),
            None,
        )
        .unwrap();
        assert_eq!(plans[0].required_limits.max_color_attachments, 4);
        assert!(plans[0].required_limits.max_color_attachments <= 5);
    }

    #[test]
    fn preferred_high_end_request_has_a_featureless_modern_fallback() {
        let supported = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::INDIRECT_FIRST_INSTANCE
            | wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        let mut limits = linux_asahi_limits();
        limits.max_binding_array_elements_per_shader_stage = 500_000;
        limits.max_binding_array_sampler_elements_per_shader_stage = 1_000;
        limits = limits.using_minimum_supported_acceleration_structure_values();
        let plans = build_device_request_plans_with_override(
            supported,
            &limits,
            DeviceRequestOptions::default(),
            None,
        )
        .unwrap();
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].tier, RendererCapabilityTier::HighEnd);
        assert!(plans[0]
            .required_features
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY));
        assert_eq!(
            plans[0]
                .required_limits
                .max_binding_array_elements_per_shader_stage,
            4_162
        );
        assert_eq!(
            plans[0]
                .required_limits
                .max_storage_buffers_per_shader_stage,
            9
        );
        assert_eq!(plans[1].tier, RendererCapabilityTier::Modern);
        assert!(plans[1].required_features.is_empty());
        assert_eq!(plans[1].required_limits.max_color_attachments, 4);

        #[cfg(feature = "models3d")]
        serde_json::from_str::<serde_json::Value>(
            &report_for(
                &plans[0],
                DeviceRequestProfile::NativeFull,
                plans[0].tier,
                None,
            )
            .report_json(),
        )
        .expect("device negotiation report is valid JSON");
    }

    #[test]
    fn forced_baseline_requests_no_high_end_features() {
        let supported = wgpu::Features::all();
        let plans = build_device_request_plans_with_override(
            supported,
            &linux_asahi_limits(),
            DeviceRequestOptions::default(),
            Some(RendererCapabilityTier::Baseline),
        )
        .unwrap();
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].tier, RendererCapabilityTier::Baseline);
        assert!(!plans[0]
            .required_features
            .intersects(wgpu::Features::EXPERIMENTAL_RAY_QUERY));
        assert!(!plans[0]
            .required_features
            .intersects(super::super::material_indirection::TIER_A_FEATURES));
        assert!(!plans[0]
            .required_features
            .intersects(wgpu::Features::INDIRECT_FIRST_INSTANCE));
    }

    #[test]
    fn missing_active_profile_limit_fails_before_backend_device_creation() {
        let mut limits = linux_asahi_limits();
        limits.max_bind_groups = 4;
        let error = build_device_request_plans_with_override(
            wgpu::Features::empty(),
            &limits,
            DeviceRequestOptions::default(),
            None,
        )
        .unwrap_err();
        assert!(error.contains("max_bind_groups requires 5, adapter has 4"));
    }

    #[test]
    fn folded_mobile_profile_accepts_four_bind_groups_and_storage_buffers() {
        let mut limits = linux_asahi_limits();
        limits.max_bind_groups = 4;
        limits.max_storage_buffers_per_shader_stage = 4;
        let plans = build_device_request_plans_with_override(
            wgpu::Features::empty(),
            &limits,
            DeviceRequestOptions {
                allow_ray_query: false,
                profile: DeviceRequestProfile::FoldedMobile,
            },
            Some(RendererCapabilityTier::Modern),
        )
        .unwrap();
        assert_eq!(plans[0].required_limits.max_bind_groups, 4);
        assert_eq!(plans[0].required_limits.max_color_attachments, 4);
        assert_eq!(
            plans[0]
                .required_limits
                .max_sampled_textures_per_shader_stage,
            19
        );
        assert_eq!(plans[0].required_limits.max_samplers_per_shader_stage, 16);
        assert_eq!(
            plans[0]
                .required_limits
                .max_storage_buffers_per_shader_stage,
            4
        );
        assert_eq!(
            plans[0].required_limits.max_uniform_buffer_binding_size,
            64 * 1024
        );
    }
}
