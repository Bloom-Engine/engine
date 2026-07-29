//! Data-driven renderer capability tiers (#138).
//!
//! Tiers describe the resource-binding model that every renderer subsystem can
//! rely on. Independent optional paths (GPU-driven submission and ray query)
//! remain feature-detected within the selected tier so an adapter never loses a
//! supported fast path merely because it lacks an unrelated feature.

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RendererCapabilityTier {
    Baseline = 1,
    Modern = 2,
    HighEnd = 3,
}

impl RendererCapabilityTier {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Modern => "modern",
            Self::HighEnd => "high-end",
        }
    }

    pub fn from_override(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "baseline" | "base" | "low" | "1" => Some(Self::Baseline),
            "modern" | "medium" | "mid" | "2" => Some(Self::Modern),
            "high-end" | "high_end" | "highend" | "high" | "3" => Some(Self::HighEnd),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapabilityTierDefinition {
    pub tier: RendererCapabilityTier,
    pub required_features: wgpu::Features,
    pub min_binding_array_elements: u32,
    pub min_binding_array_samplers: u32,
    pub min_texture_array_layers: u32,
    pub min_sampled_textures: u32,
    pub material_bindings: &'static str,
    pub geometry_submission: &'static str,
    pub shadows: &'static str,
    pub gi: &'static str,
    pub reflections: &'static str,
    pub anti_aliasing: &'static str,
    pub textures: &'static str,
    pub path_tracing: &'static str,
    pub minimum_contract: &'static str,
}

/// This is the source of truth for both runtime selection and the public table
/// in `docs/renderer-capability-tiers.md`.
pub const CAPABILITY_TIER_DEFINITIONS: [CapabilityTierDefinition; 3] = [
    CapabilityTierDefinition {
        tier: RendererCapabilityTier::Baseline,
        required_features: wgpu::Features::empty(),
        min_binding_array_elements: 0,
        min_binding_array_samplers: 0,
        min_texture_array_layers: 0,
        min_sampled_textures: 0,
        material_bindings: "Tier C per-material bind groups",
        geometry_submission: "CPU direct draws",
        shadows: "Cascaded raster shadows (VSM capability fallback)",
        gi: "Software SDF, probes, and SSGI",
        reflections: "SSR, planar, and probe fallbacks",
        anti_aliasing: "TAA/CAS/FXAA",
        textures: "Per-material resident textures",
        path_tracing: "Disabled when this tier is forced",
        minimum_contract: "Active platform profile",
    },
    CapabilityTierDefinition {
        tier: RendererCapabilityTier::Modern,
        required_features: wgpu::Features::empty(),
        min_binding_array_elements: 0,
        min_binding_array_samplers: 0,
        min_texture_array_layers: 16,
        min_sampled_textures: 8,
        material_bindings: "Tier B deterministic paged arrays",
        geometry_submission: "CPU direct draws",
        shadows: "Cascaded raster shadows (VSM capability fallback)",
        gi: "Software SDF, probes, and SSGI",
        reflections: "SSR, planar, and probe fallbacks",
        anti_aliasing: "TAA/CAS/FXAA",
        textures: "Paged texture arrays/atlases",
        path_tracing: "Disabled when this tier is forced",
        minimum_contract: "16 texture-array layers; 8 sampled textures/stage",
    },
    CapabilityTierDefinition {
        tier: RendererCapabilityTier::HighEnd,
        required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
            .union(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING),
        min_binding_array_elements: 2,
        min_binding_array_samplers: 2,
        min_texture_array_layers: 0,
        min_sampled_textures: 0,
        material_bindings: "Tier A descriptor-indexed global tables",
        geometry_submission: "GPU indirect when supported; CPU oracle fallback",
        shadows: "VSM page cache with cascaded fallback",
        gi: "Ray query when supported; software SDF/SSGI fallback",
        reflections: "Ray query when supported; SSR/planar/probe fallback",
        anti_aliasing: "TAA/CAS/FXAA",
        textures: "Descriptor-indexed texture/sampler arrays",
        path_tracing: "Available only with ray query and required limits",
        minimum_contract: "Texture-binding arrays + non-uniform indexing; 2 array elements",
    },
];

impl CapabilityTierDefinition {
    fn supported_by(self, features: wgpu::Features, limits: &wgpu::Limits) -> bool {
        features.contains(self.required_features)
            && limits.max_binding_array_elements_per_shader_stage >= self.min_binding_array_elements
            && limits.max_binding_array_sampler_elements_per_shader_stage
                >= self.min_binding_array_samplers
            && limits.max_texture_array_layers >= self.min_texture_array_layers
            && limits.max_sampled_textures_per_shader_stage >= self.min_sampled_textures
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RendererCapabilities {
    pub detected_tier: RendererCapabilityTier,
    pub selected_tier: RendererCapabilityTier,
    pub requested_tier: Option<RendererCapabilityTier>,
    pub forced_tier: Option<RendererCapabilityTier>,
    pub texture_binding_array: bool,
    pub non_uniform_indexing: bool,
    pub indirect_first_instance: bool,
    pub ray_query: bool,
    pub max_binding_array_elements: u32,
    pub max_binding_array_samplers: u32,
    pub max_texture_array_layers: u32,
    pub max_sampled_textures: u32,
    pub max_samplers: u32,
    pub max_bind_groups: u32,
    pub max_color_attachments: u32,
    pub diagnostic: Option<String>,
}

impl RendererCapabilities {
    pub fn detect(features: wgpu::Features, limits: &wgpu::Limits) -> Self {
        let raw_override = std::env::var("BLOOM_FORCE_RENDER_TIER").ok();
        let parsed_override = raw_override
            .as_deref()
            .and_then(RendererCapabilityTier::from_override);
        let mut capabilities = Self::detect_with_override(features, limits, parsed_override);
        if let Some(invalid) = raw_override.filter(|_| parsed_override.is_none()) {
            capabilities.diagnostic = Some(format!(
                "invalid BLOOM_FORCE_RENDER_TIER value {invalid:?}; using detected tier {}",
                capabilities.detected_tier.name()
            ));
        }
        capabilities
    }

    pub fn detect_with_override(
        features: wgpu::Features,
        limits: &wgpu::Limits,
        forced_tier: Option<RendererCapabilityTier>,
    ) -> Self {
        let texture_binding_array = features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY);
        let non_uniform_indexing = features.contains(
            wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        );
        let detected_tier = CAPABILITY_TIER_DEFINITIONS
            .iter()
            .rev()
            .copied()
            .find(|definition| definition.supported_by(features, limits))
            .map(|definition| definition.tier)
            .expect("baseline renderer capability tier is unconditional");
        let (selected_tier, accepted_force, diagnostic) = match forced_tier {
            Some(requested) if requested <= detected_tier => (requested, Some(requested), None),
            Some(requested) => (
                detected_tier,
                None,
                Some(format!(
                    "requested renderer tier {} exceeds detected tier {}; using {}",
                    requested.name(),
                    detected_tier.name(),
                    detected_tier.name()
                )),
            ),
            None => (detected_tier, None, None),
        };

        Self {
            detected_tier,
            selected_tier,
            requested_tier: forced_tier,
            forced_tier: accepted_force,
            texture_binding_array,
            non_uniform_indexing,
            indirect_first_instance: features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
            ray_query: features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
            max_binding_array_elements: limits.max_binding_array_elements_per_shader_stage,
            max_binding_array_samplers: limits.max_binding_array_sampler_elements_per_shader_stage,
            max_texture_array_layers: limits.max_texture_array_layers,
            max_sampled_textures: limits.max_sampled_textures_per_shader_stage,
            max_samplers: limits.max_samplers_per_shader_stage,
            max_bind_groups: limits.max_bind_groups,
            max_color_attachments: limits.max_color_attachments,
            diagnostic,
        }
    }

    /// Optional paths are not restricted during automatic detection. A forced
    /// lower tier is a test/debug contract and must disable paths above it.
    pub fn forced_path_allowed(required: RendererCapabilityTier) -> bool {
        forced_renderer_tier().is_none_or(|forced| forced >= required)
    }

    pub fn report_json(&self) -> String {
        let selected_definition = CAPABILITY_TIER_DEFINITIONS
            .iter()
            .find(|definition| definition.tier == self.selected_tier)
            .expect("selected renderer tier has a definition");
        let requested = self
            .requested_tier
            .map(|tier| format!("\"{}\"", tier.name()))
            .unwrap_or_else(|| "null".to_owned());
        let forced = self
            .forced_tier
            .map(|tier| format!("\"{}\"", tier.name()))
            .unwrap_or_else(|| "null".to_owned());
        let diagnostic = self
            .diagnostic
            .as_deref()
            .map(json_string)
            .unwrap_or_else(|| "null".to_owned());
        let mut out = format!(
            concat!(
                "{{\"detected\":\"{}\",\"selected\":\"{}\",\"requested\":{},\"forced\":{},",
                "\"diagnostic\":{},\"available\":{{\"features\":{{",
                "\"texture_binding_array\":{},\"non_uniform_indexing\":{},",
                "\"indirect_first_instance\":{},\"ray_query\":{}}},\"limits\":{{",
                "\"max_binding_array_elements_per_shader_stage\":{},",
                "\"max_binding_array_sampler_elements_per_shader_stage\":{},",
                "\"max_texture_array_layers\":{},",
                "\"max_sampled_textures_per_shader_stage\":{},",
                "\"max_samplers_per_shader_stage\":{},\"max_bind_groups\":{},",
                "\"max_color_attachments\":{}}}}}}}"
            ),
            self.detected_tier.name(),
            self.selected_tier.name(),
            requested,
            forced,
            diagnostic,
            self.texture_binding_array,
            self.non_uniform_indexing,
            self.indirect_first_instance,
            self.ray_query,
            self.max_binding_array_elements,
            self.max_binding_array_samplers,
            self.max_texture_array_layers,
            self.max_sampled_textures,
            self.max_samplers,
            self.max_bind_groups,
            self.max_color_attachments,
        );
        out.pop();
        out.push_str(",\"paths\":{\"materials\":");
        out.push_str(&json_string(selected_definition.material_bindings));
        out.push_str(",\"geometry\":");
        out.push_str(&json_string(selected_definition.geometry_submission));
        out.push_str(",\"shadows\":");
        out.push_str(&json_string(selected_definition.shadows));
        out.push_str(",\"gi\":");
        out.push_str(&json_string(selected_definition.gi));
        out.push_str(",\"reflections\":");
        out.push_str(&json_string(selected_definition.reflections));
        out.push_str(",\"anti_aliasing\":");
        out.push_str(&json_string(selected_definition.anti_aliasing));
        out.push_str(",\"textures\":");
        out.push_str(&json_string(selected_definition.textures));
        out.push_str(",\"path_tracing\":");
        out.push_str(&json_string(selected_definition.path_tracing));
        out.push_str("}}");
        out
    }
}

pub fn forced_renderer_tier() -> Option<RendererCapabilityTier> {
    std::env::var("BLOOM_FORCE_RENDER_TIER")
        .ok()
        .and_then(|value| RendererCapabilityTier::from_override(&value))
}

pub fn hardware_ray_query_enabled(features: wgpu::Features) -> bool {
    features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        && RendererCapabilities::forced_path_allowed(RendererCapabilityTier::HighEnd)
}

pub fn capability_tier_markdown() -> String {
    let mut table = String::from(
        "| Tier | Materials | Geometry | Shadows | GI | Reflections | AA | Textures | Path tracing | Minimum contract |\n\
         | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    for definition in CAPABILITY_TIER_DEFINITIONS {
        use std::fmt::Write as _;
        let _ = writeln!(
            table,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            definition.tier.name(),
            definition.material_bindings,
            definition.geometry_submission,
            definition.shadows,
            definition.gi,
            definition.reflections,
            definition.anti_aliasing,
            definition.textures,
            definition.path_tracing,
            definition.minimum_contract,
        );
    }
    table
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

    #[test]
    fn selection_depends_on_features_and_limits_not_platform_names() {
        let mut baseline_limits = wgpu::Limits::downlevel_webgl2_defaults();
        baseline_limits.max_texture_array_layers = 1;
        baseline_limits.max_sampled_textures_per_shader_stage = 4;
        let baseline = RendererCapabilities::detect_with_override(
            wgpu::Features::empty(),
            &baseline_limits,
            None,
        );
        assert_eq!(baseline.detected_tier, RendererCapabilityTier::Baseline);

        let mut modern_limits = wgpu::Limits::downlevel_defaults();
        modern_limits.max_texture_array_layers = 16;
        modern_limits.max_sampled_textures_per_shader_stage = 8;
        let modern = RendererCapabilities::detect_with_override(
            wgpu::Features::empty(),
            &modern_limits,
            None,
        );
        assert_eq!(modern.detected_tier, RendererCapabilityTier::Modern);

        modern_limits.max_binding_array_elements_per_shader_stage = 2;
        modern_limits.max_binding_array_sampler_elements_per_shader_stage = 2;
        let high_end = RendererCapabilities::detect_with_override(
            wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            &modern_limits,
            None,
        );
        assert_eq!(high_end.detected_tier, RendererCapabilityTier::HighEnd);
    }

    #[test]
    fn lower_force_is_selected_and_unsupported_upward_force_is_rejected() {
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_texture_array_layers = 16;
        limits.max_sampled_textures_per_shader_stage = 8;
        let forced_baseline = RendererCapabilities::detect_with_override(
            wgpu::Features::empty(),
            &limits,
            Some(RendererCapabilityTier::Baseline),
        );
        assert_eq!(
            forced_baseline.selected_tier,
            RendererCapabilityTier::Baseline
        );
        assert_eq!(
            forced_baseline.forced_tier,
            Some(RendererCapabilityTier::Baseline)
        );

        let rejected = RendererCapabilities::detect_with_override(
            wgpu::Features::empty(),
            &limits,
            Some(RendererCapabilityTier::HighEnd),
        );
        assert_eq!(rejected.selected_tier, RendererCapabilityTier::Modern);
        assert_eq!(rejected.forced_tier, None);
        assert!(rejected.diagnostic.is_some());
    }

    #[test]
    fn checked_documentation_table_matches_runtime_definitions() {
        let document = include_str!("../../../../docs/renderer-capability-tiers.md");
        let begin = "<!-- BEGIN GENERATED CAPABILITY TIERS -->\n";
        let end = "<!-- END GENERATED CAPABILITY TIERS -->";
        let generated = document
            .split_once(begin)
            .expect("generated tier table start marker")
            .1
            .split_once(end)
            .expect("generated tier table end marker")
            .0;
        assert_eq!(generated, capability_tier_markdown());
    }

    #[cfg(feature = "models3d")]
    #[test]
    fn capability_report_is_valid_json_with_requested_and_available_state() {
        let report = RendererCapabilities::detect_with_override(
            wgpu::Features::INDIRECT_FIRST_INSTANCE,
            &wgpu::Limits::downlevel_defaults(),
            Some(RendererCapabilityTier::Baseline),
        );
        let json: serde_json::Value =
            serde_json::from_str(&report.report_json()).expect("valid capability report JSON");
        assert_eq!(json["selected"], "baseline");
        assert_eq!(json["requested"], "baseline");
        assert!(json["available"]["features"]["indirect_first_instance"]
            .as_bool()
            .unwrap());
        assert!(json["available"]["limits"]["max_bind_groups"]
            .as_u64()
            .is_some());
        assert_eq!(
            json["paths"]["materials"],
            "Tier C per-material bind groups"
        );
        assert_eq!(json["paths"]["anti_aliasing"], "TAA/CAS/FXAA");
    }
}
