//! Read-only renderer capability reporting shared by native and web targets.
//!
//! Reports are built only when the public API is queried, so exposing this
//! module does not add work to renderer initialization or the frame loop.

use super::capabilities::RendererCapabilities;
use super::Renderer;

fn json_string(out: &mut String, value: &str) {
    out.push('"');
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
}

fn enum_name(value: impl std::fmt::Debug) -> String {
    format!("{value:?}").to_ascii_lowercase().replace('_', "-")
}

impl Renderer {
    pub fn set_device_negotiation_report(&mut self, report: String) {
        self.device_negotiation_report = Some(report);
    }

    pub fn quality_adapter_json(&self) -> String {
        let info = self.device.adapter_info();
        let features = self.device.features();
        let renderer_capabilities = RendererCapabilities::detect(features, &self.device.limits());
        let mut semantic_features = Vec::new();
        for (feature, enabled) in [
            (
                "timestamp-query",
                features.contains(wgpu::Features::TIMESTAMP_QUERY),
            ),
            (
                "ray-query",
                features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
            ),
            (
                "texture-binding-array",
                features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY),
            ),
            (
                "non-uniform-indexing",
                features.contains(
                    wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                ),
            ),
            (
                "texture-compression-bc",
                features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
            ),
        ] {
            if enabled {
                semantic_features.push(feature);
            }
        }
        let mut out = String::from("{\"availability\":\"reported\",\"name\":");
        json_string(&mut out, &info.name);
        out.push_str(",\"vendor_id\":");
        out.push_str(&info.vendor.to_string());
        out.push_str(",\"device_id\":");
        out.push_str(&info.device.to_string());
        out.push_str(",\"device_type\":");
        json_string(&mut out, &enum_name(info.device_type));
        out.push_str(",\"driver\":");
        json_string(&mut out, &info.driver);
        out.push_str(",\"driver_info\":");
        json_string(&mut out, &info.driver_info);
        out.push_str(",\"backend\":");
        json_string(&mut out, &enum_name(info.backend));
        out.push_str(",\"capability_tier\":");
        json_string(&mut out, renderer_capabilities.selected_tier.name());
        out.push_str(",\"renderer_capabilities\":");
        out.push_str(&renderer_capabilities.report_json());
        out.push_str(",\"device_negotiation\":");
        out.push_str(self.device_negotiation_report.as_deref().unwrap_or("null"));
        out.push_str(",\"features\":[");
        for (index, feature) in semantic_features.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            json_string(&mut out, feature);
        }
        out.push_str("]}");
        out
    }

    /// Public, read-only renderer capability report. This composes the same
    /// adapter/tier evidence used by qualification artifacts with live
    /// material capacities and the optional paths actually built at startup.
    pub fn renderer_capability_report_json(&self) -> String {
        let imported_refraction = match self.imported_refraction_mode_code() {
            1 => "scene-snapshot",
            2 => "environment-fallback",
            _ => "disabled-legacy",
        };
        let mut out = String::from(
            "{\"version\":1,\"availability\":\"available\",\"reason\":null,\"adapter\":",
        );
        out.push_str(&self.quality_adapter_json());
        out.push_str(",\"material_binding\":");
        out.push_str(&self.material_binding_report_json());
        out.push_str(",\"runtime_support\":{\"hardware_ray_query\":");
        out.push_str(if self.hw_rt_enabled { "true" } else { "false" });
        out.push_str(",\"path_tracing\":");
        out.push_str(if self.pt_pipeline.is_some() {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"gpu_driven\":");
        out.push_str(&self.gpu_driven.report_json());
        out.push_str(",\"imported_refraction\":");
        json_string(&mut out, imported_refraction);
        out.push_str(",\"transparency_modes\":[\"sorted\",\"auto\",\"weighted\"]}}");
        out
    }
}
