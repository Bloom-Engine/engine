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

    /// Present mode: 0 = Fifo (vsync), 1 = Mailbox (uncapped, no tearing),
    /// 2 = Immediate (uncapped, tearing allowed), 3 = AutoNoVsync (portable
    /// uncapped preference with backend fallback). Headless renderers retain
    /// the requested mode as qualification metadata without configuring a
    /// surface. Returns false only for invalid requests.
    pub fn set_present_mode(&mut self, mode: u32) -> bool {
        if mode > 3 {
            return false;
        }
        let requested = match mode {
            1 => wgpu::PresentMode::Mailbox,
            2 => wgpu::PresentMode::Immediate,
            3 => wgpu::PresentMode::AutoNoVsync,
            _ => wgpu::PresentMode::Fifo,
        };
        if self.surface_config.present_mode == requested {
            return true;
        }
        self.surface_config.present_mode = requested;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.surface_config);
        }
        eprintln!("bloom: present mode = {:?}", requested);
        true
    }

    /// Stable numeric form of the configured present-mode request. Quality
    /// qualification records this alongside wall/GPU timings so a vsync cap
    /// cannot masquerade as unchanged performance.
    pub fn present_mode_code(&self) -> u32 {
        match self.surface_config.present_mode {
            wgpu::PresentMode::Fifo => 0,
            wgpu::PresentMode::Mailbox => 1,
            wgpu::PresentMode::Immediate => 2,
            wgpu::PresentMode::AutoNoVsync => 3,
            wgpu::PresentMode::FifoRelaxed => 4,
            wgpu::PresentMode::AutoVsync => 5,
        }
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
        out.push_str(",\"vsm_gpu_casters\":");
        out.push_str(&self.vsm_gpu_casters.report_json());
        out.push_str(",\"imported_refraction\":");
        json_string(&mut out, imported_refraction);
        out.push_str(",\"transparency_modes\":[\"sorted\",\"auto\",\"weighted\"]}}");
        out
    }
}
