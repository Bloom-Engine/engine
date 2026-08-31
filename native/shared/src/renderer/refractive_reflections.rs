//! Bounded reflection-source routing for imported physical transmission.
//!
//! The first production tier is a fragment-local screen-space march against
//! the immutable opaque color/depth snapshots already required by native
//! refraction. It creates no graph pass or image allocation, and the shader,
//! layout, uniform, and per-frame work remain absent until a physical
//! transmission draw is actually submitted.

#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

use super::Renderer;

pub(super) const REFRACTIVE_REFLECTION_MAX_DISTANCE: f32 = 8.0;
pub(super) const REFRACTIVE_REFLECTION_STEPS: f32 = 8.0;
pub(super) const REFRACTIVE_REFLECTION_MAX_ROUGHNESS: f32 = 0.45;
pub(super) const REFRACTIVE_REFLECTION_PERSISTENT_BYTES: u64 =
    std::mem::size_of::<RefractiveReflectionParams>() as u64;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RefractiveReflectionParams {
    /// Current jittered world-to-view and view-to-clip transforms. Keeping the
    /// two stages explicit mirrors the established opaque SSR math and avoids
    /// accumulating a world-space ray over large scene coordinates.
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    /// x = screen-space tier active, y = maximum world-space distance,
    /// z = fixed march steps, w = maximum participating roughness.
    params: [f32; 4],
    /// xyz = explicit planar-probe normal, w = plane equation d.
    /// A zero normal is the no-probe sentinel.
    planar_plane: [f32; 4],
}

fn enabled_from(value: Option<&str>) -> bool {
    !matches!(
        value.unwrap_or("on").trim().to_ascii_lowercase().as_str(),
        "0" | "off" | "false" | "disabled"
    )
}

/// Startup-selected exact A/B gate. When disabled, the refractive pipeline is
/// compiled with the established environment-only reflection expression and
/// does not allocate this module's layout, uniform, or bind group.
pub(super) fn refractive_reflection_hierarchy_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        enabled_from(
            std::env::var("BLOOM_REFRACTIVE_REFLECTIONS")
                .ok()
                .as_deref(),
        )
    })
}

#[cfg(not(fold_scene_inputs))]
pub(super) fn create_inputs_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_refractive_inputs_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(REFRACTIVE_REFLECTION_PERSISTENT_BYTES)
                            .expect("reflection params are non-empty"),
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

#[cfg(not(fold_scene_inputs))]
pub(super) fn create_params_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scene_refractive_reflection_params"),
        size: REFRACTIVE_REFLECTION_PERSISTENT_BYTES,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(not(fold_scene_inputs))]
pub(super) struct RefractiveInputs<'a> {
    pub(super) layout: &'a wgpu::BindGroupLayout,
    pub(super) params_buffer: &'a wgpu::Buffer,
    pub(super) scene_color: &'a wgpu::TextureView,
    pub(super) scene_color_sampler: &'a wgpu::Sampler,
    pub(super) scene_depth: &'a wgpu::TextureView,
    pub(super) planar_reflection: &'a wgpu::TextureView,
}

#[cfg(not(fold_scene_inputs))]
pub(super) fn materialize_inputs(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    inputs: RefractiveInputs<'_>,
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    screen_space_active: bool,
    planar_plane: Option<([f32; 3], f32)>,
) -> wgpu::BindGroup {
    let planar_plane = planar_plane.map_or([0.0; 4], |(normal, plane_y)| {
        [normal[0], normal[1], normal[2], normal[1] * plane_y]
    });
    let params = RefractiveReflectionParams {
        view,
        proj,
        params: [
            if screen_space_active { 1.0 } else { 0.0 },
            REFRACTIVE_REFLECTION_MAX_DISTANCE,
            REFRACTIVE_REFLECTION_STEPS,
            REFRACTIVE_REFLECTION_MAX_ROUGHNESS,
        ],
        planar_plane,
    };
    queue.write_buffer(inputs.params_buffer, 0, bytemuck::bytes_of(&params));
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene_refractive_inputs_bg"),
        layout: inputs.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(inputs.scene_color),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(inputs.scene_color_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(inputs.scene_depth),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: inputs.params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(inputs.planar_reflection),
            },
        ],
    })
}

impl Renderer {
    pub(super) fn refractive_reflection_source_name(&self) -> &'static str {
        #[cfg(fold_scene_inputs)]
        {
            return "environment";
        }
        #[cfg(not(fold_scene_inputs))]
        if self.scene_refractive_inputs_layout.is_some() {
            if self.planar_probes.iter().any(Option::is_some) {
                "planar-then-screen-space-then-environment"
            } else if self.ssr_enabled {
                "screen-space-then-environment"
            } else {
                "environment-runtime-fallback"
            }
        } else {
            "environment"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_gate_defaults_on_and_accepts_documented_false_values() {
        assert!(enabled_from(None));
        assert!(enabled_from(Some("yes")));
        for value in ["0", "off", " false ", "DISABLED"] {
            assert!(!enabled_from(Some(value)), "{value}");
        }
    }

    #[test]
    fn reflection_uniform_abi_is_exactly_one_hundred_sixty_bytes() {
        assert_eq!(REFRACTIVE_REFLECTION_PERSISTENT_BYTES, 160);
    }
}
