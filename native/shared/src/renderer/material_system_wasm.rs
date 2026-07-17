//! wasm32-only material bind-group helpers, split out of
//! `material_system.rs` to keep that file under the 2000-line policy (the
//! same reason `material_system_tests.rs` is a `#[path]` child). The whole
//! module is gated `#[cfg(target_arch = "wasm32")]` at its declaration in
//! `renderer/mod.rs`, so nothing here compiles on native.

/// EN-063 — wasm32-only per_frame bind group builder: the PerFrame UBO
/// at binding 0 plus the seven folded SceneInputs resources at
/// `WASM_SCENE_INPUTS_BASE..+6` (order and types mirror
/// `update_scene_inputs` / `create_scene_inputs_layout` exactly). The
/// single creation path for every bind group made against the wasm32
/// `abi_per_frame` layout — that layout requires all eight entries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_per_frame_bg_wasm(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    per_frame_buffer: &wgpu::Buffer,
    scene_color_view: &wgpu::TextureView,
    scene_color_samp: &wgpu::Sampler,
    scene_depth_view: &wgpu::TextureView,
    scene_depth_samp: &wgpu::Sampler,
    impulse_view: &wgpu::TextureView,
    impulse_samp: &wgpu::Sampler,
    motion_vectors_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    use super::material_pipeline::WASM_SCENE_INPUTS_BASE as B;
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("material_per_frame_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0,     resource: per_frame_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: B,     resource: wgpu::BindingResource::TextureView(scene_color_view) },
            wgpu::BindGroupEntry { binding: B + 1, resource: wgpu::BindingResource::Sampler(scene_color_samp) },
            wgpu::BindGroupEntry { binding: B + 2, resource: wgpu::BindingResource::TextureView(scene_depth_view) },
            wgpu::BindGroupEntry { binding: B + 3, resource: wgpu::BindingResource::Sampler(scene_depth_samp) },
            wgpu::BindGroupEntry { binding: B + 4, resource: wgpu::BindingResource::TextureView(impulse_view) },
            wgpu::BindGroupEntry { binding: B + 5, resource: wgpu::BindingResource::Sampler(impulse_samp) },
            wgpu::BindGroupEntry { binding: B + 6, resource: wgpu::BindingResource::TextureView(motion_vectors_view) },
        ],
    })
}
