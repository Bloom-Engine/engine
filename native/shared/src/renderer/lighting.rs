//! Group-1 lighting bind group — layout and construction.
//!
//! The scene + immediate-mode 3D pipelines share one bind-group layout
//! for lighting data: the Lighting UBO, env/IBL textures, the shadow
//! cascade, and (on clustered devices) the froxel buffers at bindings
//! 10-12. The bind group is rebuilt whenever the env source changes
//! (HDR load, panorama, procedural sky); every rebuild goes through
//! [`Renderer::make_lighting_bind_group`] so the entry list exists in
//! exactly one place and cannot drift between call sites.

use super::{froxel, Renderer};

/// Create the group-1 layout. `clustered` appends the froxel bindings —
/// set when [`froxel::FroxelPass::supported`] holds for the device.
/// Pipelines whose shaders don't reference bindings 10-12 (pipeline_3d)
/// share the layout unaffected; extra entries are legal as long as the
/// bind group provides them.
pub(super) fn create_lighting_layout(
    device: &wgpu::Device,
    clustered: bool,
) -> wgpu::BindGroupLayout {
    let tex_float = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    let tex_depth = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Depth,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    let frag = wgpu::ShaderStages::FRAGMENT;
    let mut entries = vec![
        // 0: Lighting UBO. VERTEX_FRAGMENT so the scene vertex shader can read
        // `wind` (foliage sway); the fragment stage uses the full struct.
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // 1/2: env (IBL specular) texture + sampler
        wgpu::BindGroupLayoutEntry { binding: 1, visibility: frag, ty: tex_float, count: None },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: frag,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        // 3/4: BRDF LUT + sampler
        wgpu::BindGroupLayoutEntry { binding: 3, visibility: frag, ty: tex_float, count: None },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: frag,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        // 5-7: shadow cascades, 8: comparison sampler
        wgpu::BindGroupLayoutEntry { binding: 5, visibility: frag, ty: tex_depth, count: None },
        wgpu::BindGroupLayoutEntry { binding: 6, visibility: frag, ty: tex_depth, count: None },
        wgpu::BindGroupLayoutEntry { binding: 7, visibility: frag, ty: tex_depth, count: None },
        wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: frag,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            count: None,
        },
        // 9: env diffuse (IBL irradiance)
        wgpu::BindGroupLayoutEntry { binding: 9, visibility: frag, ty: tex_float, count: None },
    ];
    if clustered {
        entries.extend(froxel::extra_lighting_layout_entries());
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lighting_layout"),
        entries: &entries,
    })
}

/// Everything a lighting bind group references besides the env views.
/// `Renderer::new` builds one from constructor locals (before `self`
/// exists); [`Renderer::make_lighting_bind_group`] from fields.
pub(super) struct LightingBindSources<'a> {
    pub lighting_buffer: &'a wgpu::Buffer,
    pub env_sampler: &'a wgpu::Sampler,
    pub brdf_lut_view: &'a wgpu::TextureView,
    pub brdf_lut_sampler: &'a wgpu::Sampler,
    pub shadow_map: &'a crate::shadows::ShadowMap,
    pub froxel: Option<&'a froxel::FroxelPass>,
}

/// The single source of truth for the group-1 entry list — every
/// lighting bind group the renderer ever creates goes through here.
pub(super) fn create_lighting_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    src: &LightingBindSources<'_>,
    env_view: &wgpu::TextureView,
    diffuse_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let mut entries = vec![
        wgpu::BindGroupEntry { binding: 0, resource: src.lighting_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(src.env_sampler) },
        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(src.brdf_lut_view) },
        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(src.brdf_lut_sampler) },
        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&src.shadow_map.depth_views[0]) },
        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&src.shadow_map.depth_views[1]) },
        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&src.shadow_map.depth_views[2]) },
        wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&src.shadow_map.sampler) },
        wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(diffuse_view) },
    ];
    if let Some(f) = src.froxel {
        entries.extend(f.extra_lighting_bind_entries());
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

impl Renderer {
    /// Build a group-1 lighting bind group for the given env-specular /
    /// env-diffuse views. Everything else (UBO, BRDF LUT, shadow
    /// cascade, froxel buffers when clustered) comes from `self`.
    pub(super) fn make_lighting_bind_group(
        &self,
        label: &str,
        env_view: &wgpu::TextureView,
        diffuse_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        create_lighting_bind_group(
            &self.device,
            &self.lighting_layout,
            label,
            &LightingBindSources {
                lighting_buffer: &self.lighting_buffer,
                env_sampler: &self.env_sampler,
                brdf_lut_view: &self.brdf_lut_view,
                brdf_lut_sampler: &self.brdf_lut_sampler,
                shadow_map: &self.shadow_map,
                froxel: self.froxel.as_ref(),
            },
            env_view,
            diffuse_view,
        )
    }
}
