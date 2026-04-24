// Material system — the runtime state behind Phase 1c.
//
// Owns the compiled `MaterialPipeline`s, the per-frame / per-view /
// per-material / per-draw uniform buffers, the bind groups wiring
// them to the ABI layouts, and the per-frame draw command list.
//
// The Renderer owns one `MaterialSystem` instance. Games interact via
// three methods: `compile_material`, `submit_draw`, and (internally)
// `dispatch` which runs inside the main HDR pass.

use wgpu::util::DeviceExt;

use super::material_pipeline::{
    MaterialAbiLayouts, MaterialPipeline, MaterialCompileDesc, FragmentProfile,
    compile_material, MaterialCompileError,
};
use super::types::Vertex3D;

// =====================================================================
// Uniform structs — repr(C), bytemuck-Pod, mirror the WGSL in
// material_abi.wgsl. Kept local to this module so changes to ABI
// struct layouts happen in exactly one place per language.
// =====================================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerFrameUniforms {
    pub time:              f32,
    pub delta_time:        f32,
    pub frame_index:       u32,
    pub _pad0:             u32,
    pub screen_resolution: [f32; 2],
    pub render_resolution: [f32; 2],
    pub taa_jitter:        [f32; 2],
    pub _pad1:             [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerViewDirLight {
    pub direction: [f32; 4],  // xyz + intensity
    pub color:     [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerViewPointLight {
    pub position: [f32; 4],
    pub color:    [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerViewUniforms {
    pub view:           [[f32; 4]; 4],
    pub proj:           [[f32; 4]; 4],
    pub view_proj:      [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub inv_proj:       [[f32; 4]; 4],
    pub camera_pos:     [f32; 4],
    pub camera_dir:     [f32; 4],
    pub ambient:        [f32; 4],
    pub fog:            [f32; 4],
    pub sun_dir:        [f32; 4],
    pub sun_color:      [f32; 4],
    pub dir_light_count:   [f32; 4],
    pub dir_lights:        [PerViewDirLight; 4],
    pub point_light_count: [f32; 4],
    pub point_lights:      [PerViewPointLight; 16],
    pub shadow_splits:   [f32; 4],
    pub shadow_view:     [[f32; 4]; 4],
    pub shadow_cascades: [[[f32; 4]; 4]; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialFactorsUniforms {
    pub metal_rough: [f32; 4],
    pub emissive:    [f32; 4],
    pub base_color:  [f32; 4],
    pub _reserved:   [f32; 4],
}

impl Default for MaterialFactorsUniforms {
    fn default() -> Self {
        Self {
            metal_rough: [0.0, 1.0, 0.0, 0.0],          // non-metal, rough, no MR tex, no cutoff
            emissive:    [0.0, 0.0, 0.0, 0.0],
            base_color:  [1.0, 1.0, 1.0, 1.0],
            _reserved:   [0.0, 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerDrawUniforms {
    pub mvp:        [[f32; 4]; 4],
    pub model:      [[f32; 4]; 4],
    pub prev_mvp:   [[f32; 4]; 4],
    pub model_tint: [f32; 4],
    pub skin_info:  [u32; 4],
}

// =====================================================================
// Handles + draw commands
// =====================================================================

pub type MaterialHandle = u32;

pub struct MaterialDrawCommand {
    pub material:    MaterialHandle,
    pub mesh_handle: u64,       // matches model_gpu_cache keys
    pub mesh_idx:    usize,     // sub-mesh index within that cached model
    pub draw_slot:   usize,     // which slot in per_draw_buffers to bind
}

// =====================================================================
// The system
// =====================================================================

pub struct MaterialSystem {
    pub layouts: MaterialAbiLayouts,

    // Compiled pipelines, indexed by MaterialHandle (1-based; 0 = invalid).
    pub pipelines: Vec<Option<MaterialPipeline>>,

    // Per-frame UBO + bind group (rewritten at the start of every frame).
    pub per_frame_buffer: wgpu::Buffer,
    pub per_frame_bg:     wgpu::BindGroup,

    // Per-view UBO — one for now (single camera). Phase 2 may add more
    // for split-screen / shadow cascades.
    pub per_view_buffer: wgpu::Buffer,
    pub per_view_bg:     wgpu::BindGroup,

    // Default per-material bind group: all white 1×1 textures, default
    // factors, zero-initialised user-params. Materials that don't
    // provide their own share this.
    pub default_per_material_bg: wgpu::BindGroup,
    /// Kept alive so the BG it backs doesn't dangle.
    _default_material_factors_buffer: wgpu::Buffer,
    _default_user_params_buffer:      wgpu::Buffer,
    _default_white_tex:               wgpu::Texture,
    _default_sampler:                 wgpu::Sampler,

    // Per-draw UBO pool. Buffers grow 1-by-1 as draws pile up.
    // Each entry is `(PerDraw UBO, bind group binding it + the global
    // joint buffer at binding 1)`.
    pub per_draw_buffers: Vec<wgpu::Buffer>,
    pub per_draw_bgs:     Vec<wgpu::BindGroup>,

    // Frame state
    pub commands:   Vec<MaterialDrawCommand>,
    next_draw_slot: usize,
}

impl MaterialSystem {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        joint_buffer: &wgpu::Buffer,
    ) -> Self {
        let layouts = MaterialAbiLayouts::create(device);

        // Per-frame UBO.
        let per_frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_per_frame"),
            size: std::mem::size_of::<PerFrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let per_frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_per_frame_bg"),
            layout: &layouts.per_frame,
            entries: &[wgpu::BindGroupEntry {
                binding: 0, resource: per_frame_buffer.as_entire_binding(),
            }],
        });

        // Default white 1×1 texture + sampler shared across optional PBR bindings.
        let default_white_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("material_default_white"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            Default::default(),
            &[255, 255, 255, 255],
        );
        let default_white_view = default_white_tex.create_view(&Default::default());
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material_default_samp"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Per-view UBO — zero init; write the real data each frame in `begin_frame`.
        let per_view_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_per_view"),
            size: std::mem::size_of::<PerViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // We don't have the env / BRDF LUT / shadow textures plumbed through
        // the material-system layout yet — Phase 2 wires them from the
        // existing renderer state. For now the PerView bind group uses the
        // default white texture + sampler for the env / BRDF / shadow slots
        // as placeholders so draws validate.
        let white_samp = &default_sampler;
        let white_view = &default_white_view;
        let cmp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material_default_cmp_samp"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        // PerView layout wants a depth texture for shadow bindings 6-8.
        // Use a 1×1 depth as a stub.
        let stub_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material_stub_depth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let stub_depth_view = stub_depth.create_view(&Default::default());
        let per_view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_per_view_bg"),
            layout: &layouts.per_view,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: per_view_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(white_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(white_samp) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(white_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(white_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(white_samp) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&stub_depth_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&stub_depth_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&stub_depth_view) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&cmp_sampler) },
            ],
        });
        // The stub_depth texture and cmp_sampler outlive the bind group via
        // wgpu internal Arc; we don't need to hold them in the struct.
        std::mem::forget(stub_depth);
        std::mem::forget(stub_depth_view);
        std::mem::forget(cmp_sampler);

        // Default MaterialFactors UBO.
        let default_mf = MaterialFactorsUniforms::default();
        let default_material_factors_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("material_default_factors"),
                contents: bytemuck::bytes_of(&default_mf),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        // Default user-params UBO — 256 bytes of zeros (ABI §1.4).
        let default_user_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material_default_user_params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Need a SECOND sampler here because wgpu requires distinct sampler
        // handles when binding the same logical sampler to multiple slots
        // within one BG — actually it doesn't, the same handle works fine.
        // We reuse default_sampler for every material-tex sampler slot.
        let default_per_material_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_default_per_material_bg"),
            layout: &layouts.per_material,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 1,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 2,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 3,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 4,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 5,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 6,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 7,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 8,  resource: wgpu::BindingResource::TextureView(&default_white_view) },
                wgpu::BindGroupEntry { binding: 9,  resource: wgpu::BindingResource::Sampler(&default_sampler) },
                wgpu::BindGroupEntry { binding: 10, resource: default_material_factors_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: default_user_params_buffer.as_entire_binding() },
            ],
        });

        Self {
            layouts,
            pipelines: Vec::new(),
            per_frame_buffer,
            per_frame_bg,
            per_view_buffer,
            per_view_bg,
            default_per_material_bg,
            _default_material_factors_buffer: default_material_factors_buffer,
            _default_user_params_buffer: default_user_params_buffer,
            _default_white_tex: default_white_tex,
            _default_sampler: default_sampler,
            per_draw_buffers: Vec::new(),
            per_draw_bgs: Vec::new(),
            commands: Vec::new(),
            next_draw_slot: 0,
        }
    }

    // --- Material registry --------------------------------------------

    /// Compile a material and return its handle. Handles are 1-based;
    /// 0 is reserved for "invalid material".
    pub fn compile(
        &mut self,
        device: &wgpu::Device,
        wgsl_source: &str,
        profile: FragmentProfile,
        reads_scene: bool,
        hdr_format: wgpu::TextureFormat,
        material_format: wgpu::TextureFormat,
        velocity_format: wgpu::TextureFormat,
        albedo_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Result<MaterialHandle, MaterialCompileError> {
        // Inject the user's WGSL under a synthetic path so the
        // preprocessor can resolve `#include "material_abi.wgsl"` etc.
        let entry_path = "__user_material.wgsl";
        let desc = MaterialCompileDesc {
            label: "user_material",
            entry_path,
            extra_sources: &[(entry_path, wgsl_source)],
            profile,
            reads_scene,
            hdr_format,
            material_format,
            velocity_format,
            albedo_format,
            depth_format,
            vertex_buffers: &[Vertex3D::desc()],
        };
        let pipeline = compile_material(device, &self.layouts, &desc)?;
        self.pipelines.push(Some(pipeline));
        Ok(self.pipelines.len() as MaterialHandle)
    }

    // --- Frame lifecycle ----------------------------------------------

    /// Write the per-frame + per-view UBOs. Safe to call any time
    /// before `dispatch`; callers should call exactly once per frame
    /// to keep `PerFrame.time` accurate. Does NOT clear the commands
    /// list — that's `reset_frame`'s job (called at begin_frame, not
    /// end_frame, so draws submitted during the frame survive).
    pub fn update_frame_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        per_frame: &PerFrameUniforms,
        per_view:  &PerViewUniforms,
    ) {
        queue.write_buffer(&self.per_frame_buffer, 0, bytemuck::bytes_of(per_frame));
        queue.write_buffer(&self.per_view_buffer,  0, bytemuck::bytes_of(per_view));
    }

    /// Reset the per-draw slot cursor. Commands list is cleared by the
    /// Renderer from its own `begin_frame` so the order of reset vs.
    /// submit is deterministic.
    pub fn reset_draw_slot(&mut self) {
        self.next_draw_slot = 0;
    }

    /// Submit a draw against a compiled material. Allocates (or reuses)
    /// a per-draw UBO slot, writes the MVP / model / tint / skin info,
    /// and queues the command for dispatch.
    pub fn submit_draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        joint_buffer: &wgpu::Buffer,
        material: MaterialHandle,
        mesh_handle: u64,
        mesh_idx: usize,
        mvp: [[f32; 4]; 4],
        model: [[f32; 4]; 4],
        prev_mvp: [[f32; 4]; 4],
        tint: [f32; 4],
        skin_info: [u32; 4],
    ) {
        if material == 0 || (material as usize) > self.pipelines.len() { return; }
        let slot = self.next_draw_slot;
        self.next_draw_slot += 1;
        self.ensure_draw_slot(device, joint_buffer, slot);

        let per_draw = PerDrawUniforms { mvp, model, prev_mvp, model_tint: tint, skin_info };
        queue.write_buffer(&self.per_draw_buffers[slot], 0, bytemuck::bytes_of(&per_draw));

        self.commands.push(MaterialDrawCommand {
            material, mesh_handle, mesh_idx, draw_slot: slot,
        });
    }

    fn ensure_draw_slot(
        &mut self,
        device: &wgpu::Device,
        joint_buffer: &wgpu::Buffer,
        slot: usize,
    ) {
        while self.per_draw_buffers.len() <= slot {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("material_per_draw"),
                size: std::mem::size_of::<PerDrawUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material_per_draw_bg"),
                layout: &self.layouts.per_draw,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: joint_buffer.as_entire_binding() },
                ],
            });
            self.per_draw_buffers.push(buf);
            self.per_draw_bgs.push(bg);
        }
    }

    /// Dispatch all queued material draws. Caller owns the render pass;
    /// this method binds the pipelines + groups + meshes and issues
    /// indexed draws. `mesh_fetch` is a closure that returns
    /// `(vertex_buffer, index_buffer, index_count)` for a given
    /// (mesh_handle, mesh_idx) — lets the renderer hand over its
    /// `model_gpu_cache` without this module taking a dependency on it.
    pub fn dispatch<'pass, F>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        mut mesh_fetch: F,
    )
    where F: FnMut(u64, usize) -> Option<(&'pass wgpu::Buffer, &'pass wgpu::Buffer, u32)>
    {
        if self.commands.is_empty() { return; }

        let mut last_material: MaterialHandle = 0;
        for cmd in &self.commands {
            if cmd.material != last_material {
                let mat = match self.pipelines.get(cmd.material as usize - 1) {
                    Some(Some(m)) => m,
                    _ => continue,
                };
                pass.set_pipeline(&mat.pipeline);
                pass.set_bind_group(0, &self.per_frame_bg, &[]);
                pass.set_bind_group(1, &self.per_view_bg, &[]);
                pass.set_bind_group(2, &self.default_per_material_bg, &[]);
                last_material = cmd.material;
            }
            if let Some((vb, ib, icount)) = mesh_fetch(cmd.mesh_handle, cmd.mesh_idx) {
                pass.set_bind_group(3, &self.per_draw_bgs[cmd.draw_slot], &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..icount, 0, 0..1);
            }
        }
    }
}
