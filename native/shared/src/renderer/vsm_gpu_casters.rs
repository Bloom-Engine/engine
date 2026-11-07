//! Workload-gated indirect submission for rigid opaque VSM casters.
//!
//! Metal has no indirect-count draw in the negotiated WebGPU feature set, so
//! the exact CPU-visible page list remains authoritative. Large qualifying
//! lists are uploaded as compact indirect commands; cutout, skinned,
//! dynamic-overlay, dedicated-buffer, overflow, small, and unsupported cases
//! remain on the compatibility renderer.

use super::Vertex3D;
use std::ops::Range;

/// Below this per-page count the compatibility loop is lower overhead.
pub(super) const VSM_GPU_CASTER_MIN_DRAWS: usize = 48;
const MAX_CASTER_RECORDS: usize = crate::virtual_shadows::VSM_MAX_PAGE_RENDER_BUDGET as usize
    * crate::shadows::SHADOW_MAX_NODES as usize;

#[repr(C, align(16))]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct VsmGpuCaster {
    pub(super) clip_from_local: [[f32; 4]; 4],
    /// x=index count, y=first index, z=bitcast base vertex.
    pub(super) draw: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

const INDIRECT_COMMAND_BYTES: u64 = std::mem::size_of::<DrawIndexedIndirect>() as u64;

#[derive(Copy, Clone, Debug, Default)]
pub(super) struct VsmGpuCasterStats {
    pub(super) considered_pages: u32,
    pub(super) max_page_candidates: u32,
    pub(super) pages: u32,
    pub(super) casters: u32,
    pub(super) candidate_pairs: u32,
    pub(super) indirect_calls: u32,
}

struct Resources {
    caster_capacity: usize,
    indirect_capacity: usize,
    casters: wgpu::Buffer,
    indirect: wgpu::Buffer,
    draw_bind_group: wgpu::BindGroup,
}

pub(super) struct VsmGpuCasters {
    enabled: bool,
    draw_layout: Option<wgpu::BindGroupLayout>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    resources: Option<Resources>,
    indirect_upload: Vec<DrawIndexedIndirect>,
    pub(super) stats: VsmGpuCasterStats,
}

impl VsmGpuCasters {
    pub(super) fn new(device: &wgpu::Device, gpu_driven_enabled: bool) -> Self {
        let limits = device.limits();
        let max_caster_bytes =
            MAX_CASTER_RECORDS as u64 * std::mem::size_of::<VsmGpuCaster>() as u64;
        let max_indirect_bytes = MAX_CASTER_RECORDS as u64 * INDIRECT_COMMAND_BYTES;
        Self {
            enabled: cfg!(not(target_arch = "wasm32"))
                && gpu_driven_enabled
                && gpu_casters_requested()
                && limits.max_storage_buffers_per_shader_stage >= 1
                && limits.max_storage_buffer_binding_size >= max_caster_bytes
                && limits.max_buffer_size >= max_caster_bytes.max(max_indirect_bytes),
            draw_layout: None,
            render_pipeline: None,
            resources: None,
            indirect_upload: Vec::new(),
            stats: VsmGpuCasterStats::default(),
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn memory_bytes(&self) -> u64 {
        self.resources.as_ref().map_or(0, |resources| {
            resources.caster_capacity as u64 * std::mem::size_of::<VsmGpuCaster>() as u64
                + resources.indirect_capacity as u64 * INDIRECT_COMMAND_BYTES
        })
    }

    pub(super) fn reset_stats(&mut self) {
        self.stats = VsmGpuCasterStats::default();
    }

    pub(super) fn record_scan(&mut self, considered_pages: usize, max_page_candidates: usize) {
        self.stats.considered_pages = considered_pages as u32;
        self.stats.max_page_candidates = max_page_candidates as u32;
    }

    pub(super) fn report_json(&self) -> String {
        format!(
            concat!(
                "{{\"enabled\":{},\"active\":{},\"considered_pages\":{},",
                "\"max_page_candidates\":{},\"pages\":{},\"casters\":{},",
                "\"candidate_pairs\":{},\"indirect_calls\":{},",
                "\"classification_source\":\"cpu-exact-prefilter+gpu-indirect-submit\",",
                "\"gpu_bytes\":{}}}"
            ),
            self.enabled,
            self.stats.candidate_pairs > 0,
            self.stats.considered_pages,
            self.stats.max_page_candidates,
            self.stats.pages,
            self.stats.casters,
            self.stats.candidate_pairs,
            self.stats.indirect_calls,
            self.memory_bytes(),
        )
    }

    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        page_count: usize,
        casters: &[VsmGpuCaster],
    ) -> bool {
        self.stats.pages = page_count as u32;
        self.stats.casters = casters.len() as u32;
        self.stats.candidate_pairs = casters.len() as u32;
        self.stats.indirect_calls = page_count as u32;
        if !self.enabled
            || page_count == 0
            || casters.len() < VSM_GPU_CASTER_MIN_DRAWS
            || casters.len() > MAX_CASTER_RECORDS
        {
            self.stats.pages = 0;
            self.stats.casters = 0;
            self.stats.candidate_pairs = 0;
            self.stats.indirect_calls = 0;
            return false;
        }

        self.ensure_resources(device, casters.len());
        self.indirect_upload.clear();
        self.indirect_upload
            .extend(
                casters
                    .iter()
                    .enumerate()
                    .map(|(index, caster)| DrawIndexedIndirect {
                        index_count: caster.draw[0],
                        instance_count: 1,
                        first_index: caster.draw[1],
                        base_vertex: caster.draw[2] as i32,
                        first_instance: index as u32,
                    }),
            );
        let resources = self
            .resources
            .as_ref()
            .expect("enabled VSM indirect caster path owns resources");
        queue.write_buffer(&resources.casters, 0, bytemuck::cast_slice(casters));
        queue.write_buffer(
            &resources.indirect,
            0,
            bytemuck::cast_slice(&self.indirect_upload),
        );
        true
    }

    pub(super) fn draw_page<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        range: Range<u32>,
        vertex: &'a wgpu::Buffer,
        index: &'a wgpu::Buffer,
    ) {
        if range.is_empty() {
            return;
        }
        let Some(resources) = self.resources.as_ref() else {
            return;
        };
        pass.set_pipeline(
            self.render_pipeline
                .as_ref()
                .expect("enabled VSM indirect caster path owns render pipeline"),
        );
        pass.set_bind_group(0, &resources.draw_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex.slice(..));
        pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
        pass.multi_draw_indexed_indirect(
            &resources.indirect,
            range.start as u64 * INDIRECT_COMMAND_BYTES,
            range.end - range.start,
        );
    }

    fn ensure_resources(&mut self, device: &wgpu::Device, required_casters: usize) {
        if self.draw_layout.is_none() {
            let draw_layout = create_draw_layout(device);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vsm_indirect_caster_shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vsm_indirect_caster_pipeline_layout"),
                bind_group_layouts: &[Some(&draw_layout)],
                immediate_size: 0,
            });
            self.render_pipeline = Some(device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("vsm_indirect_caster_pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_shadow"),
                        buffers: &[Vertex3D::desc()],
                        compilation_options: Default::default(),
                    },
                    fragment: None,
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: Default::default(),
                        bias: wgpu::DepthBiasState {
                            constant: 1,
                            slope_scale: 1.0,
                            clamp: 0.0,
                        },
                    }),
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                },
            ));
            self.draw_layout = Some(draw_layout);
        }

        let required_capacity = required_casters
            .next_power_of_two()
            .max(VSM_GPU_CASTER_MIN_DRAWS);
        if self.resources.as_ref().is_some_and(|resources| {
            resources.caster_capacity >= required_capacity
                && resources.indirect_capacity >= required_capacity
        }) {
            return;
        }
        let capacity = self
            .resources
            .as_ref()
            .map_or(required_capacity, |resources| {
                resources.caster_capacity.max(required_capacity)
            });
        let casters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm_indirect_casters"),
            size: capacity as u64 * std::mem::size_of::<VsmGpuCaster>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm_indirect_commands"),
            size: capacity as u64 * INDIRECT_COMMAND_BYTES,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vsm_indirect_caster_bind_group"),
            layout: self
                .draw_layout
                .as_ref()
                .expect("enabled VSM indirect caster path owns layout"),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: casters.as_entire_binding(),
            }],
        });
        self.resources = Some(Resources {
            caster_capacity: capacity,
            indirect_capacity: capacity,
            casters,
            indirect,
            draw_bind_group,
        });
    }
}

fn gpu_casters_requested() -> bool {
    std::env::var("BLOOM_VSM_GPU_CASTERS")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false" | "disabled"
            )
        })
        .unwrap_or(true)
}

fn create_draw_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vsm_indirect_caster_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

const SHADER: &str = r#"
struct Caster {
    clip_from_local: mat4x4<f32>,
    draw: vec4<u32>,
};
struct CasterTable { values: array<Caster>, };

@group(0) @binding(0) var<storage, read> casters: CasterTable;

struct ShadowVertexInput {
    @location(0) position: vec3<f32>,
};

@vertex
fn vs_shadow(
    in: ShadowVertexInput,
    @builtin(instance_index) caster_index: u32,
) -> @builtin(position) vec4<f32> {
    return casters.values[caster_index].clip_from_local * vec4<f32>(in.position, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_caster_abi_is_bounded_and_aligned() {
        assert_eq!(std::mem::size_of::<VsmGpuCaster>(), 80);
        assert_eq!(std::mem::size_of::<DrawIndexedIndirect>(), 20);
        assert_eq!(INDIRECT_COMMAND_BYTES, 20);
        assert_eq!(
            MAX_CASTER_RECORDS,
            crate::virtual_shadows::VSM_MAX_PAGE_RENDER_BUDGET as usize
                * crate::shadows::SHADOW_MAX_NODES as usize
        );
    }

    #[test]
    fn gpu_caster_shader_parses() {
        if let Err(error) = wgpu::naga::front::wgsl::parse_str(SHADER) {
            panic!("{}", error.emit_to_string(SHADER));
        }
    }
}
