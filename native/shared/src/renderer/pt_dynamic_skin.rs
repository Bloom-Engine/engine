//! Skinned meshes in the ray scene: the per-frame compute pre-skin into the
//! PT megabuffer and the pooled BLAS each dynamic window is built into.

use super::{gpu_driven, shaders, Renderer};

/// One pooled BLAS for a skinned mesh in the ray scene, and the mesh it
/// was sized (and, with refit, first built) for.
pub(super) struct PtDynBlas {
    pub(super) blas: wgpu::Blas,
    v_count: u32,
    i_count: u32,
    cache_handle: u64,
    mesh_idx: usize,
}

/// Whether a per-frame build of a dynamic BLAS can be an in-place refit.
/// wgpu-core 29 always rebuilds; the Apple platform crates link the patched
/// copy in `native/third_party/wgpu-core-patched`, which refits a BLAS
/// created with `PreferUpdate` + `ALLOW_UPDATE`. A crate that asks for
/// refit without the patch still renders correctly, with full rebuilds.
const DYN_BLAS_REFIT: bool = cfg!(target_vendor = "apple");

/// Frames between full rebuilds of a refitted dynamic BLAS.
const DYN_BLAS_REBUILD_INTERVAL: u32 = 240;

impl Renderer {
    /// PT-6 — encode the compute pre-skin dispatches for this frame's
    /// dynamic windows (posed world-space position/normal written over
    /// the bind-pose data in the megabuffer) and (re)create each slot's
    /// BLAS when its geometry size changed. Runs before the combined
    /// BLAS+TLAS build on the same encoder, so the build reads posed
    /// vertices.
    pub(super) fn encode_dynamic_skin(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dyn_count: usize,
    ) {
        if dyn_count == 0 {
            return;
        }
        if self.pt_skin_pipeline.is_none() {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("pt_skin_shader"),
                    source: wgpu::ShaderSource::Wgsl(shaders::PT_SKIN_WGSL.into()),
                });
            let layout = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("pt_skin_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
            let pl = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pt_skin_pl"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
            self.pt_skin_pipeline = Some(self.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("pt_skin_pipeline"),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some("cs_skin"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.pt_skin_layout = Some(layout);
            self.created_pipelines(1);
        }
        while self.pt_skin_params.len() < dyn_count {
            self.pt_skin_params
                .push(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pt_skin_params"),
                    size: 80,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }

        self.pt_dyn_blas_frame = self.pt_dyn_blas_frame.wrapping_add(1);
        // Per-slot: params upload, BLAS pool upkeep, bind group.
        let mut bind_groups: Vec<wgpu::BindGroup> = Vec::with_capacity(dyn_count);
        let mut dispatch_counts: Vec<u32> = Vec::with_capacity(dyn_count);
        for i in 0..dyn_count {
            let w = &self.pt_dyn_windows[i];
            let mut words = [0f32; 20];
            for c in 0..4 {
                for r in 0..4 {
                    words[c * 4 + r] = w.model[c][r];
                }
            }
            words[16] = f32::from_bits(w.v_base);
            words[17] = f32::from_bits(w.v_count);
            words[18] = f32::from_bits(w.joint_offset.max(0.0) as u32);
            words[19] = 0.0;
            self.queue
                .write_buffer(&self.pt_skin_params[i], 0, bytemuck::cast_slice(&words));

            // A slot keeps its BLAS while it holds the same mesh. Where the
            // backend can refit (see `DYN_BLAS_REFIT`), the per-frame build of
            // a kept BLAS is an in-place refit, which keeps the tree of the
            // pose it was first built in; a staggered full rebuild every
            // `DYN_BLAS_REBUILD_INTERVAL` frames stops that tree from
            // drifting too far from the current pose.
            let stale = DYN_BLAS_REFIT
                && (self.pt_dyn_blas_frame.wrapping_add(i as u32 * 37)) % DYN_BLAS_REBUILD_INTERVAL
                    == 0;
            let need_new = match self.pt_dyn_blas.get(i) {
                Some(slot) => {
                    slot.v_count != w.v_count
                        || slot.i_count != w.i_count
                        || slot.cache_handle != w.cache_handle
                        || slot.mesh_idx != w.mesh_idx
                        || stale
                }
                None => true,
            };
            if need_new {
                // Without refit the BLAS is rebuilt from scratch every frame,
                // so build time dominates: on an M1 Max five characters (72k
                // triangles) cost 14.5 ms per frame as FAST_TRACE builds,
                // 4.5 ms as FAST_BUILD and under 1 ms as refits.
                let (flags, update_mode) = if DYN_BLAS_REFIT {
                    (
                        wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD
                            | wgpu::AccelerationStructureFlags::ALLOW_UPDATE,
                        wgpu::AccelerationStructureUpdateMode::PreferUpdate,
                    )
                } else {
                    (
                        wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
                        wgpu::AccelerationStructureUpdateMode::Build,
                    )
                };
                let blas = self.device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("pt_dyn_blas"),
                        flags,
                        update_mode,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![wgpu::BlasTriangleGeometrySizeDescriptor {
                            vertex_format: wgpu::VertexFormat::Float32x3,
                            vertex_count: w.v_count,
                            index_format: Some(wgpu::IndexFormat::Uint32),
                            index_count: Some(w.i_count),
                            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                        }],
                    },
                );
                let slot = PtDynBlas {
                    blas,
                    v_count: w.v_count,
                    i_count: w.i_count,
                    cache_handle: w.cache_handle,
                    mesh_idx: w.mesh_idx,
                };
                if i < self.pt_dyn_blas.len() {
                    self.pt_dyn_blas[i] = slot;
                } else {
                    self.pt_dyn_blas.push(slot);
                }
            }

            let src_vb = match self.model_gpu_cache.get(&w.cache_handle) {
                Some(Some(meshes)) => match meshes.get(w.mesh_idx) {
                    Some(m) => match &m.geometry {
                        gpu_driven::MeshGeometry::Dedicated { vertex, .. } => vertex,
                        // Dynamic PT windows are registered only for skinned
                        // cached meshes, which always retain dedicated input.
                        gpu_driven::MeshGeometry::Shared(_) => continue,
                    },
                    None => continue,
                },
                _ => continue,
            };
            bind_groups.push(
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pt_skin_bg"),
                    layout: self.pt_skin_layout.as_ref().unwrap(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.pt_skin_params[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: src_vb.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self
                                .pt_geo_vertex_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.joint_buffer.as_entire_binding(),
                        },
                    ],
                }),
            );
            dispatch_counts.push(w.v_count);
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pt_skin"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.pt_skin_pipeline.as_ref().unwrap());
        for (bg, vc) in bind_groups.iter().zip(dispatch_counts.iter()) {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(vc.div_ceil(64), 1, 1);
        }
    }
}
