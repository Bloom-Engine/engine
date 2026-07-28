use super::*;
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

pub(super) const GPU_RECEIVER_MIN_BOUNDS: usize = 1024;
pub(super) const GPU_RECEIVER_MAX_BOUNDS: usize = 4096;
const COVERAGE_BYTES: u64 = (receiver_demand::COVERAGE_ENTRIES * std::mem::size_of::<u32>()) as u64;

#[repr(C, align(16))]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReceiverParams {
    level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
    level_planes: [[[f32; 4]; 6]; VSM_CLIP_LEVELS as usize],
    meta: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReceiverBounds {
    bmin: [f32; 4],
    bmax: [f32; 4],
}

pub(super) struct CompletedReceiverDemand {
    pub(super) bounds_signature: u64,
    pub(super) level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
    pub(super) demand: Vec<VirtualShadowPage>,
    sequence: u64,
}

struct Readback {
    buffer: wgpu::Buffer,
    in_flight: bool,
    status: Arc<AtomicU8>,
    bounds_signature: u64,
    level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
    expected: Option<Vec<VirtualShadowPage>>,
    sequence: u64,
}

struct Resources {
    bounds_capacity: usize,
    bounds: wgpu::Buffer,
    params: wgpu::Buffer,
    coverage: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    readbacks: [Readback; 2],
}

#[derive(Copy, Clone, Debug, Default)]
pub(super) struct GpuReceiverStats {
    pub(super) dispatches: u32,
    pub(super) completions: u32,
    pub(super) validation_failures: u32,
}

pub(super) struct GpuReceiverDemand {
    enabled: bool,
    validated: bool,
    layout: Option<wgpu::BindGroupLayout>,
    pipeline: Option<wgpu::ComputePipeline>,
    resources: Option<Resources>,
    bounds_scratch: Vec<ReceiverBounds>,
    parity: usize,
    recorded_slot: Option<usize>,
    next_sequence: u64,
    latest_consumed_sequence: u64,
    pub(super) stats: GpuReceiverStats,
}

impl GpuReceiverDemand {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        let max_bounds_bytes =
            GPU_RECEIVER_MAX_BOUNDS as u64 * std::mem::size_of::<ReceiverBounds>() as u64;
        let enabled = cfg!(not(target_arch = "wasm32"))
            && gpu_receiver_requested()
            && limits.max_storage_buffers_per_shader_stage >= 2
            && limits.max_storage_buffer_binding_size >= max_bounds_bytes
            && limits.max_uniform_buffer_binding_size
                >= std::mem::size_of::<ReceiverParams>() as u64
            && limits.max_buffer_size >= max_bounds_bytes
            && limits.max_compute_invocations_per_workgroup >= 64
            && limits.max_compute_workgroup_size_x >= 64
            && limits.max_compute_workgroups_per_dimension
                >= (GPU_RECEIVER_MAX_BOUNDS as u32).div_ceil(64);
        Self {
            enabled,
            validated: false,
            layout: None,
            pipeline: None,
            resources: None,
            bounds_scratch: Vec::new(),
            parity: 0,
            recorded_slot: None,
            next_sequence: 1,
            latest_consumed_sequence: 0,
            stats: GpuReceiverStats::default(),
        }
    }

    pub(super) fn wants_gpu(&self, receiver_count: usize) -> bool {
        self.enabled
            && (GPU_RECEIVER_MIN_BOUNDS..=GPU_RECEIVER_MAX_BOUNDS).contains(&receiver_count)
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn validated(&self) -> bool {
        self.validated
    }

    pub(super) fn in_flight(&self) -> usize {
        self.resources.as_ref().map_or(0, |resources| {
            resources
                .readbacks
                .iter()
                .filter(|readback| readback.in_flight)
                .count()
        })
    }

    pub(super) fn memory_bytes(&self) -> u64 {
        self.resources.as_ref().map_or(0, |resources| {
            resources.bounds_capacity as u64 * std::mem::size_of::<ReceiverBounds>() as u64
                + std::mem::size_of::<ReceiverParams>() as u64
                + COVERAGE_BYTES * 3
        })
    }

    pub(super) fn poll(&mut self, device: &wgpu::Device) -> Option<CompletedReceiverDemand> {
        if !self.enabled {
            return None;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let resources = self.resources.as_mut()?;
        let mut latest = None;
        let mut failed = false;
        for readback in &mut resources.readbacks {
            if !readback.in_flight {
                continue;
            }
            match readback.status.load(Ordering::Acquire) {
                0 => continue,
                1 => {
                    let demand = {
                        let mapped = readback.buffer.slice(..).get_mapped_range();
                        let words: &[u32] = bytemuck::cast_slice(&mapped);
                        receiver_demand::compact_directional_coverage(
                            &words[..receiver_demand::COVERAGE_ENTRIES],
                            0,
                        )
                    };
                    readback.buffer.unmap();
                    if let Some(expected) = readback.expected.take() {
                        if expected != demand {
                            failed = true;
                            self.stats.validation_failures =
                                self.stats.validation_failures.saturating_add(1);
                        } else {
                            self.validated = true;
                        }
                    }
                    if !failed
                        && readback.sequence > self.latest_consumed_sequence
                        && latest
                            .as_ref()
                            .is_none_or(|completed: &CompletedReceiverDemand| {
                                readback.sequence > completed.sequence
                            })
                    {
                        latest = Some(CompletedReceiverDemand {
                            bounds_signature: readback.bounds_signature,
                            level_vps: readback.level_vps,
                            demand,
                            sequence: readback.sequence,
                        });
                    }
                    self.stats.completions = self.stats.completions.saturating_add(1);
                }
                _ => {
                    failed = true;
                    self.stats.validation_failures =
                        self.stats.validation_failures.saturating_add(1);
                }
            }
            readback.in_flight = false;
            readback.status.store(0, Ordering::Release);
        }
        if failed {
            self.enabled = false;
            self.validated = false;
            log::error!(
                "bloom: VSM GPU receiver marking failed validation; using fixed CPU oracle"
            );
            return None;
        }
        if let Some(completed) = latest.as_ref() {
            self.latest_consumed_sequence = completed.sequence;
        }
        latest
    }

    pub(super) fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        level_vps: [[[f32; 4]; 4]; VSM_CLIP_LEVELS as usize],
        receiver_bounds: &[([f32; 3], [f32; 3])],
        bounds_signature: u64,
        expected: Option<Vec<VirtualShadowPage>>,
    ) -> bool {
        if !self.wants_gpu(receiver_bounds.len()) || self.recorded_slot.is_some() {
            return false;
        }
        if !self.validated
            && self.resources.as_ref().is_some_and(|resources| {
                resources
                    .readbacks
                    .iter()
                    .any(|readback| readback.in_flight && readback.expected.is_some())
            })
        {
            return false;
        }
        if self.resources.as_ref().is_some_and(|resources| {
            resources.readbacks.iter().any(|readback| {
                readback.in_flight
                    && readback.bounds_signature == bounds_signature
                    && readback.level_vps == level_vps
            })
        }) {
            return false;
        }

        self.bounds_scratch.clear();
        self.bounds_scratch
            .extend(receiver_bounds.iter().map(|&(bmin, bmax)| ReceiverBounds {
                bmin: [bmin[0], bmin[1], bmin[2], 0.0],
                bmax: [bmax[0], bmax[1], bmax[2], 0.0],
            }));
        self.ensure_resources(device, receiver_bounds.len());
        let resources = self
            .resources
            .as_mut()
            .expect("GPU receiver resources were initialized");
        let slot = [self.parity, 1 - self.parity]
            .into_iter()
            .find(|&slot| !resources.readbacks[slot].in_flight);
        let Some(slot) = slot else {
            return false;
        };

        let level_planes =
            std::array::from_fn(|level| crate::scene::extract_frustum_planes(&level_vps[level]));
        let params = ReceiverParams {
            level_vps,
            level_planes,
            meta: [receiver_bounds.len() as u32, 0, 0, 0],
        };
        queue.write_buffer(
            &resources.bounds,
            0,
            bytemuck::cast_slice(&self.bounds_scratch),
        );
        queue.write_buffer(&resources.params, 0, bytemuck::bytes_of(&params));
        encoder.clear_buffer(&resources.coverage, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vsm_gpu_receiver_mark"),
                timestamp_writes: None,
            });
            pass.set_pipeline(
                self.pipeline
                    .as_ref()
                    .expect("enabled GPU receiver path owns its pipeline"),
            );
            pass.set_bind_group(0, &resources.bind_group, &[]);
            pass.dispatch_workgroups((receiver_bounds.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &resources.coverage,
            0,
            &resources.readbacks[slot].buffer,
            0,
            COVERAGE_BYTES,
        );

        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1).max(1);
        let readback = &mut resources.readbacks[slot];
        readback.in_flight = true;
        readback.bounds_signature = bounds_signature;
        readback.level_vps = level_vps;
        readback.expected = expected;
        readback.sequence = sequence;
        readback.status.store(0, Ordering::Release);
        self.recorded_slot = Some(slot);
        self.parity = 1 - slot;
        self.stats.dispatches = self.stats.dispatches.saturating_add(1);
        true
    }

    pub(super) fn after_submit(&mut self) {
        let Some(slot) = self.recorded_slot.take() else {
            return;
        };
        let Some(resources) = self.resources.as_mut() else {
            return;
        };
        let status = resources.readbacks[slot].status.clone();
        resources.readbacks[slot]
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                status.store(u8::from(result.is_err()) + 1, Ordering::Release);
            });
    }

    fn ensure_resources(&mut self, device: &wgpu::Device, required: usize) {
        if self.layout.is_none() {
            let layout = create_layout(device);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vsm_gpu_receiver_shader"),
                source: wgpu::ShaderSource::Wgsl(RECEIVER_SHADER.into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vsm_gpu_receiver_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vsm_gpu_receiver_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("cs_mark"),
                compilation_options: Default::default(),
                cache: None,
            });
            self.layout = Some(layout);
            self.pipeline = Some(pipeline);
        }
        let required = required
            .next_power_of_two()
            .clamp(GPU_RECEIVER_MIN_BOUNDS, GPU_RECEIVER_MAX_BOUNDS);
        if self
            .resources
            .as_ref()
            .is_some_and(|resources| resources.bounds_capacity >= required)
        {
            return;
        }
        let bounds = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm_gpu_receiver_bounds"),
            size: (required * std::mem::size_of::<ReceiverBounds>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(resources) = self.resources.as_mut() {
            resources.bounds_capacity = required;
            resources.bounds = bounds;
            resources.bind_group = create_bind_group(
                device,
                self.layout
                    .as_ref()
                    .expect("enabled GPU receiver path owns its layout"),
                &resources.bounds,
                &resources.params,
                &resources.coverage,
            );
            return;
        }

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm_gpu_receiver_params"),
            size: std::mem::size_of::<ReceiverParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let coverage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm_gpu_receiver_coverage"),
            size: COVERAGE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = self
            .layout
            .as_ref()
            .expect("enabled GPU receiver path owns its layout");
        let bind_group = create_bind_group(device, layout, &bounds, &params, &coverage);
        let readbacks = std::array::from_fn(|_| Readback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vsm_gpu_receiver_readback"),
                size: COVERAGE_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            in_flight: false,
            status: Arc::new(AtomicU8::new(0)),
            bounds_signature: 0,
            level_vps: [[[0.0; 4]; 4]; VSM_CLIP_LEVELS as usize],
            expected: None,
            sequence: 0,
        });
        self.resources = Some(Resources {
            bounds_capacity: required,
            bounds,
            params,
            coverage,
            bind_group,
            readbacks,
        });
    }
}

fn gpu_receiver_requested() -> bool {
    std::env::var("BLOOM_VSM_GPU_RECEIVER")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false" | "disabled"
            )
        })
        .unwrap_or(true)
}

fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vsm_gpu_receiver_layout"),
        entries: &[
            storage(0, true),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(2, false),
        ],
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bounds: &wgpu::Buffer,
    params: &wgpu::Buffer,
    coverage: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vsm_gpu_receiver_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: bounds.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: coverage.as_entire_binding(),
            },
        ],
    })
}

const RECEIVER_SHADER: &str = r#"
struct ReceiverBounds {
    bmin: vec4<f32>,
    bmax: vec4<f32>,
};
struct BoundsTable { values: array<ReceiverBounds>, };
struct ReceiverParams {
    level_vps: array<mat4x4<f32>, 3>,
    level_planes: array<array<vec4<f32>, 6>, 3>,
    counts: vec4<u32>,
};
struct CoverageTable { values: array<atomic<u32>>, };

@group(0) @binding(0) var<storage, read> bounds: BoundsTable;
@group(0) @binding(1) var<uniform> params: ReceiverParams;
@group(0) @binding(2) var<storage, read_write> coverage: CoverageTable;

fn outside_level(level: u32, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = params.level_planes[level][i];
        let p = select(bmin, bmax, plane.xyz >= vec3<f32>(0.0));
        if (dot(plane.xyz, p) + plane.w < 0.0) {
            return true;
        }
    }
    return false;
}

fn mark_level(level: u32, receiver: ReceiverBounds) {
    let bmin = receiver.bmin.xyz;
    let bmax = receiver.bmax.xyz;
    if (outside_level(level, bmin, bmax)) {
        return;
    }
    var ndc_min = vec2<f32>(1.0e30);
    var ndc_max = vec2<f32>(-1.0e30);
    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        let world = vec4<f32>(
            select(bmin.x, bmax.x, (corner & 1u) != 0u),
            select(bmin.y, bmax.y, (corner & 2u) != 0u),
            select(bmin.z, bmax.z, (corner & 4u) != 0u),
            1.0
        );
        let clip = params.level_vps[level] * world;
        if (abs(clip.w) > 1.0e-8) {
            let ndc = clip.xy / clip.w;
            ndc_min = min(ndc_min, ndc);
            ndc_max = max(ndc_max, ndc);
        }
    }
    let axis = 32.0;
    let uv_min = clamp(
        vec2<f32>(ndc_min.x * 0.5 + 0.5, 1.0 - (ndc_max.y * 0.5 + 0.5)),
        vec2<f32>(0.0),
        vec2<f32>(1.0)
    );
    let uv_max = clamp(
        vec2<f32>(ndc_max.x * 0.5 + 0.5, 1.0 - (ndc_min.y * 0.5 + 0.5)),
        vec2<f32>(0.0),
        vec2<f32>(1.0)
    );
    let min_page = clamp(
        vec2<i32>(floor(uv_min * axis)) - vec2<i32>(1),
        vec2<i32>(0),
        vec2<i32>(31)
    );
    let max_page = clamp(
        vec2<i32>(floor(uv_max * axis)) + vec2<i32>(1),
        vec2<i32>(0),
        vec2<i32>(31)
    );
    for (var y = min_page.y; y <= max_page.y; y = y + 1) {
        for (var x = min_page.x; x <= max_page.x; x = x + 1) {
            let index = level * 1024u + u32(y) * 32u + u32(x);
            atomicAdd(&coverage.values[index], 1u);
        }
    }
}

@compute @workgroup_size(64)
fn cs_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.x) {
        return;
    }
    let receiver = bounds.values[gid.x];
    if (receiver.bmin.x > receiver.bmax.x) {
        return;
    }
    for (var level = 0u; level < 3u; level = level + 1u) {
        mark_level(level, receiver);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    #[test]
    fn receiver_gpu_abi_is_bounded_and_aligned() {
        assert_eq!(std::mem::size_of::<ReceiverBounds>(), 32);
        assert_eq!(std::mem::size_of::<ReceiverParams>(), 496);
        assert_eq!(COVERAGE_BYTES, 12_288);
        assert_eq!(
            GPU_RECEIVER_MAX_BOUNDS * std::mem::size_of::<ReceiverBounds>(),
            131_072,
        );
    }

    #[test]
    fn receiver_marking_shader_parses() {
        if let Err(error) = wgpu::naga::front::wgsl::parse_str(RECEIVER_SHADER) {
            panic!("{}", error.emit_to_string(RECEIVER_SHADER));
        }
    }

    #[test]
    fn gpu_receiver_marking_matches_cpu_oracle_exactly() {
        let Some((device, queue)) = try_device() else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut marker = GpuReceiverDemand::new(&device);
        if !marker.wants_gpu(GPU_RECEIVER_MIN_BOUNDS) {
            eprintln!("skip: adapter does not support bounded GPU receiver marking");
            return;
        }
        let bounds: Vec<_> = (0..GPU_RECEIVER_MIN_BOUNDS)
            .map(|index| {
                let x = (index % 16) as f32 * 0.11 - 0.86;
                let y = (index / 16) as f32 * 0.11 - 0.86;
                ([x, y, 0.2], [x + 0.17, y + 0.19, 0.8])
            })
            .collect();
        let level_vps = [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize];
        let expected = directional_receiver_demand(level_vps, &bounds, 0);
        let signature = receiver_bounds_signature(&bounds);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vsm_gpu_receiver_test"),
        });
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        assert!(marker.record(
            &device,
            &queue,
            &mut encoder,
            level_vps,
            &bounds,
            signature,
            Some(expected.clone()),
        ));
        queue.submit(std::iter::once(encoder.finish()));
        marker.after_submit();
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let error = pollster::block_on(error_scope.pop());
        assert!(error.is_none(), "GPU receiver validation error: {error:?}");
        let completed = marker
            .poll(&device)
            .expect("submitted GPU receiver result completed");
        assert_eq!(completed.demand, expected);
        assert!(marker.validated());
        assert_eq!(marker.stats.validation_failures, 0);
    }
}
