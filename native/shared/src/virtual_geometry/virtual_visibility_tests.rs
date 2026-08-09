use super::*;
use crate::renderer::visibility_buffer::{
    VisibilityDraw, VisibilityRecord, GEOMETRY_WGSL, INVALID_DRAW_ID, RECONSTRUCTION_WGSL,
    VISIBILITY_FORMAT,
};
use crate::renderer::Renderer;

const VIRTUAL_RENDER_ABI_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/render_abi.wgsl");
const VIRTUAL_DECODE_WGSL: &str = include_str!("../../shaders/virtual_geometry/decode.wgsl");
const VIRTUAL_VISIBILITY_RECONSTRUCT_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/visibility_reconstruct.wgsl");
const VIRTUAL_VISIBILITY_SHADING_PROBE_WGSL: &str =
    include_str!("../../shaders/virtual_geometry/visibility_shading_probe.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VirtualShadingProbeRecord {
    result_info: [u32; 4],
    identity: [u32; 4],
    barycentrics: [f32; 4],
    current_clip: [f32; 4],
    previous_clip: [f32; 4],
    world_position: [f32; 4],
    world_normal: [f32; 4],
    world_tangent: [f32; 4],
    uv: [f32; 4],
    color: [f32; 4],
}

#[test]
fn raw_virtual_clusters_rasterize_namespaced_visibility_ids_on_the_real_gpu() {
    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 16;
    const ROW_BYTES: u32 = 256;

    let Some((device, queue)) = try_traversal_device() else {
        eprintln!("no GPU adapter — skipping virtual visibility raster oracle");
        return;
    };
    let required = wgpu::Features::PRIMITIVE_INDEX | wgpu::Features::INDIRECT_FIRST_INSTANCE;
    if !device.features().contains(required) {
        eprintln!("adapter lacks primitive-index/indirect-first-instance — skipping oracle");
        return;
    }

    let mut archive = hierarchy_archive();
    for cluster in &mut archive.clusters {
        cluster.flags |= bloom_geometry_format::FLAG_DOUBLE_SIDED;
    }
    let mut pool = GpuVirtualGeometryPool::new(&device, gpu_config(5)).unwrap();
    let mesh = pool
        .register_mesh(&queue, hierarchy_asset(archive))
        .unwrap();
    make_hierarchy_fully_resident(&mut pool, &queue, mesh);
    let selector = GpuVirtualHierarchySelector::new(&device, &pool, traversal_config()).unwrap();
    let emitter = GpuVirtualDrawEmitter::new(&device, &selector).unwrap();
    let raster = GpuVirtualVisibilityRaster::new(&device, &pool, &selector, &emitter).unwrap();
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    raster
        .prepare_frame(
            &queue,
            GpuVirtualVisibilityFrame::new(identity, identity).unwrap(),
        )
        .unwrap();

    let current_model = [
        [-0.8, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.2, -0.2, 0.0, 1.0],
    ];
    let previous_model = [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.1, -0.1, 0.0, 1.0],
    ];
    let tint = [0.25, 0.5, 0.75, 0.8];
    let instance =
        GpuVirtualInstance::with_render_state(mesh, 901, current_model, previous_model, tint)
            .unwrap();

    let visibility = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("virtual_visibility_oracle_ids"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: VISIBILITY_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("virtual_visibility_oracle_depth"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::renderer::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let readback_source = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("virtual_visibility_oracle_copy"),
        size: u64::from(ROW_BYTES * HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let visibility_view = visibility.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());
    let probe_source = [
        RECONSTRUCTION_WGSL,
        GEOMETRY_WGSL,
        VIRTUAL_RENDER_ABI_WGSL,
        VIRTUAL_DECODE_WGSL,
        VIRTUAL_VISIBILITY_RECONSTRUCT_WGSL,
        VIRTUAL_VISIBILITY_SHADING_PROBE_WGSL,
    ]
    .join("\n");
    wgpu::naga::front::wgsl::parse_str(&probe_source)
        .unwrap_or_else(|error| panic!("virtual PBR reconstruction probe WGSL failed: {error:?}"));
    let probe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("virtual_visibility_pbr_reconstruction_probe_shader"),
        source: wgpu::ShaderSource::Wgsl(probe_source.into()),
    });
    let probe_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("virtual_visibility_pbr_reconstruction_probe_pipeline"),
        layout: None,
        module: &probe_shader,
        entry_point: Some("cs_virtual_visibility_shading_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let probe_bytes =
        u64::from(WIDTH * HEIGHT) * std::mem::size_of::<VirtualShadingProbeRecord>() as u64;
    let probe_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("virtual_visibility_pbr_reconstruction_probe_output"),
        size: probe_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let probe_layout = probe_pipeline.get_bind_group_layout(0);
    let probe_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("virtual_visibility_pbr_reconstruction_probe_bind_group"),
        layout: &probe_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&visibility_view),
            },
            gpu_buffer_binding(1, pool.physical_buffer()),
            gpu_buffer_binding(2, pool.cluster_table_buffer()),
            gpu_buffer_binding(3, selector.selected_buffer()),
            gpu_buffer_binding(4, selector.instance_buffer()),
            gpu_buffer_binding(5, raster.frame_buffer()),
            gpu_buffer_binding(6, &probe_output),
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("virtual_visibility_oracle_encoder"),
    });
    selector
        .record(
            &queue,
            &mut encoder,
            &pool,
            &[instance],
            traversal_view(50.0),
        )
        .unwrap();
    emitter.record(&queue, &mut encoder, &selector).unwrap();
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("virtual_visibility_oracle_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &visibility_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(u32::MAX),
                        g: f64::from(u32::MAX),
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        raster.draw_fixed_for_test(&mut pass, &emitter, 4).unwrap();
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("virtual_visibility_pbr_reconstruction_probe_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&probe_pipeline);
        pass.set_bind_group(0, &probe_bind_group, &[]);
        pass.dispatch_workgroups(WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &visibility,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_source,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let bytes = read_gpu_buffer(
        &device,
        &queue,
        &readback_source,
        u64::from(ROW_BYTES * HEIGHT),
    );
    let probe_raw = read_gpu_buffer(&device, &queue, &probe_output, probe_bytes);
    let probe: &[VirtualShadingProbeRecord] = bytemuck::cast_slice(&probe_raw);
    let mut covered = 0usize;
    let mut background = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = (y * ROW_BYTES + x * 8) as usize;
            let words: &[u32] = bytemuck::cast_slice(&bytes[offset..offset + 8]);
            let record = VisibilityRecord {
                draw_id: words[0],
                primitive_and_face: words[1],
            };
            if record.draw_id == INVALID_DRAW_ID {
                background += 1;
                continue;
            }
            let Some((VisibilityDraw::Virtual(draw_index), primitive, _)) = record.decode_draw()
            else {
                panic!("virtual raster emitted a compatibility or invalid visibility ID");
            };
            assert!(draw_index < 4);
            assert_eq!(primitive, 0);
            let shading = probe[(y * WIDTH + x) as usize];
            assert_eq!(shading.result_info, [1, draw_index, 0, 1]);
            assert_eq!(shading.identity[1], 0);
            assert_ne!(shading.identity[2] & 2, 0);
            assert_eq!(shading.identity[3] & 2, 0);
            let bary = shading.barycentrics;
            probe_close(bary[0] + bary[1] + bary[2], 1.0);
            let local_x = bary[1];
            let local_y = bary[2];
            let expected_world = [-0.8 * local_x + 0.2, 0.8 * local_y - 0.2, 0.0, 1.0];
            let expected_previous = [0.5 * local_x + 0.1, 0.5 * local_y - 0.1, 0.0, 1.0];
            for lane in 0..4 {
                probe_close(shading.world_position[lane], expected_world[lane]);
                probe_close(shading.current_clip[lane], expected_world[lane]);
                probe_close(shading.previous_clip[lane], expected_previous[lane]);
            }
            for (actual, expected) in shading.world_normal.iter().zip([0.0, 0.0, 1.0, 0.0]) {
                probe_close(*actual, expected);
            }
            for (actual, expected) in shading.world_tangent.iter().zip([-1.0, 0.0, 0.0, -1.0]) {
                probe_close(*actual, expected);
            }
            probe_close(shading.uv[0], local_x);
            probe_close(shading.uv[1], local_y);
            for (actual, expected) in shading.color.iter().zip([0.25, 0.25, 0.1875, 0.8]) {
                probe_close(*actual, expected);
            }
            covered += 1;
        }
    }
    assert!(covered > 0);
    assert!(background > 0);
    assert_eq!(
        raster.counted_submission_supported(),
        device
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
    );
}

fn probe_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 2.0e-5,
        "virtual shading probe mismatch: GPU {actual}, expected {expected}"
    );
}

#[test]
fn production_renderer_constructs_virtual_four_mrt_pipeline_on_the_real_gpu() {
    let Some(renderer) = try_virtual_pbr_renderer() else {
        eprintln!("adapter lacks the native Tier-A virtual-PBR contract — skipping oracle");
        return;
    };
    let config = GpuVirtualTraversalConfig {
        max_instances: 1,
        max_selected_clusters: 4,
        max_page_requests: 1,
    };
    let pool = GpuVirtualGeometryPool::new(&renderer.device, gpu_config(1)).unwrap();
    let selector = GpuVirtualHierarchySelector::new(&renderer.device, &pool, config).unwrap();
    let emitter = GpuVirtualDrawEmitter::new(&renderer.device, &selector).unwrap();
    let raster =
        GpuVirtualVisibilityRaster::new(&renderer.device, &pool, &selector, &emitter).unwrap();
    let visibility = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("virtual_visibility_production_pbr_pipeline_ids"),
        size: wgpu::Extent3d {
            width: 16,
            height: 16,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: VISIBILITY_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    renderer
        .create_virtual_visibility_shading(&pool, &selector, &raster, &visibility)
        .expect("production renderer layouts must compile the virtual four-MRT PBR pipeline");
}

fn try_virtual_pbr_renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let required = wgpu::Features::PRIMITIVE_INDEX
        | wgpu::Features::INDIRECT_FIRST_INSTANCE
        | crate::renderer::material_indirection::TIER_A_FEATURES;
    if !adapter.features().contains(required) {
        return None;
    }
    let options = crate::renderer::device_negotiation::DeviceRequestOptions {
        allow_ray_query: false,
        ..Default::default()
    };
    let plans = crate::renderer::device_negotiation::build_device_request_plans(
        adapter.features(),
        &adapter.limits(),
        options,
    )
    .ok()?;
    let plan = plans.first()?;
    if plan.required_limits.max_bind_groups < 5
        || plan.required_limits.max_storage_buffers_per_shader_stage < 8
        || plan.required_limits.max_color_attachments < 4
    {
        return None;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("virtual_visibility_production_pbr_test_device"),
        required_features: plan.required_features | required,
        required_limits: plan.required_limits.clone(),
        experimental_features: plan.experimental_features.clone(),
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))
    .ok()?;
    Some(Renderer::new_headless(device, queue, 16, 16))
}
