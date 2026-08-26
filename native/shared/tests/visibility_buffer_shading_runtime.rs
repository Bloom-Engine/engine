//! Opt-in #27 full-PBR visibility shading smoke.
//!
//! This test has its own process because device feature selection reads the
//! runtime mode once. Eligible forward fragments are suppressed in `shade`
//! mode, so visible green coverage proves the reconstructed fragment inputs
//! reached Bloom's authoritative PBR evaluator and all four MRT outputs.

use bloom_geometry_format::{
    sha256, CLUSTER_RECORD_BYTES, ENDIAN_TAG, FLAG_COARSE_ROOT, FLAG_DOUBLE_SIDED, HEADER_BYTES,
    MAGIC, MIN_PAGE_BYTES, NO_RELATION, PAGE_RECORD_BYTES, VERSION,
};
use bloom_shared::{
    models::MaterialLayeredPbr,
    renderer::Vertex3D,
    virtual_geometry::{
        GpuVirtualGeometryConfig, GpuVirtualInstance, GpuVirtualTraversalConfig,
        GpuVirtualVisibilityFrame, VirtualGeometryAsset, VirtualGeometryView,
        VirtualMaterialBinding,
    },
};
use std::sync::Arc;

#[test]
fn opt_in_visibility_shading_replaces_eligible_forward_pixels() {
    unsafe { std::env::set_var("BLOOM_VISIBILITY_BUFFER", "shade") };
    unsafe { std::env::set_var("BLOOM_SKIP_SKY", "1") };
    let mut engine =
        match bloom_shared::attach::attach_headless_engine(wgpu::Backends::PRIMARY, 128, 128) {
            Ok(engine) => engine,
            Err(error) if error.contains("no compatible adapter") => {
                eprintln!("skip: no native GPU adapter ({error})");
                return;
            }
            Err(error) => panic!("production renderer device negotiation failed: {error}"),
        };
    engine.renderer.set_render_scale(1.0);
    engine.renderer.set_taa_enabled(false);

    let initial: serde_json::Value =
        serde_json::from_str(&engine.renderer.renderer_capability_report_json())
            .expect("initial capability report is JSON");
    let runtime = &initial["runtime_support"]["gpu_driven"]["visibility_buffer_runtime"];
    if runtime["enabled"] != true {
        eprintln!(
            "skip: requested visibility shading unavailable ({})",
            runtime["disabled_reason"]
        );
        return;
    }

    let vertices = vec![
        Vertex3D {
            position: [-0.35, -0.35, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.1, 0.9, 0.2, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
        Vertex3D {
            position: [0.35, -0.35, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.1, 0.9, 0.2, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
        Vertex3D {
            position: [0.0, 0.35, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [0.1, 0.9, 0.2, 1.0],
            uv: [0.5, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        },
    ];
    for index in 0..32 {
        let node = engine.scene.create_node();
        engine
            .scene
            .update_geometry(node, vertices.clone(), vec![0, 1, 2]);
        let column = (index % 8) as f32;
        let row = (index / 8) as f32;
        engine
            .scene
            .set_trs(node, (column - 3.5) * 0.7, (row - 1.5) * 0.7, 0.0, 0.0, 1.0);
    }
    let compatibility = engine.scene.create_node();
    engine
        .scene
        .update_geometry(compatibility, vertices.clone(), vec![0, 1, 2]);
    engine.scene.set_trs(compatibility, 0.0, 1.8, 0.0, 0.0, 1.0);
    engine
        .scene
        .set_material_color(compatibility, 0.8, 0.15, 0.7, 1.0);
    engine.scene.set_material_layered_pbr(
        compatibility,
        MaterialLayeredPbr {
            clearcoat_authored: true,
            clearcoat_factor: 0.8,
            clearcoat_roughness_factor: 0.25,
            ..Default::default()
        },
    );

    engine.begin_frame();
    engine.renderer.set_clear_color(0.08, 0.01, 0.01, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(255.0, 255.0, 255.0, 1.0);
    engine.renderer.screenshot_requested = true;
    engine.end_frame();

    let (_, _, mut rgba) = engine
        .renderer
        .screenshot_data
        .take()
        .expect("visibility-shaded frame produced a screenshot");
    if matches!(
        engine.renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    let green_pixels = rgba
        .chunks_exact(4)
        .filter(|pixel| {
            pixel[1] > pixel[0].saturating_add(30)
                && pixel[1] > pixel[2].saturating_add(30)
                && pixel[3] > 200
        })
        .count();
    assert!(
        green_pixels > 256,
        "eligible forward pixels were suppressed but visibility PBR produced no coverage"
    );
    let report: serde_json::Value =
        serde_json::from_str(&engine.renderer.renderer_capability_report_json())
            .expect("post-frame capability report is JSON");
    let gpu_driven = &report["runtime_support"]["gpu_driven"];
    assert_eq!(gpu_driven["visibility_routed_indirect_streams"], true);
    assert!(gpu_driven["visibility_routed_indirect_bytes"]
        .as_u64()
        .is_some_and(|bytes| bytes > 0));
    let runtime = &gpu_driven["visibility_buffer_runtime"];
    assert_eq!(runtime["requested_mode"], "shade");
    assert_eq!(runtime["pbr_shading"], true);
    assert_eq!(runtime["forward_authoritative"], false);
    assert_eq!(
        runtime["composition"],
        "visibility-eligible+forward-compatibility"
    );
    assert!(runtime["eligible_draws"]
        .as_u64()
        .is_some_and(|value| value >= 32));
    assert_eq!(runtime["compatibility_draws"], 1);
    assert_eq!(runtime["allocated_bytes"], 128 * 128 * 8);
    assert_eq!(
        report["runtime_support"]["virtual_geometry"]["enabled"],
        false
    );

    engine
        .renderer
        .enable_virtual_geometry(
            GpuVirtualGeometryConfig {
                capacity_bytes: u64::from(bloom_geometry_format::MIN_PAGE_BYTES),
                page_stride_bytes: bloom_geometry_format::MIN_PAGE_BYTES,
                max_meshes: 1,
                max_page_records: 4,
                max_cluster_records: 4,
                max_clusters_per_group: 4,
                max_hierarchy_levels: 4,
                max_upload_bytes_per_frame: u64::from(bloom_geometry_format::MIN_PAGE_BYTES),
                max_upload_pages_per_frame: 1,
                max_evictions_per_frame: 1,
            },
            GpuVirtualTraversalConfig {
                max_instances: 1,
                max_selected_clusters: 4,
                max_page_requests: 4,
            },
        )
        .expect("the negotiated visibility-shade device accepts virtual registration");
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    engine.begin_frame();
    engine.renderer.set_clear_color(0.08, 0.01, 0.01, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(255.0, 255.0, 255.0, 1.0);
    engine
        .renderer
        .submit_virtual_geometry(
            &[],
            VirtualGeometryView {
                frustum_planes: [[0.0; 4]; 6],
                view_projection: identity,
                camera_position: [0.0, 0.0, 6.0],
                projection_scale: 64.0,
                target_error_pixels: 1.0,
            },
            GpuVirtualVisibilityFrame::new(identity, identity).unwrap(),
        )
        .expect("empty virtual batches are valid and preserve ordinary routing");
    engine.renderer.screenshot_requested = true;
    engine.end_frame();
    let (_, _, mut virtual_noop_rgba) = engine
        .renderer
        .screenshot_data
        .take()
        .expect("registered virtual no-op frame produced a screenshot");
    if matches!(
        engine.renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in virtual_noop_rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    assert_eq!(
        virtual_noop_rgba, rgba,
        "an empty registered virtual batch changed ordinary visibility/compatibility pixels"
    );

    let queue = engine.renderer.queue.clone();
    let virtual_mesh = {
        let pool = engine
            .renderer
            .virtual_geometry_pool_mut()
            .expect("enabled renderer owns its virtual pool");
        let mesh = pool
            .register_mesh(&queue, Arc::new(virtual_triangle_asset()))
            .expect("minimal virtual triangle registers");
        pool.bind_mesh_materials(
            &queue,
            mesh,
            &[VirtualMaterialBinding {
                source_material_index: Some(0),
                material_id: 1,
            }],
        )
        .expect("virtual triangle maps to the renderer's default material");
        mesh
    };
    let model = [
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-0.4, -0.4, 0.0, 1.0],
    ];
    let virtual_instance =
        GpuVirtualInstance::with_render_state(virtual_mesh, 7, model, model, [1.0, 1.0, 1.0, 1.0])
            .unwrap();
    engine.begin_frame();
    engine.renderer.set_clear_color(0.08, 0.01, 0.01, 1.0);
    engine
        .renderer
        .begin_mode_3d(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
    engine.renderer.set_ambient_light(255.0, 255.0, 255.0, 1.0);
    engine
        .renderer
        .submit_virtual_geometry(
            &[virtual_instance],
            VirtualGeometryView {
                frustum_planes: [[0.0; 4]; 6],
                view_projection: identity,
                camera_position: [0.0, 0.0, 6.0],
                projection_scale: 64.0,
                target_error_pixels: 1.0,
            },
            GpuVirtualVisibilityFrame::new(identity, identity).unwrap(),
        )
        .expect("virtual triangle frame submission is bounded");
    engine.renderer.screenshot_requested = true;
    engine.end_frame();
    let (_, _, mut virtual_rgba) = engine
        .renderer
        .screenshot_data
        .take()
        .expect("virtual geometry frame produced a screenshot");
    if matches!(
        engine.renderer.surface_format(),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for pixel in virtual_rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    let changed_pixels = virtual_rgba
        .chunks_exact(4)
        .zip(rgba.chunks_exact(4))
        .filter(|(virtual_pixel, ordinary_pixel)| virtual_pixel != ordinary_pixel)
        .count();
    assert!(
        changed_pixels > 256,
        "registered virtual producer did not compose visible four-MRT coverage"
    );
    for (pixel_index, (virtual_pixel, ordinary_pixel)) in virtual_rgba
        .chunks_exact(4)
        .zip(rgba.chunks_exact(4))
        .enumerate()
    {
        let x = pixel_index % 128;
        let y = pixel_index / 128;
        if !(24..104).contains(&x) || !(24..104).contains(&y) {
            assert_eq!(
                virtual_pixel, ordinary_pixel,
                "virtual composition changed an unrelated pixel at ({x}, {y})"
            );
        }
    }
    let registered_report: serde_json::Value =
        serde_json::from_str(&engine.renderer.renderer_capability_report_json())
            .expect("registered virtual capability report is JSON");
    let virtual_geometry = &registered_report["runtime_support"]["virtual_geometry"];
    assert_eq!(virtual_geometry["enabled"], true);
    assert_eq!(virtual_geometry["frame_requested"], true);
    assert_eq!(virtual_geometry["frame_prepared"], true);
    assert_eq!(virtual_geometry["instances"], 1);
    assert_eq!(virtual_geometry["active_meshes"], 1);
    assert!(virtual_geometry["total_gpu_bytes"]
        .as_u64()
        .is_some_and(|bytes| bytes > 0));
    let steady: serde_json::Value =
        serde_json::from_str(&engine.renderer.quality_runtime_paths_json())
            .expect("steady-state runtime report is JSON");
    assert_eq!(
        steady["steady_state_resources"]["bind_group_creations"]["sites"]["visibility_buffer"],
        0
    );
    assert_eq!(
        steady["steady_state_resources"]["transient_physical_creations"]["textures"],
        0
    );
}

fn virtual_triangle_asset() -> VirtualGeometryAsset {
    let mut payload = Vec::new();
    for position in [
        [0.0_f32, 0.0, 0.0],
        [1.0_f32, 0.0, 0.0],
        [0.0_f32, 1.0, 0.0],
    ] {
        for value in position
            .into_iter()
            .chain([0.0, 0.0, 1.0])
            .chain([1.0, 0.0, 0.0, 1.0])
            .chain([position[0], position[1]])
            .chain([position[0] * 0.5, position[1] * 0.5])
            .chain([1.0, 0.5, 0.25, 1.0])
        {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    let index_offset = payload.len() as u64;
    payload.extend_from_slice(&[0, 1, 2]);
    payload.resize(payload.len().div_ceil(16) * 16, 0);
    let payload_hash = sha256(&payload);
    let page_table_offset = HEADER_BYTES + CLUSTER_RECORD_BYTES;
    let payload_offset = page_table_offset + PAGE_RECORD_BYTES;
    let file_bytes = payload_offset + payload.len();
    let mut bytes = Vec::with_capacity(file_bytes);
    bytes.extend_from_slice(&MAGIC);
    push_u32(&mut bytes, VERSION);
    push_u32(&mut bytes, HEADER_BYTES as u32);
    push_u32(&mut bytes, ENDIAN_TAG);
    push_u32(&mut bytes, 0);
    bytes.extend_from_slice(&[1; 32]);
    bytes.extend_from_slice(&payload_hash);
    push_u32(&mut bytes, 1);
    push_u32(&mut bytes, 1);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u64(&mut bytes, HEADER_BYTES as u64);
    push_u64(&mut bytes, page_table_offset as u64);
    push_u64(&mut bytes, payload_offset as u64);
    push_u64(&mut bytes, payload_offset as u64);
    push_u64(&mut bytes, payload.len() as u64);
    push_u64(&mut bytes, file_bytes as u64);
    push_u32(&mut bytes, MIN_PAGE_BYTES);
    push_u32(&mut bytes, 0);

    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, FLAG_COARSE_ROOT | FLAG_DOUBLE_SIDED);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 3);
    push_u32(&mut bytes, 1);
    push_u32(&mut bytes, 0);
    push_u64(&mut bytes, 0);
    push_u64(&mut bytes, index_offset);
    push_f32x3(&mut bytes, [0.0, 0.0, 0.0]);
    push_f32x3(&mut bytes, [1.0, 1.0, 0.0]);
    push_f32x3(&mut bytes, [0.5, 0.5, 0.0]);
    push_f32(&mut bytes, 1.0);
    push_f32x3(&mut bytes, [0.0, 0.0, 1.0]);
    push_f32(&mut bytes, -1.0);
    push_f32(&mut bytes, 0.0);
    push_u32(&mut bytes, NO_RELATION);
    push_u32(&mut bytes, NO_RELATION);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 72);
    push_u32(&mut bytes, 0);
    assert_eq!(bytes.len(), page_table_offset);

    push_u64(&mut bytes, 0);
    push_u32(&mut bytes, payload.len() as u32);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 1);
    push_u32(&mut bytes, 0);
    bytes.extend_from_slice(&payload_hash);
    push_u64(&mut bytes, 0);
    assert_eq!(bytes.len(), payload_offset);
    bytes.extend_from_slice(&payload);
    VirtualGeometryAsset::from_bytes(bytes).expect("minimal virtual triangle fixture is valid")
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32x3(bytes: &mut Vec<u8>, value: [f32; 3]) {
    for component in value {
        push_f32(bytes, component);
    }
}
