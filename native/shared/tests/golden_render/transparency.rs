use super::*;

#[test]
fn layered_pbr_runs_on_opaque_sorted_reactive_and_weighted_paths() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    // This test compares submission paths, not GI history. SceneGraph and
    // cached-model visibility changes legitimately select different SSGI
    // tracing backends, so remove that unrelated temporal signal.
    eng.renderer.set_ssgi_enabled(false);
    let (vertices, indices) = cube_verts(0.85, [0.82, 0.18, 0.05, 1.0]);
    let layered = MaterialLayeredPbr {
        clearcoat_authored: true,
        clearcoat_factor: 0.9,
        clearcoat_roughness_factor: 0.14,
        specular_authored: true,
        specular_factor: 0.7,
        ior_authored: true,
        ior: 1.5,
        sheen_authored: true,
        sheen_color_factor: [0.24, 0.055, 0.018],
        sheen_roughness_factor: 0.42,
        anisotropy_authored: true,
        anisotropy_strength: 0.68,
        anisotropy_rotation: 0.37,
        ..Default::default()
    };
    let node = eng.scene.create_node();
    eng.scene
        .update_geometry(node, vertices.clone(), indices.clone());
    eng.scene.set_material_pbr(node, 0.35, 0.0);
    eng.scene.set_material_layered_pbr(node, layered);

    let draw_scene = |eng: &mut EngineState| {
        let renderer = &mut eng.renderer;
        renderer.set_clear_color(0.02, 0.025, 0.04, 1.0);
        renderer.begin_mode_3d(0.0, 0.0, 4.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
        renderer.set_ambient_light(45.0, 50.0, 60.0, 0.35);
        renderer.set_directional_light(-0.4, 0.75, 0.5, 255.0, 245.0, 225.0, 2.2);
    };
    let (_, _, retained_opaque) = render(&mut eng, 2, draw_scene);

    // The cached-model route has separate material/geometry selection logic;
    // require it to remain visually equivalent to the retained route.
    eng.scene.set_visible(node, false);
    const HANDLE: u64 = 0x1340_0001;
    assert!(eng.renderer.cache_model_if_static(
        HANDLE,
        &[MeshData {
            vertices,
            secondary_tex_coords: None,
            indices,
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 0.35,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: false,
            transmission: Default::default(),
            layered_pbr: layered,
        }]
    ));
    let (_, _, cached_opaque) = render(&mut eng, 2, |eng| {
        draw_scene(eng);
        eng.renderer
            .draw_model_cached(HANDLE, [0.0; 3], 1.0, [1.0; 4]);
    });
    let parity = calculate_diff_metrics(&retained_opaque, &cached_opaque, W, H);
    assert!(
        parity.mean_rgb <= 0.5,
        "retained/cached layered-PBR paths diverged: {parity:?}"
    );

    eng.scene.set_visible(node, true);
    eng.scene
        .set_material_gltf_alpha(node, MaterialAlphaMode::Blend, 0.0, false);
    eng.scene.set_material_color(node, 1.0, 1.0, 1.0, 0.62);
    eng.renderer.set_transparency_composition_mode(0);
    eng.renderer.set_taa_enabled(false);
    let (_, _, sorted) = render(&mut eng, 2, draw_scene);
    assert_eq!(eng.renderer.active_transparency_composition_mode_code(), 0);

    eng.renderer.set_taa_enabled(true);
    let (_, _, reactive) = render(&mut eng, 3, draw_scene);
    assert_eq!(eng.renderer.active_transparency_composition_mode_code(), 0);

    eng.renderer.set_transparency_composition_mode(2);
    let (_, _, weighted) = render(&mut eng, 3, draw_scene);
    assert_eq!(eng.renderer.active_transparency_composition_mode_code(), 1);

    let center = ((H / 2 * W + W / 2) * 4) as usize;
    for (label, image) in [
        ("sorted", sorted),
        ("reactive", reactive),
        ("weighted", weighted),
    ] {
        let center_rgb = &image[center..center + 3];
        let corner_rgb = &image[..3];
        let contrast = center_rgb
            .iter()
            .zip(corner_rgb)
            .map(|(center, corner)| center.abs_diff(*corner) as u32)
            .sum::<u32>();
        assert!(
            contrast >= 12,
            "{label} layered transparency did not contribute at the center: \
             center={center_rgb:?}, corner={corner_rgb:?}"
        );
    }

    // Transmission owns a dedicated scene-snapshot composition pass. Verify
    // that all current layered lobes remain active there for both submission
    // APIs and for the TAA-reactive attachment variant.
    eng.renderer.set_transparency_composition_mode(0);
    eng.renderer.set_taa_enabled(true);
    eng.scene
        .set_material_gltf_alpha(node, MaterialAlphaMode::Opaque, 0.0, false);
    eng.scene.set_material_color(node, 1.0, 1.0, 1.0, 1.0);
    let transmission = MaterialTransmission {
        authored: true,
        factor: 0.82,
        ior_authored: true,
        ior: 1.5,
        ..Default::default()
    };
    eng.scene.set_material_transmission(node, transmission);
    let draw_refractive_background = |eng: &mut EngineState| {
        draw_scene(eng);
        eng.renderer
            .draw_cube(0.0, 0.0, -1.7, 2.0, 2.0, 0.25, 40.0, 150.0, 235.0, 255.0);
    };
    let (_, _, retained_refractive) = render(&mut eng, 3, draw_refractive_background);

    eng.scene.set_visible(node, false);
    let (refractive_vertices, refractive_indices) = cube_verts(0.85, [0.82, 0.18, 0.05, 1.0]);
    const REFRACTIVE_HANDLE: u64 = 0x1340_0002;
    assert!(eng.renderer.cache_model_if_static(
        REFRACTIVE_HANDLE,
        &[MeshData {
            vertices: refractive_vertices,
            secondary_tex_coords: None,
            indices: refractive_indices,
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 0.35,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: false,
            transmission,
            layered_pbr: layered,
        }]
    ));
    let (_, _, cached_refractive) = render(&mut eng, 3, |eng| {
        draw_refractive_background(eng);
        eng.renderer
            .draw_model_cached(REFRACTIVE_HANDLE, [0.0; 3], 1.0, [1.0; 4]);
    });
    let refractive_parity = calculate_diff_metrics(&retained_refractive, &cached_refractive, W, H);
    assert!(
        refractive_parity.mean_rgb <= 1.0
            && refractive_parity.outlier_pixel_fraction <= 0.01
            && refractive_parity.ssim >= 0.98,
        "retained/cached layered refraction diverged: {refractive_parity:?}"
    );

    let white_layer_texture = eng
        .renderer
        .register_texture_kind(2, 2, &[255; 2 * 2 * 4], false);
    let anisotropy_texels = [255, 128, 255, 255].repeat(4);
    let neutral_anisotropy_texture =
        eng.renderer
            .register_texture_kind(2, 2, &anisotropy_texels, false);
    let mut layered_uv1 = layered;
    layered_uv1.clearcoat_texture = Some(MaterialTextureBinding {
        source_texture_index: 0,
        source_image_index: 0,
        runtime_texture_idx: Some(white_layer_texture),
        transform: MaterialTextureTransform {
            tex_coord: 1,
            ..Default::default()
        },
    });
    layered_uv1.sheen_color_texture = Some(MaterialTextureBinding {
        source_texture_index: 0,
        source_image_index: 0,
        runtime_texture_idx: Some(white_layer_texture),
        transform: MaterialTextureTransform {
            tex_coord: 1,
            ..Default::default()
        },
    });
    layered_uv1.sheen_roughness_texture = Some(MaterialTextureBinding {
        source_texture_index: 0,
        source_image_index: 0,
        runtime_texture_idx: Some(white_layer_texture),
        transform: MaterialTextureTransform {
            tex_coord: 1,
            ..Default::default()
        },
    });
    layered_uv1.anisotropy_texture = Some(MaterialTextureBinding {
        source_texture_index: 0,
        source_image_index: 0,
        runtime_texture_idx: Some(neutral_anisotropy_texture),
        transform: MaterialTextureTransform {
            tex_coord: 1,
            ..Default::default()
        },
    });
    let (uv1_vertices, uv1_indices) = cube_verts(0.85, [0.82, 0.18, 0.05, 1.0]);
    let uv1 = vec![[0.5, 0.5]; uv1_vertices.len()];
    const REFRACTIVE_UV1_HANDLE: u64 = 0x1340_0003;
    assert!(eng.renderer.cache_model_if_static(
        REFRACTIVE_UV1_HANDLE,
        &[MeshData {
            vertices: uv1_vertices,
            secondary_tex_coords: Some(uv1),
            indices: uv1_indices,
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 0.35,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: false,
            transmission,
            layered_pbr: layered_uv1,
        }]
    ));
    let (_, _, cached_refractive_uv1) = render(&mut eng, 3, |eng| {
        draw_refractive_background(eng);
        eng.renderer
            .draw_model_cached(REFRACTIVE_UV1_HANDLE, [0.0; 3], 1.0, [1.0; 4]);
    });
    let uv1_parity = calculate_diff_metrics(&cached_refractive, &cached_refractive_uv1, W, H);
    assert!(
        uv1_parity.mean_rgb <= 1.25
            && uv1_parity.outlier_pixel_fraction <= 0.015
            && uv1_parity.ssim >= 0.97,
        "white UV1 layered refraction lost scalar-path image coherence: {uv1_parity:?}"
    );
}

#[test]
fn anisotropy_follows_explicit_mirrored_tangent_handedness() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_ssgi_enabled(false);
    eng.renderer.set_taa_enabled(false);

    let tangented_cube = |handedness: f32| {
        let (mut vertices, indices) = cube_verts(0.9, [0.72, 0.72, 0.72, 1.0]);
        for face in vertices.chunks_exact_mut(4) {
            let normal = face[0].normal;
            let tangent = if normal[0].abs() > 0.5 {
                [0.0, 0.0, 1.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            for (vertex, uv) in
                face.iter_mut()
                    .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
            {
                vertex.uv = uv;
                vertex.tangent = [tangent[0], tangent[1], tangent[2], handedness];
            }
        }
        (vertices, indices)
    };
    let material = |rotation: f32| MaterialLayeredPbr {
        anisotropy_authored: true,
        anisotropy_strength: 0.92,
        anisotropy_rotation: rotation,
        ..Default::default()
    };
    let draw_scene = |eng: &mut EngineState| {
        let renderer = &mut eng.renderer;
        renderer.set_clear_color(0.015, 0.02, 0.03, 1.0);
        renderer.begin_mode_3d(0.0, 0.2, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 42.0, 0.0);
        renderer.set_ambient_light(38.0, 44.0, 55.0, 0.22);
        renderer.set_directional_light(-0.7, 0.55, 0.28, 255.0, 244.0, 222.0, 3.4);
    };

    let node = eng.scene.create_node();
    let (positive_vertices, indices) = tangented_cube(1.0);
    eng.scene
        .update_geometry(node, positive_vertices, indices.clone());
    eng.scene.set_material_color(node, 0.72, 0.72, 0.72, 1.0);
    eng.scene.set_material_pbr(node, 0.24, 1.0);
    const ROTATION: f32 = 0.61;
    eng.scene.set_material_layered_pbr(node, material(ROTATION));
    let (_, _, positive) = render(&mut eng, 2, draw_scene);

    // Mirroring the UV frame flips tangent.w. Negating the authored
    // counter-clockwise rotation must recover the same world-space
    // anisotropy axis.
    let (mirrored_vertices, _) = tangented_cube(-1.0);
    eng.scene
        .update_geometry(node, mirrored_vertices, indices.clone());
    eng.scene
        .set_material_layered_pbr(node, material(-ROTATION));
    let (_, _, mirrored_compensated) = render(&mut eng, 2, draw_scene);
    let mirrored_parity = calculate_diff_metrics(&positive, &mirrored_compensated, W, H);
    assert!(
        mirrored_parity.mean_rgb <= 0.5
            && mirrored_parity.outlier_pixel_fraction <= 0.005
            && mirrored_parity.ssim >= 0.995,
        "mirrored tangent handedness changed the world anisotropy axis: {mirrored_parity:?}"
    );

    // Negative control: keeping the rotation sign while flipping tangent.w
    // rotates the world-space axis the other way and must be visible.
    eng.scene.set_material_layered_pbr(node, material(ROTATION));
    let (_, _, mirrored_uncompensated) = render(&mut eng, 2, draw_scene);
    let negative_control = calculate_diff_metrics(&positive, &mirrored_uncompensated, W, H);
    assert!(
        negative_control.mean_rgb >= 0.05 && negative_control.max_diff >= 1,
        "anisotropy tangent negative control was not orientation-sensitive: \
         {negative_control:?}"
    );
}

#[test]
fn anisotropy_follows_negative_model_scale_handedness() {
    let Some(mut eng) = try_engine() else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    eng.renderer.set_ssgi_enabled(false);
    eng.renderer.set_taa_enabled(false);

    // Duplicate each triangle with opposite winding. A negative-determinant
    // model transform reverses raster winding, so this keeps the test focused
    // on the tangent frame rather than the renderer's single-sided cull policy.
    let vertices = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ]
    .into_iter()
    .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    .map(|(position, uv)| Vertex3D {
        position,
        normal: [0.0, 0.0, 1.0],
        color: [0.72, 0.72, 0.72, 1.0],
        uv,
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    })
    .collect::<Vec<_>>();
    let indices = vec![0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2];
    let material = |rotation: f32| MaterialLayeredPbr {
        anisotropy_authored: true,
        anisotropy_strength: 0.92,
        anisotropy_rotation: rotation,
        ..Default::default()
    };
    let transform = |scale_x: f32| {
        [
            [scale_x, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };
    let draw_scene = |eng: &mut EngineState| {
        let renderer = &mut eng.renderer;
        renderer.set_clear_color(0.015, 0.02, 0.03, 1.0);
        renderer.begin_mode_3d(0.0, 0.0, 3.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 42.0, 0.0);
        renderer.set_ambient_light(32.0, 38.0, 48.0, 0.18);
        renderer.set_directional_light(-0.7, 0.55, 0.45, 255.0, 244.0, 222.0, 3.8);
    };

    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_material_pbr(node, 0.18, 1.0);
    const ROTATION: f32 = 0.58;
    eng.scene.set_material_layered_pbr(node, material(ROTATION));
    let (_, _, positive) = render(&mut eng, 2, draw_scene);

    // Reflecting the model's X axis flips tangent.xyz and the model
    // determinant. Negating the authored rotation recovers the same
    // unoriented world-space anisotropy axis only when tangent.w also
    // incorporates that determinant sign.
    eng.scene.set_transform(node, transform(-1.0));
    eng.scene
        .set_material_layered_pbr(node, material(-ROTATION));
    let (_, _, mirrored_compensated) = render(&mut eng, 2, draw_scene);
    let mirrored_parity = calculate_diff_metrics(&positive, &mirrored_compensated, W, H);
    assert!(
        mirrored_parity.mean_rgb <= 0.5
            && mirrored_parity.outlier_pixel_fraction <= 0.005
            && mirrored_parity.ssim >= 0.995,
        "negative model scale changed the compensated anisotropy axis: {mirrored_parity:?}"
    );

    // Negative control: without the corresponding rotation sign change the
    // mirrored tangent frame must produce a visibly different highlight.
    eng.scene.set_material_layered_pbr(node, material(ROTATION));
    let (_, _, mirrored_uncompensated) = render(&mut eng, 2, draw_scene);
    let negative_control = calculate_diff_metrics(&positive, &mirrored_uncompensated, W, H);
    assert!(
        negative_control.mean_rgb >= 0.05 && negative_control.max_diff >= 1,
        "negative-scale anisotropy control was not orientation-sensitive: \
         {negative_control:?}"
    );
}

#[test]
fn weighted_transparency_is_order_independent_for_intersecting_imported_blend() {
    fn render_order(
        mode: u32,
        reverse: bool,
        custom_material_z: Option<f32>,
        taa_enabled: bool,
    ) -> Option<(Vec<u8>, u32, String)> {
        let mut eng = try_engine()?;
        eng.renderer.set_transparency_composition_mode(mode);
        eng.renderer.set_taa_enabled(taa_enabled);

        let (retained_vertices, retained_indices) = cube_verts(0.95, [0.08, 0.9, 0.16, 0.35]);
        let retained = eng.scene.create_node();
        eng.scene
            .update_geometry(retained, retained_vertices, retained_indices);
        eng.scene
            .set_material_gltf_alpha(retained, MaterialAlphaMode::Blend, 0.0, false);

        let (red_vertices, indices) = cube_verts(0.8, [1.0, 0.04, 0.02, 0.55]);
        let (blue_vertices, _) = cube_verts(0.8, [0.02, 0.12, 1.0, 0.62]);
        let blend_mesh = |vertices| MeshData {
            vertices,
            secondary_tex_coords: None,
            indices: indices.clone(),
            texture_idx: None,
            normal_texture_idx: None,
            metallic_roughness_texture_idx: None,
            specular_glossiness_factor: None,
            emissive_texture_idx: None,
            occlusion_texture_idx: None,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0; 3],
            alpha_mode: MaterialAlphaMode::Blend,
            alpha_cutoff: 0.0,
            alpha_coverage_mips: false,
            double_sided: false,
            transmission: Default::default(),
            layered_pbr: Default::default(),
        };
        const RED_HANDLE: u64 = 0x0170_0001;
        const BLUE_HANDLE: u64 = 0x0170_0002;
        assert!(eng
            .renderer
            .cache_model_if_static(RED_HANDLE, &[blend_mesh(red_vertices)]));
        assert!(eng
            .renderer
            .cache_model_if_static(BLUE_HANDLE, &[blend_mesh(blue_vertices)]));
        let custom_material = custom_material_z
            .map(|_| {
                eng.renderer.compile_material_with_options(
                    r#"
#include "material_abi.wgsl"

struct VsOut {
  @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VsOut {
  var out: VsOut;
  out.clip_position = draw.mvp * vec4<f32>(in.position, 1.0);
  return out;
}

@fragment
fn fs_main(_in: VsOut) -> TranslucentOut {
  var out: TranslucentOut;
  out.hdr = vec4<f32>(0.04, 0.8, 0.12, 0.18);
  return out;
}
"#,
                    bloom_shared::renderer::material_pipeline::FragmentProfile::Translucent,
                    bloom_shared::renderer::material_pipeline::Bucket::Transparent,
                    false,
                    false,
                )
            })
            .transpose()
            .expect("custom sorted transparent material compiles alongside imported OIT");

        let (_, _, rgba) = render(&mut eng, 2, |eng| {
            let renderer = &mut eng.renderer;
            renderer.set_clear_color(0.03, 0.035, 0.05, 1.0);
            renderer.begin_mode_3d(0.0, 0.0, 4.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
            renderer.set_ambient_light(255.0, 255.0, 255.0, 1.0);
            let draw = |renderer: &mut Renderer, handle| {
                renderer.draw_model_cached(handle, [0.0; 3], 1.0, [1.0; 4]);
            };
            if reverse {
                draw(renderer, BLUE_HANDLE);
                draw(renderer, RED_HANDLE);
            } else {
                draw(renderer, RED_HANDLE);
                draw(renderer, BLUE_HANDLE);
            }
            // Custom materials retain their sorted pass after the imported OIT
            // resolve. Any attachment/layout leakage between the routes turns
            // this into a wgpu validation failure or an AB/BA mismatch.
            if let (Some(custom_material), Some(custom_material_z)) =
                (custom_material, custom_material_z)
            {
                renderer.submit_material_draw(
                    custom_material,
                    RED_HANDLE,
                    0,
                    [0.0, 0.0, custom_material_z],
                    0.72,
                    [1.0; 4],
                );
            }
        });
        Some((
            rgba,
            eng.renderer.active_transparency_composition_mode_code(),
            eng.renderer.render_graph_json().unwrap(),
        ))
    }

    let Some((sorted_ab, sorted_mode, _)) = render_order(0, false, None, false) else {
        eprintln!("skip: no GPU adapter");
        return;
    };
    let (sorted_ba, _, _) =
        render_order(0, true, None, false).expect("same adapter remains available");
    assert_eq!(sorted_mode, 0);
    let sorted_diff = calculate_diff_metrics(&sorted_ab, &sorted_ba, W, H);
    assert!(
        sorted_diff.mean_rgb > 0.25,
        "negative control was not order-sensitive: {sorted_diff:?}"
    );

    let (weighted_ab, weighted_mode, weighted_graph) =
        render_order(2, false, None, false).expect("same adapter remains available");
    let (weighted_ba, weighted_mode_reversed, _) =
        render_order(2, true, None, false).expect("same adapter remains available");
    assert_eq!(weighted_mode, 1);
    assert_eq!(weighted_mode_reversed, 1);
    assert!(
        weighted_graph.contains("transparency-accumulation")
            && weighted_graph.contains("transparency-revealage"),
        "weighted route must declare both graph-owned transient targets"
    );
    let weighted_diff = calculate_diff_metrics(&weighted_ab, &weighted_ba, W, H);
    assert!(
        weighted_diff.mean_rgb <= 0.02 && weighted_diff.max_diff <= 2,
        "weighted composition changed with submission order: {weighted_diff:?}"
    );

    let (weighted_with_custom, custom_mode, _) =
        render_order(2, false, Some(-0.05), false).expect("same adapter remains available");
    assert_eq!(custom_mode, 1);
    let custom_diff = calculate_diff_metrics(&weighted_ab, &weighted_with_custom, W, H);
    assert!(
        custom_diff.mean_rgb > 0.05,
        "custom sorted material did not render after imported OIT: {custom_diff:?}"
    );

    let (_, reactive_mode, reactive_graph) =
        render_order(2, false, Some(-0.05), true).expect("same adapter remains available");
    assert_eq!(reactive_mode, 1);
    assert!(
        reactive_graph.contains("transparency-accumulation")
            && reactive_graph.contains("transparency-revealage")
            && reactive_graph.contains("transparency-reactive"),
        "TAA-active OIT plus custom translucency must retain all graph contracts"
    );

    // Conventional imported and custom translucency must respond to their
    // cross-list depth relationship. Before the shared dispatcher, the custom
    // list always rendered after the complete imported list, so moving this
    // constant-color custom cube from behind to in front left the fully
    // overlapped center pixel unchanged. TAA-on also exercises the lazy
    // two-attachment-compatible custom sibling.
    let (sorted_custom_far, far_mode, far_graph) =
        render_order(0, false, Some(-0.6), true).expect("same adapter remains available");
    let (sorted_custom_near, near_mode, near_graph) =
        render_order(0, false, Some(0.6), true).expect("same adapter remains available");
    assert_eq!((far_mode, near_mode), (0, 0));
    assert!(
        far_graph.contains("transparency-reactive")
            && near_graph.contains("transparency-reactive")
            && !far_graph.contains("transparency-accumulation")
            && !near_graph.contains("transparency-accumulation"),
        "mixed sorted translucency must retain the reactive target without enabling OIT"
    );
    let center = (((H / 2) * W + W / 2) * 4) as usize;
    let center_delta = (0..3)
        .map(|channel| {
            sorted_custom_far[center + channel].abs_diff(sorted_custom_near[center + channel])
                as u32
        })
        .sum::<u32>();
    assert!(
        center_delta >= 8,
        "custom/imported global depth order did not affect the overlapped pixel: \
         far={:?}, near={:?}, delta={center_delta}",
        &sorted_custom_far[center..center + 4],
        &sorted_custom_near[center..center + 4],
    );
}

#[test]
fn layered_path_tracing_scalar_lobes_are_isolated_and_energy_bounded() {
    fn render_variant_with_layered_textures(
        mut layered: Option<MaterialLayeredPbr>,
        specular_textures: Option<(
            &[u8],
            &[u8],
            MaterialTextureTransform,
            MaterialTextureTransform,
        )>,
        clearcoat_textures: Option<(
            &[u8],
            &[u8],
            MaterialTextureTransform,
            MaterialTextureTransform,
        )>,
        clearcoat_normal_texture: Option<(&[u8], MaterialTextureTransform)>,
        sheen_textures: Option<(
            &[u8],
            &[u8],
            MaterialTextureTransform,
            MaterialTextureTransform,
        )>,
        iridescence_textures: Option<(
            &[u8],
            &[u8],
            MaterialTextureTransform,
            MaterialTextureTransform,
        )>,
        anisotropy_texture: Option<(&[u8], MaterialTextureTransform)>,
    ) -> Result<Option<(Vec<u8>, String)>, String> {
        let Some((mut eng, _adapter)) = try_engine_rt()? else {
            return Ok(None);
        };
        if let (Some(material), Some((factor, color, factor_transform, color_transform))) =
            (layered.as_mut(), specular_textures)
        {
            let factor_index = eng.renderer.register_texture_kind(2, 2, factor, false);
            let color_index = eng.renderer.register_texture_kind(2, 2, color, false);
            material.specular_texture = Some(MaterialTextureBinding {
                source_texture_index: 0,
                source_image_index: 0,
                runtime_texture_idx: Some(factor_index),
                transform: factor_transform,
            });
            material.specular_color_texture = Some(MaterialTextureBinding {
                source_texture_index: 1,
                source_image_index: 1,
                runtime_texture_idx: Some(color_index),
                transform: color_transform,
            });
        }
        if let (Some(material), Some((factor, roughness, factor_transform, roughness_transform))) =
            (layered.as_mut(), clearcoat_textures)
        {
            let factor_index = eng.renderer.register_texture_kind(2, 2, factor, false);
            let roughness_index = eng.renderer.register_texture_kind(2, 2, roughness, false);
            material.clearcoat_texture = Some(MaterialTextureBinding {
                source_texture_index: 0,
                source_image_index: 0,
                runtime_texture_idx: Some(factor_index),
                transform: factor_transform,
            });
            material.clearcoat_roughness_texture = Some(MaterialTextureBinding {
                source_texture_index: 1,
                source_image_index: 1,
                runtime_texture_idx: Some(roughness_index),
                transform: roughness_transform,
            });
        }
        if let (Some(material), Some((normal, transform))) =
            (layered.as_mut(), clearcoat_normal_texture)
        {
            let normal_index = eng.renderer.register_texture_kind(2, 2, normal, true);
            material.clearcoat_normal_texture = Some(MaterialTextureBinding {
                source_texture_index: 2,
                source_image_index: 2,
                runtime_texture_idx: Some(normal_index),
                transform,
            });
        }
        if let (Some(material), Some((color, roughness, color_transform, roughness_transform))) =
            (layered.as_mut(), sheen_textures)
        {
            let color_index = eng.renderer.register_texture_kind(2, 2, color, false);
            let roughness_index = eng.renderer.register_texture_kind(2, 2, roughness, false);
            material.sheen_color_texture = Some(MaterialTextureBinding {
                source_texture_index: 0,
                source_image_index: 0,
                runtime_texture_idx: Some(color_index),
                transform: color_transform,
            });
            material.sheen_roughness_texture = Some(MaterialTextureBinding {
                source_texture_index: 1,
                source_image_index: 1,
                runtime_texture_idx: Some(roughness_index),
                transform: roughness_transform,
            });
        }
        if let (Some(material), Some((factor, thickness, factor_transform, thickness_transform))) =
            (layered.as_mut(), iridescence_textures)
        {
            let factor_index = eng.renderer.register_texture_kind(2, 2, factor, false);
            let thickness_index = eng.renderer.register_texture_kind(2, 2, thickness, false);
            material.iridescence_texture = Some(MaterialTextureBinding {
                source_texture_index: 0,
                source_image_index: 0,
                runtime_texture_idx: Some(factor_index),
                transform: factor_transform,
            });
            material.iridescence_thickness_texture = Some(MaterialTextureBinding {
                source_texture_index: 1,
                source_image_index: 1,
                runtime_texture_idx: Some(thickness_index),
                transform: thickness_transform,
            });
        }
        if let (Some(material), Some((texture, transform))) = (layered.as_mut(), anisotropy_texture)
        {
            let texture_index = eng.renderer.register_texture_kind(2, 2, texture, false);
            material.anisotropy_texture = Some(MaterialTextureBinding {
                source_texture_index: 0,
                source_image_index: 0,
                runtime_texture_idx: Some(texture_index),
                transform,
            });
        }
        build_pt_scene(&mut eng);
        // Replace the material target with the same cube carrying an explicit
        // per-face tangent. The base PT shader ignores this attribute, while
        // scalar anisotropy must recover it from the geometry megabuffer at
        // both the primary and bounce intersections.
        let (mut vertices, indices) = cube_verts(0.5, [0.85, 0.2, 0.15, 1.0]);
        let face_uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for (index, vertex) in vertices.iter_mut().enumerate() {
            vertex.uv = face_uvs[index % face_uvs.len()];
            vertex.tangent = if vertex.normal[0].abs() > 0.5 {
                [0.0, 0.0, 1.0, 1.0]
            } else {
                [1.0, 0.0, 0.0, 1.0]
            };
        }
        let node = eng
            .scene
            .nodes
            .iter()
            .nth(1)
            .map(|(handle, _)| handle)
            .expect("PT scene has a test object");
        let uses_uv1 = layered.is_some_and(|material| {
            material
                .specular_texture
                .is_some_and(|binding| binding.transform.tex_coord == 1)
                || material
                    .specular_color_texture
                    .is_some_and(|binding| binding.transform.tex_coord == 1)
                || material
                    .clearcoat_texture
                    .is_some_and(|binding| binding.transform.tex_coord == 1)
                || material
                    .clearcoat_roughness_texture
                    .is_some_and(|binding| binding.transform.tex_coord == 1)
                || material
                    .clearcoat_normal_texture
                    .is_some_and(|binding| binding.transform.tex_coord == 1)
                || material
                    .sheen_color_texture
                    .is_some_and(|binding| binding.transform.tex_coord == 1)
                || material
                    .sheen_roughness_texture
                    .is_some_and(|binding| binding.transform.tex_coord == 1)
                || material
                    .iridescence_texture
                    .is_some_and(|binding| binding.transform.tex_coord == 1)
                || material
                    .iridescence_thickness_texture
                    .is_some_and(|binding| binding.transform.tex_coord == 1)
                || material
                    .anisotropy_texture
                    .is_some_and(|binding| binding.transform.tex_coord == 1)
        });
        let secondary_uvs = uses_uv1.then(|| {
            vertices
                .iter()
                .map(|vertex| [1.0 - vertex.uv[1], vertex.uv[0]])
                .collect()
        });
        eng.scene
            .update_geometry_with_secondary_uv(node, vertices, secondary_uvs, indices);
        if let Some(material) = layered {
            eng.scene.set_material_layered_pbr(node, material);
        }
        eng.renderer.set_path_tracing(1);
        eng.renderer.set_path_tracing_seed(0);
        eng.renderer.reset_path_tracing_history(0);
        let (_, _, rgba) = render(&mut eng, 12, draw_pt_static_frame);
        Ok(Some((rgba, eng.renderer.quality_runtime_paths_json())))
    }

    fn render_variant_with_specular_textures(
        layered: Option<MaterialLayeredPbr>,
        specular_textures: Option<(
            &[u8],
            &[u8],
            MaterialTextureTransform,
            MaterialTextureTransform,
        )>,
    ) -> Result<Option<(Vec<u8>, String)>, String> {
        render_variant_with_layered_textures(
            layered,
            specular_textures,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn render_variant_with_clearcoat_textures(
        layered: Option<MaterialLayeredPbr>,
        clearcoat_textures: Option<(
            &[u8],
            &[u8],
            MaterialTextureTransform,
            MaterialTextureTransform,
        )>,
    ) -> Result<Option<(Vec<u8>, String)>, String> {
        render_variant_with_layered_textures(
            layered,
            None,
            clearcoat_textures,
            None,
            None,
            None,
            None,
        )
    }

    fn render_variant_with_clearcoat_normal(
        layered: Option<MaterialLayeredPbr>,
        clearcoat_normal: Option<(&[u8], MaterialTextureTransform)>,
    ) -> Result<Option<(Vec<u8>, String)>, String> {
        render_variant_with_layered_textures(
            layered,
            None,
            None,
            clearcoat_normal,
            None,
            None,
            None,
        )
    }

    fn render_variant_with_sheen_textures(
        layered: Option<MaterialLayeredPbr>,
        sheen_textures: Option<(
            &[u8],
            &[u8],
            MaterialTextureTransform,
            MaterialTextureTransform,
        )>,
    ) -> Result<Option<(Vec<u8>, String)>, String> {
        render_variant_with_layered_textures(layered, None, None, None, sheen_textures, None, None)
    }

    fn render_variant_with_iridescence_textures(
        layered: Option<MaterialLayeredPbr>,
        iridescence_textures: Option<(
            &[u8],
            &[u8],
            MaterialTextureTransform,
            MaterialTextureTransform,
        )>,
    ) -> Result<Option<(Vec<u8>, String)>, String> {
        render_variant_with_layered_textures(
            layered,
            None,
            None,
            None,
            None,
            iridescence_textures,
            None,
        )
    }

    fn render_variant_with_anisotropy_texture(
        layered: Option<MaterialLayeredPbr>,
        anisotropy_texture: Option<(&[u8], MaterialTextureTransform)>,
    ) -> Result<Option<(Vec<u8>, String)>, String> {
        render_variant_with_layered_textures(
            layered,
            None,
            None,
            None,
            None,
            None,
            anisotropy_texture,
        )
    }

    fn render_variant(
        layered: Option<MaterialLayeredPbr>,
    ) -> Result<Option<(Vec<u8>, String)>, String> {
        render_variant_with_layered_textures(layered, None, None, None, None, None, None)
    }

    fn mean_display_luminance(rgba: &[u8]) -> f64 {
        rgba.chunks_exact(4)
            .map(|pixel| {
                0.2126 * f64::from(pixel[0])
                    + 0.7152 * f64::from(pixel[1])
                    + 0.0722 * f64::from(pixel[2])
            })
            .sum::<f64>()
            / (rgba.len() / 4) as f64
    }

    fn assert_transport_response(label: &str, base: &[u8], transported: &[u8]) {
        let response = calculate_diff_metrics(base, transported, W, H);
        let changed_pixels = base
            .chunks_exact(4)
            .zip(transported.chunks_exact(4))
            .filter(|(base, transported)| {
                (0..3)
                    .map(|channel| base[channel].abs_diff(transported[channel]) as u32)
                    .sum::<u32>()
                    >= 3
            })
            .count();
        let changed_fraction = changed_pixels as f64 / (W * H) as f64;
        assert!(
            response.mean_rgb >= 0.05 && changed_fraction >= 0.002,
            "{label} did not produce a visible transport response: \
             metrics={response:?}, changed_fraction={changed_fraction:.6}"
        );

        // A layered interface redistributes energy; it must not behave as a
        // second unattenuated BRDF. This display-space guard complements the
        // CPU white-furnace oracle and catches catastrophic GPU energy gain
        // without over-constraining a legitimate sharp highlight.
        let base_luminance = mean_display_luminance(base);
        let transported_luminance = mean_display_luminance(transported);
        assert!(
            transported_luminance <= base_luminance * 1.10 + 0.25,
            "{label} created unbounded display energy: \
             base={base_luminance:.4}, transported={transported_luminance:.4}"
        );
    }

    let _guard = lock_rt_goldens();
    let Some((base, base_paths)) = render_variant(None).expect("base PT variant initializes")
    else {
        skip_rt_golden(
            "layered_path_tracing_scalar_lobes_are_isolated_and_energy_bounded",
            "no-non-cpu-ray-query-adapter",
        );
        return;
    };
    let (neutral, neutral_paths) = render_variant(Some(MaterialLayeredPbr {
        iridescence_authored: true,
        iridescence_factor: 0.65,
        iridescence_thickness_minimum: 120.0,
        iridescence_thickness_maximum: 360.0,
        iridescence_texture: Some(MaterialTextureBinding {
            source_texture_index: 0,
            source_image_index: 0,
            runtime_texture_idx: None,
            transform: Default::default(),
        }),
        ..Default::default()
    }))
    .expect("neutral layered PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (clearcoat, clearcoat_paths) = render_variant(Some(MaterialLayeredPbr {
        clearcoat_authored: true,
        clearcoat_factor: 0.85,
        clearcoat_roughness_factor: 0.2,
        ..Default::default()
    }))
    .expect("clearcoat PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (specular_ior, specular_ior_paths) = render_variant(Some(MaterialLayeredPbr {
        specular_authored: true,
        specular_factor: 0.8,
        specular_color_factor: [1.2, 0.6, 0.3],
        ior_authored: true,
        ior: 2.0,
        ..Default::default()
    }))
    .expect("specular/IOR PT variant initializes")
    .expect("same ray-query adapter remains available");
    let white_specular_texels = [255u8; 2 * 2 * 4];
    let directional_specular_factor = [
        255, 255, 255, 24, 255, 255, 255, 255, 255, 255, 255, 24, 255, 255, 255, 255,
    ];
    let directional_clearcoat_channels = [
        24, 255, 255, 255, 255, 24, 255, 255, 24, 255, 255, 255, 255, 24, 255, 255,
    ];
    let textured_clearcoat_material = MaterialLayeredPbr {
        clearcoat_authored: true,
        clearcoat_factor: 0.85,
        clearcoat_roughness_factor: 0.2,
        ..Default::default()
    };
    let (textured_clearcoat_white, textured_clearcoat_white_paths) =
        render_variant_with_clearcoat_textures(
            Some(textured_clearcoat_material),
            Some((
                &white_specular_texels,
                &white_specular_texels,
                Default::default(),
                Default::default(),
            )),
        )
        .expect("neutral textured clearcoat PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_clearcoat, textured_clearcoat_paths) = render_variant_with_clearcoat_textures(
        Some(textured_clearcoat_material),
        Some((
            &directional_clearcoat_channels,
            &directional_clearcoat_channels,
            Default::default(),
            Default::default(),
        )),
    )
    .expect("textured clearcoat PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (textured_clearcoat_rotated, textured_clearcoat_rotated_paths) =
        render_variant_with_clearcoat_textures(
            Some(textured_clearcoat_material),
            Some((
                &directional_clearcoat_channels,
                &directional_clearcoat_channels,
                MaterialTextureTransform {
                    rotation: std::f32::consts::FRAC_PI_2,
                    ..Default::default()
                },
                Default::default(),
            )),
        )
        .expect("rotated textured clearcoat PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_clearcoat_uv1, textured_clearcoat_uv1_paths) =
        render_variant_with_clearcoat_textures(
            Some(textured_clearcoat_material),
            Some((
                &directional_clearcoat_channels,
                &directional_clearcoat_channels,
                MaterialTextureTransform {
                    tex_coord: 1,
                    ..Default::default()
                },
                MaterialTextureTransform {
                    tex_coord: 1,
                    ..Default::default()
                },
            )),
        )
        .expect("UV1 textured clearcoat PT variant initializes")
        .expect("same ray-query adapter remains available");
    let flat_clearcoat_normals = [
        128, 128, 255, 255, 128, 128, 255, 255, 128, 128, 255, 255, 128, 128, 255, 255,
    ];
    let directional_clearcoat_normals = [
        230, 128, 220, 255, 128, 230, 220, 255, 26, 128, 220, 255, 128, 26, 220, 255,
    ];
    let (flat_clearcoat_normal, flat_clearcoat_normal_paths) =
        render_variant_with_clearcoat_normal(
            Some(MaterialLayeredPbr {
                clearcoat_authored: true,
                clearcoat_factor: 0.85,
                clearcoat_roughness_factor: 0.2,
                clearcoat_normal_scale: 0.0,
                ..Default::default()
            }),
            Some((&flat_clearcoat_normals, Default::default())),
        )
        .expect("flat clearcoat normal PT variant initializes")
        .expect("same ray-query adapter remains available");
    let clearcoat_normal_material = MaterialLayeredPbr {
        clearcoat_authored: true,
        clearcoat_factor: 0.85,
        clearcoat_roughness_factor: 0.2,
        clearcoat_normal_scale: 0.75,
        ..Default::default()
    };
    let (textured_clearcoat_normal, textured_clearcoat_normal_paths) =
        render_variant_with_clearcoat_normal(
            Some(clearcoat_normal_material),
            Some((&directional_clearcoat_normals, Default::default())),
        )
        .expect("textured clearcoat normal PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (rotated_clearcoat_normal, rotated_clearcoat_normal_paths) =
        render_variant_with_clearcoat_normal(
            Some(clearcoat_normal_material),
            Some((
                &directional_clearcoat_normals,
                MaterialTextureTransform {
                    rotation: std::f32::consts::FRAC_PI_2,
                    ..Default::default()
                },
            )),
        )
        .expect("rotated clearcoat normal PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (uv1_clearcoat_normal, uv1_clearcoat_normal_paths) = render_variant_with_clearcoat_normal(
        Some(clearcoat_normal_material),
        Some((
            &directional_clearcoat_normals,
            MaterialTextureTransform {
                tex_coord: 1,
                ..Default::default()
            },
        )),
    )
    .expect("UV1 clearcoat normal PT variant initializes")
    .expect("same ray-query adapter remains available");
    let textured_specular_material = MaterialLayeredPbr {
        specular_authored: true,
        specular_factor: 0.8,
        specular_color_factor: [1.2, 0.6, 0.3],
        ior_authored: true,
        ior: 2.0,
        ..Default::default()
    };
    let (textured_specular_white, textured_specular_white_paths) =
        render_variant_with_specular_textures(
            Some(textured_specular_material),
            Some((
                &white_specular_texels,
                &white_specular_texels,
                Default::default(),
                Default::default(),
            )),
        )
        .expect("neutral textured specular/IOR PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_specular, textured_specular_paths) = render_variant_with_specular_textures(
        Some(textured_specular_material),
        Some((
            &directional_specular_factor,
            &white_specular_texels,
            Default::default(),
            Default::default(),
        )),
    )
    .expect("textured specular/IOR PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (textured_specular_rotated, textured_specular_rotated_paths) =
        render_variant_with_specular_textures(
            Some(textured_specular_material),
            Some((
                &directional_specular_factor,
                &white_specular_texels,
                MaterialTextureTransform {
                    rotation: std::f32::consts::FRAC_PI_2,
                    ..Default::default()
                },
                Default::default(),
            )),
        )
        .expect("rotated textured specular/IOR PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_specular_uv1, textured_specular_uv1_paths) =
        render_variant_with_specular_textures(
            Some(textured_specular_material),
            Some((
                &directional_specular_factor,
                &white_specular_texels,
                MaterialTextureTransform {
                    tex_coord: 1,
                    ..Default::default()
                },
                MaterialTextureTransform {
                    tex_coord: 1,
                    ..Default::default()
                },
            )),
        )
        .expect("UV1 textured specular/IOR PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (sheen, sheen_paths) = render_variant(Some(MaterialLayeredPbr {
        sheen_authored: true,
        sheen_color_factor: [0.45, 0.12, 0.04],
        sheen_roughness_factor: 0.4,
        ..Default::default()
    }))
    .expect("sheen PT variant initializes")
    .expect("same ray-query adapter remains available");
    let directional_sheen_channels = [
        255, 64, 32, 255, 32, 255, 64, 32, 255, 64, 32, 255, 32, 255, 64, 32,
    ];
    let textured_sheen_material = MaterialLayeredPbr {
        sheen_authored: true,
        sheen_color_factor: [0.45, 0.12, 0.04],
        sheen_roughness_factor: 0.4,
        ..Default::default()
    };
    let (textured_sheen_white, textured_sheen_white_paths) = render_variant_with_sheen_textures(
        Some(textured_sheen_material),
        Some((
            &white_specular_texels,
            &white_specular_texels,
            Default::default(),
            Default::default(),
        )),
    )
    .expect("neutral textured sheen PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (textured_sheen, textured_sheen_paths) = render_variant_with_sheen_textures(
        Some(textured_sheen_material),
        Some((
            &directional_sheen_channels,
            &directional_sheen_channels,
            Default::default(),
            Default::default(),
        )),
    )
    .expect("textured sheen PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (textured_sheen_rotated, textured_sheen_rotated_paths) =
        render_variant_with_sheen_textures(
            Some(textured_sheen_material),
            Some((
                &directional_sheen_channels,
                &directional_sheen_channels,
                MaterialTextureTransform {
                    rotation: std::f32::consts::FRAC_PI_2,
                    ..Default::default()
                },
                Default::default(),
            )),
        )
        .expect("rotated textured sheen PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_sheen_uv1, textured_sheen_uv1_paths) = render_variant_with_sheen_textures(
        Some(textured_sheen_material),
        Some((
            &directional_sheen_channels,
            &directional_sheen_channels,
            MaterialTextureTransform {
                tex_coord: 1,
                ..Default::default()
            },
            MaterialTextureTransform {
                tex_coord: 1,
                ..Default::default()
            },
        )),
    )
    .expect("UV1 textured sheen PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (anisotropy, anisotropy_paths) = render_variant(Some(MaterialLayeredPbr {
        anisotropy_authored: true,
        anisotropy_strength: 1.0,
        anisotropy_rotation: 0.3,
        ..Default::default()
    }))
    .expect("anisotropy PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (anisotropy_rotated, anisotropy_rotated_paths) = render_variant(Some(MaterialLayeredPbr {
        anisotropy_authored: true,
        anisotropy_strength: 1.0,
        anisotropy_rotation: 0.3 + std::f32::consts::FRAC_PI_2,
        ..Default::default()
    }))
    .expect("rotated anisotropy PT variant initializes")
    .expect("same ray-query adapter remains available");
    let neutral_anisotropy_channels = [
        255, 128, 255, 255, 255, 128, 255, 255, 255, 128, 255, 255, 255, 128, 255, 255,
    ];
    let directional_anisotropy_channels = [
        255, 128, 255, 255, 128, 255, 255, 255, 128, 255, 255, 255, 128, 255, 255, 255,
    ];
    let textured_anisotropy_material = MaterialLayeredPbr {
        anisotropy_authored: true,
        anisotropy_strength: 1.0,
        anisotropy_rotation: 0.3,
        ..Default::default()
    };
    let (textured_anisotropy_neutral, textured_anisotropy_neutral_paths) =
        render_variant_with_anisotropy_texture(
            Some(textured_anisotropy_material),
            Some((&neutral_anisotropy_channels, Default::default())),
        )
        .expect("neutral textured anisotropy PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_anisotropy, textured_anisotropy_paths) = render_variant_with_anisotropy_texture(
        Some(textured_anisotropy_material),
        Some((&directional_anisotropy_channels, Default::default())),
    )
    .expect("textured anisotropy PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (textured_anisotropy_rotated, textured_anisotropy_rotated_paths) =
        render_variant_with_anisotropy_texture(
            Some(textured_anisotropy_material),
            Some((
                &directional_anisotropy_channels,
                MaterialTextureTransform {
                    rotation: std::f32::consts::FRAC_PI_2,
                    ..Default::default()
                },
            )),
        )
        .expect("rotated textured anisotropy PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_anisotropy_uv1, textured_anisotropy_uv1_paths) =
        render_variant_with_anisotropy_texture(
            Some(textured_anisotropy_material),
            Some((
                &directional_anisotropy_channels,
                MaterialTextureTransform {
                    tex_coord: 1,
                    ..Default::default()
                },
            )),
        )
        .expect("UV1 textured anisotropy PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (iridescence, iridescence_paths) = render_variant(Some(MaterialLayeredPbr {
        iridescence_authored: true,
        iridescence_factor: 1.0,
        iridescence_ior: 1.3,
        iridescence_thickness_minimum: 100.0,
        iridescence_thickness_maximum: 180.0,
        ..Default::default()
    }))
    .expect("iridescence PT variant initializes")
    .expect("same ray-query adapter remains available");
    let directional_iridescence_channels = [
        24, 255, 255, 255, 255, 24, 255, 255, 24, 255, 255, 255, 255, 24, 255, 255,
    ];
    let textured_iridescence_material = MaterialLayeredPbr {
        iridescence_authored: true,
        iridescence_factor: 1.0,
        iridescence_ior: 1.3,
        iridescence_thickness_minimum: 100.0,
        iridescence_thickness_maximum: 180.0,
        ..Default::default()
    };
    let (textured_iridescence_white, textured_iridescence_white_paths) =
        render_variant_with_iridescence_textures(
            Some(textured_iridescence_material),
            Some((
                &white_specular_texels,
                &white_specular_texels,
                Default::default(),
                Default::default(),
            )),
        )
        .expect("neutral textured iridescence PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_iridescence, textured_iridescence_paths) =
        render_variant_with_iridescence_textures(
            Some(textured_iridescence_material),
            Some((
                &directional_iridescence_channels,
                &directional_iridescence_channels,
                Default::default(),
                Default::default(),
            )),
        )
        .expect("textured iridescence PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_iridescence_rotated, textured_iridescence_rotated_paths) =
        render_variant_with_iridescence_textures(
            Some(textured_iridescence_material),
            Some((
                &directional_iridescence_channels,
                &directional_iridescence_channels,
                MaterialTextureTransform {
                    rotation: std::f32::consts::FRAC_PI_2,
                    ..Default::default()
                },
                Default::default(),
            )),
        )
        .expect("rotated textured iridescence PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (textured_iridescence_uv1, textured_iridescence_uv1_paths) =
        render_variant_with_iridescence_textures(
            Some(textured_iridescence_material),
            Some((
                &directional_iridescence_channels,
                &directional_iridescence_channels,
                MaterialTextureTransform {
                    tex_coord: 1,
                    ..Default::default()
                },
                MaterialTextureTransform {
                    tex_coord: 1,
                    ..Default::default()
                },
            )),
        )
        .expect("UV1 textured iridescence PT variant initializes")
        .expect("same ray-query adapter remains available");
    let (iridescence_thick, iridescence_thick_paths) = render_variant(Some(MaterialLayeredPbr {
        iridescence_authored: true,
        iridescence_factor: 1.0,
        iridescence_ior: 1.3,
        iridescence_thickness_minimum: 100.0,
        iridescence_thickness_maximum: 520.0,
        ..Default::default()
    }))
    .expect("thick iridescence PT variant initializes")
    .expect("same ray-query adapter remains available");
    let (combined, combined_paths) = render_variant(Some(MaterialLayeredPbr {
        clearcoat_authored: true,
        clearcoat_factor: 0.7,
        clearcoat_roughness_factor: 0.16,
        specular_authored: true,
        specular_factor: 0.8,
        specular_color_factor: [1.2, 0.6, 0.3],
        ior_authored: true,
        ior: 2.0,
        sheen_authored: true,
        sheen_color_factor: [0.35, 0.1, 0.04],
        sheen_roughness_factor: 0.4,
        anisotropy_authored: true,
        anisotropy_strength: 0.6,
        anisotropy_rotation: 0.3,
        iridescence_authored: true,
        iridescence_factor: 0.8,
        iridescence_ior: 1.3,
        iridescence_thickness_minimum: 100.0,
        iridescence_thickness_maximum: 360.0,
        ..Default::default()
    }))
    .expect("combined clearcoat/specular PT variant initializes")
    .expect("same ray-query adapter remains available");

    assert!(
        base == neutral,
        "an unqualified layered lobe changed PT output before its transport landed"
    );
    assert!(base_paths.contains("\"path_tracing_specialization_initialized\":false"));
    assert!(base_paths.contains("\"path_tracing_sheen_specialization_initialized\":false"));
    assert!(base_paths.contains("\"path_tracing_iridescence_specialization_initialized\":false"));
    assert!(base_paths.contains("\"path_tracing_active_instance_count\":0"));
    assert!(base_paths.contains("\"path_tracing_sidecar_allocated_bytes\":0"));
    assert!(
        base_paths.contains("\"path_tracing_clearcoat_texture_specialization_initialized\":false")
    );
    assert!(base_paths.contains("\"path_tracing_clearcoat_texture_sidecar_allocated_bytes\":0"));
    assert!(
        base_paths.contains("\"path_tracing_clearcoat_normal_specialization_initialized\":false")
    );
    assert!(base_paths.contains("\"path_tracing_clearcoat_normal_sidecar_allocated_bytes\":0"));
    assert!(base_paths.contains("\"path_tracing_sheen_texture_specialization_initialized\":false"));
    assert!(base_paths.contains("\"path_tracing_sheen_texture_sidecar_allocated_bytes\":0"));
    assert!(base_paths
        .contains("\"path_tracing_iridescence_texture_specialization_initialized\":false"));
    assert!(base_paths.contains("\"path_tracing_iridescence_texture_sidecar_allocated_bytes\":0"));
    assert!(
        base_paths.contains("\"path_tracing_anisotropy_texture_specialization_initialized\":false")
    );
    assert!(base_paths.contains("\"path_tracing_anisotropy_texture_sidecar_allocated_bytes\":0"));
    assert!(neutral_paths.contains("\"path_tracing_specialization_initialized\":false"));
    assert!(neutral_paths.contains("\"path_tracing_iridescence_specialization_initialized\":false"));
    assert!(neutral_paths.contains("\"path_tracing_active_instance_count\":1"));
    assert!(neutral_paths.contains("\"path_tracing_sidecar_allocated_bytes\":0"));
    assert!(clearcoat_paths.contains("\"path_tracing_specialization_initialized\":true"));
    assert!(clearcoat_paths.contains("\"path_tracing_sheen_specialization_initialized\":false"));
    assert!(
        clearcoat_paths.contains("\"path_tracing_anisotropy_specialization_initialized\":false")
    );
    assert!(
        clearcoat_paths.contains("\"path_tracing_iridescence_specialization_initialized\":false")
    );
    assert!(clearcoat_paths.contains("\"sheen_lut_initialized\":false"));
    assert!(clearcoat_paths.contains("\"path_tracing_active_instance_count\":1"));
    assert!(clearcoat_paths
        .contains("\"path_tracing_clearcoat_normal_specialization_initialized\":false"));
    assert!(clearcoat_paths.contains("\"path_tracing_clearcoat_normal_sidecar_allocated_bytes\":0"));
    let textured_clearcoat_supported = textured_clearcoat_white_paths
        .contains("\"path_tracing_clearcoat_texture_specialization_initialized\":true");
    if textured_clearcoat_supported {
        assert_eq!(
            clearcoat, textured_clearcoat_white,
            "neutral UV0 clearcoat textures changed scalar path-traced transport"
        );
        assert!(textured_clearcoat_paths
            .contains("\"path_tracing_clearcoat_texture_specialization_initialized\":true"));
        assert!(textured_clearcoat_rotated_paths
            .contains("\"path_tracing_clearcoat_texture_specialization_initialized\":true"));
        assert!(textured_clearcoat_paths
            .contains("\"path_tracing_texture_specialization_initialized\":false"));
        assert!(
            textured_clearcoat_paths.contains("\"path_tracing_texture_sidecar_allocated_bytes\":0")
        );
        assert!(textured_clearcoat_paths
            .contains("\"path_tracing_sheen_texture_specialization_initialized\":false"));
        assert!(textured_clearcoat_paths
            .contains("\"path_tracing_sheen_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_clearcoat_paths
            .contains("\"path_tracing_clearcoat_texture_sidecar_record_bytes\":64"));
        assert!(!textured_clearcoat_paths
            .contains("\"path_tracing_clearcoat_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_clearcoat_paths
            .contains("\"path_tracing_clearcoat_normal_sidecar_allocated_bytes\":0"));
        assert!(textured_clearcoat_uv1_paths
            .contains("\"path_tracing_uv1_specialization_initialized\":true"));
        assert!(!textured_clearcoat_uv1_paths
            .contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0"));
    } else {
        assert_eq!(
            base, textured_clearcoat_white,
            "an adapter without PT texture arrays must preserve clearcoat fallback"
        );
        assert_eq!(
            base, textured_clearcoat_uv1,
            "an adapter without PT texture arrays must not partially enable clearcoat UV1"
        );
        assert!(textured_clearcoat_white_paths
            .contains("\"path_tracing_clearcoat_texture_sidecar_allocated_bytes\":0"));
        assert!(
            textured_clearcoat_uv1_paths.contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0")
        );
    }
    let clearcoat_normal_supported = flat_clearcoat_normal_paths
        .contains("\"path_tracing_clearcoat_normal_specialization_initialized\":true");
    if clearcoat_normal_supported {
        assert_eq!(
            clearcoat, flat_clearcoat_normal,
            "a flat zero-scale coat normal changed scalar path-traced transport"
        );
        assert!(textured_clearcoat_normal_paths
            .contains("\"path_tracing_clearcoat_normal_specialization_initialized\":true"));
        assert!(rotated_clearcoat_normal_paths
            .contains("\"path_tracing_clearcoat_normal_specialization_initialized\":true"));
        assert!(textured_clearcoat_normal_paths
            .contains("\"path_tracing_clearcoat_normal_sidecar_record_bytes\":48"));
        assert!(!textured_clearcoat_normal_paths
            .contains("\"path_tracing_clearcoat_normal_sidecar_allocated_bytes\":0"));
        assert!(textured_clearcoat_normal_paths
            .contains("\"path_tracing_clearcoat_texture_sidecar_allocated_bytes\":0"));
        assert!(uv1_clearcoat_normal_paths
            .contains("\"path_tracing_uv1_specialization_initialized\":true"));
        assert!(
            !uv1_clearcoat_normal_paths.contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0")
        );
    } else {
        assert_eq!(
            clearcoat, flat_clearcoat_normal,
            "an adapter without PT texture arrays must preserve scalar clearcoat fallback"
        );
        assert_eq!(
            clearcoat, textured_clearcoat_normal,
            "an adapter without PT texture arrays must ignore only the coat normal texture"
        );
        assert_eq!(
            clearcoat, uv1_clearcoat_normal,
            "an adapter without PT texture arrays must not partially enable coat-normal UV1"
        );
        assert!(flat_clearcoat_normal_paths
            .contains("\"path_tracing_clearcoat_normal_sidecar_allocated_bytes\":0"));
        assert!(
            uv1_clearcoat_normal_paths.contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0")
        );
    }
    assert!(specular_ior_paths.contains("\"path_tracing_specialization_initialized\":true"));
    assert!(specular_ior_paths.contains("\"path_tracing_sheen_specialization_initialized\":false"));
    assert!(
        specular_ior_paths.contains("\"path_tracing_anisotropy_specialization_initialized\":false")
    );
    assert!(specular_ior_paths
        .contains("\"path_tracing_iridescence_specialization_initialized\":false"));
    assert!(specular_ior_paths.contains("\"sheen_lut_initialized\":false"));
    assert!(specular_ior_paths.contains("\"path_tracing_active_instance_count\":1"));
    let textured_specular_supported = textured_specular_white_paths
        .contains("\"path_tracing_texture_specialization_initialized\":true");
    if textured_specular_supported {
        assert_eq!(
            specular_ior, textured_specular_white,
            "neutral UV0 specular textures changed scalar path-traced transport"
        );
        assert!(textured_specular_paths
            .contains("\"path_tracing_texture_specialization_initialized\":true"));
        assert!(textured_specular_rotated_paths
            .contains("\"path_tracing_texture_specialization_initialized\":true"));
        assert!(
            textured_specular_paths.contains("\"path_tracing_texture_sidecar_record_bytes\":64")
        );
        assert!(
            !textured_specular_paths.contains("\"path_tracing_texture_sidecar_allocated_bytes\":0")
        );
        assert!(textured_specular_paths
            .contains("\"path_tracing_clearcoat_texture_specialization_initialized\":false"));
        assert!(textured_specular_paths
            .contains("\"path_tracing_clearcoat_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_specular_paths
            .contains("\"path_tracing_sheen_texture_specialization_initialized\":false"));
        assert!(textured_specular_paths
            .contains("\"path_tracing_sheen_texture_sidecar_allocated_bytes\":0"));
    } else {
        assert_eq!(
            base, textured_specular_white,
            "an adapter without PT texture arrays must preserve the exact base path"
        );
        assert!(textured_specular_white_paths
            .contains("\"path_tracing_texture_sidecar_allocated_bytes\":0"));
    }
    if textured_specular_supported {
        assert!(textured_specular_uv1_paths
            .contains("\"path_tracing_texture_specialization_initialized\":true"));
        assert!(textured_specular_uv1_paths
            .contains("\"path_tracing_uv1_specialization_initialized\":true"));
        assert!(
            !textured_specular_uv1_paths.contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0")
        );
    } else {
        assert_eq!(
            base, textured_specular_uv1,
            "an adapter without PT texture arrays must not partially enable UV1"
        );
        assert!(textured_specular_uv1_paths
            .contains("\"path_tracing_uv1_specialization_initialized\":false"));
        assert!(
            textured_specular_uv1_paths.contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0")
        );
    }
    assert!(sheen_paths.contains("\"path_tracing_specialization_initialized\":true"));
    assert!(sheen_paths.contains("\"path_tracing_sheen_specialization_initialized\":true"));
    assert!(sheen_paths.contains("\"path_tracing_anisotropy_specialization_initialized\":false"));
    assert!(sheen_paths.contains("\"path_tracing_iridescence_specialization_initialized\":false"));
    assert!(sheen_paths.contains("\"sheen_lut_initialized\":true"));
    assert!(sheen_paths.contains("\"path_tracing_active_instance_count\":1"));
    let textured_sheen_supported = textured_sheen_white_paths
        .contains("\"path_tracing_sheen_texture_specialization_initialized\":true");
    if textured_sheen_supported {
        assert_eq!(
            sheen, textured_sheen_white,
            "neutral UV0 sheen textures changed scalar path-traced transport"
        );
        assert!(textured_sheen_paths
            .contains("\"path_tracing_sheen_texture_specialization_initialized\":true"));
        assert!(
            textured_sheen_paths.contains("\"path_tracing_sheen_specialization_initialized\":true")
        );
        assert!(textured_sheen_rotated_paths
            .contains("\"path_tracing_sheen_texture_specialization_initialized\":true"));
        assert!(textured_sheen_paths.contains("\"sheen_lut_initialized\":true"));
        assert!(textured_sheen_paths.contains("\"path_tracing_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_sheen_paths
            .contains("\"path_tracing_clearcoat_texture_sidecar_allocated_bytes\":0"));
        assert!(
            textured_sheen_paths.contains("\"path_tracing_sheen_texture_sidecar_record_bytes\":64")
        );
        assert!(!textured_sheen_paths
            .contains("\"path_tracing_sheen_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_sheen_uv1_paths
            .contains("\"path_tracing_uv1_specialization_initialized\":true"));
        assert!(
            !textured_sheen_uv1_paths.contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0")
        );
    } else {
        assert_eq!(
            base, textured_sheen_white,
            "an adapter without PT texture arrays must preserve sheen fallback"
        );
        assert_eq!(
            base, textured_sheen_uv1,
            "an adapter without PT texture arrays must not partially enable sheen UV1"
        );
        assert!(textured_sheen_white_paths
            .contains("\"path_tracing_sheen_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_sheen_uv1_paths.contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0"));
    }
    assert!(anisotropy_paths.contains("\"path_tracing_specialization_initialized\":true"));
    assert!(anisotropy_paths.contains("\"path_tracing_sheen_specialization_initialized\":false"));
    assert!(
        anisotropy_paths.contains("\"path_tracing_anisotropy_specialization_initialized\":true")
    );
    assert!(
        anisotropy_paths.contains("\"path_tracing_iridescence_specialization_initialized\":false")
    );
    assert!(anisotropy_paths.contains("\"sheen_lut_initialized\":false"));
    assert!(anisotropy_paths.contains("\"path_tracing_active_instance_count\":1"));
    assert!(anisotropy_rotated_paths.contains("\"path_tracing_specialization_initialized\":true"));
    assert!(anisotropy_rotated_paths
        .contains("\"path_tracing_anisotropy_specialization_initialized\":true"));
    assert!(anisotropy_rotated_paths.contains("\"sheen_lut_initialized\":false"));
    let textured_anisotropy_supported = textured_anisotropy_neutral_paths
        .contains("\"path_tracing_anisotropy_texture_specialization_initialized\":true");
    if textured_anisotropy_supported {
        // KHR_materials_anisotropy stores a centered direction in UNORM
        // RG. Eight-bit 128 decodes to 0.50196 rather than exact 0.5, so the
        // closest representable +X control uses an image-tolerance contract
        // instead of the byte-exact neutral contract used by scalar channels.
        let neutral_parity =
            calculate_diff_metrics(&anisotropy, &textured_anisotropy_neutral, W, H);
        assert!(
            neutral_parity.mean_rgb <= 0.5 && neutral_parity.ssim >= 0.995,
            "nearest representable neutral anisotropy texture drifted from scalar transport: \
             {neutral_parity:?}"
        );
        assert!(textured_anisotropy_paths
            .contains("\"path_tracing_anisotropy_texture_specialization_initialized\":true"));
        assert!(textured_anisotropy_paths
            .contains("\"path_tracing_anisotropy_specialization_initialized\":true"));
        assert!(textured_anisotropy_rotated_paths
            .contains("\"path_tracing_anisotropy_texture_specialization_initialized\":true"));
        assert!(textured_anisotropy_paths
            .contains("\"path_tracing_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_anisotropy_paths
            .contains("\"path_tracing_clearcoat_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_anisotropy_paths
            .contains("\"path_tracing_sheen_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_anisotropy_paths
            .contains("\"path_tracing_iridescence_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_anisotropy_paths
            .contains("\"path_tracing_anisotropy_texture_sidecar_record_bytes\":64"));
        assert!(!textured_anisotropy_paths
            .contains("\"path_tracing_anisotropy_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_anisotropy_uv1_paths
            .contains("\"path_tracing_uv1_specialization_initialized\":true"));
        assert!(!textured_anisotropy_uv1_paths
            .contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0"));
    } else {
        assert_eq!(
            base, textured_anisotropy_neutral,
            "an adapter without PT texture arrays must preserve anisotropy fallback"
        );
        assert_eq!(
            base, textured_anisotropy_uv1,
            "an adapter without PT texture arrays must not partially enable anisotropy UV1"
        );
        assert!(textured_anisotropy_neutral_paths
            .contains("\"path_tracing_anisotropy_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_anisotropy_uv1_paths
            .contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0"));
    }
    assert!(iridescence_paths.contains("\"path_tracing_specialization_initialized\":true"));
    assert!(iridescence_paths.contains("\"path_tracing_sheen_specialization_initialized\":false"));
    assert!(
        iridescence_paths.contains("\"path_tracing_anisotropy_specialization_initialized\":false")
    );
    assert!(
        iridescence_paths.contains("\"path_tracing_iridescence_specialization_initialized\":true")
    );
    assert!(iridescence_paths.contains("\"sheen_lut_initialized\":false"));
    assert!(iridescence_thick_paths
        .contains("\"path_tracing_iridescence_specialization_initialized\":true"));
    let textured_iridescence_supported = textured_iridescence_white_paths
        .contains("\"path_tracing_iridescence_texture_specialization_initialized\":true");
    if textured_iridescence_supported {
        assert_eq!(
            iridescence, textured_iridescence_white,
            "neutral UV0 iridescence textures changed scalar path-traced transport"
        );
        assert!(textured_iridescence_paths
            .contains("\"path_tracing_iridescence_texture_specialization_initialized\":true"));
        assert!(textured_iridescence_paths
            .contains("\"path_tracing_iridescence_specialization_initialized\":true"));
        assert!(textured_iridescence_rotated_paths
            .contains("\"path_tracing_iridescence_texture_specialization_initialized\":true"));
        assert!(textured_iridescence_paths
            .contains("\"path_tracing_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_iridescence_paths
            .contains("\"path_tracing_clearcoat_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_iridescence_paths
            .contains("\"path_tracing_sheen_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_iridescence_paths
            .contains("\"path_tracing_iridescence_texture_sidecar_record_bytes\":64"));
        assert!(!textured_iridescence_paths
            .contains("\"path_tracing_iridescence_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_iridescence_uv1_paths
            .contains("\"path_tracing_uv1_specialization_initialized\":true"));
        assert!(!textured_iridescence_uv1_paths
            .contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0"));
    } else {
        assert_eq!(
            base, textured_iridescence_white,
            "an adapter without PT texture arrays must preserve iridescence fallback"
        );
        assert_eq!(
            base, textured_iridescence_uv1,
            "an adapter without PT texture arrays must not partially enable iridescence UV1"
        );
        assert!(textured_iridescence_white_paths
            .contains("\"path_tracing_iridescence_texture_sidecar_allocated_bytes\":0"));
        assert!(textured_iridescence_uv1_paths
            .contains("\"path_tracing_uv1_sidecar_allocated_bytes\":0"));
    }
    assert!(combined_paths.contains("\"path_tracing_specialization_initialized\":true"));
    assert!(combined_paths.contains("\"path_tracing_sheen_specialization_initialized\":true"));
    assert!(combined_paths.contains("\"path_tracing_anisotropy_specialization_initialized\":true"));
    assert!(combined_paths.contains("\"path_tracing_iridescence_specialization_initialized\":true"));
    assert!(combined_paths.contains("\"path_tracing_active_instance_count\":1"));

    assert_transport_response("clearcoat", &base, &clearcoat);
    if textured_clearcoat_supported {
        assert_transport_response(
            "UV0 textured clearcoat",
            &textured_clearcoat_white,
            &textured_clearcoat,
        );
        let transform_response =
            calculate_diff_metrics(&textured_clearcoat, &textured_clearcoat_rotated, W, H);
        assert!(
            transform_response.mean_rgb >= 0.02,
            "clearcoat texture UV rotation did not turn the path-traced response: \
             {transform_response:?}"
        );
        let uv_set_response =
            calculate_diff_metrics(&textured_clearcoat, &textured_clearcoat_uv1, W, H);
        assert!(
            uv_set_response.mean_rgb >= 0.02,
            "clearcoat UV1 did not select retained secondary coordinates: {uv_set_response:?}"
        );
    }
    if clearcoat_normal_supported {
        let normal_response =
            calculate_diff_metrics(&flat_clearcoat_normal, &textured_clearcoat_normal, W, H);
        assert_transport_response(
            "textured clearcoat normal",
            &flat_clearcoat_normal,
            &textured_clearcoat_normal,
        );
        let transform_response =
            calculate_diff_metrics(&textured_clearcoat_normal, &rotated_clearcoat_normal, W, H);
        assert!(
            transform_response.mean_rgb >= 0.02,
            "clearcoat normal UV rotation did not turn the path-traced response: \
             {transform_response:?}"
        );
        let uv_set_response =
            calculate_diff_metrics(&textured_clearcoat_normal, &uv1_clearcoat_normal, W, H);
        eprintln!(
            "clearcoat-normal PT qualification: normal={normal_response:?}, \
             rotation={transform_response:?}, uv1={uv_set_response:?}, \
             flat_luma={:.6}, mapped_luma={:.6}",
            mean_display_luminance(&flat_clearcoat_normal),
            mean_display_luminance(&textured_clearcoat_normal),
        );
        assert!(
            uv_set_response.mean_rgb >= 0.02,
            "clearcoat normal UV1 did not select retained secondary coordinates: \
             {uv_set_response:?}"
        );
    }
    assert_transport_response("specular/IOR", &base, &specular_ior);
    if textured_specular_supported {
        assert_transport_response(
            "UV0 textured specular/IOR",
            &textured_specular_white,
            &textured_specular,
        );
        let transform_response =
            calculate_diff_metrics(&textured_specular, &textured_specular_rotated, W, H);
        assert!(
            transform_response.mean_rgb >= 0.02,
            "specular texture UV rotation did not turn the path-traced response: \
             {transform_response:?}"
        );
        assert_transport_response(
            "UV0 rotated textured specular/IOR",
            &base,
            &textured_specular_rotated,
        );
        assert_transport_response("UV1 textured specular/IOR", &base, &textured_specular_uv1);
        let uv_set_response =
            calculate_diff_metrics(&textured_specular, &textured_specular_uv1, W, H);
        assert!(
            uv_set_response.mean_rgb >= 0.02,
            "UV1 did not select the retained secondary coordinates: {uv_set_response:?}"
        );
    }
    assert_transport_response("sheen", &base, &sheen);
    if textured_sheen_supported {
        assert_transport_response("UV0 textured sheen", &textured_sheen_white, &textured_sheen);
        let transform_response =
            calculate_diff_metrics(&textured_sheen, &textured_sheen_rotated, W, H);
        assert!(
            transform_response.mean_rgb >= 0.02,
            "sheen texture UV rotation did not turn the path-traced response: \
             {transform_response:?}"
        );
        let uv_set_response = calculate_diff_metrics(&textured_sheen, &textured_sheen_uv1, W, H);
        assert!(
            uv_set_response.mean_rgb >= 0.02,
            "sheen UV1 did not select retained secondary coordinates: {uv_set_response:?}"
        );
    }
    assert_transport_response("anisotropy", &base, &anisotropy);
    assert_transport_response("rotated anisotropy", &base, &anisotropy_rotated);
    let rotation_response = calculate_diff_metrics(&anisotropy, &anisotropy_rotated, W, H);
    assert!(
        rotation_response.mean_rgb >= 0.02,
        "anisotropy rotation did not turn the path-traced highlight: {rotation_response:?}"
    );
    if textured_anisotropy_supported {
        assert_transport_response(
            "UV0 textured anisotropy",
            &textured_anisotropy_neutral,
            &textured_anisotropy,
        );
        let transform_response =
            calculate_diff_metrics(&textured_anisotropy, &textured_anisotropy_rotated, W, H);
        assert!(
            transform_response.mean_rgb >= 0.02,
            "anisotropy texture UV rotation did not turn the path-traced response: \
             {transform_response:?}"
        );
        let uv_set_response =
            calculate_diff_metrics(&textured_anisotropy, &textured_anisotropy_uv1, W, H);
        assert!(
            uv_set_response.mean_rgb >= 0.02,
            "anisotropy UV1 did not select retained secondary coordinates: {uv_set_response:?}"
        );
    }
    assert_transport_response("iridescence", &base, &iridescence);
    if textured_iridescence_supported {
        assert_transport_response(
            "UV0 textured iridescence",
            &textured_iridescence_white,
            &textured_iridescence,
        );
        let transform_response =
            calculate_diff_metrics(&textured_iridescence, &textured_iridescence_rotated, W, H);
        assert!(
            transform_response.mean_rgb >= 0.02,
            "iridescence texture UV rotation did not turn the path-traced response: \
             {transform_response:?}"
        );
        let uv_set_response =
            calculate_diff_metrics(&textured_iridescence, &textured_iridescence_uv1, W, H);
        assert!(
            uv_set_response.mean_rgb >= 0.02,
            "iridescence UV1 did not select retained secondary coordinates: {uv_set_response:?}"
        );
    }
    assert_transport_response("thick iridescence", &base, &iridescence_thick);
    let thickness_response = calculate_diff_metrics(&iridescence, &iridescence_thick, W, H);
    assert!(
        thickness_response.mean_rgb >= 0.02,
        "thin-film thickness did not shift the path-traced spectrum: {thickness_response:?}"
    );
    assert_transport_response("combined scalar lobes", &base, &combined);
}
