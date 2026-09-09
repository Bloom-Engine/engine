use super::*;

const ANGULAR_REFERENCE: &str =
    include_str!("../../../../tools/bloom-reference/reference/layered-pbr-angular-v1.json");
const VIEW_SAMPLE: usize = 1;
const LIGHT_SAMPLE: usize = 2;
const PT_ACCUMULATION_FRAMES: u32 = 64;

#[derive(Clone, Copy)]
struct ParityScenario {
    label: &'static str,
    base_color: [f32; 3],
    metallic: f32,
    roughness: f32,
    layered: MaterialLayeredPbr,
    view_n_dot: f32,
    view_azimuth: f32,
    light_n_dot: f32,
    light_azimuth: f32,
    reference_brdf_cos: [f64; 3],
}

fn vec3(value: &serde_json::Value, field: &str) -> [f32; 3] {
    let values = value[field]
        .as_array()
        .unwrap_or_else(|| panic!("angular reference field {field} is not an array"));
    [
        values[0].as_f64().unwrap() as f32,
        values[1].as_f64().unwrap() as f32,
        values[2].as_f64().unwrap() as f32,
    ]
}

fn angular_scenarios() -> Vec<ParityScenario> {
    let report: serde_json::Value =
        serde_json::from_str(ANGULAR_REFERENCE).expect("checked angular reference is valid JSON");
    [
        "base",
        "specular-ior",
        "clearcoat",
        "sheen",
        "anisotropy",
        "iridescence",
        "combined",
    ]
    .into_iter()
    .map(|label| {
        let id = format!("{label}-v{VIEW_SAMPLE}-l{LIGHT_SAMPLE}");
        let sample = report["samples"]
            .as_array()
            .unwrap()
            .iter()
            .find(|sample| sample["id"].as_str() == Some(&id))
            .unwrap_or_else(|| panic!("angular reference is missing {id}"));
        let material = &sample["material"];
        let specular_authored = matches!(label, "specular-ior" | "combined");
        let clearcoat_authored = matches!(label, "clearcoat" | "combined");
        let sheen_authored = matches!(label, "sheen" | "combined");
        let anisotropy_authored = matches!(label, "anisotropy" | "combined");
        let iridescence_authored = matches!(label, "iridescence" | "combined");
        ParityScenario {
            label,
            base_color: vec3(material, "base_color"),
            metallic: material["metallic"].as_f64().unwrap() as f32,
            roughness: material["perceptual_roughness"].as_f64().unwrap() as f32,
            layered: MaterialLayeredPbr {
                clearcoat_authored,
                clearcoat_factor: material["clearcoat_factor"].as_f64().unwrap() as f32,
                clearcoat_roughness_factor: material["clearcoat_perceptual_roughness"]
                    .as_f64()
                    .unwrap() as f32,
                specular_authored,
                specular_factor: material["specular_factor"].as_f64().unwrap() as f32,
                specular_color_factor: vec3(material, "specular_color"),
                ior_authored: specular_authored,
                ior: material["ior"].as_f64().unwrap() as f32,
                sheen_authored,
                sheen_color_factor: vec3(material, "sheen_color"),
                sheen_roughness_factor: material["sheen_perceptual_roughness"].as_f64().unwrap()
                    as f32,
                anisotropy_authored,
                anisotropy_strength: material["anisotropy_strength"].as_f64().unwrap() as f32,
                anisotropy_rotation: material["anisotropy_rotation"].as_f64().unwrap() as f32,
                iridescence_authored,
                iridescence_factor: material["iridescence_factor"].as_f64().unwrap() as f32,
                iridescence_ior: material["iridescence_ior"].as_f64().unwrap() as f32,
                iridescence_thickness_minimum: material["iridescence_thickness_nm"]
                    .as_f64()
                    .unwrap() as f32,
                iridescence_thickness_maximum: material["iridescence_thickness_nm"]
                    .as_f64()
                    .unwrap() as f32,
                ..Default::default()
            },
            view_n_dot: sample["n_dot_v"].as_f64().unwrap() as f32,
            view_azimuth: sample["view_azimuth_radians"].as_f64().unwrap() as f32,
            light_n_dot: sample["n_dot_l"].as_f64().unwrap() as f32,
            light_azimuth: sample["light_azimuth_radians"].as_f64().unwrap() as f32,
            reference_brdf_cos: {
                let values = sample["direct_brdf_cos"].as_array().unwrap();
                [
                    values[0].as_f64().unwrap(),
                    values[1].as_f64().unwrap(),
                    values[2].as_f64().unwrap(),
                ]
            },
        }
    })
    .collect()
}

fn comparison_quad() -> (Vec<Vertex3D>, Vec<u32>) {
    let vertices = [
        [-2.2, -2.2, 0.0],
        [2.2, -2.2, 0.0],
        [2.2, 2.2, 0.0],
        [-2.2, 2.2, 0.0],
    ]
    .into_iter()
    .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    .map(|(position, uv)| Vertex3D {
        position,
        normal: [0.0, 0.0, 1.0],
        color: [1.0; 4],
        uv,
        joints: [0.0; 4],
        weights: [0.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
    })
    .collect();
    (vertices, vec![0, 1, 2, 0, 2, 3])
}

fn direction(n_dot: f32, azimuth: f32) -> [f32; 3] {
    let sin_theta = (1.0 - n_dot * n_dot).sqrt();
    [sin_theta * azimuth.cos(), sin_theta * azimuth.sin(), n_dot]
}

fn render_forward_and_path(
    scenario: ParityScenario,
) -> Result<Option<(Vec<u8>, Vec<u8>, String)>, String> {
    let Some((mut eng, _adapter)) = try_engine_rt()? else {
        return Ok(None);
    };
    eng.renderer.set_taa_enabled(false);
    eng.renderer.set_render_scale(1.0);
    eng.renderer.set_ssao_enabled(false);
    eng.renderer.set_ssr_enabled(false);
    eng.renderer.set_ssgi_enabled(false);
    eng.renderer.set_bloom_enabled(false);
    eng.renderer.set_auto_exposure(false);
    eng.renderer.set_manual_exposure(1.0);
    eng.renderer.set_motion_blur_enabled(false);
    eng.renderer.set_shadows_enabled(false);
    eng.renderer.set_env_intensity(0.0);

    let (vertices, indices) = comparison_quad();
    let node = eng.scene.create_node();
    eng.scene.update_geometry(node, vertices, indices);
    eng.scene.set_material_color(
        node,
        scenario.base_color[0],
        scenario.base_color[1],
        scenario.base_color[2],
        1.0,
    );
    eng.scene
        .set_material_pbr(node, scenario.roughness, scenario.metallic);
    eng.scene.set_material_layered_pbr(node, scenario.layered);

    let view = direction(scenario.view_n_dot, scenario.view_azimuth);
    let light = direction(scenario.light_n_dot, scenario.light_azimuth);
    let draw = |eng: &mut EngineState| {
        let renderer = &mut eng.renderer;
        renderer.set_clear_color(0.0, 0.0, 0.0, 1.0);
        renderer.begin_mode_3d(
            view[0] * 5.0,
            view[1] * 5.0,
            view[2] * 5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            42.0,
            0.0,
        );
        renderer.set_ambient_light(255.0, 255.0, 255.0, 0.0);
        renderer.set_directional_light(
            f64::from(light[0]),
            f64::from(light[1]),
            f64::from(light[2]),
            255.0,
            255.0,
            255.0,
            1.0,
        );
    };

    let forward = render(&mut eng, 3, draw).2;
    eng.renderer.set_path_tracing(1);
    eng.renderer.set_path_tracing_seed(0);
    eng.renderer.reset_path_tracing_history(0);
    let path = render(&mut eng, PT_ACCUMULATION_FRAMES, draw).2;
    Ok(Some((
        forward,
        path,
        eng.renderer.quality_runtime_paths_json(),
    )))
}

fn center_mean_rgb(image: &[u8]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    let mut count = 0.0;
    for y in H / 2 - 16..H / 2 + 16 {
        for x in W / 2 - 16..W / 2 + 16 {
            let pixel = ((y * W + x) * 4) as usize;
            for channel in 0..3 {
                sum[channel] += f64::from(image[pixel + channel]);
            }
            count += 1.0;
        }
    }
    sum.map(|value| value / count)
}

fn subtract(value: [f64; 3], base: [f64; 3]) -> [f64; 3] {
    [value[0] - base[0], value[1] - base[1], value[2] - base[2]]
}

fn length(value: [f64; 3]) -> f64 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn cosine_similarity(a: [f64; 3], b: [f64; 3]) -> f64 {
    let denominator = length(a) * length(b);
    if denominator <= 1.0e-12 {
        1.0
    } else {
        (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / denominator
    }
}

fn mean_absolute_rgb(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()) / 3.0
}

fn display_luminance(value: [f64; 3]) -> f64 {
    0.2126 * value[0] + 0.7152 * value[1] + 0.0722 * value[2]
}

#[test]
fn layered_forward_and_path_responses_track_the_angular_reference() {
    let _guard = lock_rt_goldens();
    let scenarios = angular_scenarios();
    let mut captures = Vec::with_capacity(scenarios.len());
    for scenario in scenarios.iter().copied() {
        let Some((forward, path, runtime_paths)) = render_forward_and_path(scenario)
            .unwrap_or_else(|error| panic!("{} parity scene failed: {error}", scenario.label))
        else {
            skip_rt_golden(
                "layered_forward_and_path_responses_track_the_angular_reference",
                "no-non-cpu-ray-query-adapter",
            );
            return;
        };
        captures.push((
            scenario,
            center_mean_rgb(&forward),
            center_mean_rgb(&path),
            runtime_paths,
        ));
    }

    let (_, base_forward, base_path, base_runtime) = &captures[0];
    assert!(base_runtime.contains("\"path_tracing_specialization_initialized\":false"));
    eprintln!(
        "layered-parity base forward={base_forward:?} path={base_path:?}, frames={PT_ACCUMULATION_FRAMES}"
    );
    for (scenario, forward, path, runtime_paths) in &captures {
        let display_mae = mean_absolute_rgb(*forward, *path);
        let display_cosine = cosine_similarity(*forward, *path);
        let forward_luminance = display_luminance(*forward);
        let path_luminance = display_luminance(*path);
        let luminance_relative_error = (forward_luminance - path_luminance).abs()
            / forward_luminance.max(path_luminance).max(1.0);
        assert!(
            display_mae <= 24.0 && display_cosine >= 0.96 && luminance_relative_error <= 0.30,
            "{} forward/path direct-light display agreement exceeded the approved model \
             tolerance: mae={display_mae:.4}, cosine={display_cosine:.6}, \
             luma_relative={luminance_relative_error:.4}, \
             forward={forward:?}, path={path:?}",
            scenario.label,
        );
        if scenario.label == "base" {
            continue;
        }
        let reference_response = subtract(
            scenario.reference_brdf_cos,
            captures[0].0.reference_brdf_cos,
        );
        let forward_response = subtract(*forward, *base_forward);
        let path_response = subtract(*path, *base_path);
        let reference_path_cosine = cosine_similarity(reference_response, path_response);
        eprintln!(
            "layered-parity {} reference={reference_response:?} \
             forward={forward_response:?} path={path_response:?} \
             display_mae={display_mae:.4} display_cosine={display_cosine:.6} \
             luma_relative={luminance_relative_error:.4} \
             reference_path_cosine={reference_path_cosine:.6}",
            scenario.label,
        );
        assert!(
            runtime_paths.contains("\"path_tracing_specialization_initialized\":true"),
            "{} did not select layered path transport",
            scenario.label,
        );
        assert!(
            length(forward_response) >= 0.01 && length(path_response) >= 0.01,
            "{} did not produce a measurable response in both renderers",
            scenario.label,
        );
        if length(reference_response) >= 0.01 {
            assert!(
                reference_path_cosine >= 0.85,
                "{} path-traced lobe response diverged from the linear angular reference: \
                 reference={reference_response:?}, path={path_response:?}, \
                 cosine={reference_path_cosine:.6}",
                scenario.label,
            );
        } else {
            let maximum_display_response = forward_response
                .into_iter()
                .chain(path_response)
                .map(f64::abs)
                .fold(0.0f64, f64::max);
            assert!(
                maximum_display_response <= 12.0,
                "{} sub-threshold reference response exceeded the bounded display allowance: \
                 {maximum_display_response:.4}",
                scenario.label,
            );
        }
    }
}
