//! Production device-request smoke: proves the granted minimum limits can
//! construct the complete renderer, not only a bare wgpu device.

#[test]
fn negotiated_headless_device_constructs_the_complete_renderer() {
    let engine = bloom_shared::attach::attach_headless_engine(wgpu::Backends::PRIMARY, 64, 64);
    let mut engine = match engine {
        Ok(engine) => engine,
        Err(error) if error.contains("no compatible adapter") => {
            eprintln!("skip: no native GPU adapter ({error})");
            return;
        }
        Err(error) => panic!("production renderer device negotiation failed: {error}"),
    };
    let report = engine.renderer.quality_adapter_json();
    let json: serde_json::Value =
        serde_json::from_str(&report).expect("quality adapter snapshot is valid JSON");
    assert_eq!(
        json["device_negotiation"]["required_limits"]["max_color_attachments"],
        4
    );
    assert!(
        json["device_negotiation"]["required_limits"]["max_sampled_textures_per_shader_stage"]
            .as_u64()
            .is_some_and(|value| value >= 19)
    );

    let public_report = engine.renderer.renderer_capability_report_json();
    let public: serde_json::Value =
        serde_json::from_str(&public_report).expect("public renderer capability report is valid");
    assert_eq!(public["version"], 1);
    assert_eq!(public["availability"], "available");
    assert!(public["adapter"]["renderer_capabilities"]["paths"]["materials"].is_string());
    assert!(public["material_binding"]["selected_tier"].is_string());
    assert!(public["runtime_support"]["path_tracing"].is_boolean());
    assert!(public["runtime_support"]["gpu_driven"]["enabled"].is_boolean());
    let layered = &public["runtime_support"]["layered_pbr"];
    assert!(layered["granted_sampled_textures_per_stage"].is_number());
    assert!(layered["scene_required_sampled_textures"].is_number());
    assert!(layered["scene_available"].is_boolean());
    assert!(layered["combined_refraction_full_layout_available"].is_boolean());
    let visibility = &public["runtime_support"]["gpu_driven"]["visibility_buffer_runtime"];
    assert_eq!(visibility["requested_mode"], "off");
    assert_eq!(visibility["enabled"], false);
    assert_eq!(visibility["allocated_bytes"], 0);
    assert_eq!(visibility["frame_recorded"], false);
    assert!(!json["device_negotiation"]["required_features"]
        .as_str()
        .unwrap_or_default()
        .contains("PRIMITIVE_INDEX"));
    assert!(public["runtime_support"]["virtual_shadows"]["requested"].is_boolean());
    assert!(public["runtime_support"]["virtual_shadows"]["capability_eligible"].is_boolean());
    assert!(public["runtime_support"]["virtual_shadows"]["enabled"].is_boolean());
    assert!(public["runtime_support"]["virtual_shadows"]["selection_reason"].is_string());
    assert!(
        public["runtime_support"]["virtual_shadows"]["debug_views"]["capture_only"].is_boolean()
    );
    assert!(public["runtime_support"]["virtual_shadows"]["debug_views"]["legend_order"].is_array());
    assert_eq!(
        public["runtime_support"]["virtual_shadows"]["debug_views"]["report"],
        "virtual-shadow-report.json"
    );
    assert_eq!(
        public["runtime_support"]["virtual_shadows"]["per_light_cost"][0]["light"],
        0
    );
    assert!(
        public["runtime_support"]["virtual_shadows"]["per_light_cost"][0]
            ["physical_depth_bytes_owned"]
            .is_number()
    );

    // Headless qualification has no swapchain to configure, but it is still
    // uncapped and must retain the requested mode for truthful telemetry.
    assert!(engine.renderer.set_present_mode(3));
    assert_eq!(engine.renderer.present_mode_code(), 3);
    assert!(!engine.renderer.vsync_active());
    assert!(!engine.renderer.set_present_mode(4));
    assert_eq!(engine.renderer.present_mode_code(), 3);
}
