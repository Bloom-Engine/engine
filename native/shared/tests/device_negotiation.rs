//! Production device-request smoke: proves the granted minimum limits can
//! construct the complete renderer, not only a bare wgpu device.

#[test]
fn negotiated_headless_device_constructs_the_complete_renderer() {
    let engine = bloom_shared::attach::attach_headless_engine(wgpu::Backends::PRIMARY, 64, 64);
    let engine = match engine {
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
}
