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
}
