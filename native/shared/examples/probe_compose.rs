use bloom_shared::engine::EngineState;
use bloom_shared::renderer::Renderer;
fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(), ..Default::default()
    })).unwrap();
    let renderer = Renderer::new_headless(device, queue, 256, 256);
    let mut eng = EngineState::new(renderer);
    eng.renderer.set_taa_enabled(true);
    
    let n: u32 = std::env::args().nth(1).map(|v| v.parse().unwrap()).unwrap_or(1); for i in 0..n {
        eng.begin_frame();
        eng.renderer.set_clear_color(13.0, 18.0, 26.0, 255.0);
        eng.renderer.begin_mode_3d(4.0, 3.0, 6.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 45.0, 0.0);
        eng.renderer.add_directional_light(-0.5, -1.0, -0.3, 1.0, 0.95, 0.9, 1.2);
        eng.renderer.add_point_light(2.0, 2.0, 2.0, 10.0, 0.2, 0.4, 1.0, 2.0);
        eng.renderer.draw_plane(0.0, 0.0, 0.0, 10.0, 10.0, 120.0, 120.0, 125.0, 255.0);
        eng.renderer.draw_cube(-1.2, 0.5, 0.0, 1.0, 1.0, 1.0, 230.0, 41.0, 55.0, 255.0);
        if i + 1 == n { eng.renderer.screenshot_requested = true; }
        eng.end_frame();
    }
    let (w, h, mut data) = eng.renderer.screenshot_data.take().unwrap();
    for px in data.chunks_exact_mut(4) { px.swap(0, 2); }
    image::save_buffer("/tmp/probe_compose.png", &data, w, h, image::ColorType::Rgba8).unwrap();
    println!("sky pixel: {:?}", &data[..4]);
}
