use crate::audio::AudioMixer;
use crate::input::InputState;
use crate::renderer::Renderer;
use crate::text_renderer::TextRenderer;
use crate::textures::TextureManager;
use crate::models::ModelManager;
use crate::scene::SceneGraph;
use crate::frame_callbacks::FrameCallbackSystem;

pub struct EngineState {
    pub renderer: Renderer,
    pub text: TextRenderer,
    pub input: InputState,
    pub audio: AudioMixer,
    pub textures: TextureManager,
    pub models: ModelManager,
    pub scene: SceneGraph,
    pub frame_callbacks: FrameCallbackSystem,

    // Timing
    pub target_fps: f64,
    pub delta_time: f64,
    last_frame_time: std::time::Instant,
    frame_count: u64,
    fps_timer: std::time::Instant,
    fps_frame_count: u64,
    current_fps: f64,
    start_time: std::time::Instant,

    pub should_close: bool,
}

impl EngineState {
    pub fn new(renderer: Renderer) -> Self {
        let now = std::time::Instant::now();
        Self {
            renderer,
            text: TextRenderer::new(),
            input: InputState::new(),
            audio: AudioMixer::new(),
            textures: TextureManager::new(),
            models: ModelManager::new(),
            scene: SceneGraph::new(),
            frame_callbacks: FrameCallbackSystem::new(),
            target_fps: 60.0,
            delta_time: 0.0,
            last_frame_time: now,
            frame_count: 0,
            fps_timer: now,
            fps_frame_count: 0,
            current_fps: 0.0,
            start_time: now,
            should_close: false,
        }
    }

    pub fn begin_frame(&mut self) {
        let now = std::time::Instant::now();
        self.delta_time = now.duration_since(self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;

        self.fps_frame_count += 1;
        let fps_elapsed = now.duration_since(self.fps_timer).as_secs_f64();
        if fps_elapsed >= 1.0 {
            self.current_fps = self.fps_frame_count as f64 / fps_elapsed;
            self.fps_frame_count = 0;
            self.fps_timer = now;
        }

        self.input.begin_frame();
        self.renderer.begin_frame();
        self.frame_count += 1;

        // Run frame callbacks after begin_frame (matching R3F's useFrame timing)
        self.frame_callbacks.run_all(self.delta_time);
    }

    pub fn end_frame(&mut self) {
        // Prepare scene graph GPU resources before rendering
        self.scene.prepare(
            &self.renderer.device,
            &self.renderer.queue,
            &self.renderer.vp_matrix(),
            self.renderer.uniform_3d_layout(),
        );
        self.renderer.end_frame_with_scene(&self.scene);
        self.input.end_frame();

        // Vsync (PresentMode::Fifo, the wgpu default) already caps frame rate.
        // Only apply CPU sleep-based cap when vsync is not active.
        if self.target_fps > 0.0 && !self.renderer.vsync_active() {
            let target_frame_time = 1.0 / self.target_fps;
            let elapsed = self.last_frame_time.elapsed().as_secs_f64();
            if elapsed < target_frame_time {
                let sleep_time = target_frame_time - elapsed;
                std::thread::sleep(std::time::Duration::from_secs_f64(sleep_time));
            }
        }
    }

    pub fn get_fps(&self) -> f64 { self.current_fps }
    pub fn get_time(&self) -> f64 { self.start_time.elapsed().as_secs_f64() }
    pub fn screen_width(&self) -> f64 { self.renderer.width() as f64 }
    pub fn screen_height(&self) -> f64 { self.renderer.height() as f64 }
}
