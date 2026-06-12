use crate::audio::AudioMixer;
use crate::input::InputState;
use crate::renderer::Renderer;
use crate::text_renderer::TextRenderer;
use crate::textures::TextureManager;
#[cfg(feature = "models3d")]
use crate::models::ModelManager;
use crate::scene::SceneGraph;
use crate::frame_callbacks::FrameCallbackSystem;
use crate::postfx::PostFxPipeline;
use crate::profiler::Profiler;
use crate::drs::DrsController;
#[cfg(all(feature = "jolt", not(target_arch = "wasm32")))]
use crate::physics_jolt::JoltPhysics;

#[cfg(feature = "web")]
use web_time::Instant;
#[cfg(not(feature = "web"))]
use std::time::Instant;

pub struct EngineState {
    pub renderer: Renderer,
    pub text: TextRenderer,
    pub input: InputState,
    pub audio: AudioMixer,
    pub textures: TextureManager,
    #[cfg(feature = "models3d")]
    pub models: ModelManager,
    pub scene: SceneGraph,
    pub frame_callbacks: FrameCallbackSystem,
    pub postfx: Option<PostFxPipeline>,
    pub profiler: Profiler,
    pub screenshot_pending: bool,
    pub drs: DrsController,

    // Timing
    pub target_fps: f64,
    pub delta_time: f64,
    last_frame_time: Instant,
    frame_count: u64,
    fps_timer: Instant,
    fps_frame_count: u64,
    current_fps: f64,
    start_time: Instant,

    pub should_close: bool,

    // Read-back state for the split FFI getters: bloom_scene_pick /
    // bloom_scene_pick_all / bloom_project_to_screen store here, and the
    // bloom_pick_hit_* / bloom_pick_all_* / bloom_project_screen_y getters
    // read the components back one f64 at a time (Perry FFI returns are
    // scalar). Lived in per-crate `static mut`s before the FFI
    // unification.
    pub last_pick: Option<crate::picking::PickResult>,
    pub last_pick_all: Vec<crate::picking::PickResult>,
    pub last_project: (f64, f64),

    // When true, end_frame() takes a direct-to-swapchain path that skips
    // scene graph prep, shadow maps, HDR/post-FX, SDF/WSRC bakes, etc.
    // Intended for pure-2D games that only need the batched 2D pipeline.
    // Off by default to preserve existing behaviour.
    pub direct_2d_mode: bool,

    #[cfg(all(feature = "jolt", not(target_arch = "wasm32")))]
    pub jolt: JoltPhysics,
}

impl EngineState {
    pub fn new(renderer: Renderer) -> Self {
        let now = Instant::now();
        let mut profiler = Profiler::new();
        profiler.init_gpu(&renderer.device, &renderer.queue);
        // Ticket 007b: tell the scene graph whether the device was
        // created with HW ray-query so geometry upload pays the BLAS /
        // BLAS_INPUT cost only when it can be used.
        let mut scene = SceneGraph::new();
        scene.hw_rt_enabled = renderer.hw_rt_enabled;
        Self {
            renderer,
            text: TextRenderer::new(),
            input: InputState::new(),
            audio: AudioMixer::new(),
            textures: TextureManager::new(),
            #[cfg(feature = "models3d")]
            models: ModelManager::new(),
            scene,
            frame_callbacks: FrameCallbackSystem::new(),
            postfx: None,
            profiler,
            screenshot_pending: false,
            drs: DrsController::new(),
            target_fps: 60.0,
            delta_time: 0.0,
            last_frame_time: now,
            frame_count: 0,
            fps_timer: now,
            fps_frame_count: 0,
            current_fps: 0.0,
            start_time: now,
            should_close: false,
            last_pick: None,
            last_pick_all: Vec::new(),
            last_project: (0.0, 0.0),
            direct_2d_mode: false,
            #[cfg(all(feature = "jolt", not(target_arch = "wasm32")))]
            jolt: JoltPhysics::new(),
        }
    }

    /// Hard ceiling on the delta time games observe. A debugger pause, OS
    /// hitch, or backgrounded tab produces one slowed-down frame instead
    /// of a quarter-hour dt that explodes physics, animation, and any
    /// game-side `pos += vel * dt`. Matches Unity's maximumDeltaTime
    /// default order of magnitude.
    pub const MAX_DELTA_TIME: f64 = 0.25;

    pub fn begin_frame(&mut self) {
        let now = Instant::now();
        self.delta_time = now
            .duration_since(self.last_frame_time)
            .as_secs_f64()
            .min(Self::MAX_DELTA_TIME);
        self.last_frame_time = now;

        // DRS samples wall-clock frame time and may step render_scale
        // up or down before this frame's renderer work begins. No-op
        // when disabled (default) — and the controller's own cooldown
        // + hysteresis bound how often a step actually happens.
        self.drs.tick(self.delta_time, &mut self.renderer);

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
        self.profiler.begin("frame_callbacks");
        self.frame_callbacks.run_all(self.delta_time);
        self.profiler.end("frame_callbacks");
    }

    pub fn end_frame(&mut self) {
        if self.direct_2d_mode {
            // Fast path for pure-2D games: render direct to the swapchain,
            // skipping scene prep, shadow maps, HDR/tonemap, SSAO, bloom,
            // SDF/WSRC bakes and mesh-card capture. On mobile GPUs the
            // full deferred pipeline easily costs tens of ms/frame even
            // when the scene graph is empty; this path typically hits
            // 60 fps on the same device.
            self.profiler.begin("render_total");
            self.renderer.end_frame();
            self.profiler.end("render_total");
        } else {
            self.profiler.begin("scene_prepare");
            // Collect last frame's occlusion-grid readback (if it
            // completed) before culling against it.
            self.renderer.occlusion.poll(&self.renderer.device);
            self.scene.prepare(
                &self.renderer.device,
                &self.renderer.queue,
                &self.renderer.vp_matrix(),
                &self.renderer.prev_vp_matrix,
                self.renderer.uniform_3d_layout(),
                Some(&self.renderer.occlusion),
            );
            self.scene.prepare_materials(&self.renderer);
            self.profiler.end("scene_prepare");

            // Phase 6 — drain hot-reload events and rebuild any
            // material whose .wgsl file changed on disk. Cheap when
            // nothing has changed (just a try_recv on an empty queue).
            self.renderer.poll_material_hot_reload();

            // Sync material-system PerFrame + PerView UBOs with the
            // current clock + camera before dispatching any queued
            // material draws. Without this, group 1 (view/proj/camera/
            // lights/shadow) is zero and shaders that read e.g.
            // `view.view_proj` produce offscreen geometry.
            let t  = self.get_time() as f32;
            let dt = self.delta_time as f32;
            self.renderer.material_system_begin_frame(t, dt);

            self.profiler.begin("render_total");
            self.renderer.end_frame_with_scene(&mut self.scene, &mut self.profiler);
            self.profiler.end("render_total");
        }

        self.profiler.frame_end(&self.renderer.device);
        self.input.end_frame();

        // Vsync (PresentMode::Fifo, the wgpu default) already caps frame rate.
        // Only apply CPU sleep-based cap when vsync is not active.
        // On WASM, frame pacing is handled by requestAnimationFrame.
        #[cfg(not(target_arch = "wasm32"))]
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
