//! Pipelines that compile when they are first used, or ahead of that on
//! background threads.
//!
//! Creating a pipeline compiles its shaders for the GPU (on Metal: WGSL ->
//! MSL -> `newLibraryWithSource`, synchronously). `Renderer::new` used to
//! create every pipeline of every feature up front, one after another: 57
//! pipelines and ~10.8 s on an M1 Max with a cold shader cache, of which a
//! game at the Medium preset used 16. A `LazyRenderPipeline` keeps an owned
//! copy of its descriptor (shader modules and layouts are reference-counted
//! handles) and compiles on the first `Deref`, so a feature that never runs
//! never compiles. `prewarm` compiles a set of pipelines in parallel on
//! background threads; a draw that needs one of them waits for that one only.
//!
//! Correctness never depends on prewarming: whatever path a frame takes, its
//! pipelines are built on first use. Prewarming (`Renderer::prewarm_pipelines`)
//! only decides when.
//!
//! `BLOOM_PIPELINE_LOG=1` prints each compile with its time and thread, so a
//! compile on the render thread (a hitch) shows up.

use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(not(target_arch = "wasm32"))]
type Init<P> = Box<dyn FnOnce() -> P + Send>;
// wgpu objects are not Send on the web; nothing is built off-thread there.
#[cfg(target_arch = "wasm32")]
type Init<P> = Box<dyn FnOnce() -> P>;

struct Inner<P> {
    label: String,
    cell: OnceLock<P>,
    init: Mutex<Option<Init<P>>>,
}

/// A pipeline built on first use. Derefs to the wgpu pipeline; cloning shares it.
pub struct Lazy<P>(Arc<Inner<P>>);

impl<P> Clone for Lazy<P> {
    fn clone(&self) -> Self {
        Lazy(self.0.clone())
    }
}

pub type LazyRenderPipeline = Lazy<wgpu::RenderPipeline>;
pub type LazyComputePipeline = Lazy<wgpu::ComputePipeline>;

impl<P> Lazy<P> {
    fn from_init(label: String, init: Init<P>) -> Self {
        Lazy(Arc::new(Inner {
            label,
            cell: OnceLock::new(),
            init: Mutex::new(Some(init)),
        }))
    }

    /// An already-built pipeline (one created at runtime, where it is needed at once).
    pub fn ready(pipeline: P) -> Self {
        let cell = OnceLock::new();
        let _ = cell.set(pipeline);
        Lazy(Arc::new(Inner {
            label: String::new(),
            cell,
            init: Mutex::new(None),
        }))
    }

    /// The pipeline, compiling it on this thread unless it is built or being built.
    pub fn get(&self) -> &P {
        self.0.cell.get_or_init(|| {
            let init = self
                .0
                .init
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .expect("lazy pipeline initializer already taken");
            let started = log_enabled().then(std::time::Instant::now);
            let pipeline = init();
            if let Some(t) = started {
                let thread = std::thread::current();
                eprintln!(
                    "bloom: pipeline {} compiled in {:.1} ms, {}",
                    self.0.label,
                    t.elapsed().as_secs_f64() * 1000.0,
                    match thread.name() {
                        Some("bloom-pipeline-warm") => "a prewarm thread",
                        _ => "first use (the thread waited)",
                    }
                );
            }
            pipeline
        })
    }

    pub fn is_built(&self) -> bool {
        self.0.cell.get().is_some()
    }

    pub fn label(&self) -> &str {
        &self.0.label
    }
}

impl<P> Deref for Lazy<P> {
    type Target = P;
    fn deref(&self) -> &P {
        self.get()
    }
}

fn log_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BLOOM_PIPELINE_LOG").is_some())
}

// ------------------------------------------------------------ owned descriptors

struct OwnedCompilation {
    constants: Vec<(String, f64)>,
    zero_initialize_workgroup_memory: bool,
}

impl OwnedCompilation {
    fn new(o: &wgpu::PipelineCompilationOptions<'_>) -> Self {
        OwnedCompilation {
            constants: o.constants.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            zero_initialize_workgroup_memory: o.zero_initialize_workgroup_memory,
        }
    }
}

struct OwnedVertexBuffer {
    array_stride: wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode,
    attributes: Vec<wgpu::VertexAttribute>,
}

struct OwnedFragment {
    module: wgpu::ShaderModule,
    entry_point: Option<String>,
    compilation: OwnedCompilation,
    targets: Vec<Option<wgpu::ColorTargetState>>,
}

impl LazyRenderPipeline {
    /// Records `desc` and compiles it on first use (see the module docs).
    pub fn new(device: &wgpu::Device, desc: &wgpu::RenderPipelineDescriptor<'_>) -> Self {
        let device = device.clone();
        let label = desc.label.map(str::to_owned);
        let layout = desc.layout.cloned();
        let vs_module = desc.vertex.module.clone();
        let vs_entry = desc.vertex.entry_point.map(str::to_owned);
        let vs_compilation = OwnedCompilation::new(&desc.vertex.compilation_options);
        let buffers: Vec<OwnedVertexBuffer> = desc
            .vertex
            .buffers
            .iter()
            .map(|b| OwnedVertexBuffer {
                array_stride: b.array_stride,
                step_mode: b.step_mode,
                attributes: b.attributes.to_vec(),
            })
            .collect();
        let primitive = desc.primitive;
        let depth_stencil = desc.depth_stencil.clone();
        let multisample = desc.multisample;
        let fragment = desc.fragment.as_ref().map(|f| OwnedFragment {
            module: f.module.clone(),
            entry_point: f.entry_point.map(str::to_owned),
            compilation: OwnedCompilation::new(&f.compilation_options),
            targets: f.targets.to_vec(),
        });
        let multiview_mask = desc.multiview_mask;
        let cache = desc.cache.cloned();
        let name = label.clone().unwrap_or_else(|| "(unlabeled)".to_owned());
        Lazy::from_init(
            name,
            Box::new(move || {
                let vs_constants: Vec<(&str, f64)> =
                    vs_compilation.constants.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                let vertex_buffers: Vec<wgpu::VertexBufferLayout<'_>> = buffers
                    .iter()
                    .map(|b| wgpu::VertexBufferLayout {
                        array_stride: b.array_stride,
                        step_mode: b.step_mode,
                        attributes: &b.attributes,
                    })
                    .collect();
                let fs_constants: Vec<(&str, f64)> = fragment
                    .as_ref()
                    .map(|f| f.compilation.constants.iter().map(|(k, v)| (k.as_str(), *v)).collect())
                    .unwrap_or_default();
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: label.as_deref(),
                    layout: layout.as_ref(),
                    vertex: wgpu::VertexState {
                        module: &vs_module,
                        entry_point: vs_entry.as_deref(),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &vs_constants,
                            zero_initialize_workgroup_memory: vs_compilation.zero_initialize_workgroup_memory,
                        },
                        buffers: &vertex_buffers,
                    },
                    primitive,
                    depth_stencil,
                    multisample,
                    fragment: fragment.as_ref().map(|f| wgpu::FragmentState {
                        module: &f.module,
                        entry_point: f.entry_point.as_deref(),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &fs_constants,
                            zero_initialize_workgroup_memory: f.compilation.zero_initialize_workgroup_memory,
                        },
                        targets: &f.targets,
                    }),
                    multiview_mask,
                    cache: cache.as_ref(),
                })
            }),
        )
    }
}

impl LazyComputePipeline {
    /// Records `desc` and compiles it on first use (see the module docs).
    pub fn new(device: &wgpu::Device, desc: &wgpu::ComputePipelineDescriptor<'_>) -> Self {
        let device = device.clone();
        let label = desc.label.map(str::to_owned);
        let layout = desc.layout.cloned();
        let module = desc.module.clone();
        let entry_point = desc.entry_point.map(str::to_owned);
        let compilation = OwnedCompilation::new(&desc.compilation_options);
        let cache = desc.cache.cloned();
        let name = label.clone().unwrap_or_else(|| "(unlabeled)".to_owned());
        Lazy::from_init(
            name,
            Box::new(move || {
                let constants: Vec<(&str, f64)> =
                    compilation.constants.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: label.as_deref(),
                    layout: layout.as_ref(),
                    module: &module,
                    entry_point: entry_point.as_deref(),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        zero_initialize_workgroup_memory: compilation.zero_initialize_workgroup_memory,
                    },
                    cache: cache.as_ref(),
                })
            }),
        )
    }
}

// ------------------------------------------------------------ prewarming

/// Something `prewarm` can build: a lazy pipeline of either kind.
pub trait Warm {
    fn build(&self);
    fn is_built(&self) -> bool;
}

impl<P> Warm for Lazy<P> {
    fn build(&self) {
        let _ = self.get();
    }
    fn is_built(&self) -> bool {
        Lazy::is_built(self)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub type WarmHandle = Arc<dyn Warm + Send + Sync>;
#[cfg(target_arch = "wasm32")]
pub type WarmHandle = Arc<dyn Warm>;

#[cfg(not(target_arch = "wasm32"))]
impl<P: Send + Sync + 'static> Lazy<P> {
    pub fn warm_handle(&self) -> WarmHandle {
        Arc::new(self.clone())
    }
}

#[cfg(target_arch = "wasm32")]
impl<P: 'static> Lazy<P> {
    pub fn warm_handle(&self) -> WarmHandle {
        Arc::new(self.clone())
    }
}

/// Compile these pipelines in parallel on background threads (a few at a
/// time: the GPU driver compiles in its own processes). Returns at once.
/// On the web this does nothing; pipelines build on first use there.
#[cfg(not(target_arch = "wasm32"))]
pub fn prewarm(pipelines: Vec<WarmHandle>) {
    use std::collections::VecDeque;
    let queue: VecDeque<WarmHandle> = pipelines.into_iter().filter(|p| !p.is_built()).collect();
    if queue.is_empty() {
        return;
    }
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
        .min(queue.len());
    let queue = Arc::new(Mutex::new(queue));
    for _ in 0..workers {
        let queue = queue.clone();
        let spawned = std::thread::Builder::new()
            .name("bloom-pipeline-warm".to_owned())
            .spawn(move || loop {
                let next = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
                match next {
                    Some(p) => p.build(),
                    None => break,
                }
            });
        // no thread: the pipelines build on first use instead
        if spawned.is_err() {
            break;
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub fn prewarm(_pipelines: Vec<WarmHandle>) {}

// ------------------------------------------------------------ what a frame needs

impl super::Renderer {
    /// Starts compiling, in the background, the pipelines the current
    /// settings draw with, whenever those settings change (the first frame,
    /// a quality preset, a feature toggled at runtime). Called at the start
    /// of each frame; cheap when nothing changed.
    ///
    /// The conditions mirror the frame code (measured per quality preset with
    /// a pipeline-use trace). If they drift, a frame compiles what it misses
    /// on first use: slower, never wrong.
    pub(super) fn prewarm_pipelines(&mut self, scene_3d: bool) {
        // the previous frame queued GPU-driven draws (meshes that qualify: not
        // skinned-compat, blended, refractive or layered); the first such frame
        // starts these, later ones find them built
        let gpu_driven_draws = scene_3d && self.gpu_driven.enabled() && !self.gpu_driven.draw_scratch.is_empty();
        let upscale = self.render_scale < 0.999 && !self.taa_enabled;
        let flags = [
            scene_3d,
            gpu_driven_draws,
            self.shadow_map.enabled,
            self.taa_enabled,
            self.ssao_enabled,
            self.bloom_enabled,
            self.auto_exposure,
            self.cas_strength > 0.0,
            upscale,
            self.ssr_enabled,
            self.ssgi_enabled,
            self.dof_enabled,
            self.motion_blur_enabled,
            self.sss_enabled,
            self.pt_mode > 0,
        ];
        let signature = flags.iter().enumerate().fold(0u64, |acc, (i, on)| acc | ((*on as u64) << i));
        if signature == self.prewarm_signature {
            return;
        }
        self.prewarm_signature = signature;

        let mut wanted: Vec<WarmHandle> = vec![self.pipeline_2d.warm_handle()];
        if scene_3d {
            wanted.extend([
                self.scene_depth_pipeline.warm_handle(),
                self.scene_pipeline_prepassed.warm_handle(),
                self.scene_pipeline.warm_handle(),
                self.scene_compose_pipeline.warm_handle(),
                self.composite_pipeline.warm_handle(),
            ]);
            if let Some(froxel) = &self.froxel {
                wanted.push(froxel.assign_pipeline.warm_handle());
            }
            if self.shadow_map.enabled {
                wanted.push(self.shadow_map.pipeline.warm_handle());
                wanted.push(self.shadow_map.pipeline_skinned.warm_handle());
            }
            if gpu_driven_draws {
                wanted.extend(self.gpu_driven.warm_handles());
            }
            if self.taa_enabled {
                wanted.push(self.taa_pipeline.warm_handle());
            }
            if upscale {
                wanted.push(self.upscale_pipeline.warm_handle());
            }
            if self.ssao_enabled {
                // SSAO reads the Hi-Z pyramid, which the occlusion culler reduces too
                wanted.extend([
                    self.hiz_linearize_pipeline.warm_handle(),
                    self.hiz_downsample_pipeline.warm_handle(),
                    self.ssao_pipeline.warm_handle(),
                    self.ssao_blur_pipeline.warm_handle(),
                    self.occlusion.warm_handle(),
                ]);
            }
            if self.bloom_enabled {
                wanted.extend([
                    self.bloom_pipeline_threshold_downsample.warm_handle(),
                    self.bloom_pipeline_downsample.warm_handle(),
                    self.bloom_pipeline_upsample.warm_handle(),
                ]);
            }
            if self.auto_exposure {
                wanted.push(self.exposure_pipeline.warm_handle());
            }
            if self.cas_strength > 0.0 {
                wanted.push(self.cas_pipeline.warm_handle());
            }
            if self.ssr_enabled {
                wanted.push(self.ssr_pipeline.warm_handle());
                wanted.push(self.ssr_temporal_pipeline.warm_handle());
            }
            if self.ssgi_enabled {
                wanted.extend([
                    self.probe_place_pipeline.warm_handle(),
                    self.probe_temporal_pipeline.warm_handle(),
                    self.probe_spatial_pipeline.warm_handle(),
                    self.probe_resolve_pipeline.warm_handle(),
                    self.card_capture_pipeline.warm_handle(),
                ]);
                // hardware ray queries where the device has them, else the SDF tracers
                match &self.probe_trace_hw_pipeline {
                    Some(hw) => wanted.push(hw.warm_handle()),
                    None => wanted.push(self.probe_trace_pipeline.warm_handle()),
                }
                match &self.card_light_hw_pipeline {
                    Some(hw) => wanted.push(hw.warm_handle()),
                    None => wanted.push(self.card_light_pipeline.warm_handle()),
                }
                match &self.wsrc_bake_hw_pipeline {
                    Some(hw) => wanted.push(hw.warm_handle()),
                    None => wanted.push(self.wsrc_bake_pipeline.warm_handle()),
                }
            }
            if self.dof_enabled {
                wanted.push(self.dof_pipeline.warm_handle());
            }
            if self.motion_blur_enabled {
                wanted.push(self.motion_blur_pipeline.warm_handle());
            }
            if self.sss_enabled {
                wanted.push(self.sss_pipeline.warm_handle());
            }
            if self.pt_mode > 0 {
                wanted.extend(
                    [&self.pt_pipeline, &self.pt_atrous_mid_pipeline, &self.pt_atrous_final_pipeline]
                        .into_iter()
                        .flatten()
                        .map(|p| p.warm_handle()),
                );
            }
        }
        prewarm(wanted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counting(label: &str, counter: Arc<AtomicUsize>) -> Lazy<u32> {
        Lazy::from_init(
            label.to_owned(),
            Box::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                7
            }),
        )
    }

    #[test]
    fn builds_once_on_first_use() {
        let n = Arc::new(AtomicUsize::new(0));
        let p = counting("p", n.clone());
        assert!(!p.is_built());
        assert_eq!(n.load(Ordering::SeqCst), 0, "creating a lazy pipeline must not build it");
        assert_eq!(*p, 7);
        let q = p.clone();
        assert_eq!(*q, 7);
        assert_eq!(n.load(Ordering::SeqCst), 1);
        assert!(q.is_built());
    }

    #[test]
    fn ready_is_built() {
        assert!(Lazy::ready(3u32).is_built());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn prewarm_builds_in_background_and_first_use_waits() {
        let n = Arc::new(AtomicUsize::new(0));
        let slow: Lazy<u32> = {
            let n = n.clone();
            Lazy::from_init(
                "slow".to_owned(),
                Box::new(move || {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    n.fetch_add(1, Ordering::SeqCst);
                    9
                }),
            )
        };
        let others: Vec<Lazy<u32>> = (0..6).map(|i| counting(&format!("p{i}"), n.clone())).collect();
        let mut handles = vec![slow.warm_handle()];
        handles.extend(others.iter().map(|p| p.warm_handle()));
        prewarm(handles);
        // racing the warm thread: either way it is built exactly once
        assert_eq!(*slow, 9);
        for p in &others {
            assert_eq!(**p, 7);
        }
        assert_eq!(n.load(Ordering::SeqCst), 7);
    }
}
