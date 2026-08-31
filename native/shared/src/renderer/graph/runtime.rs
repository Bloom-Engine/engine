//! Compiled-plan cache and frame-local execution binding.

use super::{CompileError, CompileOptions, CompiledGraph, GraphBuilder, PassId};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

/// Backend-agnostic hook used by the compiled executor to expose stable pass
/// names to GPU capture tools without coupling graph code to wgpu.
pub trait GraphDebugMarkerContext {
    fn push_graph_debug_group(&mut self, label: &str);
    fn pop_graph_debug_group(&mut self);
}

/// Dynamic draw content requires the two scene snapshots used by refractive
/// materials. It changes the allocation contract and therefore belongs in the
/// topology key.
pub const FRAME_FEATURE_SSAO: u64 = 1 << 1;
pub const FRAME_FEATURE_BLOOM: u64 = 1 << 2;
pub const FRAME_FEATURE_SSR: u64 = 1 << 4;
pub const FRAME_FEATURE_SSGI: u64 = 1 << 5;
pub const FRAME_FEATURE_SCENE_SNAPSHOTS: u64 = 1 << 16;
/// The final output must be copied to a CPU-visible staging buffer this frame.
/// Capture is deliberately part of the topology key: ordinary frames pay no
/// execution-node or transition cost, while repeated captures reuse one plan.
pub const FRAME_FEATURE_CAPTURE_OUTPUT: u64 = 1 << 17;
/// Qualification capture additionally reads named HDR/depth graph resources.
pub const FRAME_FEATURE_CAPTURE_QUALITY: u64 = 1 << 18;
/// Imported physical transmission writes per-object velocity in its dedicated
/// forward pass. This versions the persistent velocity target even on folded
/// platforms that do not allocate scene snapshots.
pub const FRAME_FEATURE_IMPORTED_REFRACTION: u64 = 1 << 19;
/// Imported glTF BLEND draws use the bounded weighted-blended OIT path.
/// The two accumulation targets are transient and must not exist in plans
/// for sorted-only or opaque frames.
pub const FRAME_FEATURE_WEIGHTED_TRANSPARENCY: u64 = 1 << 20;
/// A visible imported BLEND/transmission draw feeds TAA this frame. Coverage
/// is a render-resolution transient and must not exist for opaque, custom-only,
/// or TAA-disabled topologies.
pub const FRAME_FEATURE_TEMPORAL_REACTIVE: u64 = 1 << 21;
/// Physical transmission contributes a lazy persistent light-space
/// transmittance/depth cascade and an additive post-opaque sun correction.
pub const FRAME_FEATURE_TRANSMITTED_SHADOWS: u64 = 1 << 22;
/// Qualification-only native MRT readback samples the renderer-owned HDR,
/// material, velocity, and albedo textures in a terminal compute pass. The
/// bit is absent from ordinary plans, so normal frames gain no resource
/// transitions or execution nodes.
pub const FRAME_FEATURE_CAPTURE_MRT: u64 = 1 << 23;
/// An explicitly enabled virtual-geometry submission needs a current max-depth
/// pyramid to conservatively cull its next frame. Ordinary plans omit it.
pub const FRAME_FEATURE_VIRTUAL_HIZ: u64 = 1 << 24;

/// Coarse output class used in topology keys. Exact dimensions belong to the
/// allocation/resize generation; topology does not rebuild for every resize.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum ResolutionClass {
    Tiny,
    Small,
    Medium,
    Large,
    Ultra,
}

impl ResolutionClass {
    pub fn from_extent(width: u32, height: u32) -> Self {
        match u64::from(width).saturating_mul(u64::from(height)) {
            0..=409_599 => Self::Tiny,
            409_600..=921_599 => Self::Small,
            921_600..=2_073_599 => Self::Medium,
            2_073_600..=4_194_303 => Self::Large,
            _ => Self::Ultra,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum CapabilityTier {
    Constrained,
    Raster,
    HardwareRayQuery,
    HardwareRayQueryTextureArrays,
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum PathTracingMode {
    Off,
    Progressive,
    Realtime,
}

impl PathTracingMode {
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::Off,
            1 => Self::Progressive,
            _ => Self::Realtime,
        }
    }
}

/// Hashable topology key. Values that only affect uniforms are deliberately
/// absent. `feature_mask` contains only switches that add/remove passes or
/// change resource contracts.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct FramePlanKey {
    pub resolution: ResolutionClass,
    pub quality_tier: u8,
    pub feature_mask: u64,
    pub capability: CapabilityTier,
    pub path_tracing: PathTracingMode,
    pub post_pass_count: u16,
    pub render_target_output: bool,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct PlanCacheStats {
    pub compile_count: u64,
    pub hit_count: u64,
    pub plan_count: usize,
}

/// Cache of immutable compiled topologies. Physical allocations are cached
/// separately by the transient pool because they also depend on resize
/// generation and exact resolved dimensions.
pub struct PlanCache<K> {
    plans: HashMap<K, Arc<CompiledGraph>>,
    compile_count: u64,
    hit_count: u64,
}

impl<K> PlanCache<K>
where
    K: Copy + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            plans: HashMap::new(),
            compile_count: 0,
            hit_count: 0,
        }
    }

    pub fn get_or_compile(
        &mut self,
        key: K,
        options: CompileOptions,
        build: impl FnOnce() -> GraphBuilder,
    ) -> Result<Arc<CompiledGraph>, CompileError> {
        if let Some(plan) = self.plans.get(&key) {
            self.hit_count = self.hit_count.saturating_add(1);
            return Ok(Arc::clone(plan));
        }
        let plan = Arc::new(build().compile(options)?);
        self.compile_count = self.compile_count.saturating_add(1);
        self.plans.insert(key, Arc::clone(&plan));
        Ok(plan)
    }

    pub fn get(&self, key: &K) -> Option<Arc<CompiledGraph>> {
        self.plans.get(key).map(Arc::clone)
    }

    pub fn clear(&mut self) {
        self.plans.clear();
    }

    pub fn stats(&self) -> PlanCacheStats {
        PlanCacheStats {
            compile_count: self.compile_count,
            hit_count: self.hit_count,
            plan_count: self.plans.len(),
        }
    }
}

impl<K> Default for PlanCache<K>
where
    K: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

pub type ExecutionRunFn<'a, Ctx> = Box<dyn FnOnce(&mut Ctx) + 'a>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionError {
    UnknownPass(String),
    DuplicateBinding(String),
    MissingBinding(String),
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPass(name) => write!(f, "compiled graph has no pass '{name}'"),
            Self::DuplicateBinding(name) => write!(f, "pass '{name}' was bound twice"),
            Self::MissingBinding(name) => {
                write!(f, "compiled pass '{name}' has no execution closure")
            }
        }
    }
}

impl std::error::Error for ExecutionError {}

/// Frame-local closure table backed by an immutable compiled topology.
pub struct ExecutableGraph<'a, Ctx> {
    plan: Arc<CompiledGraph>,
    runs: Vec<Option<ExecutionRunFn<'a, Ctx>>>,
}

impl<'a, Ctx> ExecutableGraph<'a, Ctx> {
    pub fn new(plan: Arc<CompiledGraph>) -> Self {
        let runs = (0..plan.passes.len()).map(|_| None).collect();
        Self { plan, runs }
    }

    pub fn plan(&self) -> &CompiledGraph {
        &self.plan
    }

    pub fn contains(&self, name: &str) -> bool {
        self.plan.pass(name).is_some()
    }

    pub fn bind(&mut self, name: &str, run: ExecutionRunFn<'a, Ctx>) -> Result<(), ExecutionError> {
        let pass = self
            .plan
            .pass(name)
            .ok_or_else(|| ExecutionError::UnknownPass(name.to_string()))?;
        let position = self
            .plan
            .pass_position(pass.id)
            .expect("compiled pass has a schedule position");
        if self.runs[position].is_some() {
            return Err(ExecutionError::DuplicateBinding(name.to_string()));
        }
        self.runs[position] = Some(run);
        Ok(())
    }

    /// Bind only when an optional pass is present in this topology.
    pub fn bind_optional(&mut self, name: &str, run: ExecutionRunFn<'a, Ctx>) {
        if self.contains(name) {
            self.bind(name, run)
                .expect("optional pass is bound at most once");
        }
    }

    pub fn execute(mut self, context: &mut Ctx) -> Result<(), ExecutionError> {
        for (position, pass) in self.plan.passes.iter().enumerate() {
            let run = self.runs[position]
                .take()
                .ok_or_else(|| ExecutionError::MissingBinding(pass.name.clone()))?;
            run(context);
        }
        Ok(())
    }

    pub fn execute_marked(mut self, context: &mut Ctx) -> Result<(), ExecutionError>
    where
        Ctx: GraphDebugMarkerContext,
    {
        for (position, pass) in self.plan.passes.iter().enumerate() {
            let run = self.runs[position]
                .take()
                .ok_or_else(|| ExecutionError::MissingBinding(pass.name.clone()))?;
            context.push_graph_debug_group(&pass.name);
            run(context);
            context.pop_graph_debug_group();
        }
        Ok(())
    }

    pub fn pass_id(&self, name: &str) -> Option<PassId> {
        self.plan.pass(name).map(|pass| pass.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::graph::{Extent, TextureDesc, TextureUsage};

    #[test]
    fn stable_key_compiles_once_and_executes_many_times() {
        let key = FramePlanKey {
            resolution: ResolutionClass::Medium,
            quality_tier: 3,
            feature_mask: 7,
            capability: CapabilityTier::Raster,
            path_tracing: PathTracingMode::Off,
            post_pass_count: 0,
            render_target_output: false,
        };
        let build = || {
            let mut graph = GraphBuilder::new("cached");
            let resource = graph.create_texture(
                "color",
                TextureDesc::color(
                    wgpu::TextureFormat::Rgba16Float,
                    Extent::RenderRelative {
                        numerator: 1,
                        denominator: 1,
                        layers: 1,
                    },
                    TextureUsage::COLOR_ATTACHMENT,
                ),
            );
            let pass = graph.add_pass("draw");
            let _ = graph.write_texture(pass, resource, TextureUsage::COLOR_ATTACHMENT);
            graph
        };
        let mut cache = PlanCache::new();
        let first = cache
            .get_or_compile(key, CompileOptions::NO_ALIASING, build)
            .unwrap();
        let second = cache
            .get_or_compile(key, CompileOptions::NO_ALIASING, build)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(
            cache.stats(),
            PlanCacheStats {
                compile_count: 1,
                hit_count: 1,
                plan_count: 1
            }
        );

        let mut output = Vec::new();
        for plan in [first, second] {
            let mut executable = ExecutableGraph::new(plan);
            executable
                .bind(
                    "draw",
                    Box::new(|out: &mut Vec<&'static str>| out.push("draw")),
                )
                .unwrap();
            executable.execute(&mut output).unwrap();
        }
        assert_eq!(output, vec!["draw", "draw"]);
    }

    #[test]
    fn missing_closure_is_rejected_before_silent_pass_loss() {
        let mut graph = GraphBuilder::new("missing");
        graph.add_pass("required");
        let plan = Arc::new(graph.compile(CompileOptions::NO_ALIASING).unwrap());
        let executable: ExecutableGraph<'_, ()> = ExecutableGraph::new(plan);
        assert_eq!(
            executable.execute(&mut ()),
            Err(ExecutionError::MissingBinding("required".to_string()))
        );
    }

    #[test]
    fn marked_execution_brackets_the_compiled_pass_name() {
        struct Context(Vec<String>);
        impl GraphDebugMarkerContext for Context {
            fn push_graph_debug_group(&mut self, label: &str) {
                self.0.push(format!("push:{label}"));
            }

            fn pop_graph_debug_group(&mut self) {
                self.0.push("pop".to_string());
            }
        }

        let mut graph = GraphBuilder::new("marked");
        graph.add_pass("stable-name");
        let plan = Arc::new(graph.compile(CompileOptions::NO_ALIASING).unwrap());
        let mut executable = ExecutableGraph::new(plan);
        executable
            .bind(
                "stable-name",
                Box::new(|context: &mut Context| context.0.push("run".to_string())),
            )
            .unwrap();
        let mut context = Context(Vec::new());
        executable.execute_marked(&mut context).unwrap();
        assert_eq!(context.0, ["push:stable-name", "run", "pop"]);
    }
}
