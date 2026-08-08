//! Renderer integration for compiled/cached frame topologies.

use super::{graph, Renderer};
use std::sync::Arc;

const FEATURE_SHADOWS: u64 = 1 << 0;
const FEATURE_TAA: u64 = 1 << 3;
const FEATURE_MOTION_BLUR: u64 = 1 << 6;
const FEATURE_SSS: u64 = 1 << 7;
const FEATURE_AUTO_EXPOSURE: u64 = 1 << 8;

impl Renderer {
    pub(super) fn graph_debug_markers_enabled(&self) -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("BLOOM_GRAPH_DEBUG_MARKERS")
                .map(|value| value != "0")
                .unwrap_or(false)
        })
    }

    pub(super) fn frame_plan_key(&self, render_target_output: bool) -> graph::FramePlanKey {
        let mut feature_mask = 0;
        let imported_refraction = self.imported_refraction_enabled
            && (self.has_refractive_model_draws || self.has_refractive_scene_nodes);
        for (enabled, bit) in [
            (self.shadow_map.enabled, FEATURE_SHADOWS),
            (self.ssao_enabled, graph::FRAME_FEATURE_SSAO),
            (self.bloom_enabled, graph::FRAME_FEATURE_BLOOM),
            (self.taa_enabled, FEATURE_TAA),
            (self.ssr_enabled, graph::FRAME_FEATURE_SSR),
            (self.ssgi_enabled, graph::FRAME_FEATURE_SSGI),
            (self.motion_blur_enabled, FEATURE_MOTION_BLUR),
            (self.sss_enabled, FEATURE_SSS),
            (self.auto_exposure, FEATURE_AUTO_EXPOSURE),
            (
                self.material_system
                    .translucent_commands
                    .iter()
                    .any(|command| {
                        self.material_system
                            .pipelines
                            .get(command.material as usize - 1)
                            .and_then(|pipeline| pipeline.as_ref())
                            .map(|pipeline| pipeline.reads_scene)
                            .unwrap_or(false)
                    })
                    || (cfg!(not(fold_scene_inputs)) && imported_refraction),
                graph::FRAME_FEATURE_SCENE_SNAPSHOTS,
            ),
            (
                imported_refraction,
                graph::FRAME_FEATURE_IMPORTED_REFRACTION,
            ),
            (
                self.weighted_transparency_active,
                graph::FRAME_FEATURE_WEIGHTED_TRANSPARENCY,
            ),
            (
                self.temporal_reactive_active,
                graph::FRAME_FEATURE_TEMPORAL_REACTIVE,
            ),
            (
                self.transmitted_shadows_active,
                graph::FRAME_FEATURE_TRANSMITTED_SHADOWS,
            ),
        ] {
            if enabled {
                feature_mask |= bit;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if !render_target_output
            && (self.screenshot_requested
                || self.pending_quality_capture_dir.is_some()
                || self.pending_mrt_capture_dir.is_some())
        {
            feature_mask |= graph::FRAME_FEATURE_CAPTURE_OUTPUT;
            if self.pending_quality_capture_dir.is_some() {
                feature_mask |= graph::FRAME_FEATURE_CAPTURE_QUALITY;
            }
            if self.pending_mrt_capture_dir.is_some() {
                feature_mask |= graph::FRAME_FEATURE_CAPTURE_MRT;
            }
        }

        let quality_tier = if feature_mask
            & (FEATURE_SHADOWS
                | graph::FRAME_FEATURE_SSAO
                | graph::FRAME_FEATURE_BLOOM
                | FEATURE_TAA
                | graph::FRAME_FEATURE_SSR
                | graph::FRAME_FEATURE_SSGI)
            == 0
        {
            0
        } else if feature_mask
            & (graph::FRAME_FEATURE_SSAO
                | FEATURE_TAA
                | graph::FRAME_FEATURE_SSR
                | graph::FRAME_FEATURE_SSGI)
            == 0
        {
            1
        } else if feature_mask & (graph::FRAME_FEATURE_SSR | graph::FRAME_FEATURE_SSGI) == 0 {
            2
        } else if feature_mask & (FEATURE_MOTION_BLUR | FEATURE_SSS) == 0 {
            3
        } else {
            4
        };

        let capability = if cfg!(lean_mrt) {
            graph::CapabilityTier::Constrained
        } else if self.hw_rt_enabled && self.pt_texture_arrays_enabled {
            graph::CapabilityTier::HardwareRayQueryTextureArrays
        } else if self.hw_rt_enabled {
            graph::CapabilityTier::HardwareRayQuery
        } else {
            graph::CapabilityTier::Raster
        };

        graph::FramePlanKey {
            resolution: graph::ResolutionClass::from_extent(
                self.surface_config.width,
                self.surface_config.height,
            ),
            quality_tier,
            feature_mask,
            capability,
            path_tracing: graph::PathTracingMode::from_u32(if self.pt_active() {
                self.pt_mode
            } else {
                0
            }),
            post_pass_count: self.post_passes.len().min(u16::MAX as usize) as u16,
            render_target_output,
        }
    }

    pub(super) fn compiled_frame_plan(
        &mut self,
        render_target_output: bool,
    ) -> Result<Arc<graph::CompiledGraph>, graph::CompileError> {
        let key = self.frame_plan_key(render_target_output);
        let output_format = self.output_format;
        let compile_count_before = self.frame_plan_cache.stats().compile_count;
        let plan = self.frame_plan_cache.get_or_compile(
            key,
            // The allocator is conservative: exact descriptor/alias-class
            // match plus strictly non-overlapping lifetimes. Persistent and
            // temporal resources are imports and can never enter this set.
            graph::CompileOptions::CONSERVATIVE_ALIASING,
            || graph::build_renderer_frame_plan(key, output_format),
        )?;
        let compile_count_after = self.frame_plan_cache.stats().compile_count;
        self.frame_resource_stats
            .created_graph_compiles(compile_count_after.saturating_sub(compile_count_before));
        self.maybe_dump_frame_plan(&plan);
        self.last_frame_plan = Some(Arc::clone(&plan));
        Ok(plan)
    }

    pub fn render_graph_cache_stats(&self) -> graph::PlanCacheStats {
        self.frame_plan_cache.stats()
    }

    pub fn render_graph_json(&self) -> Option<String> {
        self.last_frame_plan.as_ref().map(|plan| plan.to_json())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn maybe_dump_frame_plan(&mut self, plan: &graph::CompiledGraph) {
        if self.dumped_frame_plans.contains(&plan.plan_id) {
            return;
        }
        let Some(directory) = std::env::var_os("BLOOM_GRAPH_DUMP_DIR") else {
            return;
        };
        let directory = std::path::PathBuf::from(directory);
        if let Err(error) = std::fs::create_dir_all(&directory) {
            eprintln!(
                "bloom graph: cannot create dump directory '{}': {error}",
                directory.display()
            );
            return;
        }
        let stem = format!("bloom-frame-{:016x}", plan.plan_id);
        let json_path = directory.join(format!("{stem}.json"));
        let dot_path = directory.join(format!("{stem}.dot"));
        if let Err(error) = std::fs::write(&json_path, plan.to_json()) {
            eprintln!(
                "bloom graph: cannot write '{}': {error}",
                json_path.display()
            );
            return;
        }
        if let Err(error) = std::fs::write(&dot_path, plan.to_dot()) {
            eprintln!(
                "bloom graph: cannot write '{}': {error}",
                dot_path.display()
            );
            return;
        }
        self.dumped_frame_plans.insert(plan.plan_id);
    }

    #[cfg(target_arch = "wasm32")]
    fn maybe_dump_frame_plan(&mut self, plan: &graph::CompiledGraph) {
        self.dumped_frame_plans.insert(plan.plan_id);
    }
}
