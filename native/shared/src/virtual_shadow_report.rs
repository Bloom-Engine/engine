use super::*;
use std::fmt::Write as _;

pub(super) fn json(vsm: &DirectionalVirtualShadowMap) -> String {
    let stats = vsm.cache.stats();
    let levels = vsm.cache.level_counts(0);
    let page_table_bytes = VSM_VIRTUAL_PAGES_PER_AXIS as u64
        * VSM_VIRTUAL_PAGES_PER_AXIS as u64
        * VSM_CLIP_LEVELS as u64
        * std::mem::size_of::<u32>() as u64;
    let render_staging_bytes = crate::shadows::SHADOW_UNIFORM_STRIDE as u64
        * crate::shadows::SHADOW_MAX_NODES as u64
        * vsm.render_budget as u64;
    let gpu_receiver_bytes = vsm
        .gpu
        .as_ref()
        .map_or(0, |gpu| gpu.receiver_demand.memory_bytes());
    let gpu_receiver_stats = vsm
        .gpu
        .as_ref()
        .map(|gpu| gpu.receiver_demand.stats)
        .unwrap_or_default();
    let gpu_receiver_validated = vsm
        .gpu
        .as_ref()
        .is_some_and(|gpu| gpu.receiver_demand.validated());
    let gpu_receiver_enabled = vsm
        .gpu
        .as_ref()
        .is_some_and(|gpu| gpu.receiver_demand.enabled());
    let gpu_receiver_in_flight = vsm
        .gpu
        .as_ref()
        .map_or(0, |gpu| gpu.receiver_demand.in_flight());
    let gpu_overhead_bytes =
        page_table_bytes + render_staging_bytes + VSM_SAMPLING_PARAMS_BYTES + gpu_receiver_bytes;
    let page_depth_bytes = VSM_PHYSICAL_PAGE_SIZE as u64
        * VSM_PHYSICAL_PAGE_SIZE as u64
        * std::mem::size_of::<f32>() as u64;
    let (physical_capacity, physical_bytes, gpu_overhead_bytes, render_budget) = if vsm.enabled {
        (
            stats.capacity,
            vsm.cache.memory_bytes(),
            gpu_overhead_bytes,
            vsm.render_budget,
        )
    } else {
        (0, 0, 0, 0)
    };
    let (local_resident_pages, local_dirty_pages) = vsm
        .cache
        .physical
        .iter()
        .filter_map(|slot| slot.owner.filter(|owner| owner.light > 0).map(|_| slot))
        .fold((0u16, 0u16), |(resident, dirty), slot| {
            (
                resident.saturating_add(1),
                dirty.saturating_add(u16::from(slot.dirty)),
            )
        });
    let local_requested_pages: u32 = vsm
        .local_page_stats
        .iter()
        .map(|stats| stats.requested)
        .sum();
    let local_cache_hits: u32 = vsm.local_page_stats.iter().map(|stats| stats.hits).sum();
    let local_cache_misses: u32 = vsm.local_page_stats.iter().map(|stats| stats.misses).sum();
    let local_denied_pages: u32 = vsm.local_page_stats.iter().map(|stats| stats.denied).sum();
    let local_invalidated_pages: u32 = vsm
        .local_page_stats
        .iter()
        .map(|stats| stats.invalidated)
        .sum();
    let local_rendered_pages: u32 = vsm
        .local_page_stats
        .iter()
        .map(|stats| stats.rendered)
        .sum();
    let local_active_lights = vsm
        .local_selected
        .iter()
        .filter(|local| {
            (0..local.request.face_count()).all(|face| {
                let page = VirtualShadowPage::new_local(local.request.light_index, face)
                    .expect("selected local light has valid page addresses");
                vsm.cache.encoded_page(page) != VSM_PAGE_TABLE_MISSING
            })
        })
        .count();
    let local_point_submitted = vsm
        .local_requests
        .iter()
        .filter(|request| matches!(request.projection, LocalShadowProjection::PointCube))
        .count();
    let local_spot_submitted = vsm
        .local_requests
        .iter()
        .filter(|request| matches!(request.projection, LocalShadowProjection::Spot { .. }))
        .count();
    let directional_requested_pages = stats.requested.saturating_sub(local_requested_pages);
    let directional_cache_hits = stats.hits.saturating_sub(local_cache_hits);
    let directional_cache_misses = stats.misses.saturating_sub(local_cache_misses);
    let directional_denied_pages = stats.denied.saturating_sub(local_denied_pages);
    let directional_invalidated_pages = stats.invalidated.saturating_sub(local_invalidated_pages);
    let directional_rendered_pages = stats.rendered.saturating_sub(local_rendered_pages);
    let directional_resident_pages = stats.resident.saturating_sub(local_resident_pages);
    let directional_dirty_pages = stats.dirty.saturating_sub(local_dirty_pages);
    let mut local_cost_rows = String::new();
    for request in &vsm.local_requests {
        let point_index = request.light_index as usize;
        let cache_light = request.light_index + 1;
        let page_stats = vsm.local_page_stats[point_index];
        let (resident_pages, dirty_pages) =
            (0..request.face_count()).fold((0u16, 0u16), |(resident, dirty), face| {
                let Some(page) = VirtualShadowPage::new_local(request.light_index, face) else {
                    return (resident, dirty);
                };
                let Some((page_dirty, _)) = vsm.cache.request_state(page) else {
                    return (resident, dirty);
                };
                (
                    resident.saturating_add(1),
                    dirty.saturating_add(u16::from(page_dirty)),
                )
            });
        let selected = vsm
            .local_selected
            .iter()
            .any(|local| local.request.light_index == request.light_index);
        let active = selected
            && (0..request.face_count()).all(|face| {
                VirtualShadowPage::new_local(request.light_index, face)
                    .is_some_and(|page| vsm.cache.encoded_page(page) != VSM_PAGE_TABLE_MISSING)
            });
        let state = if active {
            "active"
        } else if selected {
            "admitted-pending"
        } else {
            "suppressed"
        };
        write!(
            local_cost_rows,
            concat!(
                ",{{\"light\":{},\"cache_light\":{},\"kind\":\"{}\",",
                "\"state\":\"{}\",\"requested_pages\":{},\"cache_hits\":{},",
                "\"cache_misses\":{},\"denied_pages\":{},\"invalidated_pages\":{},",
                "\"rendered_pages\":{},\"resident_pages\":{},\"dirty_pages\":{},",
                "\"physical_depth_bytes_owned\":{},\"shared_pool_bytes\":{},",
                "\"shared_metadata_staging_bytes\":{},\"render_budget_pages\":{}}}"
            ),
            point_index,
            cache_light,
            request.kind(),
            state,
            page_stats.requested,
            page_stats.hits,
            page_stats.misses,
            page_stats.denied,
            page_stats.invalidated,
            page_stats.rendered,
            resident_pages,
            dirty_pages,
            u64::from(resident_pages) * page_depth_bytes,
            physical_bytes,
            gpu_overhead_bytes,
            render_budget,
        )
        .expect("writing JSON to a String cannot fail");
    }
    format!(
        concat!(
            "{{\"requested\":{},\"capability_eligible\":{},\"enabled\":{},\"active\":{},",
            "\"selection_reason\":\"{}\",",
            "\"projection\":\"camera-centered-page-snapped-clipmap\",",
            "\"fallback\":\"csm\",\"dynamic_fallback\":{},",
            "\"dynamic_fallback_mode\":\"{}\",\"dynamic_fallback_pages\":{},",
            "\"dynamic_overlay_pages\":{},\"dynamic_overlay_rendered_pages\":{},",
            "\"dynamic_overlay_draws\":{},\"dynamic_overlay_deferred_pages\":{},",
            "\"page_cutout_draws\":{},\"page_skinned_draws\":{},",
            "\"dynamic_overlay_page_budget\":{},\"dynamic_overlay_draw_budget\":{},",
            "\"local_lights\":{{\"submission_limit\":{},\"admission_limit\":{},",
            "\"faces_per_light\":{},\"point_faces_per_light\":{},",
            "\"spot_faces_per_light\":1,\"point_submitted\":{},\"spot_submitted\":{},",
            "\"submitted\":{},\"visible\":{},\"admitted\":{},",
            "\"active_shaded\":{},\"visibility_rejected\":{},\"budget_suppressed\":{},",
            "\"requested_pages\":{},\"resident_pages\":{},\"dirty_pages\":{},",
            "\"rendered_pages\":{},\"shared_page_budget\":true,",
            "\"fallback\":\"suppress-direct-contribution\"}},",
            "\"physical_capacity\":{},\"physical_bytes\":{},",
            "\"gpu_overhead_bytes\":{},\"gpu_total_bytes\":{},",
            "\"resident\":{},\"dirty\":{},\"requested_pages\":{},",
            "\"cache_hits\":{},\"cache_misses\":{},\"evictions\":{},",
            "\"denied\":{},\"invalidated\":{},\"rendered\":{},",
            "\"clipmap_level_rebases\":{},\"clipmap_pages_preserved\":{},",
            "\"clipmap_pages_dropped\":{},",
            "\"pending_render\":{},\"render_budget\":{},",
            "\"demand_source\":\"{}\",\"demand_count\":{},",
            "\"receiver_bounds_count\":{},\"receiver_marking_backend\":\"{}\",",
            "\"gpu_receiver_min_bounds\":{},\"gpu_receiver_max_bounds\":{},",
            "\"gpu_receiver_enabled\":{},\"gpu_receiver_validated\":{},",
            "\"gpu_receiver_in_flight\":{},",
            "\"gpu_receiver_dispatches\":{},\"gpu_receiver_completions\":{},",
            "\"gpu_receiver_validation_failures\":{},\"gpu_receiver_bytes\":{},",
            "\"debug_views\":{{\"available\":{},\"capture_only\":true,",
            "\"virtual_pages\":\"virtual-shadow-pages.png\",",
            "\"physical_occupancy\":\"virtual-shadow-physical.png\",",
            "\"legend\":\"virtual-shadow-legend.png\",",
            "\"report\":\"virtual-shadow-report.json\",",
            "\"legend_order\":[\"free\",\"miss-unrendered\",\"invalidated\",",
            "\"clip-level-0\",\"clip-level-1\",\"clip-level-2\"],",
            "\"colors\":[\"#080808\",\"#ffb423\",\"#ff37be\",",
            "\"#46d26e\",\"#4696ff\",\"#be64ff\"]}},",
            "\"per_light_cost\":[{{\"light\":0,\"kind\":\"directional\",",
            "\"requested_pages\":{},\"cache_hits\":{},\"cache_misses\":{},",
            "\"denied_pages\":{},\"invalidated_pages\":{},\"rendered_pages\":{},",
            "\"resident_pages\":{},\"dirty_pages\":{},",
            "\"clipmap_level_rebases\":{},\"dynamic_overlay_draws\":{},",
            "\"physical_depth_bytes_owned\":{},\"shared_pool_bytes\":{},",
            "\"shared_metadata_staging_bytes\":{},\"render_budget_pages\":{}}}{}],",
            "\"levels\":[",
            "{{\"level\":0,\"resident\":{},\"dirty\":{}}},",
            "{{\"level\":1,\"resident\":{},\"dirty\":{}}},",
            "{{\"level\":2,\"resident\":{},\"dirty\":{}}}]}}"
        ),
        vsm.requested_by_user,
        vsm.capability_eligible,
        vsm.enabled,
        vsm.sampling_active,
        vsm.selection_reason,
        vsm.dynamic_global_fallback || vsm.dynamic_overlay_deferred_pages > 0,
        if vsm.dynamic_global_fallback {
            "whole-frame-csm"
        } else if vsm.dynamic_overlay_pages.is_empty() {
            "none"
        } else if vsm.dynamic_overlay_deferred_pages > 0 {
            "bounded-page-overlay-with-csm"
        } else {
            "page-overlay"
        },
        vsm.dynamic_overlay_deferred_pages,
        vsm.dynamic_overlay_pages.len(),
        vsm.dynamic_overlay_rendered_pages,
        vsm.dynamic_overlay_draws,
        vsm.dynamic_overlay_deferred_pages,
        vsm.page_cutout_draws,
        vsm.page_skinned_draws,
        VSM_DYNAMIC_OVERLAY_PAGE_BUDGET,
        VSM_DYNAMIC_OVERLAY_DRAW_BUDGET,
        VSM_MAX_LOCAL_SHADOW_REQUESTS,
        VSM_MAX_LOCAL_SHADOW_LIGHTS,
        VSM_LOCAL_FACES,
        VSM_LOCAL_FACES,
        local_point_submitted,
        local_spot_submitted,
        vsm.local_admission_stats.submitted,
        vsm.local_admission_stats.visible,
        vsm.local_admission_stats.admitted,
        local_active_lights,
        vsm.local_admission_stats.visibility_rejected,
        vsm.local_admission_stats.budget_suppressed,
        local_requested_pages,
        local_resident_pages,
        local_dirty_pages,
        local_rendered_pages,
        physical_capacity,
        physical_bytes,
        gpu_overhead_bytes,
        physical_bytes + gpu_overhead_bytes,
        stats.resident,
        stats.dirty,
        stats.requested,
        stats.hits,
        stats.misses,
        stats.evictions,
        stats.denied,
        stats.invalidated,
        stats.rendered,
        stats.clipmap_level_rebases,
        stats.clipmap_pages_preserved,
        stats.clipmap_pages_dropped,
        vsm.pending.len(),
        render_budget,
        if !vsm.enabled {
            "disabled"
        } else if vsm.receiver_demand_active {
            "receiver-bounds"
        } else {
            "bounded-center-fallback"
        },
        vsm.last_demand_count,
        vsm.receiver_bounds_count,
        vsm.receiver_marking_backend,
        gpu_receiver::GPU_RECEIVER_MIN_BOUNDS,
        gpu_receiver::GPU_RECEIVER_MAX_BOUNDS,
        gpu_receiver_enabled,
        gpu_receiver_validated,
        gpu_receiver_in_flight,
        gpu_receiver_stats.dispatches,
        gpu_receiver_stats.completions,
        gpu_receiver_stats.validation_failures,
        gpu_receiver_bytes,
        vsm.enabled,
        directional_requested_pages,
        directional_cache_hits,
        directional_cache_misses,
        directional_denied_pages,
        directional_invalidated_pages,
        directional_rendered_pages,
        directional_resident_pages,
        directional_dirty_pages,
        stats.clipmap_level_rebases,
        vsm.dynamic_overlay_draws,
        u64::from(directional_resident_pages) * page_depth_bytes,
        physical_bytes,
        gpu_overhead_bytes,
        render_budget,
        local_cost_rows,
        levels[0].0,
        levels[0].1,
        levels[1].0,
        levels[1].1,
        levels[2].0,
        levels[2].1,
    )
}
