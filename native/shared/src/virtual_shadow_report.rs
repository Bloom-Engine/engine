use super::*;

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
    let gpu_overhead_bytes = page_table_bytes + render_staging_bytes + VSM_SAMPLING_PARAMS_BYTES;
    let (physical_capacity, physical_bytes, gpu_overhead_bytes, render_budget) = if vsm.requested {
        (
            stats.capacity,
            vsm.cache.memory_bytes(),
            gpu_overhead_bytes,
            vsm.render_budget,
        )
    } else {
        (0, 0, 0, 0)
    };
    format!(
        concat!(
            "{{\"requested\":{},\"active\":{},",
            "\"projection\":\"camera-centered-page-snapped-clipmap\",",
            "\"fallback\":\"csm\",\"dynamic_fallback\":{},",
            "\"dynamic_fallback_mode\":\"{}\",\"dynamic_fallback_pages\":{},",
            "\"dynamic_overlay_pages\":{},\"dynamic_overlay_rendered_pages\":{},",
            "\"dynamic_overlay_draws\":{},\"dynamic_overlay_deferred_pages\":{},",
            "\"dynamic_overlay_page_budget\":{},\"dynamic_overlay_draw_budget\":{},",
            "\"physical_capacity\":{},\"physical_bytes\":{},",
            "\"gpu_overhead_bytes\":{},\"gpu_total_bytes\":{},",
            "\"resident\":{},\"dirty\":{},\"requested_pages\":{},",
            "\"cache_hits\":{},\"cache_misses\":{},\"evictions\":{},",
            "\"denied\":{},\"invalidated\":{},\"rendered\":{},",
            "\"clipmap_level_rebases\":{},\"clipmap_pages_preserved\":{},",
            "\"clipmap_pages_dropped\":{},",
            "\"pending_render\":{},\"render_budget\":{},",
            "\"demand_source\":\"{}\",\"demand_count\":{},",
            "\"levels\":[",
            "{{\"level\":0,\"resident\":{},\"dirty\":{}}},",
            "{{\"level\":1,\"resident\":{},\"dirty\":{}}},",
            "{{\"level\":2,\"resident\":{},\"dirty\":{}}}]}}"
        ),
        vsm.requested,
        vsm.sampling_active,
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
        VSM_DYNAMIC_OVERLAY_PAGE_BUDGET,
        VSM_DYNAMIC_OVERLAY_DRAW_BUDGET,
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
        if !vsm.requested {
            "disabled"
        } else if vsm.receiver_demand_active {
            "receiver-bounds"
        } else {
            "bounded-center-fallback"
        },
        vsm.last_demand_count,
        levels[0].0,
        levels[0].1,
        levels[1].0,
        levels[1].1,
        levels[2].0,
        levels[2].1,
    )
}
