use super::*;

fn transform_ndc(matrix: &[[f32; 4]; 4], point: [f32; 4]) -> [f32; 3] {
    let clip = crate::renderer::mat4_mul_vec4(matrix, &point);
    [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
}

fn page(x: u16) -> VirtualShadowPage {
    VirtualShadowPage::new(0, 0, x, 0).unwrap()
}

fn rgb_at(rgb: &[u8], width: u32, x: u32, y: u32) -> [u8; 3] {
    let offset = ((y * width + x) * 3) as usize;
    rgb[offset..offset + 3].try_into().unwrap()
}

#[test]
fn invalid_virtual_coordinates_are_rejected() {
    assert!(VirtualShadowPage::new(0, VSM_CLIP_LEVELS, 0, 0).is_none());
    assert!(VirtualShadowPage::new(0, 0, VSM_VIRTUAL_PAGES_PER_AXIS, 0).is_none());
}

#[test]
fn reuse_is_stable_and_signature_changes_dirty_the_page() {
    let mut cache = VirtualShadowPageCache::new(2);
    cache.begin_frame(1);
    let first = cache.request(page(0), 7).unwrap();
    assert!(first.needs_render);
    assert!(cache.mark_rendered(page(0), 7));
    let hit = cache.request(page(0), 7).unwrap();
    assert_eq!(first.physical_page, hit.physical_page);
    assert!(!hit.needs_render);
    assert!(cache.request(page(0), 8).unwrap().needs_render);
}

#[test]
fn current_frame_pages_are_never_evicted() {
    let mut cache = VirtualShadowPageCache::new(2);
    cache.begin_frame(1);
    cache.request(page(0), 1).unwrap();
    cache.request(page(1), 1).unwrap();
    assert!(cache.request(page(2), 1).is_none());
    assert_eq!(cache.stats().denied, 1);
}

#[test]
fn stable_request_accounting_skips_cache_walk_without_hiding_fallbacks() {
    let mut cache = VirtualShadowPageCache::new(2);
    cache.begin_frame(1);
    cache.request(page(0), 1).unwrap();
    cache.request(page(1), 1).unwrap();
    cache.finish_requests();
    cache.begin_frame(2);
    cache.record_stable_requests(3);
    assert_eq!(cache.stats().requested, 3);
    assert_eq!(cache.stats().hits, 2);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().denied, 1);
}

#[test]
fn debug_images_distinguish_misses_invalidations_levels_and_free_pages() {
    let mut cache = VirtualShadowPageCache::new(4);
    cache.begin_frame(1);
    for level in 0..VSM_CLIP_LEVELS {
        let page = VirtualShadowPage::new(0, level, 0, 0).unwrap();
        cache.request(page, 1).unwrap();
        if level > 0 {
            cache.mark_rendered(page, 1);
        }
    }
    cache.finish_requests();

    let (virtual_width, virtual_height, virtual_rgb) = cache.debug_virtual_rgb(0, 2);
    assert_eq!(virtual_width, u32::from(VSM_VIRTUAL_PAGES_PER_AXIS) * 2);
    assert_eq!(
        virtual_height,
        u32::from(VSM_VIRTUAL_PAGES_PER_AXIS) * u32::from(VSM_CLIP_LEVELS) * 2,
    );
    assert_eq!(
        virtual_rgb.len(),
        (virtual_width * virtual_height * 3) as usize
    );
    assert_eq!(
        rgb_at(&virtual_rgb, virtual_width, 0, 0),
        debug::MISS_UNRENDERED
    );
    assert_eq!(
        rgb_at(
            &virtual_rgb,
            virtual_width,
            0,
            u32::from(VSM_VIRTUAL_PAGES_PER_AXIS) * 2,
        ),
        debug::LEVELS[1],
    );
    assert_eq!(
        rgb_at(
            &virtual_rgb,
            virtual_width,
            0,
            u32::from(VSM_VIRTUAL_PAGES_PER_AXIS) * 4,
        ),
        debug::LEVELS[2],
    );

    let (physical_width, physical_height, physical_rgb) = cache.debug_physical_rgb(2);
    assert_eq!(
        physical_rgb.len(),
        (physical_width * physical_height * 3) as usize
    );
    assert_eq!(
        rgb_at(&physical_rgb, physical_width, 0, 0),
        debug::MISS_UNRENDERED
    );
    assert_eq!(
        rgb_at(&physical_rgb, physical_width, 2, 0),
        debug::LEVELS[1]
    );
    assert_eq!(
        rgb_at(&physical_rgb, physical_width, 4, 0),
        debug::LEVELS[2]
    );
    assert_eq!(rgb_at(&physical_rgb, physical_width, 6, 0), debug::FREE);

    cache.mark_rendered(page(0), 1);
    cache.begin_frame(2);
    assert!(cache.request(page(0), 2).unwrap().needs_render);
    let (virtual_width, _, virtual_rgb) = cache.debug_virtual_rgb(0, 1);
    assert_eq!(
        rgb_at(&virtual_rgb, virtual_width, 0, 0),
        debug::INVALIDATED
    );
    let (physical_width, _, physical_rgb) = cache.debug_physical_rgb(1);
    assert_eq!(
        rgb_at(&physical_rgb, physical_width, 0, 0),
        debug::INVALIDATED
    );
}

#[test]
fn debug_legend_is_a_stable_machine_readable_palette() {
    let (width, height, rgb) = VirtualShadowPageCache::debug_legend_rgb(2);
    assert_eq!((width, height), (12, 2));
    let expected = [
        debug::FREE,
        debug::MISS_UNRENDERED,
        debug::INVALIDATED,
        debug::LEVELS[0],
        debug::LEVELS[1],
        debug::LEVELS[2],
    ];
    for (index, color) in expected.into_iter().enumerate() {
        assert_eq!(rgb_at(&rgb, width, index as u32 * 2, 0), color);
    }
}

#[test]
fn lru_eviction_is_deterministic() {
    let mut cache = VirtualShadowPageCache::new(2);
    cache.begin_frame(1);
    cache.request(page(0), 1).unwrap();
    cache.request(page(1), 1).unwrap();
    cache.begin_frame(2);
    cache.request(page(1), 1).unwrap();
    let request = cache.request(page(2), 1).unwrap();
    assert_eq!(request.evicted, Some(page(0)));
    assert_eq!(request.physical_page, 0);
}

#[test]
fn clipmap_scroll_preserves_overlap_and_drops_only_the_boundary() {
    let mut cache = VirtualShadowPageCache::new(3);
    cache.begin_frame(1);
    for x in [0, 1, VSM_VIRTUAL_PAGES_PER_AXIS - 1] {
        let page = page(x);
        cache.request(page, 7).unwrap();
        cache.mark_rendered(page, 7);
    }
    cache.finish_requests();

    cache.begin_frame(2);
    cache.scroll_level(0, 0, [-1, 0]);
    let table = cache.page_table(0);
    assert_ne!(table[page(0).table_index()], VSM_PAGE_TABLE_MISSING);
    assert_ne!(
        table[page(VSM_VIRTUAL_PAGES_PER_AXIS - 2).table_index()],
        VSM_PAGE_TABLE_MISSING,
    );
    assert_eq!(
        table[page(VSM_VIRTUAL_PAGES_PER_AXIS - 1).table_index()],
        VSM_PAGE_TABLE_MISSING,
    );
    assert_eq!(cache.stats().resident, 2);
    assert_eq!(cache.stats().dirty, 0);
    assert_eq!(cache.stats().clipmap_level_rebases, 1);
    assert_eq!(cache.stats().clipmap_pages_preserved, 2);
    assert_eq!(cache.stats().clipmap_pages_dropped, 1);
}

#[test]
fn dirty_pages_are_missing_until_rendered_and_then_age() {
    let mut cache = VirtualShadowPageCache::new(1);
    cache.begin_frame(1);
    cache.request(page(0), 42).unwrap();
    assert_eq!(cache.page_table(0)[page(0).table_index()], 0);
    cache.mark_rendered(page(0), 42);
    let first = cache.page_table(0)[page(0).table_index()];
    assert_eq!(first & 0xffff, 1);
    assert_eq!(first >> 16, 1);
    cache.begin_frame(4);
    let aged = cache.page_table(0)[page(0).table_index()];
    assert_eq!(aged >> 16, 4);
    cache.begin_frame(100);
    let saturated = cache.page_table(0)[page(0).table_index()];
    assert_eq!(saturated >> 16, 8);
    cache.invalidate_light(0);
    assert_eq!(cache.page_table(0)[page(0).table_index()], 0);
}

#[test]
fn configured_capacity_is_a_hard_memory_bound() {
    let cache = VirtualShadowPageCache::new(8);
    let edge = VSM_PHYSICAL_PAGE_SIZE as u64;
    assert_eq!(cache.memory_bytes(), edge * edge * 4 * 8);
    assert_eq!(cache.stats().capacity, 8);
}

#[test]
fn centered_demand_fits_default_pool() {
    let demand = centered_directional_demand(0);
    assert_eq!(demand.len(), 224);
    assert!(demand.len() <= VSM_DEFAULT_PHYSICAL_PAGES as usize);
    let mut sorted = demand.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), demand.len());
    assert_eq!(
        demand[..3]
            .iter()
            .map(|page| page.level)
            .collect::<Vec<_>>(),
        vec![0, 1, 2],
    );
    assert!(demand[..3]
        .iter()
        .all(|page| (15..=16).contains(&page.x) && (15..=16).contains(&page.y)));
}

#[test]
fn receiver_demand_is_bounded_unique_and_fair_across_levels() {
    let demand = directional_receiver_demand(
        [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
        &[([-1.0; 3], [1.0; 3])],
        7,
    );
    assert_eq!(
        demand.len(),
        VSM_DIRECTIONAL_LEVEL_PAGE_CAPS
            .iter()
            .copied()
            .sum::<usize>()
    );
    let mut sorted = demand.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), demand.len());
    assert_eq!(
        demand[..3]
            .iter()
            .map(|page| page.level)
            .collect::<Vec<_>>(),
        vec![0, 1, 2],
    );
    assert!(demand.iter().all(|page| page.light == 7));
}

#[test]
fn receiver_demand_marks_local_footprint_with_guard_pages() {
    let demand = directional_receiver_demand(
        [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
        &[([-0.02, -0.02, 0.4], [0.02, 0.02, 0.6])],
        0,
    );
    assert_eq!(demand.iter().filter(|page| page.level == 0).count(), 16);
    assert!(demand
        .iter()
        .all(|page| { (14..=17).contains(&page.x) && (14..=17).contains(&page.y) }));
}

#[test]
fn receiver_demand_rejects_bounds_outside_light_volume() {
    let demand = directional_receiver_demand(
        [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
        &[([2.0, 2.0, 2.0], [3.0, 3.0, 3.0])],
        0,
    );
    assert!(demand.is_empty());
}

#[test]
fn dynamic_overlays_cover_only_guarded_caster_pages() {
    let pages = directional_dynamic_fallback_pages(
        [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
        &[([-0.02, -0.02, 0.4], [0.02, 0.02, 0.6])],
        0,
    );
    assert_eq!(pages.len(), 36 * VSM_CLIP_LEVELS as usize);
    assert!(pages
        .iter()
        .all(|page| { (13..=18).contains(&page.x) && (13..=18).contains(&page.y) }));
    assert!(pages[..4]
        .iter()
        .all(|page| page.level == 0 && (15..=16).contains(&page.x) && (15..=16).contains(&page.y)));

    let mut table = vec![
        99;
        VSM_VIRTUAL_PAGES_PER_AXIS as usize
            * VSM_VIRTUAL_PAGES_PER_AXIS as usize
            * VSM_CLIP_LEVELS as usize
    ];
    force_dynamic_overlay_age(&mut table, &pages);
    assert_eq!(
        table
            .iter()
            .filter(|entry| **entry == 99 | (8 << 16))
            .count(),
        pages.len()
    );
    assert_eq!(
        table.iter().filter(|entry| **entry == 99).count(),
        table.len() - pages.len()
    );
}

#[test]
fn targeted_invalidation_never_exposes_stale_dynamic_depth() {
    let mut cache = VirtualShadowPageCache::new(2);
    cache.begin_frame(1);
    for x in 0..2 {
        cache.request(page(x), 7).unwrap();
        cache.mark_rendered(page(x), 7);
    }
    cache.invalidate_pages(&[page(1)]);
    let table = cache.page_table(0);
    assert_ne!(table[page(0).table_index()], VSM_PAGE_TABLE_MISSING);
    assert_eq!(table[page(1).table_index()], VSM_PAGE_TABLE_MISSING);
    assert_eq!(cache.stats().dirty, 1);
}

#[test]
fn unbounded_dynamic_caster_preserves_whole_frame_fallback() {
    let pages = directional_dynamic_fallback_pages(
        [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
        &[([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0])],
        0,
    );
    assert_eq!(
        pages.len(),
        VSM_VIRTUAL_PAGES_PER_AXIS as usize
            * VSM_VIRTUAL_PAGES_PER_AXIS as usize
            * VSM_CLIP_LEVELS as usize
    );
}

#[test]
fn offscreen_dynamic_caster_does_not_mask_resident_pages() {
    let pages = directional_dynamic_fallback_pages(
        [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize],
        &[([2.0, 2.0, 2.0], [3.0, 3.0, 3.0])],
        0,
    );
    assert!(pages.is_empty());
}

#[test]
fn receiver_demand_and_signature_are_deterministic() {
    let bounds = [
        ([-0.8, -0.4, 0.2], [-0.2, 0.1, 0.8]),
        ([0.1, -0.2, 0.1], [0.7, 0.6, 0.9]),
    ];
    let vps = [crate::renderer::IDENTITY_MAT4; VSM_CLIP_LEVELS as usize];
    let first = directional_receiver_demand(vps, &bounds, 0);
    let second = directional_receiver_demand(vps, &bounds, 0);
    assert_eq!(first, second);
    assert_eq!(demand_signature(&first), demand_signature(&second));
    assert_ne!(
        demand_signature(&first),
        demand_signature(&centered_directional_demand(0))
    );
}

#[test]
fn coordinator_is_inert_without_explicit_request() {
    if virtual_shadows_requested() {
        return;
    }
    // Runtime construction requires a device; the non-GPU cache already
    // proves inert behavior above. Keep the environment contract here.
    let cache = VirtualShadowPageCache::new(1);
    assert_eq!(cache.stats().resident, 0);
    assert_eq!(
        cache
            .page_table(0)
            .iter()
            .filter(|entry| **entry != 0)
            .count(),
        0
    );
}

#[test]
fn page_crop_maps_interior_edges_to_guard_texels() {
    let page = VirtualShadowPage::new(0, 0, 9, 12).unwrap();
    let crop = directional_page_vp(crate::renderer::IDENTITY_MAT4, page);
    let axis = VSM_VIRTUAL_PAGES_PER_AXIS as f32;
    let left = f32::from(page.x) * (2.0 / axis) - 1.0;
    let right = f32::from(page.x + 1) * (2.0 / axis) - 1.0;
    let top = 1.0 - f32::from(page.y) * (2.0 / axis);
    let bottom = 1.0 - f32::from(page.y + 1) * (2.0 / axis);
    let expected = VSM_PAGE_INTERIOR as f32 / VSM_PHYSICAL_PAGE_SIZE as f32;

    let left_ndc = transform_ndc(&crop, [left, 0.5 * (top + bottom), 0.5, 1.0]);
    let right_ndc = transform_ndc(&crop, [right, 0.5 * (top + bottom), 0.5, 1.0]);
    let top_ndc = transform_ndc(&crop, [0.5 * (left + right), top, 0.5, 1.0]);
    let bottom_ndc = transform_ndc(&crop, [0.5 * (left + right), bottom, 0.5, 1.0]);

    assert!((left_ndc[0] + expected).abs() < 1.0e-5);
    assert!((right_ndc[0] - expected).abs() < 1.0e-5);
    assert!((top_ndc[1] - expected).abs() < 1.0e-5);
    assert!((bottom_ndc[1] + expected).abs() < 1.0e-5);
}

#[test]
fn page_crop_preserves_depth() {
    let page = VirtualShadowPage::new(0, 2, 16, 16).unwrap();
    let crop = directional_page_vp(crate::renderer::IDENTITY_MAT4, page);
    let transformed = transform_ndc(&crop, [0.03125, -0.03125, 0.37, 1.0]);
    assert!((transformed[2] - 0.37).abs() < 1.0e-6);
}

#[test]
fn scene_shader_variant_injects_bindings_and_both_cascade_samples() {
    let source = r#"
let shadow_val = sample_cascade(cascade, shadow_uv, depth_ref);
fn sample_shadow(world_pos: vec3<f32>, geo_n: vec3<f32>) -> f32 {
let next_val = sample_cascade(next_cascade, next_uv, next_depth_ref);
}
"#;
    let variant = directional_scene_shader(source);
    assert!(variant.contains("@binding(13) var vsm_page_table"));
    assert!(variant.contains("@binding(14) var vsm_physical_pages"));
    assert!(variant.contains("@binding(15) var<uniform> vsm_params"));
    assert_eq!(variant.matches("sample_virtual_shadow(").count(), 3);
    assert!(variant.contains("sample_virtual_shadow(cascade, recv_pos,"));
    assert!(variant.contains("sample_virtual_shadow(next_cascade, recv_pos,"));
    assert!(variant.contains("(vsm_params.words.x & 2u) == 0u"));
    assert!(variant.contains("level_vps: array<mat4x4<f32>, 3>"));
    assert!(!source.contains("vsm_page_table"));
}

#[test]
fn material_shader_variant_wraps_the_canonical_cascade_sampler() {
    let source = r#"
fn sample_shadow_cascade(
  cascade_idx: u32, world_pos: vec3<f32>, outside_value: f32,
) -> f32 {
  return 1.0;
}
fn sample_sun_shadow(world_pos: vec3<f32>) -> f32 {
  return sample_shadow_cascade(0u, world_pos, -1.0);
}
"#;
    let variant = directional_material_shader(source.to_owned());
    assert!(variant.contains("@binding(10) var vsm_page_table"));
    assert!(variant.contains("fn sample_shadow_cascade_csm("));
    assert_eq!(variant.matches("fn sample_shadow_cascade(").count(), 1);
    assert!(variant.contains("sample_shadow_cascade_csm(cascade_idx, world_pos, outside_value)"));
    assert!(variant.contains("level_vps: array<mat4x4<f32>, 3>"));
}

#[test]
fn sampling_uniform_matches_wgsl_layout() {
    assert_eq!(
        VSM_SAMPLING_PARAMS_BYTES,
        3 * 64
            + 16
            + VSM_MAX_LOCAL_SHADOW_REQUESTS as u64 * 16
            + VSM_MAX_LOCAL_SHADOW_LIGHTS as u64 * (VSM_LOCAL_FACES as u64 * 64 + 32)
    );
    assert_eq!(std::mem::align_of::<DirectionalVsmSamplingParams>(), 16);
}

#[test]
fn immediate_shader_variant_injects_local_shadow_sampling() {
    let source = r#"
struct PointLight { position: vec4<f32>, color: vec4<f32> };
struct Lighting { point_lights: array<PointLight, 256> };
struct VertexOutput3D { world_pos: vec3<f32> };
@group(1) @binding(0) var<uniform> lighting: Lighting;
fn local_lighting(in: VertexOutput3D, i: u32, diff: f32, atten2: f32) -> vec3<f32> {
    let pl = lighting.point_lights[i];
    return pl.color.rgb * pl.color.w * diff * atten2;
}
"#;
    let variant = local_immediate_shader(source);
    assert!(variant.contains("@binding(14) var vsm_physical_pages"));
    assert!(variant.contains("@binding(8) var shadow_samp"));
    assert!(variant.contains("diff * atten2 * sample_local_shadow(i, in.world_pos)"));
    let result = wgpu::naga::front::wgsl::parse_str(&variant);
    if let Err(error) = result.as_ref() {
        panic!(
            "VSM immediate variant failed WGSL parsing:\n{}",
            error.emit_to_string(&variant),
        );
    }
}

#[test]
fn material_shadow_variant_parses_through_naga() {
    let source = format!(
        "{}\n{}",
        include_str!("../shaders/material_abi.wgsl"),
        include_str!("../shaders/common/shadows.wgsl"),
    );
    let variant = directional_material_shader(source);
    let result = wgpu::naga::front::wgsl::parse_str(&variant);
    if let Err(error) = result.as_ref() {
        panic!(
            "VSM material shadow variant failed WGSL parsing:\n{}",
            error.emit_to_string(&variant),
        );
    }
}
