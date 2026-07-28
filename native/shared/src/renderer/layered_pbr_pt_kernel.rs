//! Exact source specialization for the layered path-tracing kernel.

fn replace_once(source: &mut String, needle: &str, replacement: &str) {
    let count = source.matches(needle).count();
    assert_eq!(
        count, 1,
        "layered PT specialization expected one source anchor, found {count}: {needle}"
    );
    *source = source.replacen(needle, replacement, 1);
}

pub(super) fn layered_kernel_variant(base: &str) -> String {
    let mut source = base.to_owned();
    replace_once(
        &mut source,
        "    var rough_cur = mr0.g;",
        "    var rough_cur = mr0.g;\n\
         \x20   let layered_primary = pt_layered_primary_surface(p0, n0);\n\
         \x20   var layered_cur = layered_primary.material;\n\
         \x20   var layered_tangent_cur = layered_primary.tangent;\n\
         \x20   var layered_clearcoat_normal_cur = layered_primary.clearcoat_normal;",
    );
    replace_once(
        &mut source,
        "    let use_restir = u.ext.w == 1u && u.cfg.x >= 2.0;",
        "    let use_restir = u.ext.w == 1u && u.cfg.x >= 2.0\n        \
         && !pt_layered_has_transport(layered_cur);",
    );
    replace_once(
        &mut source,
        "    var radiance = direct_light(\n\
         \x20       p0 + n0 * 0.02, n0, sun_r2, view_cur,\n\
         \x20       albedo0, rough_cur, metal_cur, !use_restir,\n\
         \x20   );",
        "    var radiance = pt_layered_direct_light(\n\
         \x20       p0 + n0 * 0.02, n0, layered_clearcoat_normal_cur,\n\
         \x20       layered_tangent_cur, sun_r2, view_cur,\n\
         \x20       albedo0, rough_cur, metal_cur, !use_restir, layered_cur,\n\
         \x20   );",
    );
    replace_once(
        &mut source,
        "        let s = sample_brdf(n_cur, view_cur, alb_cur, rough_cur, metal_cur);",
        "        let s = pt_sample_layered_brdf(\n\
         \x20           n_cur, layered_clearcoat_normal_cur,\n\
         \x20           layered_tangent_cur, view_cur,\n\
         \x20           alb_cur, rough_cur, metal_cur, layered_cur,\n\
         \x20       );",
    );
    replace_once(
        &mut source,
        "        radiance += throughput * direct_light(\n\
         \x20           hit_p, n_hit, rand_2f(), -dir,\n\
         \x20           alb_hit, inst.mat_params.x, inst.mat_params.y, true,\n\
         \x20       );",
        "        var layered_hit = pt_layered_materials[hit.instance_custom_data];\n\
         \x20       var layered_primary_uv = vec2<f32>(0.0);\n\
         \x20       var layered_secondary_uv = vec2<f32>(0.0);\n\
         \x20       if ((\n\
         \x20           PT_HAS_LAYERED_TEXTURES\n\
         \x20               || PT_HAS_CLEARCOAT_TEXTURES\n\
         \x20               || PT_HAS_CLEARCOAT_NORMALS\n\
         \x20               || PT_HAS_SHEEN_TEXTURES\n\
         \x20               || PT_HAS_IRIDESCENCE_TEXTURES\n\
         \x20               || PT_HAS_ANISOTROPY_TEXTURES\n\
         \x20       ) && inst.geo.z > 0u) {\n\
         \x20           let layered_attributes = fetch_hit_attrs(\n\
         \x20               inst.geo, hit.primitive_index, hit.barycentrics,\n\
         \x20           );\n\
         \x20           layered_primary_uv = layered_attributes.uv;\n\
         \x20           layered_secondary_uv = pt_layered_hit_uv1(\n\
         \x20               inst.geo, hit.primitive_index, hit.barycentrics,\n\
         \x20           );\n\
         \x20           layered_hit = pt_layered_apply_textures(\n\
         \x20               layered_hit, hit.instance_custom_data,\n\
         \x20               layered_primary_uv, layered_secondary_uv,\n\
         \x20           );\n\
         \x20           layered_hit = pt_layered_apply_clearcoat_textures(\n\
         \x20               layered_hit, hit.instance_custom_data,\n\
         \x20               layered_primary_uv, layered_secondary_uv,\n\
         \x20           );\n\
         \x20           layered_hit = pt_layered_apply_sheen_textures(\n\
         \x20               layered_hit, hit.instance_custom_data,\n\
         \x20               layered_primary_uv, layered_secondary_uv,\n\
         \x20           );\n\
         \x20           layered_hit = pt_layered_apply_anisotropy_texture(\n\
         \x20               layered_hit, hit.instance_custom_data,\n\
         \x20               layered_primary_uv, layered_secondary_uv,\n\
         \x20           );\n\
         \x20           layered_hit = pt_layered_apply_iridescence_textures(\n\
         \x20               layered_hit, hit.instance_custom_data,\n\
         \x20               layered_primary_uv, layered_secondary_uv,\n\
         \x20           );\n\
         \x20       }\n\
         \x20       var layered_tangent_hit = vec4<f32>(0.0);\n\
         \x20       if (\n\
         \x20           pt_layered_has_anisotropy(layered_hit)\n\
         \x20               || (\n\
         \x20                   PT_HAS_CLEARCOAT_NORMALS\n\
         \x20                       && pt_layered_has_clearcoat_normal(\n\
         \x20                           hit.instance_custom_data,\n\
         \x20                       )\n\
         \x20               )\n\
         \x20       ) {\n\
         \x20           layered_tangent_hit = pt_layered_hit_tangent(\n\
         \x20               inst.geo, hit.primitive_index, hit.barycentrics,\n\
         \x20               hit.object_to_world, n_hit,\n\
         \x20           );\n\
         \x20       }\n\
         \x20       let layered_coat_sample = pt_layered_apply_clearcoat_normal(\n\
         \x20           layered_hit, hit.instance_custom_data,\n\
         \x20           layered_primary_uv, layered_secondary_uv,\n\
         \x20           n_hit, layered_tangent_hit,\n\
         \x20       );\n\
         \x20       layered_hit = layered_coat_sample.material;\n\
         \x20       let layered_clearcoat_normal_hit = layered_coat_sample.normal;\n\
         \x20       radiance += throughput * pt_layered_direct_light(\n\
         \x20           hit_p, n_hit, layered_clearcoat_normal_hit,\n\
         \x20           layered_tangent_hit, rand_2f(), -dir,\n\
         \x20           alb_hit, inst.mat_params.x, inst.mat_params.y, true, layered_hit,\n\
         \x20       );",
    );
    replace_once(
        &mut source,
        "        metal_cur = inst.mat_params.y;\n        view_cur = -dir;",
        "        metal_cur = inst.mat_params.y;\n\
         \x20       layered_cur = layered_hit;\n\
         \x20       layered_tangent_cur = layered_tangent_hit;\n\
         \x20       layered_clearcoat_normal_cur = layered_clearcoat_normal_hit;\n\
         \x20       view_cur = -dir;",
    );
    source
}
