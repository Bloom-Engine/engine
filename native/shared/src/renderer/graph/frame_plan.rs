//! Typed declaration of Bloom's current serial renderer topology.
//!
//! The execution order deliberately matches the pre-#129 scheduler exactly.
//! Conservative aliasing is enabled only for exact-compatible, non-overlapping
//! transients; persistent and temporal renderer imports remain isolated.

use super::{
    AliasClass, BufferDesc, BufferUsage, Extent, FramePlanKey, GraphBuilder, Ownership,
    ResourceOrigin, TextureDesc, TextureUsage, Usage, FRAME_FEATURE_CAPTURE_OUTPUT,
    FRAME_FEATURE_CAPTURE_QUALITY, FRAME_FEATURE_IMPORTED_REFRACTION,
    FRAME_FEATURE_SCENE_SNAPSHOTS, FRAME_FEATURE_TEMPORAL_REACTIVE,
    FRAME_FEATURE_TRANSMITTED_SHADOWS, FRAME_FEATURE_WEIGHTED_TRANSPARENCY,
};

/// Qualification resources accepted by the internal debug-capture path.
/// These are graph names rather than renderer field names so requests remain
/// stable when a physical texture is replaced or aliased.
pub const QUALITY_CAPTURE_RESOURCE_NAMES: [&str; 5] = [
    "hdr-scene",
    "scene-depth",
    "shadow-cascade-0",
    "shadow-cascade-1",
    "shadow-cascade-2",
];

fn persistent_texture(initial_usage: TextureUsage, final_usage: TextureUsage) -> ResourceOrigin {
    ResourceOrigin::Persistent {
        initial_usage: Usage::Texture(initial_usage),
        final_usage: Usage::Texture(final_usage),
        ownership: Ownership::Graph,
    }
}

fn texture_desc(
    format: wgpu::TextureFormat,
    extent: Extent,
    usage: TextureUsage,
    alias_class: AliasClass,
) -> TextureDesc {
    let mut desc = TextureDesc::color(format, extent, usage);
    desc.alias_class = alias_class;
    desc
}

/// Build the renderer topology for one [`FramePlanKey`](super::FramePlanKey).
///
/// Feature methods still gate their own command recording in the parity stage,
/// so all historical pass names remain present. `FramePlanKey` nevertheless
/// prevents one configuration from accidentally reusing a topology after a
/// future migration starts selecting optional nodes at compile time.
pub fn build_renderer_frame_plan(
    key: FramePlanKey,
    output_format: wgpu::TextureFormat,
) -> GraphBuilder {
    let render_full = Extent::RenderRelative {
        numerator: 1,
        denominator: 1,
        layers: 1,
    };
    let render_half = Extent::RenderRelative {
        numerator: 1,
        denominator: 2,
        layers: 1,
    };
    let output_full = Extent::OutputRelative {
        numerator: 1,
        denominator: 1,
        layers: 1,
    };
    let sampled_color = TextureUsage::SAMPLED
        .union(TextureUsage::COLOR_ATTACHMENT)
        .union(TextureUsage::COPY_SRC)
        .union(TextureUsage::COPY_DST)
        .union(TextureUsage::STORAGE_READ)
        .union(TextureUsage::STORAGE_WRITE);
    let sampled_depth = TextureUsage::SAMPLED
        .union(TextureUsage::DEPTH_ATTACHMENT_READ)
        .union(TextureUsage::DEPTH_ATTACHMENT_WRITE)
        .union(TextureUsage::COPY_SRC)
        .union(TextureUsage::COPY_DST);

    let mut graph = GraphBuilder::new("bloom-frame");
    let froxel = graph.import_buffer(
        "froxel-clusters",
        BufferDesc {
            // Allocation is persistent and external to this plan; size here is
            // descriptive, not used for a physical allocation.
            size: 1,
            allowed_usage: BufferUsage::STORAGE_READ.union(BufferUsage::STORAGE_WRITE),
            alias_class: AliasClass::Never,
        },
        ResourceOrigin::Persistent {
            initial_usage: Usage::Buffer(BufferUsage::STORAGE_READ),
            final_usage: Usage::Buffer(BufferUsage::STORAGE_READ),
            ownership: Ownership::Graph,
        },
    );
    let shadow_desc = texture_desc(
        wgpu::TextureFormat::Depth32Float,
        Extent::Fixed {
            width: 2048,
            height: 2048,
            layers: 1,
        },
        sampled_depth,
        AliasClass::Never,
    );
    let mut shadows = [
        graph.import_texture(
            "shadow-cascade-0",
            shadow_desc.clone(),
            persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
        ),
        graph.import_texture(
            "shadow-cascade-1",
            shadow_desc.clone(),
            persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
        ),
        graph.import_texture(
            "shadow-cascade-2",
            shadow_desc,
            persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
        ),
    ];
    let mut transmitted_shadows = if key.feature_mask & FRAME_FEATURE_TRANSMITTED_SHADOWS != 0 {
        let color_desc = texture_desc(
            super::super::transmitted_shadows::TRANSMITTED_SHADOW_COLOR_FORMAT,
            Extent::Fixed {
                width: super::super::transmitted_shadows::TRANSMITTED_SHADOW_MAP_SIZE,
                height: super::super::transmitted_shadows::TRANSMITTED_SHADOW_MAP_SIZE,
                layers: 1,
            },
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED),
            AliasClass::Never,
        );
        let depth_desc = texture_desc(
            super::super::transmitted_shadows::TRANSMITTED_SHADOW_DEPTH_FORMAT,
            Extent::Fixed {
                width: super::super::transmitted_shadows::TRANSMITTED_SHADOW_MAP_SIZE,
                height: super::super::transmitted_shadows::TRANSMITTED_SHADOW_MAP_SIZE,
                layers: 1,
            },
            TextureUsage::DEPTH_ATTACHMENT_WRITE.union(TextureUsage::SAMPLED),
            AliasClass::Never,
        );
        Some((
            [
                graph.import_texture(
                    "transmitted-shadow-color-0",
                    color_desc.clone(),
                    persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
                ),
                graph.import_texture(
                    "transmitted-shadow-color-1",
                    color_desc.clone(),
                    persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
                ),
                graph.import_texture(
                    "transmitted-shadow-color-2",
                    color_desc,
                    persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
                ),
            ],
            [
                graph.import_texture(
                    "transmitted-shadow-depth-0",
                    depth_desc.clone(),
                    persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
                ),
                graph.import_texture(
                    "transmitted-shadow-depth-1",
                    depth_desc.clone(),
                    persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
                ),
                graph.import_texture(
                    "transmitted-shadow-depth-2",
                    depth_desc,
                    persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
                ),
            ],
        ))
    } else {
        None
    };
    let mut hdr = graph.import_texture(
        "hdr-scene",
        texture_desc(
            wgpu::TextureFormat::Rgba16Float,
            render_full,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut depth = graph.import_texture(
        "scene-depth",
        texture_desc(
            wgpu::TextureFormat::Depth32Float,
            render_full,
            sampled_depth,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut material = graph.import_texture(
        "material-properties",
        texture_desc(
            wgpu::TextureFormat::Rg8Unorm,
            render_full,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut velocity = graph.import_texture(
        "motion-vectors",
        texture_desc(
            wgpu::TextureFormat::Rg16Float,
            render_full,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut albedo = graph.import_texture(
        "albedo",
        texture_desc(
            wgpu::TextureFormat::Rgba8Unorm,
            render_full,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut hiz = graph.import_texture(
        "hiz-pyramid",
        texture_desc(
            wgpu::TextureFormat::R32Float,
            render_half,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut ssao_raw = graph.import_texture(
        "ssao-raw",
        texture_desc(
            wgpu::TextureFormat::R8Unorm,
            render_half,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut ssao_blur = graph.import_texture(
        "ssao-filtered",
        texture_desc(
            wgpu::TextureFormat::R8Unorm,
            render_half,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut ssr = graph.import_texture(
        "ssr",
        texture_desc(
            wgpu::TextureFormat::Rgba16Float,
            render_half,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut ssgi = graph.import_texture(
        "ssgi",
        texture_desc(
            wgpu::TextureFormat::Rgba16Float,
            render_half,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut bloom = graph.import_texture(
        "bloom-chain",
        texture_desc(
            wgpu::TextureFormat::Rgba16Float,
            render_half,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut composed = graph.import_texture(
        "composed-hdr",
        texture_desc(
            wgpu::TextureFormat::Rgba16Float,
            render_full,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut postfx = graph.import_texture(
        "postfx-hdr",
        texture_desc(
            wgpu::TextureFormat::Rgba16Float,
            output_full,
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let mut exposure = graph.import_texture(
        "exposure-history",
        texture_desc(
            wgpu::TextureFormat::Rg16Float,
            Extent::Fixed {
                width: 1,
                height: 1,
                layers: 1,
            },
            sampled_color,
            AliasClass::Never,
        ),
        persistent_texture(TextureUsage::SAMPLED, TextureUsage::SAMPLED),
    );
    let scene_snapshots = if key.feature_mask & FRAME_FEATURE_SCENE_SNAPSHOTS != 0 {
        Some((
            graph.create_texture(
                "translucent-scene-color",
                texture_desc(
                    wgpu::TextureFormat::Rgba16Float,
                    render_full,
                    TextureUsage::COPY_DST.union(TextureUsage::SAMPLED),
                    AliasClass::Color,
                ),
            ),
            graph.create_texture(
                "translucent-scene-depth",
                texture_desc(
                    wgpu::TextureFormat::Depth32Float,
                    render_full,
                    TextureUsage::COPY_DST.union(TextureUsage::SAMPLED),
                    AliasClass::Depth,
                ),
            ),
        ))
    } else {
        None
    };
    let weighted_transparency = if key.feature_mask & FRAME_FEATURE_WEIGHTED_TRANSPARENCY != 0 {
        Some((
            graph.create_texture(
                "transparency-accumulation",
                texture_desc(
                    wgpu::TextureFormat::Rgba16Float,
                    render_full,
                    TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED),
                    AliasClass::Color,
                ),
            ),
            graph.create_texture(
                "transparency-revealage",
                texture_desc(
                    wgpu::TextureFormat::R16Float,
                    render_full,
                    TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED),
                    AliasClass::Color,
                ),
            ),
        ))
    } else {
        None
    };
    let temporal_reactive = if key.feature_mask & FRAME_FEATURE_TEMPORAL_REACTIVE != 0 {
        Some(graph.create_texture(
            "transparency-reactive",
            texture_desc(
                wgpu::TextureFormat::R8Unorm,
                render_full,
                TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED),
                AliasClass::Color,
            ),
        ))
    } else {
        None
    };
    let mut output = graph.import_texture(
        "output",
        texture_desc(
            output_format,
            output_full,
            TextureUsage::COLOR_ATTACHMENT
                .union(TextureUsage::COPY_SRC)
                .union(TextureUsage::PRESENT),
            AliasClass::Never,
        ),
        ResourceOrigin::External {
            initial_usage: Usage::Texture(TextureUsage::COLOR_ATTACHMENT),
            final_usage: Usage::Texture(TextureUsage::PRESENT),
            ownership: Ownership::External,
        },
    );

    let mut previous_gi = None;
    if key.feature_mask & super::FRAME_FEATURE_SSGI != 0 {
        for name in [
            "accel_rebuild",
            "card_capture",
            "sdf_bake",
            "scene_sdf_clipmap",
            "wsrc_bake",
            "card_light",
        ] {
            let pass = graph.add_pass(name);
            if let Some(previous) = previous_gi {
                graph.after(pass, previous);
            }
            graph.set_side_effects(pass, super::SideEffects::EXTERNAL_STATE);
            previous_gi = Some(pass);
        }
    }

    let froxel_assign = graph.add_pass("froxel_assign");
    if let Some(previous) = previous_gi {
        graph.after(froxel_assign, previous);
    }
    let froxel = graph.write_buffer(froxel_assign, froxel, BufferUsage::STORAGE_WRITE);

    let shadow = graph.add_pass("shadow");
    graph.after(shadow, froxel_assign);
    for cascade in &mut shadows {
        *cascade = graph.write_texture(shadow, *cascade, TextureUsage::DEPTH_ATTACHMENT_WRITE);
    }
    if let Some((colors, depths)) = transmitted_shadows.as_mut() {
        for color in colors {
            *color = graph.write_texture(shadow, *color, TextureUsage::COLOR_ATTACHMENT);
        }
        for depth in depths {
            *depth = graph.write_texture(shadow, *depth, TextureUsage::DEPTH_ATTACHMENT_WRITE);
        }
    }

    let hdr_scene = graph.add_pass("hdr_scene");
    graph.after(hdr_scene, shadow);
    for cascade in shadows {
        graph.read_texture(hdr_scene, cascade, TextureUsage::SAMPLED);
    }
    graph.read_buffer(hdr_scene, froxel, BufferUsage::STORAGE_READ);
    hdr = graph.write_texture(hdr_scene, hdr, TextureUsage::COLOR_ATTACHMENT);
    depth = graph.write_texture(hdr_scene, depth, TextureUsage::DEPTH_ATTACHMENT_WRITE);
    material = graph.write_texture(hdr_scene, material, TextureUsage::COLOR_ATTACHMENT);
    velocity = graph.write_texture(hdr_scene, velocity, TextureUsage::COLOR_ATTACHMENT);
    albedo = graph.write_texture(hdr_scene, albedo, TextureUsage::COLOR_ATTACHMENT);

    let pt = graph.add_pass("pt");
    graph.after(pt, hdr_scene);
    graph.read_texture(pt, depth, TextureUsage::SAMPLED);
    hdr = graph.read_write_texture(
        pt,
        hdr,
        TextureUsage::STORAGE_READ,
        TextureUsage::STORAGE_WRITE,
    );
    let mut after_opaque = pt;
    if let Some((colors, depths)) = transmitted_shadows {
        let resolve = graph.add_pass("transmitted_shadow_resolve");
        graph.after(resolve, pt);
        graph.read_texture(resolve, depth, TextureUsage::SAMPLED);
        graph.read_texture(resolve, albedo, TextureUsage::SAMPLED);
        graph.read_texture(resolve, material, TextureUsage::SAMPLED);
        for color in colors {
            graph.read_texture(resolve, color, TextureUsage::SAMPLED);
        }
        for trans_depth in depths {
            graph.read_texture(resolve, trans_depth, TextureUsage::SAMPLED);
        }
        hdr = graph.read_write_texture(
            resolve,
            hdr,
            TextureUsage::COLOR_ATTACHMENT,
            TextureUsage::COLOR_ATTACHMENT,
        );
        after_opaque = resolve;
    }
    let translucent = graph.add_pass("translucent");
    graph.after(translucent, after_opaque);
    graph.read_texture(translucent, depth, TextureUsage::DEPTH_ATTACHMENT_READ);
    if let Some((scene_color, scene_depth)) = scene_snapshots {
        let scene_color = graph.write_texture(translucent, scene_color, TextureUsage::COPY_DST);
        graph.read_texture(translucent, scene_color, TextureUsage::SAMPLED);
        let scene_depth = graph.write_texture(translucent, scene_depth, TextureUsage::COPY_DST);
        graph.read_texture(translucent, scene_depth, TextureUsage::SAMPLED);
    }
    if let Some((accumulation, revealage)) = weighted_transparency {
        let accumulation =
            graph.write_texture(translucent, accumulation, TextureUsage::COLOR_ATTACHMENT);
        graph.read_texture(translucent, accumulation, TextureUsage::SAMPLED);
        let revealage = graph.write_texture(translucent, revealage, TextureUsage::COLOR_ATTACHMENT);
        graph.read_texture(translucent, revealage, TextureUsage::SAMPLED);
    }
    let temporal_reactive = temporal_reactive
        .map(|reactive| graph.write_texture(translucent, reactive, TextureUsage::COLOR_ATTACHMENT));
    hdr = graph.read_write_texture(
        translucent,
        hdr,
        TextureUsage::SAMPLED,
        TextureUsage::COLOR_ATTACHMENT,
    );
    if key.feature_mask & FRAME_FEATURE_IMPORTED_REFRACTION != 0 {
        velocity = graph.read_write_texture(
            translucent,
            velocity,
            TextureUsage::COLOR_ATTACHMENT,
            TextureUsage::COLOR_ATTACHMENT,
        );
    }

    let mut previous = translucent;
    if key.feature_mask & super::FRAME_FEATURE_SSAO != 0 {
        let hiz_build = graph.add_pass("hiz_build");
        graph.after(hiz_build, previous);
        graph.read_texture(hiz_build, depth, TextureUsage::SAMPLED);
        hiz = graph.write_texture(hiz_build, hiz, TextureUsage::STORAGE_WRITE);

        let occlusion_capture = graph.add_pass("occlusion_capture");
        graph.after(occlusion_capture, hiz_build);
        graph.read_texture(occlusion_capture, hiz, TextureUsage::SAMPLED);

        let gtao = graph.add_pass("gtao");
        graph.after(gtao, occlusion_capture);
        graph.read_texture(gtao, hiz, TextureUsage::SAMPLED);
        ssao_raw = graph.write_texture(gtao, ssao_raw, TextureUsage::STORAGE_WRITE);
        previous = gtao;
    }
    let ssao_blur_pass = graph.add_pass("ssao_blur");
    graph.after(ssao_blur_pass, previous);
    graph.read_texture(ssao_blur_pass, ssao_raw, TextureUsage::SAMPLED);
    ssao_blur = graph.write_texture(ssao_blur_pass, ssao_blur, TextureUsage::COLOR_ATTACHMENT);
    previous = ssao_blur_pass;

    if key.feature_mask & super::FRAME_FEATURE_SSR != 0 {
        let ssr_march = graph.add_pass("ssr_march");
        graph.after(ssr_march, previous);
        graph.read_texture(ssr_march, hdr, TextureUsage::SAMPLED);
        graph.read_texture(ssr_march, depth, TextureUsage::SAMPLED);
        graph.read_texture(ssr_march, material, TextureUsage::SAMPLED);
        ssr = graph.write_texture(ssr_march, ssr, TextureUsage::COLOR_ATTACHMENT);

        let ssr_temporal = graph.add_pass("ssr_temporal");
        graph.after(ssr_temporal, ssr_march);
        ssr = graph.read_write_texture(
            ssr_temporal,
            ssr,
            TextureUsage::SAMPLED,
            TextureUsage::COLOR_ATTACHMENT,
        );
        previous = ssr_temporal;
    }

    if key.feature_mask & super::FRAME_FEATURE_SSGI != 0 {
        let ssgi_pass = graph.add_pass("ssgi");
        graph.after(ssgi_pass, previous);
        graph.read_texture(ssgi_pass, hdr, TextureUsage::SAMPLED);
        graph.read_texture(ssgi_pass, depth, TextureUsage::SAMPLED);
        graph.read_texture(ssgi_pass, albedo, TextureUsage::SAMPLED);
        ssgi = graph.write_texture(ssgi_pass, ssgi, TextureUsage::STORAGE_WRITE);
        previous = ssgi_pass;
    }

    if key.feature_mask & super::FRAME_FEATURE_BLOOM != 0 {
        let bloom_pass = graph.add_pass("bloom");
        graph.after(bloom_pass, previous);
        graph.read_texture(bloom_pass, hdr, TextureUsage::SAMPLED);
        bloom = graph.write_texture(bloom_pass, bloom, TextureUsage::COLOR_ATTACHMENT);
        previous = bloom_pass;
    }

    let compose = graph.add_pass("compose");
    graph.after(compose, previous);
    for input in [hdr, ssao_blur, ssr, ssgi, bloom] {
        graph.read_texture(compose, input, TextureUsage::SAMPLED);
    }
    composed = graph.write_texture(compose, composed, TextureUsage::COLOR_ATTACHMENT);

    let postfx_tail = graph.add_pass("postfx_tail");
    graph.after(postfx_tail, compose);
    graph.read_texture(postfx_tail, composed, TextureUsage::SAMPLED);
    graph.read_texture(postfx_tail, velocity, TextureUsage::SAMPLED);
    graph.read_texture(postfx_tail, depth, TextureUsage::SAMPLED);
    if let Some(reactive) = temporal_reactive {
        graph.read_texture(postfx_tail, reactive, TextureUsage::SAMPLED);
    }
    postfx = graph.write_texture(postfx_tail, postfx, TextureUsage::COLOR_ATTACHMENT);

    let auto_exposure = graph.add_pass("auto_exposure");
    graph.after(auto_exposure, postfx_tail);
    graph.read_texture(auto_exposure, postfx, TextureUsage::SAMPLED);
    exposure = graph.write_texture(auto_exposure, exposure, TextureUsage::COLOR_ATTACHMENT);
    let final_composite = graph.add_pass("final_composite");
    graph.after(final_composite, auto_exposure);
    graph.read_texture(final_composite, postfx, TextureUsage::SAMPLED);
    graph.read_texture(final_composite, exposure, TextureUsage::SAMPLED);
    graph.read_texture(final_composite, ssao_blur, TextureUsage::SAMPLED);
    output = graph.write_texture(final_composite, output, TextureUsage::COLOR_ATTACHMENT);

    let overlay = graph.add_pass("overlay_2d");
    graph.after(overlay, final_composite);
    graph.set_side_effects(
        overlay,
        super::SideEffects::PRESENT.union(super::SideEffects::TIMESTAMP),
    );
    output = graph.read_write_texture(
        overlay,
        output,
        TextureUsage::COLOR_ATTACHMENT,
        TextureUsage::COLOR_ATTACHMENT,
    );

    if key.feature_mask & FRAME_FEATURE_CAPTURE_OUTPUT != 0 {
        let capture = graph.add_pass("capture_readback");
        graph.after(capture, overlay);
        graph.set_queue(capture, super::QueueClass::CopyCapable);
        graph.set_side_effects(capture, super::SideEffects::READBACK);
        graph.read_texture(capture, output, TextureUsage::COPY_SRC);
        if key.feature_mask & FRAME_FEATURE_CAPTURE_QUALITY != 0 {
            graph.read_texture(capture, hdr, TextureUsage::COPY_SRC);
            graph.read_texture(capture, depth, TextureUsage::COPY_SRC);
            for cascade in shadows {
                graph.read_texture(capture, cascade, TextureUsage::COPY_SRC);
            }
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::graph::{
        CapabilityTier, CompileOptions, PathTracingMode, QueueClass, ResolutionClass, SideEffects,
        FRAME_FEATURE_BLOOM, FRAME_FEATURE_SSAO, FRAME_FEATURE_SSGI, FRAME_FEATURE_SSR,
    };

    fn key(feature_mask: u64) -> FramePlanKey {
        FramePlanKey {
            resolution: ResolutionClass::Medium,
            quality_tier: 3,
            feature_mask,
            capability: CapabilityTier::Raster,
            path_tracing: PathTracingMode::Off,
            post_pass_count: 0,
            render_target_output: false,
        }
    }

    #[test]
    fn renderer_plan_preserves_the_legacy_serial_order() {
        let all_optional =
            FRAME_FEATURE_SSAO | FRAME_FEATURE_SSR | FRAME_FEATURE_SSGI | FRAME_FEATURE_BLOOM;
        let plan =
            build_renderer_frame_plan(key(all_optional), wgpu::TextureFormat::Bgra8UnormSrgb)
                .compile(CompileOptions::NO_ALIASING)
                .unwrap();
        let names = plan
            .passes
            .iter()
            .map(|pass| pass.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "accel_rebuild",
                "card_capture",
                "sdf_bake",
                "scene_sdf_clipmap",
                "wsrc_bake",
                "card_light",
                "froxel_assign",
                "shadow",
                "hdr_scene",
                "pt",
                "translucent",
                "hiz_build",
                "occlusion_capture",
                "gtao",
                "ssao_blur",
                "ssr_march",
                "ssr_temporal",
                "ssgi",
                "bloom",
                "compose",
                "postfx_tail",
                "auto_exposure",
                "final_composite",
                "overlay_2d",
            ]
        );
        assert!(plan
            .passes
            .iter()
            .all(|pass| pass.queue == QueueClass::Graphics));
        assert!(
            plan.allocations.is_empty(),
            "parity plan imports current persistent RTs"
        );
    }

    #[test]
    fn disabled_quality_passes_are_absent_from_the_compiled_topology() {
        let plan = build_renderer_frame_plan(key(0), wgpu::TextureFormat::Bgra8UnormSrgb)
            .compile(CompileOptions::CONSERVATIVE_ALIASING)
            .unwrap();
        for absent in [
            "accel_rebuild",
            "card_capture",
            "sdf_bake",
            "scene_sdf_clipmap",
            "wsrc_bake",
            "card_light",
            "hiz_build",
            "occlusion_capture",
            "gtao",
            "ssr_march",
            "ssr_temporal",
            "ssgi",
            "bloom",
        ] {
            assert!(
                plan.pass(absent).is_none(),
                "{absent} should be selected out"
            );
        }
        assert!(
            plan.pass("ssao_blur").is_some(),
            "disabled AO path still clears white"
        );
        assert!(plan.pass("compose").is_some());
        assert!(plan.pass("capture_readback").is_none());
    }

    #[test]
    fn capture_is_a_terminal_copy_pass_over_named_logical_resources() {
        let plan = build_renderer_frame_plan(
            key(FRAME_FEATURE_CAPTURE_OUTPUT | FRAME_FEATURE_CAPTURE_QUALITY),
            wgpu::TextureFormat::Bgra8UnormSrgb,
        )
        .compile(CompileOptions::NO_ALIASING)
        .unwrap();
        let capture = plan.pass("capture_readback").unwrap();
        assert_eq!(capture.queue, QueueClass::CopyCapable);
        assert!(capture.side_effects.contains(SideEffects::READBACK));
        for name in [
            "output",
            "hdr-scene",
            "scene-depth",
            "shadow-cascade-0",
            "shadow-cascade-1",
            "shadow-cascade-2",
        ] {
            let resource = plan.resource(name).unwrap();
            assert!(
                capture
                    .accesses
                    .iter()
                    .any(|access| access.resource == resource.id),
                "capture must read logical resource {name}"
            );
        }
        assert_eq!(
            plan.passes.last().map(|pass| pass.name.as_str()),
            Some("capture_readback")
        );
    }

    #[test]
    fn scene_reading_materials_add_two_non_aliasable_transients() {
        let plan = build_renderer_frame_plan(
            key(FRAME_FEATURE_SCENE_SNAPSHOTS),
            wgpu::TextureFormat::Bgra8UnormSrgb,
        )
        .compile(CompileOptions::CONSERVATIVE_ALIASING)
        .unwrap();
        assert_eq!(plan.allocations.len(), 2);
        assert_eq!(
            plan.unaliased_transient_bytes((1920, 1080), (1920, 1080)),
            plan.transient_bytes((1920, 1080), (1920, 1080)),
            "color/depth snapshots overlap and have incompatible descriptors"
        );
    }

    #[test]
    fn imported_refraction_declares_velocity_attachment_write() {
        let plan = build_renderer_frame_plan(
            key(FRAME_FEATURE_IMPORTED_REFRACTION),
            wgpu::TextureFormat::Bgra8UnormSrgb,
        )
        .compile(CompileOptions::CONSERVATIVE_ALIASING)
        .unwrap();
        let translucent = plan.pass("translucent").unwrap();
        let velocity = plan.resource("motion-vectors").unwrap();
        assert!(
            translucent
                .accesses
                .iter()
                .any(|access| access.resource == velocity.id),
            "the compiled translucent node must own the refractive velocity write"
        );
        assert!(
            plan.resource("translucent-scene-color").is_none(),
            "folded imported refraction can write velocity without allocating snapshots"
        );
    }

    #[test]
    fn weighted_transparency_allocates_only_its_two_declared_targets() {
        let disabled = build_renderer_frame_plan(key(0), wgpu::TextureFormat::Bgra8UnormSrgb)
            .compile(CompileOptions::CONSERVATIVE_ALIASING)
            .unwrap();
        assert!(disabled.resource("transparency-accumulation").is_none());
        assert!(disabled.resource("transparency-revealage").is_none());

        let plan = build_renderer_frame_plan(
            key(FRAME_FEATURE_WEIGHTED_TRANSPARENCY),
            wgpu::TextureFormat::Bgra8UnormSrgb,
        )
        .compile(CompileOptions::CONSERVATIVE_ALIASING)
        .unwrap();
        assert_eq!(plan.allocations.len(), 2);
        let translucent = plan.pass("translucent").unwrap();
        for name in ["transparency-accumulation", "transparency-revealage"] {
            let resource = plan.resource(name).unwrap();
            assert!(
                translucent
                    .accesses
                    .iter()
                    .any(|access| access.resource == resource.id),
                "translucent pass must own {name}"
            );
        }
    }

    #[test]
    fn temporal_reactive_target_is_lazy_and_connects_translucency_to_postfx() {
        let disabled = build_renderer_frame_plan(key(0), wgpu::TextureFormat::Bgra8UnormSrgb)
            .compile(CompileOptions::CONSERVATIVE_ALIASING)
            .unwrap();
        assert!(disabled.resource("transparency-reactive").is_none());

        let plan = build_renderer_frame_plan(
            key(FRAME_FEATURE_TEMPORAL_REACTIVE),
            wgpu::TextureFormat::Bgra8UnormSrgb,
        )
        .compile(CompileOptions::CONSERVATIVE_ALIASING)
        .unwrap();
        assert_eq!(plan.allocations.len(), 1);
        let reactive = plan.resource("transparency-reactive").unwrap();
        assert_eq!(
            reactive.desc,
            super::super::ResourceDesc::Texture(texture_desc(
                wgpu::TextureFormat::R8Unorm,
                Extent::RenderRelative {
                    numerator: 1,
                    denominator: 1,
                    layers: 1,
                },
                TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED),
                AliasClass::Color,
            ))
        );
        for pass_name in ["translucent", "postfx_tail"] {
            let pass = plan.pass(pass_name).unwrap();
            assert!(
                pass.accesses
                    .iter()
                    .any(|access| access.resource == reactive.id),
                "{pass_name} must own temporal reactive coverage"
            );
        }
        assert_eq!(
            plan.transient_bytes((1920, 1080), (1920, 1080)),
            1920 * 1080,
            "R8 coverage costs exactly one byte per render pixel"
        );
    }

    #[test]
    fn transmitted_shadow_cascades_are_lazy_persistent_and_resolved_after_pt() {
        let disabled = build_renderer_frame_plan(key(0), wgpu::TextureFormat::Bgra8UnormSrgb)
            .compile(CompileOptions::CONSERVATIVE_ALIASING)
            .unwrap();
        assert!(disabled.pass("transmitted_shadow_resolve").is_none());
        assert!(disabled.resource("transmitted-shadow-color-0").is_none());
        assert!(disabled.resource("transmitted-shadow-depth-0").is_none());

        let plan = build_renderer_frame_plan(
            key(FRAME_FEATURE_TRANSMITTED_SHADOWS),
            wgpu::TextureFormat::Bgra8UnormSrgb,
        )
        .compile(CompileOptions::CONSERVATIVE_ALIASING)
        .unwrap();
        assert!(
            plan.allocations.is_empty(),
            "transmitted cascades are persistent lazy imports, not frame transients"
        );
        let shadow = plan.pass("shadow").unwrap();
        let resolve = plan.pass("transmitted_shadow_resolve").unwrap();
        for cascade in 0..3 {
            for prefix in ["transmitted-shadow-color", "transmitted-shadow-depth"] {
                let name = format!("{prefix}-{cascade}");
                let resource = plan.resource(&name).unwrap();
                assert!(shadow
                    .accesses
                    .iter()
                    .any(|access| access.resource == resource.id));
                assert!(resolve
                    .accesses
                    .iter()
                    .any(|access| access.resource == resource.id));
            }
        }
        let pt_index = plan
            .passes
            .iter()
            .position(|pass| pass.name == "pt")
            .unwrap();
        let resolve_index = plan
            .passes
            .iter()
            .position(|pass| pass.name == "transmitted_shadow_resolve")
            .unwrap();
        let translucent_index = plan
            .passes
            .iter()
            .position(|pass| pass.name == "translucent")
            .unwrap();
        assert!(pt_index < resolve_index && resolve_index < translucent_index);
    }
}
