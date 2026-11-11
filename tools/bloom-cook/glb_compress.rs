//! `glb-compress`: rewrite a GLB so its embedded images are KTX2 containers
//! of ASTC blocks with the engine's own mip chains.
//!
//! The runtime uploads those blocks untouched on adapters with ASTC (every
//! Apple GPU, most mobile GPUs) and decodes level zero elsewhere, so one file
//! loads everywhere. A 4096² texture drops from 85 MB of RGBA8 plus mips to
//! 22 MB at 4×4 or 10 MB at 6×6, and load skips PNG inflate and mip filtering.
//!
//! Mips are built exactly as the runtime builds them for RGBA8 uploads:
//! sRGB-aware averaging for colour, plain averaging for data, and
//! renormalised direction with accumulated variance in alpha for normal maps.
//! Images that feed an alpha-MASK material are left alone: their coverage
//! mips depend on the material cutoff and the runtime bakes them from texels.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use bloom_shared::compressed_texture::{BlockFormat, CompressedImage};
use serde_json::Value;

const GLB_MAGIC: u32 = 0x4654_6C67;
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

#[derive(Clone, Copy)]
struct Options {
    color: (u8, u8),
    /// `None` leaves normal maps as they are.
    normal: Option<(u8, u8)>,
    preset: ctt_astcenc::Preset,
}

fn parse_block(value: &str) -> Result<(u8, u8), String> {
    let parsed = value
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u8>().ok()?, h.parse::<u8>().ok()?)));
    match parsed {
        Some((width, height))
            if BlockFormat::Astc { width, height }.vk_format(false).is_some() =>
        {
            Ok((width, height))
        }
        _ => Err(format!(
            "unsupported ASTC block {value:?}: use 4x4, 5x5, 6x6, 8x8, 10x10, 12x12 or a listed in-between size"
        )),
    }
}

fn parse_options(flags: &[String]) -> Result<Options, String> {
    let mut options = Options {
        color: (4, 4),
        normal: Some((4, 4)),
        preset: ctt_astcenc::Preset::Thorough,
    };
    let mut iter = flags.iter();
    while let Some(flag) = iter.next() {
        let mut value = || iter.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--astc" => options.color = parse_block(value()?)?,
            "--normal-astc" => {
                let value = value()?;
                options.normal = if value == "keep" {
                    None
                } else {
                    Some(parse_block(value)?)
                };
            }
            "--quality" => {
                options.preset = match value()?.as_str() {
                    "fast" => ctt_astcenc::Preset::Fast,
                    "medium" => ctt_astcenc::Preset::Medium,
                    "thorough" => ctt_astcenc::Preset::Thorough,
                    "exhaustive" => ctt_astcenc::Preset::Exhaustive,
                    other => return Err(format!("unknown --quality {other:?}")),
                }
            }
            other => return Err(format!("unknown glb-compress flag {other:?}")),
        }
    }
    Ok(options)
}

/// Image index a texture-info object (`{"index": texture}`) resolves to.
fn image_of(root: &Value, info: Option<&Value>) -> Option<usize> {
    let texture = info?.get("index")?.as_u64()? as usize;
    let source = root
        .get("textures")?
        .get(texture)?
        .get("source")?
        .as_u64()?;
    Some(source as usize)
}

#[derive(Default)]
struct Roles {
    normal: BTreeSet<usize>,
    srgb: BTreeSet<usize>,
    alpha_mask: BTreeSet<usize>,
}

/// Mirrors the runtime's classification in `models_gltf*.rs`.
fn classify(root: &Value) -> Roles {
    let mut roles = Roles::default();
    let empty = Vec::new();
    for material in root
        .get("materials")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
    {
        let extension = |name: &str, key: &str| {
            let info = material.get("extensions")?.get(name)?.get(key);
            image_of(root, info)
        };
        roles
            .normal
            .extend(image_of(root, material.get("normalTexture")));
        roles.normal.extend(extension(
            "KHR_materials_clearcoat",
            "clearcoatNormalTexture",
        ));
        let pbr = material.get("pbrMetallicRoughness");
        let base = image_of(root, pbr.and_then(|pbr| pbr.get("baseColorTexture")));
        let diffuse = extension("KHR_materials_pbrSpecularGlossiness", "diffuseTexture");
        roles.srgb.extend(base);
        roles.srgb.extend(diffuse);
        roles
            .srgb
            .extend(image_of(root, material.get("emissiveTexture")));
        roles.srgb.extend(extension(
            "KHR_materials_pbrSpecularGlossiness",
            "specularGlossinessTexture",
        ));
        roles
            .srgb
            .extend(extension("KHR_materials_specular", "specularColorTexture"));
        roles
            .srgb
            .extend(extension("KHR_materials_sheen", "sheenColorTexture"));
        if material.get("alphaMode").and_then(Value::as_str) == Some("MASK") {
            roles.alpha_mask.extend(base.or(diffuse));
        }
    }
    roles
}

/// Encode one RGBA8 level with ARM's reference encoder. Blocks are tuned for
/// the 8-bit decode the runtime's UNORM formats use, and the error is weighed
/// in the stored (display-encoded) space for every role.
fn encode_level(
    rgba: &[u8],
    width: u32,
    height: u32,
    block: (u8, u8),
    preset: ctt_astcenc::Preset,
) -> Result<Vec<u8>, String> {
    use ctt_astcenc::bindings::{astcenc_image, astcenc_type_ASTCENC_TYPE_U8};
    let failed = |error: ctt_astcenc::Error| format!("ASTC encoder: {error:?}");
    let config = ctt_astcenc::config_init(
        ctt_astcenc::Profile::Ldr,
        u32::from(block.0),
        u32::from(block.1),
        1,
        preset,
        ctt_astcenc::Flags::USE_DECODE_UNORM8,
    )
    .map_err(failed)?;
    let mut context = ctt_astcenc::Context::new_parallel(&config).map_err(failed)?;
    let mut texels = rgba.to_vec();
    let mut slices = [texels.as_mut_ptr().cast::<std::ffi::c_void>()];
    let mut image = astcenc_image {
        dim_x: width,
        dim_y: height,
        dim_z: 1,
        data_type: astcenc_type_ASTCENC_TYPE_U8,
        data: slices.as_mut_ptr(),
    };
    let format = BlockFormat::Astc {
        width: block.0,
        height: block.1,
    };
    let mut blocks = vec![0u8; format.level_bytes(width, height)];
    context
        .compress(&mut image, ctt_astcenc::Swizzle::IDENTITY, &mut blocks)
        .map_err(failed)?;
    Ok(blocks)
}

struct Encoded {
    ktx2: Vec<u8>,
    report: String,
}

fn encode_image(
    index: usize,
    bytes: &[u8],
    is_normal: bool,
    is_srgb: bool,
    block: (u8, u8),
    preset: ctt_astcenc::Preset,
) -> Result<Encoded, String> {
    let image = image::load_from_memory(bytes)
        .map_err(|error| format!("image {index}: {error}"))?
        .to_rgba8();
    let (width, height) = image.dimensions();
    let (mip_data, mip_levels) = if is_normal {
        crate::texture_cook::build_normal_mip_chain(&image)
    } else {
        bloom_shared::renderer::build_cooked_color_mip_chain(
            width,
            height,
            image.as_raw(),
            None,
            is_srgb,
        )
    };
    // Normal maps carry variance in alpha (zero at level zero).
    let alpha = is_normal || image.pixels().any(|pixel| pixel[3] != 255);
    let mut extents = Vec::with_capacity(mip_levels as usize);
    let mut offset = 0usize;
    for level in 0..mip_levels {
        let (level_width, level_height) = ((width >> level).max(1), (height >> level).max(1));
        let length = (level_width * level_height * 4) as usize;
        extents.push((offset, length, level_width, level_height));
        offset += length;
    }
    // The encoder spreads each level over every core itself.
    let mut levels = Vec::with_capacity(extents.len());
    for &(offset, length, level_width, level_height) in &extents {
        let texels = &mip_data[offset..offset + length];
        levels.push(encode_level(
            texels,
            level_width,
            level_height,
            block,
            preset,
        )?);
    }
    let compressed = CompressedImage {
        format: BlockFormat::Astc {
            width: block.0,
            height: block.1,
        },
        width,
        height,
        srgb: is_srgb && !is_normal,
        levels,
    };
    let psnr = level_zero_psnr(&compressed, &mip_data[..extents[0].1], alpha);
    let ktx2 = compressed
        .to_ktx2()
        .ok_or_else(|| format!("image {index}: cannot serialise KTX2"))?;
    let role = if is_normal {
        "normal"
    } else if is_srgb {
        "colour"
    } else {
        "data"
    };
    Ok(Encoded {
        report: format!(
            "  image {index}: {width}x{height} {role} -> ASTC {}x{} {mip_levels} mips, \
             {:.1} MB RGBA8 -> {:.1} MB on the GPU, {} -> {} bytes in the file, PSNR {psnr:.1} dB",
            block.0,
            block.1,
            mip_data.len() as f64 / 1e6,
            compressed.levels.iter().map(Vec::len).sum::<usize>() as f64 / 1e6,
            bytes.len(),
            ktx2.len(),
        ),
        ktx2,
    })
}

/// Peak signal-to-noise ratio of the decoded top level against its source.
fn level_zero_psnr(image: &CompressedImage, source: &[u8], alpha: bool) -> f64 {
    let BlockFormat::Astc { width, height } = image.format else {
        return f64::NAN;
    };
    let channels = if alpha { 4 } else { 3 };
    let row = image.width as usize * 4;
    let mut squared = 0f64;
    let decoded = astc_decode::astc_decode(
        image.levels[0].as_slice(),
        image.width,
        image.height,
        astc_decode::Footprint::new(u32::from(width), u32::from(height)),
        |x, y, texel| {
            let at = y as usize * row + x as usize * 4;
            for channel in 0..channels {
                let difference = f64::from(texel[channel]) - f64::from(source[at + channel]);
                squared += difference * difference;
            }
        },
    );
    if decoded.is_err() {
        return f64::NAN;
    }
    let mean = squared / (image.width as f64 * image.height as f64 * channels as f64);
    if mean == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mean).log10()
    }
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// The JSON document, the binary chunk, and any further chunks verbatim
/// (header included) so extension data survives the rewrite.
fn split_glb(data: &[u8]) -> Result<(Value, &[u8], Vec<u8>), String> {
    if read_u32(data, 0) != Some(GLB_MAGIC) || read_u32(data, 4) != Some(2) {
        return Err("not a glTF 2.0 binary (.glb)".to_string());
    }
    let mut json = None;
    let mut bin: Option<&[u8]> = None;
    let mut other = Vec::new();
    let mut at = 12usize;
    while let (Some(length), Some(kind)) = (read_u32(data, at), read_u32(data, at + 4)) {
        let end = at + 8 + length as usize;
        let body = data.get(at + 8..end).ok_or("truncated GLB chunk")?;
        match kind {
            CHUNK_JSON if json.is_none() => {
                json = Some(serde_json::from_slice::<Value>(body).map_err(|e| e.to_string())?)
            }
            CHUNK_BIN if bin.is_none() => bin = Some(body),
            _ => {
                other.extend_from_slice(&data[at..end]);
                other.resize(other.len().next_multiple_of(4), 0);
            }
        }
        at = end.next_multiple_of(4);
    }
    Ok((
        json.ok_or("GLB has no JSON chunk")?,
        bin.unwrap_or(&[]),
        other,
    ))
}

fn view_range(root: &Value, view: usize) -> Option<(usize, usize)> {
    let view = root.get("bufferViews")?.get(view)?;
    if view.get("buffer")?.as_u64()? != 0 {
        return None;
    }
    let offset = view.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
    Some((offset, view.get("byteLength")?.as_u64()? as usize))
}

/// Compress the images of `data` (a GLB); returns the new file and a report.
pub(crate) fn compress_glb(data: &[u8], flags: &[String]) -> Result<(Vec<u8>, String), String> {
    let options = parse_options(flags)?;
    let (mut root, bin, other_chunks) = split_glb(data)?;
    let roles = classify(&root);

    // image index -> (buffer view, block size)
    let mut jobs: BTreeMap<usize, (usize, (u8, u8))> = BTreeMap::new();
    let mut notes = Vec::new();
    let images = root.get("images").and_then(Value::as_array).cloned();
    for (index, image) in images.unwrap_or_default().iter().enumerate() {
        let Some(view) = image.get("bufferView").and_then(Value::as_u64) else {
            notes.push(format!("  image {index}: external file, left as it is"));
            continue;
        };
        let is_normal = roles.normal.contains(&index);
        if roles.alpha_mask.contains(&index) {
            notes.push(format!(
                "  image {index}: alpha-MASK base colour, left as it is"
            ));
        } else if let Some(block) = if is_normal {
            options.normal
        } else {
            Some(options.color)
        } {
            jobs.insert(index, (view as usize, block));
        } else {
            notes.push(format!("  image {index}: normal map, left as it is"));
        }
    }

    // Images may share a buffer view. Its bytes can only be replaced when
    // every image on it is compressed the same way; otherwise one of them
    // would keep a PNG MIME type over KTX2 bytes, or get the wrong mips.
    let mut by_view: BTreeMap<usize, Vec<Option<(bool, bool, (u8, u8))>>> = BTreeMap::new();
    let images = root.get("images").and_then(Value::as_array).cloned();
    for (index, image) in images.unwrap_or_default().iter().enumerate() {
        if let Some(view) = image.get("bufferView").and_then(Value::as_u64) {
            let plan = jobs.get(&index).map(|&(_, block)| {
                (
                    roles.normal.contains(&index),
                    roles.srgb.contains(&index),
                    block,
                )
            });
            by_view.entry(view as usize).or_default().push(plan);
        }
    }
    jobs.retain(|index, (view, _)| {
        let plans = &by_view[view];
        let consistent = plans.iter().all(|plan| plan.is_some() && *plan == plans[0]);
        if !consistent {
            notes.push(format!(
                "  image {index}: shares its data with an image that is used differently, \
                 left as it is"
            ));
        }
        consistent
    });

    let mut replacements: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    for (&index, &(view, block)) in &jobs {
        let (offset, length) =
            view_range(&root, view).ok_or_else(|| format!("image {index}: bad buffer view"))?;
        let source = bin
            .get(offset..offset + length)
            .ok_or_else(|| format!("image {index}: buffer view outside the binary chunk"))?;
        if bloom_shared::compressed_texture::is_ktx2(source) {
            notes.push(format!("  image {index}: already KTX2, left as it is"));
            continue;
        }
        let is_normal = roles.normal.contains(&index);
        let encoded = encode_image(
            index,
            source,
            is_normal,
            roles.srgb.contains(&index),
            block,
            options.preset,
        )?;
        notes.push(encoded.report);
        replacements.insert(view, encoded.ktx2);
        root["images"][index]["mimeType"] = Value::from("image/ktx2");
    }

    // Rebuild the binary chunk view by view so the replaced PNG bytes are gone.
    let view_count = root
        .get("bufferViews")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let mut order: Vec<usize> = (0..view_count).collect();
    order.sort_by_key(|&view| view_range(&root, view).map_or(usize::MAX, |(offset, _)| offset));
    let mut new_bin = Vec::with_capacity(bin.len());
    for view in order {
        let Some((offset, length)) = view_range(&root, view) else {
            continue;
        };
        let bytes = match replacements.get(&view) {
            Some(bytes) => bytes.as_slice(),
            None => bin
                .get(offset..offset + length)
                .ok_or_else(|| format!("buffer view {view} lies outside the binary chunk"))?,
        };
        new_bin.resize(new_bin.len().next_multiple_of(4), 0);
        root["bufferViews"][view]["byteOffset"] = Value::from(new_bin.len());
        root["bufferViews"][view]["byteLength"] = Value::from(bytes.len());
        new_bin.extend_from_slice(bytes);
    }
    new_bin.resize(new_bin.len().next_multiple_of(4), 0);
    if root
        .get("buffers")
        .and_then(|buffers| buffers.get(0))
        .is_some()
    {
        root["buffers"][0]["byteLength"] = Value::from(new_bin.len());
    }

    let mut json = serde_json::to_vec(&root).map_err(|error| error.to_string())?;
    json.resize(json.len().next_multiple_of(4), b' ');
    let total = 12 + 8 + json.len() + 8 + new_bin.len() + other_chunks.len();
    let mut out = Vec::with_capacity(total);
    for value in [GLB_MAGIC, 2, total as u32, json.len() as u32, CHUNK_JSON] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&json);
    out.extend_from_slice(&(new_bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&new_bin);
    out.extend_from_slice(&other_chunks);

    let report = format!(
        "{} image(s) compressed, {} -> {} bytes\n{}",
        replacements.len(),
        data.len(),
        out.len(),
        notes.join("\n")
    );
    Ok((out, report))
}

pub(crate) fn compress_glb_command(
    input: &Path,
    output: &Path,
    flags: &[String],
) -> Result<String, String> {
    let data =
        std::fs::read(input).map_err(|error| format!("read {}: {error}", input.display()))?;
    let (out, report) = compress_glb(&data, flags)?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    std::fs::write(output, out).map_err(|error| format!("write {}: {error}", output.display()))?;
    Ok(format!(
        "{} -> {}: {report}",
        input.display(),
        output.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let image = image::RgbaImage::from_fn(width, height, |x, y| image::Rgba(pixel(x, y)));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut bytes, image::ImageFormat::Png)
            .expect("PNG encodes");
        bytes.into_inner()
    }

    /// A one-triangle GLB with a base-colour image and a normal map.
    fn sample_glb(alpha_mode: &str) -> Vec<u8> {
        let color = png(24, 16, |x, y| [(x * 10) as u8, (y * 15) as u8, 90, 255]);
        let normal = png(16, 16, |_, _| [128, 128, 255, 255]);
        let mut bin: Vec<u8> = Vec::new();
        for value in [0f32, 0., 0., 1., 0., 0., 0., 1., 0.] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        let color_at = bin.len();
        bin.extend_from_slice(&color);
        bin.resize(bin.len().next_multiple_of(4), 0);
        let normal_at = bin.len();
        bin.extend_from_slice(&normal);
        bin.resize(bin.len().next_multiple_of(4), 0);
        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
            "materials": [{
                "alphaMode": alpha_mode,
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "normalTexture": {"index": 1}
            }],
            "textures": [{"source": 0}, {"source": 1}],
            "images": [
                {"bufferView": 1, "mimeType": "image/png"},
                {"bufferView": 2, "mimeType": "image/png"}
            ],
            "accessors": [{
                "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]
            }],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": color_at, "byteLength": color.len()},
                {"buffer": 0, "byteOffset": normal_at, "byteLength": normal.len()}
            ],
            "buffers": [{"byteLength": bin.len()}]
        });
        let mut json = serde_json::to_vec(&json).unwrap();
        json.resize(json.len().next_multiple_of(4), b' ');
        let total = 28 + json.len() + bin.len();
        let mut out = Vec::new();
        for value in [GLB_MAGIC, 2, total as u32, json.len() as u32, CHUNK_JSON] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
        out.extend_from_slice(&bin);
        out
    }

    fn image_bytes<'a>(root: &Value, bin: &'a [u8], image: usize) -> &'a [u8] {
        let view = root["images"][image]["bufferView"].as_u64().unwrap() as usize;
        let (offset, length) = view_range(root, view).unwrap();
        &bin[offset..offset + length]
    }

    #[test]
    fn images_become_ktx2_and_geometry_is_untouched() {
        let source = sample_glb("OPAQUE");
        let (out, report) = compress_glb(&source, &["--astc".into(), "6x6".into()]).unwrap();
        assert!(report.starts_with("2 image(s) compressed"), "{report}");
        let (root, bin, _) = split_glb(&out).unwrap();
        let (source_root, source_bin, _) = split_glb(&source).unwrap();
        let (offset, length) = view_range(&root, 0).unwrap();
        let (source_offset, _) = view_range(&source_root, 0).unwrap();
        assert_eq!(
            &bin[offset..offset + length],
            &source_bin[source_offset..source_offset + 36]
        );
        assert_eq!(root["images"][0]["mimeType"], "image/ktx2");
        assert_eq!(root["buffers"][0]["byteLength"], bin.len());

        let color = CompressedImage::from_ktx2(image_bytes(&root, bin, 0)).unwrap();
        assert_eq!(
            color.format,
            BlockFormat::Astc {
                width: 6,
                height: 6
            }
        );
        assert_eq!((color.width, color.height, color.levels.len()), (24, 16, 5));
        assert!(color.srgb);
        let normal = CompressedImage::from_ktx2(image_bytes(&root, bin, 1)).unwrap();
        assert_eq!(
            normal.format,
            BlockFormat::Astc {
                width: 4,
                height: 4
            }
        );
        assert_eq!(
            (normal.width, normal.height, normal.levels.len()),
            (16, 16, 5)
        );
        assert!(!normal.srgb);
    }

    #[test]
    fn flat_normal_map_keeps_direction_and_zero_variance() {
        let (out, _) = compress_glb(&sample_glb("OPAQUE"), &[]).unwrap();
        let (root, bin, _) = split_glb(&out).unwrap();
        let bytes = image_bytes(&root, bin, 1);
        let image = CompressedImage::from_ktx2(bytes).unwrap();
        let mut worst = 0i32;
        astc_decode::astc_decode(
            image.levels[0].as_slice(),
            16,
            16,
            astc_decode::Footprint::new(4, 4),
            |_, _, texel| {
                for (value, expected) in texel.iter().zip([128, 128, 255, 0]) {
                    worst = worst.max((i32::from(*value) - expected).abs());
                }
            },
        )
        .unwrap();
        assert!(worst <= 1, "flat normal drifted by {worst}");
    }

    #[test]
    fn alpha_mask_base_colour_and_kept_normals_stay_png() {
        let flags = ["--normal-astc".to_string(), "keep".to_string()];
        let (out, report) = compress_glb(&sample_glb("MASK"), &flags).unwrap();
        assert!(report.starts_with("0 image(s) compressed"), "{report}");
        let (root, bin, _) = split_glb(&out).unwrap();
        assert_eq!(root["images"][0]["mimeType"], "image/png");
        assert!(image_bytes(&root, bin, 0).starts_with(b"\x89PNG"));
        assert!(image_bytes(&root, bin, 1).starts_with(b"\x89PNG"));
    }

    #[test]
    fn shared_image_data_and_extra_chunks_are_preserved() {
        // Point the normal map at the colour image's bytes: the two images
        // now share a buffer view but need different mips.
        let source = sample_glb("OPAQUE");
        let (mut root, bin, _) = split_glb(&source).unwrap();
        root["images"][1]["bufferView"] = Value::from(1);
        let mut json = serde_json::to_vec(&root).unwrap();
        json.resize(json.len().next_multiple_of(4), b' ');
        let extra = [4u32.to_le_bytes(), 0x1234_5678u32.to_le_bytes(), *b"DATA"].concat();
        let total = 28 + json.len() + bin.len() + extra.len();
        let mut glb = Vec::new();
        for value in [GLB_MAGIC, 2, total as u32, json.len() as u32, CHUNK_JSON] {
            glb.extend_from_slice(&value.to_le_bytes());
        }
        glb.extend_from_slice(&json);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(&CHUNK_BIN.to_le_bytes());
        glb.extend_from_slice(bin);
        glb.extend_from_slice(&extra);

        let (out, report) = compress_glb(&glb, &[]).unwrap();
        assert!(report.starts_with("0 image(s) compressed"), "{report}");
        let (root, bin, other) = split_glb(&out).unwrap();
        assert_eq!(other, extra);
        assert_eq!(root["images"][0]["mimeType"], "image/png");
        assert!(image_bytes(&root, bin, 0).starts_with(b"\x89PNG"));
        assert_eq!(read_u32(&out, 8), Some(out.len() as u32));
    }

    #[test]
    fn unknown_flags_and_blocks_are_rejected() {
        let source = sample_glb("OPAQUE");
        assert!(compress_glb(&source, &["--astc".into(), "7x7".into()]).is_err());
        assert!(compress_glb(&source, &["--bogus".into()]).is_err());
        assert!(compress_glb(b"not a glb", &[]).is_err());
    }
}
