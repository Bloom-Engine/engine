//! Deterministic texture preparation shared by direct and store cooking.

use crate::asset_profile::AssetProfile;
use crate::geometry_format::sha256;
use serde_json::{json, Value};
use std::path::Path;

pub(crate) const TEXTURE_RECIPE_VERSION: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TextureArtifactFormat {
    Bc7Linear,
    Bc7Srgb,
    Rgba8Linear,
    Rgba8Srgb,
    Rgba8NormalVariance,
}

impl TextureArtifactFormat {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Bc7Linear => "bc7-rgba-unorm",
            Self::Bc7Srgb => "bc7-rgba-unorm-srgb",
            Self::Rgba8Linear => "rgba8-unorm",
            Self::Rgba8Srgb => "rgba8-unorm-srgb",
            Self::Rgba8NormalVariance => "rgba8-unorm-normal-variance",
        }
    }

    pub(crate) const fn image_format(self) -> image_dds::ImageFormat {
        match self {
            Self::Bc7Linear => image_dds::ImageFormat::BC7RgbaUnorm,
            Self::Bc7Srgb => image_dds::ImageFormat::BC7RgbaUnormSrgb,
            Self::Rgba8Linear | Self::Rgba8NormalVariance => image_dds::ImageFormat::Rgba8Unorm,
            Self::Rgba8Srgb => image_dds::ImageFormat::Rgba8UnormSrgb,
        }
    }

    pub(crate) const fn is_block_compressed(self) -> bool {
        matches!(self, Self::Bc7Linear | Self::Bc7Srgb)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TextureSettings {
    normal_map: bool,
    linear: bool,
    alpha_coverage_reference: Option<f32>,
}

impl TextureSettings {
    pub(crate) const fn from_semantics(
        is_normal: bool,
        is_srgb: bool,
        alpha_coverage_reference: Option<f32>,
    ) -> Self {
        Self {
            normal_map: is_normal,
            linear: is_normal || !is_srgb,
            alpha_coverage_reference,
        }
    }

    pub(crate) fn parse<'a>(flags: impl IntoIterator<Item = &'a str>) -> Result<Self, String> {
        let mut normal_map = false;
        let mut linear = false;
        let mut alpha_coverage_reference = None;
        let flags = flags.into_iter().collect::<Vec<_>>();
        let mut index = 0;
        while index < flags.len() {
            let flag = flags[index];
            if flag == "--alpha-coverage" {
                let value = flags
                    .get(index + 1)
                    .ok_or("--alpha-coverage requires a value")?
                    .parse::<f32>()
                    .map_err(|_| "--alpha-coverage must be a finite non-negative number")?;
                if !value.is_finite() || value < 0.0 {
                    return Err("--alpha-coverage must be a finite non-negative number".to_string());
                }
                if alpha_coverage_reference.replace(value).is_some() {
                    return Err("--alpha-coverage may only be specified once".to_string());
                }
                index += 2;
                continue;
            }
            let slot = match flag {
                "--normal" => &mut normal_map,
                "--linear" => &mut linear,
                _ => return Err(format!("unknown texture option {flag:?}")),
            };
            if *slot {
                return Err(format!("{flag} may only be specified once"));
            }
            *slot = true;
            index += 1;
        }
        if normal_map {
            linear = true;
        }
        if normal_map && alpha_coverage_reference.is_some() {
            return Err("normal maps cannot use alpha-coverage mips".to_string());
        }
        Ok(Self {
            normal_map,
            linear,
            alpha_coverage_reference,
        })
    }

    pub(crate) fn from_manifest(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or("asset manifest texture settings are missing or not an object")?;
        if object.len() != 3 {
            return Err(
                "asset manifest texture settings have unknown or missing fields".to_string(),
            );
        }
        let color_space = object
            .get("color_space")
            .and_then(Value::as_str)
            .ok_or("asset manifest texture color_space is missing or not a string")?;
        let normal_map = object
            .get("normal_map")
            .and_then(Value::as_bool)
            .ok_or("asset manifest texture normal_map is missing or not a boolean")?;
        let alpha_coverage_reference = match object.get("alpha_coverage_reference") {
            Some(Value::Null) => None,
            Some(Value::Number(value)) => value.as_f64().map(|value| value as f32),
            _ => None,
        };
        if alpha_coverage_reference.is_some_and(|value| !value.is_finite() || value < 0.0)
            || !object.contains_key("alpha_coverage_reference")
        {
            return Err("asset manifest alpha coverage reference is invalid".to_string());
        }
        let linear = match color_space {
            "linear" => true,
            "srgb" => false,
            other => return Err(format!("asset manifest has unknown color space {other:?}")),
        };
        if normal_map && !linear {
            return Err("normal-map texture manifests must use linear color space".to_string());
        }
        if normal_map && alpha_coverage_reference.is_some() {
            return Err("normal-map texture manifests cannot use alpha coverage".to_string());
        }
        Ok(Self {
            normal_map,
            linear,
            alpha_coverage_reference,
        })
    }

    pub(crate) fn as_json(self) -> Value {
        json!({
            "alpha_coverage_reference": self.alpha_coverage_reference,
            "color_space": if self.linear { "linear" } else { "srgb" },
            "normal_map": self.normal_map,
        })
    }

    pub(crate) fn artifact_format(self, profile: Option<&AssetProfile>) -> TextureArtifactFormat {
        if self.normal_map {
            TextureArtifactFormat::Rgba8NormalVariance
        } else if profile.is_some_and(|profile| profile.platform() == "portable") {
            if self.linear {
                TextureArtifactFormat::Rgba8Linear
            } else {
                TextureArtifactFormat::Rgba8Srgb
            }
        } else if self.linear {
            TextureArtifactFormat::Bc7Linear
        } else {
            TextureArtifactFormat::Bc7Srgb
        }
    }

    pub(crate) fn is_normal_map(self) -> bool {
        self.normal_map
    }

    pub(crate) fn is_srgb(self) -> bool {
        !self.linear
    }

    pub(crate) fn build_key_sha256(self, source_sha256: [u8; 32]) -> [u8; 32] {
        let mut bytes = Vec::with_capacity(80);
        bytes.extend_from_slice(b"bloom-texture-recipe\0");
        bytes.extend_from_slice(&TEXTURE_RECIPE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&source_sha256);
        bytes.push(u8::from(self.normal_map));
        bytes.push(u8::from(self.linear));
        bytes.extend_from_slice(
            &self
                .alpha_coverage_reference
                .map_or(u32::MAX, f32::to_bits)
                .to_le_bytes(),
        );
        sha256(&bytes)
    }
}

pub(crate) struct PreparedTexture {
    pub(crate) source_bytes: Vec<u8>,
    pub(crate) source_sha256: [u8; 32],
    pub(crate) settings: TextureSettings,
    decoded_rgba: Option<image::RgbaImage>,
}

impl PreparedTexture {
    pub(crate) fn read(input: &Path, settings: TextureSettings) -> Result<Self, String> {
        let source_bytes =
            std::fs::read(input).map_err(|error| format!("read {}: {error}", input.display()))?;
        let source_sha256 = sha256(&source_bytes);
        Ok(Self {
            source_bytes,
            source_sha256,
            settings,
            decoded_rgba: None,
        })
    }

    pub(crate) fn from_rgba(
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        settings: TextureSettings,
    ) -> Result<Self, String> {
        let image = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| format!("RGBA byte count does not match {width}x{height}"))?;
        let mut source_bytes = Vec::with_capacity(32 + image.len());
        source_bytes.extend_from_slice(b"bloom-decoded-rgba-v1\0");
        source_bytes.extend_from_slice(&width.to_le_bytes());
        source_bytes.extend_from_slice(&height.to_le_bytes());
        source_bytes.extend_from_slice(image.as_raw());
        let source_sha256 = sha256(&source_bytes);
        Ok(Self {
            source_bytes,
            source_sha256,
            settings,
            decoded_rgba: Some(image),
        })
    }
}

pub(crate) struct CookedTexture {
    pub(crate) bytes: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mip_levels: u32,
    pub(crate) format: TextureArtifactFormat,
}

pub(crate) fn cook_prepared_texture(
    input: &Path,
    prepared: &PreparedTexture,
    format: TextureArtifactFormat,
) -> Result<CookedTexture, String> {
    let image = match &prepared.decoded_rgba {
        Some(image) => image.clone(),
        None => image::load_from_memory(&prepared.source_bytes)
            .map_err(|error| format!("decode {}: {error}", input.display()))?
            .to_rgba8(),
    };
    if prepared.settings.is_normal_map() != (format == TextureArtifactFormat::Rgba8NormalVariance) {
        return Err("normal-map semantics do not match the selected artifact format".to_string());
    }
    let dds = if prepared.settings.is_normal_map() {
        let (mip_data, mip_levels) = build_normal_mip_chain(&image);
        normal_mip_chain_dds(
            image.width(),
            image.height(),
            mip_levels,
            &mip_data,
            format.image_format(),
        )
        .map_err(|error| format!("encode {}: {error}", input.display()))?
    } else if prepared.settings.alpha_coverage_reference.is_some() {
        let (mip_data, mip_levels) = bloom_shared::renderer::build_cooked_color_mip_chain(
            image.width(),
            image.height(),
            image.as_raw(),
            prepared.settings.alpha_coverage_reference,
            prepared.settings.is_srgb(),
        );
        let encoded = image_dds::SurfaceRgba8 {
            width: image.width(),
            height: image.height(),
            depth: 1,
            layers: 1,
            mipmaps: mip_levels,
            data: mip_data,
        }
        .encode(
            format.image_format(),
            image_dds::Quality::Normal,
            image_dds::Mipmaps::FromSurface,
        )
        .map_err(|error| format!("encode coverage mips {}: {error}", input.display()))?;
        encoded
            .to_dds()
            .map_err(|error| format!("create coverage DDS {}: {error}", input.display()))?
    } else {
        image_dds::dds_from_image(
            &image,
            format.image_format(),
            image_dds::Quality::Normal,
            image_dds::Mipmaps::GeneratedAutomatic,
        )
        .map_err(|error| format!("encode {}: {error}", input.display()))?
    };
    let mut bytes = Vec::new();
    dds.write(&mut bytes)
        .map_err(|error| format!("serialize {}: {error}", input.display()))?;
    Ok(CookedTexture {
        bytes,
        width: image.width(),
        height: image.height(),
        mip_levels: dds.get_num_mipmap_levels(),
        format,
    })
}

fn normal_mip_chain_dds(
    width: u32,
    height: u32,
    mip_levels: u32,
    mip_data: &[u8],
    format: image_dds::ImageFormat,
) -> Result<image_dds::ddsfile::Dds, String> {
    image_dds::Surface {
        width,
        height,
        depth: 1,
        layers: 1,
        mipmaps: mip_levels,
        image_format: format,
        data: mip_data,
    }
    .to_dds()
    .map_err(|error| format!("create normal-map DDS: {error}"))
}

fn build_normal_mip_chain(image: &image::RgbaImage) -> (Vec<u8>, u32) {
    let mut mip_data = Vec::with_capacity(image.as_raw().len() * 2);
    for pixel in image.pixels() {
        mip_data.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 0]);
    }
    let mut mip_levels = 1;
    let mut previous_offset = 0usize;
    let mut previous_width = image.width();
    let mut previous_height = image.height();
    while previous_width > 1 || previous_height > 1 {
        let width = (previous_width / 2).max(1);
        let height = (previous_height / 2).max(1);
        let next_offset = mip_data.len();
        for y in 0..height {
            for x in 0..width {
                let source_x = x * 2;
                let source_y = y * 2;
                let source_x1 = (source_x + 1).min(previous_width - 1);
                let source_y1 = (source_y + 1).min(previous_height - 1);
                let index =
                    |x: u32, y: u32| previous_offset + ((y * previous_width + x) * 4) as usize;
                let children = [
                    index(source_x, source_y),
                    index(source_x1, source_y),
                    index(source_x, source_y1),
                    index(source_x1, source_y1),
                ];
                let mut average = [0.0f32; 3];
                let mut child_variance = 0.0;
                for child in children {
                    for channel in 0..3 {
                        average[channel] += mip_data[child + channel] as f32 * (2.0 / 255.0) - 1.0;
                    }
                    child_variance += mip_data[child + 3] as f32 / 255.0;
                }
                for channel in &mut average {
                    *channel *= 0.25;
                }
                child_variance *= 0.25;
                let length_squared = average.iter().map(|value| value * value).sum::<f32>();
                let length = length_squared.sqrt().max(1e-6);
                let encode = |value: f32| ((value * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                mip_data.extend(average.map(|value| encode(value / length)));
                let local_variance = (1.0 - length_squared).max(0.0);
                mip_data.push(((local_variance + child_variance).min(1.0) * 255.0).round() as u8);
            }
        }
        previous_offset = next_offset;
        previous_width = width;
        previous_height = height;
        mip_levels += 1;
    }
    (mip_data, mip_levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_profiles_use_capability_neutral_rgba8() {
        let color = TextureSettings::parse(std::iter::empty()).unwrap();
        let data = TextureSettings::parse(["--linear"]).unwrap();
        let normal = TextureSettings::parse(["--normal"]).unwrap();
        let portable = AssetProfile::new("portable", "high").unwrap();
        let macos = AssetProfile::new("macos", "high").unwrap();
        assert_eq!(color.artifact_format(None), TextureArtifactFormat::Bc7Srgb);
        assert_eq!(
            color.artifact_format(Some(&portable)),
            TextureArtifactFormat::Rgba8Srgb
        );
        assert_eq!(
            data.artifact_format(Some(&portable)),
            TextureArtifactFormat::Rgba8Linear
        );
        assert_eq!(
            color.artifact_format(Some(&macos)),
            TextureArtifactFormat::Bc7Srgb
        );
        assert_eq!(
            normal.artifact_format(Some(&portable)),
            TextureArtifactFormat::Rgba8NormalVariance
        );
    }

    #[test]
    fn flat_normal_chain_keeps_direction_and_zero_variance() {
        let image = image::RgbaImage::from_pixel(2, 2, image::Rgba([128, 128, 255, 231]));
        let (mips, levels) = build_normal_mip_chain(&image);
        assert_eq!(levels, 2);
        assert_eq!(mips.len(), 20);
        for pixel in mips.chunks_exact(4) {
            assert_eq!(pixel, [128, 128, 255, 0]);
        }
    }

    #[test]
    fn disagreeing_normals_accumulate_filtered_variance() {
        let image = image::RgbaImage::from_fn(2, 2, |x, _| {
            if x == 0 {
                image::Rgba([255, 128, 128, 255])
            } else {
                image::Rgba([0, 128, 128, 255])
            }
        });
        let (mips, levels) = build_normal_mip_chain(&image);
        assert_eq!(levels, 2);
        let filtered = &mips[16..20];
        assert!(filtered[3] >= 254);
        let decoded = std::array::from_fn::<_, 3, _>(|channel| {
            filtered[channel] as f32 * (2.0 / 255.0) - 1.0
        });
        let length = decoded
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((length - 1.0).abs() < 0.01);
    }

    #[test]
    fn normal_dds_preserves_the_authored_top_mip_exactly() {
        let image = image::RgbaImage::from_fn(3, 2, |x, y| {
            image::Rgba([(x * 71) as u8, (y * 131) as u8, 207, 99])
        });
        let (mips, levels) = build_normal_mip_chain(&image);
        let dds = normal_mip_chain_dds(
            image.width(),
            image.height(),
            levels,
            &mips,
            image_dds::ImageFormat::Rgba8Unorm,
        )
        .unwrap();
        let decoded = image_dds::image_from_dds(&dds, 0).unwrap();
        for (source, candidate) in image.pixels().zip(decoded.pixels()) {
            assert_eq!(&source.0[..3], &candidate.0[..3]);
            assert_eq!(candidate[3], 0);
        }
    }

    #[test]
    fn coverage_recipe_survives_dds_serialization() {
        let mut rgba = Vec::new();
        for index in 0..64 {
            rgba.extend_from_slice(if index % 4 == 0 {
                &[255, 0, 255, 0]
            } else {
                &[40, 180, 30, 255]
            });
        }
        let settings = TextureSettings::from_semantics(false, true, Some(0.5));
        assert_eq!(settings.as_json()["alpha_coverage_reference"], 0.5);
        let prepared = PreparedTexture::from_rgba(8, 8, rgba, settings).unwrap();
        let portable = AssetProfile::new("portable", "high").unwrap();
        let cooked = cook_prepared_texture(
            Path::new("coverage-fixture"),
            &prepared,
            settings.artifact_format(Some(&portable)),
        )
        .unwrap();
        assert_eq!(cooked.mip_levels, 4);
        let dds = image_dds::ddsfile::Dds::read(std::io::Cursor::new(&cooked.bytes)).unwrap();
        let final_mip = image_dds::image_from_dds(&dds, 3).unwrap();
        assert_eq!(final_mip.dimensions(), (1, 1));
        assert!((final_mip.get_pixel(0, 0)[3] as i16 - 191).abs() <= 1);
        let pixel = final_mip.get_pixel(0, 0);
        assert!(pixel[1] > pixel[0], "transparent magenta bled into RGB");
    }
}
