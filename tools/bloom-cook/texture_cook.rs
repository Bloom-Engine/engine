//! Deterministic texture preparation shared by direct and store cooking.

use crate::geometry_format::sha256;
use serde_json::{json, Value};
use std::path::Path;

pub(crate) const TEXTURE_RECIPE_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TextureSettings {
    normal_map: bool,
    linear: bool,
}

impl TextureSettings {
    pub(crate) fn parse<'a>(flags: impl IntoIterator<Item = &'a str>) -> Result<Self, String> {
        let mut normal_map = false;
        let mut linear = false;
        for flag in flags {
            let slot = match flag {
                "--normal" => &mut normal_map,
                "--linear" => &mut linear,
                _ => return Err(format!("unknown texture option {flag:?}")),
            };
            if *slot {
                return Err(format!("{flag} may only be specified once"));
            }
            *slot = true;
        }
        if normal_map {
            linear = true;
        }
        Ok(Self { normal_map, linear })
    }

    pub(crate) fn from_manifest(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or("asset manifest texture settings are missing or not an object")?;
        if object.len() != 2 {
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
        let linear = match color_space {
            "linear" => true,
            "srgb" => false,
            other => return Err(format!("asset manifest has unknown color space {other:?}")),
        };
        if normal_map && !linear {
            return Err("normal-map texture manifests must use linear color space".to_string());
        }
        Ok(Self { normal_map, linear })
    }

    pub(crate) fn as_json(self) -> Value {
        json!({
            "color_space": if self.linear { "linear" } else { "srgb" },
            "normal_map": self.normal_map,
        })
    }

    pub(crate) fn format_name(self) -> &'static str {
        if self.normal_map {
            "rgba8-unorm-normal-variance"
        } else if self.linear {
            "bc7-rgba-unorm"
        } else {
            "bc7-rgba-unorm-srgb"
        }
    }

    pub(crate) fn is_normal_map(self) -> bool {
        self.normal_map
    }

    pub(crate) fn is_srgb(self) -> bool {
        !self.linear
    }

    fn image_format(self) -> image_dds::ImageFormat {
        if self.normal_map {
            image_dds::ImageFormat::Rgba8Unorm
        } else if self.linear {
            image_dds::ImageFormat::BC7RgbaUnorm
        } else {
            image_dds::ImageFormat::BC7RgbaUnormSrgb
        }
    }

    pub(crate) fn build_key_sha256(self, source_sha256: [u8; 32]) -> [u8; 32] {
        let mut bytes = Vec::with_capacity(80);
        bytes.extend_from_slice(b"bloom-texture-recipe\0");
        bytes.extend_from_slice(&TEXTURE_RECIPE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&source_sha256);
        bytes.push(u8::from(self.normal_map));
        bytes.push(u8::from(self.linear));
        sha256(&bytes)
    }
}

pub(crate) struct PreparedTexture {
    pub(crate) source_bytes: Vec<u8>,
    pub(crate) source_sha256: [u8; 32],
    pub(crate) settings: TextureSettings,
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
        })
    }
}

pub(crate) struct CookedTexture {
    pub(crate) bytes: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mip_levels: u32,
}

pub(crate) fn cook_prepared_texture(
    input: &Path,
    prepared: &PreparedTexture,
) -> Result<CookedTexture, String> {
    let image = image::load_from_memory(&prepared.source_bytes)
        .map_err(|error| format!("decode {}: {error}", input.display()))?
        .to_rgba8();
    let dds = if prepared.settings.is_normal_map() {
        let (mip_data, mip_levels) = build_normal_mip_chain(&image);
        normal_mip_chain_dds(
            image.width(),
            image.height(),
            mip_levels,
            &mip_data,
            prepared.settings.image_format(),
        )
        .map_err(|error| format!("encode {}: {error}", input.display()))?
    } else {
        image_dds::dds_from_image(
            &image,
            prepared.settings.image_format(),
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
}
