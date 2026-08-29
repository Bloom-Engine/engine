//! Deterministic BC7 texture preparation shared by direct and store cooking.

use crate::geometry_format::sha256;
use serde_json::{json, Value};
use std::path::Path;

pub(crate) const TEXTURE_RECIPE_VERSION: u32 = 1;

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
        if self.linear {
            "bc7-rgba-unorm"
        } else {
            "bc7-rgba-unorm-srgb"
        }
    }

    fn image_format(self) -> image_dds::ImageFormat {
        if self.linear {
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
    let dds = image_dds::dds_from_image(
        &image,
        prepared.settings.image_format(),
        image_dds::Quality::Normal,
        image_dds::Mipmaps::GeneratedAutomatic,
    )
    .map_err(|error| format!("encode {}: {error}", input.display()))?;
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
