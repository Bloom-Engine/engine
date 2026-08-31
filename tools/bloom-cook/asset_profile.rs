//! Canonical platform/quality identifiers for cooked asset variants.

use serde_json::{json, Value};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct AssetProfile {
    platform: String,
    quality: String,
}

impl AssetProfile {
    pub(crate) fn new(platform: &str, quality: &str) -> Result<Self, String> {
        validate_profile_component(platform, "platform")?;
        validate_profile_component(quality, "quality")?;
        Ok(Self {
            platform: platform.to_string(),
            quality: quality.to_string(),
        })
    }

    pub(crate) fn from_json(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or("asset profile is missing or not an object")?;
        if object.len() != 2 {
            return Err("asset profile has unknown or missing fields".to_string());
        }
        let platform = object
            .get("platform")
            .and_then(Value::as_str)
            .ok_or("asset profile platform is missing or not a string")?;
        let quality = object
            .get("quality")
            .and_then(Value::as_str)
            .ok_or("asset profile quality is missing or not a string")?;
        Self::new(platform, quality)
    }

    pub(crate) fn split_optional_flags(
        flags: &[String],
    ) -> Result<(Option<Self>, Vec<String>), String> {
        let mut platform = None;
        let mut quality = None;
        let mut remaining = Vec::new();
        let mut index = 0;
        while index < flags.len() {
            let flag = &flags[index];
            if flag != "--platform" && flag != "--quality" {
                remaining.push(flag.clone());
                index += 1;
                continue;
            }
            let value = flags
                .get(index + 1)
                .ok_or_else(|| format!("{flag} requires a value"))?;
            let slot = if flag == "--platform" {
                &mut platform
            } else {
                &mut quality
            };
            if slot.replace(value.clone()).is_some() {
                return Err(format!("{flag} may only be specified once"));
            }
            index += 2;
        }
        match (platform, quality) {
            (None, None) => Ok((None, remaining)),
            (Some(platform), Some(quality)) => {
                Ok((Some(Self::new(&platform, &quality)?), remaining))
            }
            _ => Err("--platform and --quality must be specified together".to_string()),
        }
    }

    pub(crate) fn platform(&self) -> &str {
        &self.platform
    }

    pub(crate) fn quality(&self) -> &str {
        &self.quality
    }

    pub(crate) fn as_json(&self) -> Value {
        json!({
            "platform": self.platform,
            "quality": self.quality,
        })
    }

    pub(crate) fn label(&self) -> String {
        format!("{}/{}", self.platform, self.quality)
    }
}

pub(crate) fn validate_profile_component(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 32
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
    {
        return Err(format!(
            "asset {label} profile {value:?} must be 1..=32 lowercase ASCII letters, digits, '-' or '_'"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_ids_and_paired_flags_are_canonical() {
        let flags = [
            "--hierarchy-levels".to_string(),
            "4".to_string(),
            "--platform".to_string(),
            "macos".to_string(),
            "--quality".to_string(),
            "high-end".to_string(),
        ];
        let (profile, remaining) = AssetProfile::split_optional_flags(&flags).unwrap();
        let profile = profile.unwrap();
        assert_eq!(profile.label(), "macos/high-end");
        assert_eq!(remaining, ["--hierarchy-levels", "4"]);

        let texture_flags = [
            "--normal".to_string(),
            "--platform".to_string(),
            "portable".to_string(),
            "--quality".to_string(),
            "high".to_string(),
            "--linear".to_string(),
        ];
        let (profile, remaining) = AssetProfile::split_optional_flags(&texture_flags).unwrap();
        assert_eq!(profile.unwrap().label(), "portable/high");
        assert_eq!(remaining, ["--normal", "--linear"]);

        assert!(AssetProfile::new("MacOS", "high").is_err());
        assert!(AssetProfile::new("../macos", "high").is_err());
        assert!(AssetProfile::split_optional_flags(&[
            "--platform".to_string(),
            "macos".to_string(),
        ])
        .is_err());
        assert!(AssetProfile::split_optional_flags(&[
            "--quality".to_string(),
            "high".to_string(),
            "--quality".to_string(),
            "medium".to_string(),
        ])
        .is_err());
    }
}
