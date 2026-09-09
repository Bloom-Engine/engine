//! Deterministic cooked-asset variant selection with caller-authored fallbacks.

use crate::asset_index::validated_index_value;
use crate::asset_profile::AssetProfile;
use crate::asset_store::validate_logical_id;
use serde_json::{json, Value};
use std::path::Path;

const RESOLUTION_SCHEMA: &str = "bloom-asset-resolution-v1";

pub(crate) fn resolve_asset_command(
    logical_id: &str,
    store: &Path,
    flags: &[String],
) -> Result<String, String> {
    validate_logical_id(logical_id)?;
    let request = ResolutionRequest::parse(flags)?;
    let index = validated_index_value(store)?;
    let report = resolve_from_index(logical_id, &index, &request)?;
    serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize asset resolution: {error}"))
}

struct ResolutionRequest {
    requested: AssetProfile,
    fallbacks: Vec<AssetProfile>,
    allow_unprofiled: bool,
}

impl ResolutionRequest {
    fn parse(flags: &[String]) -> Result<Self, String> {
        let mut platform = None;
        let mut quality = None;
        let mut fallbacks = Vec::new();
        let mut allow_unprofiled = false;
        let mut index = 0;
        while index < flags.len() {
            let flag = &flags[index];
            if flag == "--allow-unprofiled" {
                if allow_unprofiled {
                    return Err("--allow-unprofiled may only be specified once".to_string());
                }
                allow_unprofiled = true;
                index += 1;
                continue;
            }
            let value = flags
                .get(index + 1)
                .ok_or_else(|| format!("{flag} requires a value"))?;
            match flag.as_str() {
                "--platform" => set_once(&mut platform, value, flag)?,
                "--quality" => set_once(&mut quality, value, flag)?,
                "--fallback" => fallbacks.push(parse_profile_pair(value)?),
                _ => return Err(format!("unknown asset resolution option {flag:?}")),
            }
            index += 2;
        }
        let requested = match (platform, quality) {
            (Some(platform), Some(quality)) => AssetProfile::new(&platform, &quality)?,
            _ => return Err("asset resolution requires both --platform and --quality".to_string()),
        };
        let mut seen = vec![requested.clone()];
        for fallback in &fallbacks {
            if seen.contains(fallback) {
                return Err(format!(
                    "asset resolution profile {} is duplicated",
                    fallback.label()
                ));
            }
            seen.push(fallback.clone());
        }
        Ok(Self {
            requested,
            fallbacks,
            allow_unprofiled,
        })
    }
}

fn set_once(slot: &mut Option<String>, value: &str, flag: &str) -> Result<(), String> {
    if slot.replace(value.to_string()).is_some() {
        return Err(format!("{flag} may only be specified once"));
    }
    Ok(())
}

fn parse_profile_pair(value: &str) -> Result<AssetProfile, String> {
    let (platform, quality) = value
        .split_once('/')
        .ok_or_else(|| format!("--fallback requires PLATFORM/QUALITY, got {value:?}"))?;
    if quality.contains('/') {
        return Err(format!(
            "--fallback requires exactly PLATFORM/QUALITY, got {value:?}"
        ));
    }
    AssetProfile::new(platform, quality)
}

fn resolve_from_index(
    logical_id: &str,
    index: &Value,
    request: &ResolutionRequest,
) -> Result<Value, String> {
    let schema = index
        .get("schema")
        .and_then(Value::as_str)
        .ok_or("asset index schema is missing or not a string")?;
    if !matches!(schema, "bloom-asset-index-v1" | "bloom-asset-index-v2") {
        return Err(format!(
            "unsupported asset index schema {schema:?}; recook assets"
        ));
    }
    let entries = index
        .get("entries")
        .and_then(Value::as_array)
        .ok_or("asset index entries are missing or not an array")?;
    let candidates = entries
        .iter()
        .filter(|entry| entry.get("logical_id").and_then(Value::as_str) == Some(logical_id))
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(format!(
            "logical asset {logical_id:?} is not in the cooked index"
        ));
    }

    if let Some(entry) = find_profile(&candidates, &request.requested)? {
        return Ok(resolution_report(
            logical_id,
            request,
            entry,
            "exact",
            Some(&request.requested),
            None,
        ));
    }
    for (rank, fallback) in request.fallbacks.iter().enumerate() {
        if let Some(entry) = find_profile(&candidates, fallback)? {
            return Ok(resolution_report(
                logical_id,
                request,
                entry,
                "fallback",
                Some(fallback),
                Some(rank),
            ));
        }
    }
    if request.allow_unprofiled {
        if let Some(entry) = candidates
            .iter()
            .copied()
            .find(|entry| entry.get("profile").is_none())
        {
            return Ok(resolution_report(
                logical_id,
                request,
                entry,
                "unprofiled-fallback",
                None,
                None,
            ));
        }
    }

    let mut available = Vec::new();
    let mut has_unprofiled = false;
    for entry in candidates {
        match entry.get("profile") {
            Some(value) => available.push(AssetProfile::from_json(value)?.label()),
            None => has_unprofiled = true,
        }
    }
    available.sort();
    Err(format!(
        "logical asset {logical_id:?} has no allowed variant for {}; available profiles: {}; \
         unprofiled available: {has_unprofiled}",
        request.requested.label(),
        if available.is_empty() {
            "none".to_string()
        } else {
            available.join(", ")
        }
    ))
}

fn find_profile<'a>(
    candidates: &[&'a Value],
    profile: &AssetProfile,
) -> Result<Option<&'a Value>, String> {
    let mut found = None;
    for entry in candidates {
        let Some(value) = entry.get("profile") else {
            continue;
        };
        if AssetProfile::from_json(value)? == *profile && found.replace(*entry).is_some() {
            return Err(format!(
                "asset index contains duplicate {} variants",
                profile.label()
            ));
        }
    }
    Ok(found)
}

fn resolution_report(
    logical_id: &str,
    request: &ResolutionRequest,
    entry: &Value,
    kind: &str,
    selected_profile: Option<&AssetProfile>,
    fallback_rank: Option<usize>,
) -> Value {
    let mut selection = json!({
        "kind": kind,
        "profile": selected_profile.map(AssetProfile::as_json),
    });
    if let Some(rank) = fallback_rank {
        selection["fallback_rank"] = json!(rank);
    }
    json!({
        "artifact": entry["artifact"].clone(),
        "build_key_sha256": entry["build_key_sha256"].clone(),
        "logical_id": logical_id,
        "manifest": entry["manifest"].clone(),
        "request": {
            "allow_unprofiled": request.allow_unprofiled,
            "fallbacks": request.fallbacks.iter().map(AssetProfile::as_json).collect::<Vec<_>>(),
            "profile": request.requested.as_json(),
        },
        "schema": RESOLUTION_SCHEMA,
        "selection": selection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_parser_requires_explicit_canonical_fallbacks() {
        let request = ResolutionRequest::parse(&[
            "--platform".to_string(),
            "windows".to_string(),
            "--quality".to_string(),
            "ultra".to_string(),
            "--fallback".to_string(),
            "portable/high".to_string(),
        ])
        .unwrap();
        assert_eq!(request.requested.label(), "windows/ultra");
        assert_eq!(request.fallbacks[0].label(), "portable/high");
        assert!(!request.allow_unprofiled);

        assert!(ResolutionRequest::parse(&["--platform".to_string(), "web".to_string()]).is_err());
        assert!(ResolutionRequest::parse(&[
            "--platform".to_string(),
            "web".to_string(),
            "--quality".to_string(),
            "high".to_string(),
            "--fallback".to_string(),
            "web/high".to_string(),
        ])
        .is_err());
    }
}
