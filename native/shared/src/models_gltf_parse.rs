//! Parsing a glTF document for the loaders: the gltf crate's validation, with
//! the one case it gets wrong let through, and every rejection reported.
//!
//! glTF 2.0 (§5.1, accessor.bufferView): "When undefined, the accessor MUST be
//! initialized with zeros". Blender writes such accessors for shape keys that
//! move nothing on a primitive (`export_try_omit_sparse_sk`), and the gltf
//! crate's validator (1.4) calls them missing data, so a face rig with a
//! target that leaves the teeth alone failed to load at all, silently. When
//! every validation error is of that kind the document is taken without
//! validation; the crate yields no reader for such an accessor, which every
//! loader already treats as "nothing" (zero deltas). Anything else is still
//! rejected, now with a line saying why.

/// A GLB's JSON chunk and binary chunk, both borrowed from `data`; `None`
/// when `data` is not a GLB (a glTF JSON file, say).
pub(super) fn split_glb(data: &[u8]) -> Option<(&[u8], Option<&[u8]>)> {
    let word = |at: usize| -> Option<usize> {
        Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?) as usize)
    };
    if data.get(0..4)? != b"glTF" || data.get(16..20)? != b"JSON" {
        return None;
    }
    let json_end = 20usize.checked_add(word(12)?)?;
    let json = data.get(20..json_end)?;
    let bin = match data.get(json_end + 4..json_end + 8) {
        Some(b"BIN\0") => {
            let start = json_end + 8;
            Some(data.get(start..start.checked_add(word(json_end)?)?)?)
        }
        _ => None,
    };
    Some((json, bin))
}

/// The document, and for a GLB its binary chunk borrowed from `data`, not
/// copied (the gltf crate copies it: a model's whole file, read again). With
/// a memory-mapped file the texture levels nobody uploads are then never read.
pub(super) fn parse_borrowed(data: &[u8]) -> Option<(gltf::Gltf, Option<&[u8]>)> {
    match split_glb(data) {
        Some((json, bin)) => Some((parse_gltf(json)?, bin)),
        None => Some((parse_gltf(data)?, None)),
    }
}

pub(super) fn parse_gltf(data: &[u8]) -> Option<gltf::Gltf> {
    match gltf::Gltf::from_slice(data) {
        Ok(document) => Some(document),
        Err(gltf::Error::Validation(errors)) if only_zero_accessors(&errors) => {
            match gltf::Gltf::from_slice_without_validation(data) {
                Ok(document) => Some(document),
                Err(error) => {
                    eprintln!("bloom: glTF rejected: {error}");
                    None
                }
            }
        }
        Err(error) => {
            eprintln!("bloom: glTF rejected: {error}");
            None
        }
    }
}

/// True when every error is an accessor without a bufferView (a zero accessor).
fn only_zero_accessors(errors: &[(gltf::json::Path, gltf::json::validation::Error)]) -> bool {
    !errors.is_empty()
        && errors.iter().all(|(path, error)| {
            matches!(error, gltf::json::validation::Error::Missing)
                && path.as_str().starts_with("accessors[")
                && path.as_str().ends_with("].bufferView")
        })
}

#[cfg(test)]
mod tests {
    use super::parse_gltf;

    const ZERO_ACCESSOR: &str = r#"{"asset":{"version":"2.0"},
        "accessors":[{"componentType":5126,"count":3,"type":"VEC3"}]}"#;

    #[test]
    fn a_zero_accessor_is_accepted() {
        assert!(gltf::Gltf::from_slice(ZERO_ACCESSOR.as_bytes()).is_err());
        let document = parse_gltf(ZERO_ACCESSOR.as_bytes()).expect("zero accessor parses");
        let accessor = document.accessors().next().unwrap();
        assert!(accessor.view().is_none() && accessor.sparse().is_none());
    }

    #[test]
    fn other_validation_errors_still_reject() {
        // a zero accessor next to one with an invalid component type: the second must still reject
        let bad = r#"{"asset":{"version":"2.0"},
            "accessors":[{"componentType":5126,"count":3,"type":"VEC3"},
                         {"bufferView":0,"componentType":9999,"count":3,"type":"VEC3"}],
            "bufferViews":[{"buffer":0,"byteLength":36}],"buffers":[{"byteLength":36}]}"#;
        assert!(parse_gltf(bad.as_bytes()).is_none());
    }
}
