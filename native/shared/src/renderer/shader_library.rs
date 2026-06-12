// Baked shader library — the release-build source of truth for every
// Bloom-owned WGSL file under native/shared/shaders/.
//
// Files live on disk so humans can read them, hot-reload can watch them,
// and diffs show real changes. For release builds we `include_str!`
// them here so the engine has zero filesystem dependencies at runtime —
// every shader is embedded in the binary.
//
// Adding a new shader: (1) drop the .wgsl file under native/shared/
// shaders/, (2) add a line below. The path string is what `#include`
// directives reference.

use super::shader_include::{BakedSource, ShaderSource};

const ENTRIES: &[(&str, &str)] = &[
    ("material_abi.wgsl",           include_str!("../../../shared/shaders/material_abi.wgsl")),
    ("common/pbr.wgsl",             include_str!("../../../shared/shaders/common/pbr.wgsl")),
    ("common/shadows.wgsl",         include_str!("../../../shared/shaders/common/shadows.wgsl")),
    ("common/imposter.wgsl",        include_str!("../../../shared/shaders/common/imposter.wgsl")),
    ("common/fog.wgsl",             include_str!("../../../shared/shaders/common/fog.wgsl")),
    ("common/tonemap.wgsl",         include_str!("../../../shared/shaders/common/tonemap.wgsl")),
    ("common/sky.wgsl",             include_str!("../../../shared/shaders/common/sky.wgsl")),
    ("materials/test_minimal.wgsl", include_str!("../../../shared/shaders/materials/test_minimal.wgsl")),
    ("impulse_field.wgsl",          include_str!("../../../shared/shaders/impulse_field.wgsl")),
];

/// The single shared source resolver for built-in shaders. Phase 1
/// preps it but no pipelines consume it yet — that's Phase 1b.
pub fn library() -> impl ShaderSource {
    BakedSource { entries: ENTRIES }
}

/// Self-check: the ABI header parses to a known version. Called from
/// Renderer::new so a bad merge is caught at startup, not at the first
/// draw call.
pub fn verify_abi_version(expected: u32) -> Result<(), String> {
    let src = library();
    let body = src.fetch("material_abi.wgsl")
        .ok_or_else(|| "material_abi.wgsl missing from shader library".to_string())?;
    let actual = super::shader_include::abi_version_of(body);
    if actual != expected {
        return Err(format!(
            "shader ABI version mismatch: header declares {}, engine expects {}",
            actual, expected
        ));
    }
    Ok(())
}

/// The version the engine was built against. EN-012 bumped this to 2
/// when MaterialFactors gained `shading_model` + `foliage_params`
/// (foliage shading model). Bump together with the
/// `ABI-VERSION:` comment in `shaders/material_abi.wgsl`.
pub const EXPECTED_ABI_VERSION: u32 = 3;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::shader_include::process;

    #[test]
    fn abi_header_present_and_versioned() {
        verify_abi_version(EXPECTED_ABI_VERSION).unwrap();
    }

    #[test]
    fn abi_header_includes_through_preprocessor() {
        // A synthetic shader that pulls in the ABI header plus one
        // common helper — proves the include machinery wires to the
        // baked library end-to-end.
        let synthetic = BakedSource {
            entries: &[
                (
                    "test.wgsl",
                    "#include \"material_abi.wgsl\"\n\
                     #include \"common/pbr.wgsl\"\n\
                     // synthesised\n",
                ),
                ENTRIES[0],
                ENTRIES[1],
                ENTRIES[2],
                ENTRIES[3],
                ENTRIES[4],
                ENTRIES[5],
            ],
        };
        let out = process(&synthetic, "test.wgsl").unwrap();
        assert!(out.contains("struct PerFrame"));
        assert!(out.contains("fn d_ggx"));
        assert!(out.contains("synthesised"));
    }

    #[test]
    fn every_common_file_appears_in_entries() {
        for (path, body) in ENTRIES {
            assert!(!body.is_empty(), "{} should not be empty", path);
        }
    }

    // EN-015 V1 — imposter helper library is registered and includes
    // cleanly through the preprocessor.

    #[test]
    fn imposter_helper_in_library() {
        let src = library();
        let body = src.fetch("common/imposter.wgsl")
            .expect("common/imposter.wgsl must be registered in ENTRIES");
        assert!(body.contains("fn octahedral_encode"));
        assert!(body.contains("fn imposter_atlas_uv"));
        assert!(body.contains("fn billboard_quad"));
    }

    #[test]
    fn imposter_helper_includes_through_preprocessor() {
        // Synthetic shader that pulls in the imposter helper, proving
        // the include resolves end-to-end against the baked library.
        let lib = library();
        let imposter_src = lib.fetch("common/imposter.wgsl").unwrap();
        let synthetic = BakedSource {
            entries: &[
                (
                    "test_imposter.wgsl",
                    "#include \"common/imposter.wgsl\"\n\
                     // synthesised\n",
                ),
                ("common/imposter.wgsl", imposter_src),
            ],
        };
        let out = process(&synthetic, "test_imposter.wgsl").unwrap();
        assert!(out.contains("fn octahedral_encode"));
        assert!(out.contains("fn imposter_atlas_uv"));
        assert!(out.contains("synthesised"));
    }
}
