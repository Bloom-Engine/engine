//! Bloom reference CPU path tracer — Phase 3.
//!
//! What Phase 3 adds over Phase 2:
//!   - Normal maps. glTF files embed per-surface normal perturbations
//!     that carry the vast majority of visible surface detail. Without
//!     them, the helmet is a smooth blob; with them, every panel line,
//!     bolt, and seam shows up.
//!   - HDR environment map IBL. A real Radiance .hdr image replaces
//!     the procedural sky — metallic surfaces now reflect a real
//!     environment and the overall lighting is color-correct.
//!   - Next event estimation (NEE) for both the environment and
//!     explicit directional lights, with multiple importance sampling
//!     (MIS balance heuristic) against the BRDF sampler. Cuts the
//!     noise in direct lighting by an order of magnitude at the same
//!     spp count.
//!
//! Usage:
//!   bloom-reference --scene path/to/model.glb --out ref.png
//!                   [--env path/to/env.hdr] [--sun-dir X Y Z]
//!                   [--width 512] [--height 512] [--spp 64] [--bounces 4]

use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use rayon::prelude::*;
use serde::Deserialize;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

// ============================================================
// Shared view spec — JSON format read by BOTH the reference and the
// Bloom realtime renderer so comparisons are apples-to-apples.
// ============================================================

#[derive(Deserialize, Debug)]
struct ViewSpec {
    scene: String,
    resolution: [u32; 2],
    camera: CameraSpec,
    env: Option<EnvSpec>,
    sun: Option<SunSpec>,
    reference_defaults: Option<ReferenceDefaults>,
    // (realtime_defaults is read by the TS side, ignored here)
}

#[derive(Deserialize, Debug)]
struct CameraSpec {
    position: [f32; 3],
    target: [f32; 3],
    up: [f32; 3],
    fov_y_deg: f32,
}

#[derive(Deserialize, Debug)]
struct EnvSpec {
    path: String,
    intensity: f32,
}

#[derive(Deserialize, Debug)]
struct SunSpec {
    direction: [f32; 3],
    color: Option<[f32; 3]>,
    intensity: f32,
}

#[derive(Deserialize, Debug)]
struct ReferenceDefaults {
    spp: Option<u32>,
    bounces: Option<u32>,
}

/// Load and parse the JSON spec. Relative paths in the spec
/// (`scene`, `env.path`) are resolved against the spec file's own
/// directory so the spec can live alongside its assets and be moved
/// as a unit.
fn load_spec(path: &Path) -> Result<(ViewSpec, PathBuf), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {:?}: {e}", path))?;
    let spec: ViewSpec = serde_json::from_str(&text).map_err(|e| format!("parse {:?}: {e}", path))?;
    let base_dir = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    Ok((spec, base_dir))
}

// ============================================================
// Scene representation
// ============================================================

/// World-space triangle with per-vertex normals, tangents, and UVs.
/// Tangents are stored as Vec4 because glTF encodes the bitangent sign
/// in the W component — xyz is the tangent vector (world space after
/// the node transform), w is ±1 indicating which way the bitangent
/// points. Without this we'd have mirrored normal maps on any mesh
/// whose UVs flipped handedness.
#[derive(Clone)]
struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
    t0: Vec4,
    t1: Vec4,
    t2: Vec4,
    uv0: Vec2,
    uv1: Vec2,
    uv2: Vec2,
    material_index: u32,
}

/// Bilinear-sampled 8-bit RGBA image. Decoded from the glTF image block
/// at load time. Using RGBA8 rather than linear f32 keeps memory down;
/// we convert to linear on sample.
struct Texture {
    pixels: Vec<u8>, // RGBA8, row-major
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug)]
struct Material {
    /// Scalar multiplier applied on top of the base color texture (or
    /// used directly when no texture is present).
    base_color_factor: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive_factor: [f32; 3],
    base_color_texture: Option<u32>,
    /// glTF MR texture: green = roughness, blue = metallic, scaled by
    /// the factor fields above. Sampled per-hit so materials that mix
    /// metal/nonmetal across the surface (e.g. DamagedHelmet's visor
    /// vs. rubber straps) look right.
    metallic_roughness_texture: Option<u32>,
    emissive_texture: Option<u32>,
    /// Normal map in tangent space. Unpacked from [0,1]³ to [-1,1]³
    /// and transformed via the per-vertex TBN before use. The
    /// `normal_scale` factor from glTF lets meshes dial the effect
    /// strength down without re-authoring the texture.
    normal_texture: Option<u32>,
    normal_scale: f32,
    /// Occlusion (AO) texture — R channel is an occlusion multiplier
    /// in [0,1]. glTF's strength factor blends between "no effect" (0)
    /// and "full effect" (1). Applied to indirect lighting only
    /// (direct lights should produce their own AO via shadow rays).
    occlusion_texture: Option<u32>,
    occlusion_strength: f32,
}

impl Material {
    fn default_material() -> Self {
        Self {
            base_color_factor: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            emissive_factor: [0.0, 0.0, 0.0],
            base_color_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
        }
    }
}

struct Scene {
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
    bbox_min: Vec3,
    bbox_max: Vec3,
}

impl Scene {
    fn bbox_center(&self) -> Vec3 {
        (self.bbox_min + self.bbox_max) * 0.5
    }
    fn bbox_radius(&self) -> f32 {
        (self.bbox_max - self.bbox_min).length() * 0.5
    }

    /// Sample a texture with clamp-to-edge wrapping and bilinear
    /// filtering. Converts sRGB-encoded base-color data back to linear
    /// because all lighting math happens in linear space.
    fn sample_base_color(&self, material: &Material, uv: Vec2) -> Vec3 {
        let factor = Vec3::new(
            material.base_color_factor[0],
            material.base_color_factor[1],
            material.base_color_factor[2],
        );
        let tex_idx = match material.base_color_texture {
            Some(i) => i as usize,
            None => return factor,
        };
        let tex = match self.textures.get(tex_idx) {
            Some(t) => t,
            None => return factor,
        };

        // Repeat-wrap on U/V — glTF default wrap mode. Clamp-to-edge
        // would also be acceptable for most inputs; REPEAT matches what
        // the realtime sampler will use.
        let u = uv.x - uv.x.floor();
        let v = uv.y - uv.y.floor();

        // Bilinear tap. Sampling in the center of each texel.
        let fx = u * tex.width as f32 - 0.5;
        let fy = v * tex.height as f32 - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let w = tex.width as i32;
        let h = tex.height as i32;
        let clamp = |c: i32, max: i32| c.rem_euclid(max);

        let fetch = |x: i32, y: i32| -> Vec3 {
            let xc = clamp(x, w) as usize;
            let yc = clamp(y, h) as usize;
            let idx = (yc * tex.width as usize + xc) * 4;
            let r = srgb_u8_to_linear(tex.pixels[idx]);
            let g = srgb_u8_to_linear(tex.pixels[idx + 1]);
            let b = srgb_u8_to_linear(tex.pixels[idx + 2]);
            Vec3::new(r, g, b)
        };

        let c00 = fetch(x0, y0);
        let c10 = fetch(x0 + 1, y0);
        let c01 = fetch(x0, y0 + 1);
        let c11 = fetch(x0 + 1, y0 + 1);

        let color = c00.lerp(c10, tx).lerp(c01.lerp(c11, tx), ty);
        color * factor
    }

    /// Sample metallic-roughness texture. The glTF convention packs
    /// occlusion in R, roughness in G, metallic in B; values are
    /// LINEAR (no sRGB decode). When no texture is present, falls back
    /// to the material's scalar factors alone.
    fn sample_metallic_roughness(&self, material: &Material, uv: Vec2) -> (f32, f32) {
        let (metallic_factor, roughness_factor) = (material.metallic, material.roughness);
        let tex_idx = match material.metallic_roughness_texture {
            Some(i) => i as usize,
            None => return (metallic_factor, roughness_factor),
        };
        let (r, g, b) = match self.sample_texture_linear(tex_idx, uv) {
            Some(c) => c,
            None => return (metallic_factor, roughness_factor),
        };
        let _ = r; // occlusion — ignored here, sampled separately if needed
        let roughness = g * roughness_factor;
        let metallic = b * metallic_factor;
        (metallic, roughness)
    }

    /// Sample emissive texture × factor. Emissive textures are sRGB-
    /// encoded per the glTF spec.
    fn sample_emissive(&self, material: &Material, uv: Vec2) -> Vec3 {
        let factor = Vec3::from(material.emissive_factor);
        let tex_idx = match material.emissive_texture {
            Some(i) => i as usize,
            None => {
                // No texture: just the factor. But factor=[1,1,1]
                // without a texture is a common glTF default meaning
                // "use the texture if present, otherwise no emission".
                // Treat as non-emissive when there's no texture.
                return Vec3::ZERO;
            }
        };
        match self.sample_texture_srgb(tex_idx, uv) {
            Some(c) => Vec3::new(c.0, c.1, c.2) * factor,
            None => Vec3::ZERO,
        }
    }

    /// Bilinear sRGB-decoded texture sample at UV.
    fn sample_texture_srgb(&self, tex_idx: usize, uv: Vec2) -> Option<(f32, f32, f32)> {
        let tex = self.textures.get(tex_idx)?;
        let (x0, y0, tx, ty) = tex_sample_coords(tex, uv);
        let w = tex.width as i32;
        let h = tex.height as i32;
        let fetch = |x: i32, y: i32| -> (f32, f32, f32) {
            let idx = ((y.rem_euclid(h) as usize) * tex.width as usize
                + (x.rem_euclid(w) as usize))
                * 4;
            (
                srgb_u8_to_linear(tex.pixels[idx]),
                srgb_u8_to_linear(tex.pixels[idx + 1]),
                srgb_u8_to_linear(tex.pixels[idx + 2]),
            )
        };
        let c00 = fetch(x0, y0);
        let c10 = fetch(x0 + 1, y0);
        let c01 = fetch(x0, y0 + 1);
        let c11 = fetch(x0 + 1, y0 + 1);
        let r = lerp(lerp(c00.0, c10.0, tx), lerp(c01.0, c11.0, tx), ty);
        let g = lerp(lerp(c00.1, c10.1, tx), lerp(c01.1, c11.1, tx), ty);
        let b = lerp(lerp(c00.2, c10.2, tx), lerp(c01.2, c11.2, tx), ty);
        Some((r, g, b))
    }

    /// Sample occlusion in [0,1]. glTF packs occlusion in R of the
    /// occlusion texture; strength blends between "no AO" (0) and
    /// "full AO" (1). Returns 1.0 (no attenuation) when absent.
    fn sample_occlusion(&self, material: &Material, uv: Vec2) -> f32 {
        let tex_idx = match material.occlusion_texture {
            Some(i) => i as usize,
            None => return 1.0,
        };
        let (r, _g, _b) = match self.sample_texture_linear(tex_idx, uv) {
            Some(c) => c,
            None => return 1.0,
        };
        // mix(1.0, r, strength)
        1.0 + (r - 1.0) * material.occlusion_strength
    }

    /// Sample a tangent-space normal from the normal texture, in
    /// [-1,1]³. Returns (0,0,1) (i.e. "no perturbation") when the
    /// material has no normal map so callers can uniformly apply the
    /// TBN transform without a special case.
    fn sample_tangent_normal(&self, material: &Material, uv: Vec2) -> Vec3 {
        let tex_idx = match material.normal_texture {
            Some(i) => i as usize,
            None => return Vec3::new(0.0, 0.0, 1.0),
        };
        let (r, g, b) = match self.sample_texture_linear(tex_idx, uv) {
            Some(c) => c,
            None => return Vec3::new(0.0, 0.0, 1.0),
        };
        // glTF stores normal maps with Y-up in tangent space: the
        // sampled RGB maps linearly from [0,1] to [-1,1]. Apply the
        // material's normal_scale to X/Y (Z is derived).
        let nx = (r * 2.0 - 1.0) * material.normal_scale;
        let ny = (g * 2.0 - 1.0) * material.normal_scale;
        let nz = b * 2.0 - 1.0;
        Vec3::new(nx, ny, nz).normalize_or_zero()
    }

    /// Bilinear linear-space texture sample (no sRGB decode). Used
    /// for MR / normal / occlusion textures per glTF spec.
    fn sample_texture_linear(&self, tex_idx: usize, uv: Vec2) -> Option<(f32, f32, f32)> {
        let tex = self.textures.get(tex_idx)?;
        let (x0, y0, tx, ty) = tex_sample_coords(tex, uv);
        let w = tex.width as i32;
        let h = tex.height as i32;
        let fetch = |x: i32, y: i32| -> (f32, f32, f32) {
            let idx = ((y.rem_euclid(h) as usize) * tex.width as usize
                + (x.rem_euclid(w) as usize))
                * 4;
            let r = tex.pixels[idx] as f32 / 255.0;
            let g = tex.pixels[idx + 1] as f32 / 255.0;
            let b = tex.pixels[idx + 2] as f32 / 255.0;
            (r, g, b)
        };
        let c00 = fetch(x0, y0);
        let c10 = fetch(x0 + 1, y0);
        let c01 = fetch(x0, y0 + 1);
        let c11 = fetch(x0 + 1, y0 + 1);
        let r = lerp(lerp(c00.0, c10.0, tx), lerp(c01.0, c11.0, tx), ty);
        let g = lerp(lerp(c00.1, c10.1, tx), lerp(c01.1, c11.1, tx), ty);
        let b = lerp(lerp(c00.2, c10.2, tx), lerp(c01.2, c11.2, tx), ty);
        Some((r, g, b))
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Shared bilinear coord computation so every texture sampler does
/// the same wrapping and interpolation math.
fn tex_sample_coords(tex: &Texture, uv: Vec2) -> (i32, i32, f32, f32) {
    let u = uv.x - uv.x.floor();
    let v = uv.y - uv.y.floor();
    let fx = u * tex.width as f32 - 0.5;
    let fy = v * tex.height as f32 - 0.5;
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    (x0, y0, tx, ty)
}

fn srgb_u8_to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

// ============================================================
// glTF loading
// ============================================================

fn load_scene(path: &Path) -> Result<Scene, String> {
    let (document, buffers, images) =
        gltf::import(path).map_err(|e| format!("gltf import failed: {e}"))?;

    // Pre-decode all images. glTF's `images` array is parallel to the
    // `textures` array; each `Texture` references one image by index.
    // We store the decoded RGBA8 buffers and let `textures` become the
    // lookup index used by materials.
    let decoded_images: Vec<Texture> = images
        .into_iter()
        .map(|img| {
            // `gltf::image::Data` gives us pre-decoded pixels in the
            // format described by `img.format`. Some glTFs ship as
            // R8G8B8 without alpha; pad to RGBA8 so the sampler only
            // has one pixel format to handle.
            let pixels = match img.format {
                gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
                gltf::image::Format::R8G8B8 => {
                    let mut p = Vec::with_capacity((img.width * img.height * 4) as usize);
                    for rgb in img.pixels.chunks_exact(3) {
                        p.push(rgb[0]);
                        p.push(rgb[1]);
                        p.push(rgb[2]);
                        p.push(255);
                    }
                    p
                }
                // Other formats (R8, R8G8, and 16-bit variants) are not
                // used by the glTF samples we care about yet. Fall back
                // to white so a bad texture doesn't nuke the whole
                // render — we'd rather see the model.
                _ => vec![255u8; (img.width * img.height * 4) as usize],
            };
            Texture {
                pixels,
                width: img.width,
                height: img.height,
            }
        })
        .collect();

    // `textures` in glTF is an array of (image, sampler) pairs. We flatten
    // to just "image index this texture uses" — samplers aren't honored
    // yet (Phase 3+ concern).
    let texture_to_image: Vec<u32> = document
        .textures()
        .map(|t| t.source().index() as u32)
        .collect();

    let mut materials: Vec<Material> = document
        .materials()
        .map(|m| {
            let pbr = m.pbr_metallic_roughness();
            let base_color_texture = pbr.base_color_texture().and_then(|info| {
                let tex_idx = info.texture().index();
                texture_to_image.get(tex_idx).copied()
            });
            let metallic_roughness_texture =
                pbr.metallic_roughness_texture().and_then(|info| {
                    let tex_idx = info.texture().index();
                    texture_to_image.get(tex_idx).copied()
                });
            let emissive_texture = m.emissive_texture().and_then(|info| {
                let tex_idx = info.texture().index();
                texture_to_image.get(tex_idx).copied()
            });
            let (normal_texture, normal_scale) = match m.normal_texture() {
                Some(info) => {
                    let tex_idx = info.texture().index();
                    (texture_to_image.get(tex_idx).copied(), info.scale())
                }
                None => (None, 1.0),
            };
            let (occlusion_texture, occlusion_strength) = match m.occlusion_texture() {
                Some(info) => {
                    let tex_idx = info.texture().index();
                    (texture_to_image.get(tex_idx).copied(), info.strength())
                }
                None => (None, 1.0),
            };
            let emissive = m.emissive_factor();
            Material {
                base_color_factor: pbr.base_color_factor(),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                emissive_factor: emissive,
                base_color_texture,
                metallic_roughness_texture,
                emissive_texture,
                normal_texture,
                normal_scale,
                occlusion_texture,
                occlusion_strength,
            }
        })
        .collect();
    if materials.is_empty() {
        materials.push(Material::default_material());
    }
    let default_material_index = (materials.len() - 1) as u32;

    let mut triangles: Vec<Triangle> = Vec::new();
    let mut bbox_min = Vec3::splat(f32::INFINITY);
    let mut bbox_max = Vec3::splat(f32::NEG_INFINITY);

    for scene in document.scenes() {
        for node in scene.nodes() {
            walk_node(
                &node,
                Mat4::IDENTITY,
                &buffers,
                &materials,
                default_material_index,
                &mut triangles,
                &mut bbox_min,
                &mut bbox_max,
            );
        }
    }

    if triangles.is_empty() {
        return Err("scene contained no triangles".to_string());
    }

    Ok(Scene {
        triangles,
        materials,
        textures: decoded_images,
        bbox_min,
        bbox_max,
    })
}

fn walk_node(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    materials: &[Material],
    default_material_index: u32,
    triangles: &mut Vec<Triangle>,
    bbox_min: &mut Vec3,
    bbox_max: &mut Vec3,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_transform * local;
    let world_normal = world.inverse().transpose();

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let texcoords: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
            // glTF tangents: Vec4 (xyz = tangent, w = ±1 bitangent sign).
            // Required for normal mapping; we zero them out for meshes
            // without tangents, and the shader falls back to geometric
            // normals when the length is zero.
            let tangents: Vec<[f32; 4]> = reader
                .read_tangents()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 0.0, 0.0, 0.0]; positions.len()]);
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            let material_index = primitive
                .material()
                .index()
                .map(|i| i as u32)
                .unwrap_or(default_material_index);
            if material_index as usize >= materials.len() {
                continue;
            }

            for tri in indices.chunks_exact(3) {
                let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                if i0.max(i1).max(i2) >= positions.len() {
                    continue;
                }

                let v0 = world.transform_point3(Vec3::from(positions[i0]));
                let v1 = world.transform_point3(Vec3::from(positions[i1]));
                let v2 = world.transform_point3(Vec3::from(positions[i2]));
                let n0 = world_normal.transform_vector3(Vec3::from(normals[i0]));
                let n1 = world_normal.transform_vector3(Vec3::from(normals[i1]));
                let n2 = world_normal.transform_vector3(Vec3::from(normals[i2]));
                let uv0 = Vec2::from(texcoords[i0]);
                let uv1 = Vec2::from(texcoords[i1]);
                let uv2 = Vec2::from(texcoords[i2]);
                // Tangents transform like positions (direction, not
                // pseudo-vector) — use the world matrix, not its
                // inverse-transpose. The sign stays in .w untouched.
                let transform_tangent = |t: [f32; 4]| -> Vec4 {
                    let v = world.transform_vector3(Vec3::new(t[0], t[1], t[2]));
                    Vec4::new(v.x, v.y, v.z, t[3])
                };
                let t0 = transform_tangent(tangents[i0]);
                let t1 = transform_tangent(tangents[i1]);
                let t2 = transform_tangent(tangents[i2]);

                *bbox_min = bbox_min.min(v0).min(v1).min(v2);
                *bbox_max = bbox_max.max(v0).max(v1).max(v2);

                triangles.push(Triangle {
                    v0,
                    v1,
                    v2,
                    n0,
                    n1,
                    n2,
                    t0,
                    t1,
                    t2,
                    uv0,
                    uv1,
                    uv2,
                    material_index,
                });
            }
        }
    }

    for child in node.children() {
        walk_node(
            &child,
            world,
            buffers,
            materials,
            default_material_index,
            triangles,
            bbox_min,
            bbox_max,
        );
    }
}

// Path-tracing core (ray/BVH/camera/RNG/BRDF/environment/lights/
// integrator) lives in tracer.rs (2000-line file policy).
mod tracer;
use tracer::*;

// ============================================================
// Tone mapping
// ============================================================

fn tonemap_aces(color: Vec3) -> Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let mapped = (color * (color * a + b)) / (color * (color * c + d) + Vec3::splat(e));
    mapped.clamp(Vec3::ZERO, Vec3::ONE)
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

// ============================================================
// Render loop
// ============================================================

struct RenderOptions {
    width: u32,
    height: u32,
    spp: u32,
    max_bounces: u32,
    seed: u64,
}

fn render(
    scene: &Scene,
    bvh: &Bvh,
    environment: &Environment,
    sun: Option<SunLight>,
    camera: &Camera,
    opts: &RenderOptions,
) -> Vec<u8> {
    let w = opts.width as usize;
    let h = opts.height as usize;
    let mut pixels = vec![0u8; w * h * 3];
    let image_size = UVec2::new(opts.width, opts.height);

    let scenario = Scenario {
        scene,
        bvh,
        environment,
        sun,
    };

    pixels
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let pixel = UVec2::new(x as u32, y as u32);
                let mut accum = Vec3::ZERO;
                for s in 0..opts.spp {
                    let mut rng = Rng::new(seed_for(pixel, s, opts.seed));
                    let jitter = rng.next_vec2();
                    let ray = camera.ray_for_pixel_jittered(pixel, image_size, jitter);
                    accum += trace_path(&scenario, ray, opts.max_bounces, &mut rng);
                }
                let color_linear = accum / opts.spp as f32;
                let color_mapped = tonemap_aces(color_linear);
                let r = linear_to_srgb(color_mapped.x).clamp(0.0, 1.0);
                let g = linear_to_srgb(color_mapped.y).clamp(0.0, 1.0);
                let b = linear_to_srgb(color_mapped.z).clamp(0.0, 1.0);
                let base = x * 3;
                row[base] = (r * 255.0) as u8;
                row[base + 1] = (g * 255.0) as u8;
                row[base + 2] = (b * 255.0) as u8;
            }
        });

    pixels
}

// ============================================================
// CLI
// ============================================================

struct Args {
    scene_path: String,
    out_path: String,
    env_path: Option<String>,
    env_intensity: f32,
    sun_direction: Option<Vec3>,
    sun_color: Vec3,
    sun_intensity: f32,
    camera_override: Option<CameraSpec>,
    width: u32,
    height: u32,
    spp: u32,
    max_bounces: u32,
    seed: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut scene_path: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut env_path: Option<String> = None;
    let mut env_intensity: f32 = 1.0;
    let mut sun_direction: Option<Vec3> = Some(Vec3::new(0.4, 0.8, 0.3).normalize());
    let mut sun_color = Vec3::new(1.0, 0.98, 0.93);
    let mut sun_intensity: f32 = 2.0;
    let mut camera_override: Option<CameraSpec> = None;
    let mut width: u32 = 512;
    let mut height: u32 = 512;
    let mut spp: u32 = 64;
    let mut max_bounces: u32 = 4;
    let mut seed: u64 = 0x12345;
    let mut spec_path: Option<String> = None;

    let mut width_from_cli = false;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--spec" => spec_path = iter.next(),
            "--scene" => scene_path = iter.next(),
            "--out" => out_path = iter.next(),
            "--env" => env_path = iter.next(),
            "--env-intensity" => {
                env_intensity = iter
                    .next()
                    .ok_or("--env-intensity needs a value")?
                    .parse()
                    .map_err(|e| format!("invalid --env-intensity: {e}"))?;
            }
            "--sun-dir" => {
                let x: f32 = iter
                    .next()
                    .ok_or("--sun-dir needs 3 values")?
                    .parse()
                    .map_err(|e| format!("invalid --sun-dir x: {e}"))?;
                let y: f32 = iter
                    .next()
                    .ok_or("--sun-dir needs 3 values")?
                    .parse()
                    .map_err(|e| format!("invalid --sun-dir y: {e}"))?;
                let z: f32 = iter
                    .next()
                    .ok_or("--sun-dir needs 3 values")?
                    .parse()
                    .map_err(|e| format!("invalid --sun-dir z: {e}"))?;
                sun_direction = Some(Vec3::new(x, y, z).normalize_or_zero());
            }
            "--no-sun" => {
                sun_direction = None;
            }
            "--sun-intensity" => {
                sun_intensity = iter
                    .next()
                    .ok_or("--sun-intensity needs a value")?
                    .parse()
                    .map_err(|e| format!("invalid --sun-intensity: {e}"))?;
            }
            "--width" => {
                width = iter
                    .next()
                    .ok_or("--width needs a value")?
                    .parse()
                    .map_err(|e| format!("invalid --width: {e}"))?;
                width_from_cli = true;
            }
            "--height" => {
                height = iter
                    .next()
                    .ok_or("--height needs a value")?
                    .parse()
                    .map_err(|e| format!("invalid --height: {e}"))?;
                width_from_cli = true;
            }
            "--spp" => {
                spp = iter
                    .next()
                    .ok_or("--spp needs a value")?
                    .parse()
                    .map_err(|e| format!("invalid --spp: {e}"))?;
            }
            "--bounces" => {
                max_bounces = iter
                    .next()
                    .ok_or("--bounces needs a value")?
                    .parse()
                    .map_err(|e| format!("invalid --bounces: {e}"))?;
            }
            "--seed" => {
                seed = iter
                    .next()
                    .ok_or("--seed needs a value")?
                    .parse()
                    .map_err(|e| format!("invalid --seed: {e}"))?;
            }
            "--camera" => {
                // Matches renderer-test's CLI: --camera px py pz tx ty tz fov
                let mut vals = [0.0f32; 7];
                for (i, slot) in vals.iter_mut().enumerate() {
                    *slot = iter
                        .next()
                        .ok_or_else(|| format!("--camera needs 7 values (got {})", i))?
                        .parse()
                        .map_err(|e| format!("invalid --camera value {}: {e}", i))?;
                }
                camera_override = Some(CameraSpec {
                    position: [vals[0], vals[1], vals[2]],
                    target: [vals[3], vals[4], vals[5]],
                    up: [0.0, 1.0, 0.0],
                    fov_y_deg: vals[6],
                });
            }
            "-h" | "--help" => {
                println!("bloom-reference — CPU path tracer (reference renderer)");
                println!();
                println!("  --spec PATH          shared view spec (JSON) — populates scene,");
                println!("                       camera, env, sun, resolution. CLI flags below");
                println!("                       override individual spec fields.");
                println!("  --scene PATH         glTF/GLB file to render");
                println!("  --out PATH           output PNG path (required)");
                println!("  --env PATH           HDR (.hdr) environment map");
                println!("  --env-intensity F    env map multiplier (default 1.0)");
                println!("  --sun-dir X Y Z      sun direction toward light");
                println!("  --sun-intensity F    sun intensity");
                println!("  --no-sun             disable sun light");
                println!("  --width N            image width");
                println!("  --height N           image height");
                println!("  --spp N              samples/pixel (default 64)");
                println!("  --bounces N          max bounces   (default 4)");
                println!("  --seed N             RNG seed      (default 0x12345)");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    // --spec populates fields that weren't overridden on the CLI.
    // We give CLI flags final authority so one-off overrides (e.g.
    // bumping --spp for a cleaner render) don't require editing the
    // spec file.
    if let Some(ref sp) = spec_path {
        let (spec, base_dir) = load_spec(Path::new(sp))?;
        // Resolve relative paths against the spec's own directory.
        let resolve = |p: &str| -> String {
            let pp = Path::new(p);
            if pp.is_absolute() {
                p.to_string()
            } else {
                base_dir.join(pp).to_string_lossy().to_string()
            }
        };
        if scene_path.is_none() {
            scene_path = Some(resolve(&spec.scene));
        }
        if !width_from_cli {
            width = spec.resolution[0];
            height = spec.resolution[1];
        }
        // Spec camera only fills in when the CLI didn't override.
        // Without this guard `--camera` from validate.sh gets clobbered.
        if camera_override.is_none() {
            camera_override = Some(spec.camera);
        }
        if env_path.is_none() {
            if let Some(ref env) = spec.env {
                env_path = Some(resolve(&env.path));
                if env_intensity == 1.0 {
                    env_intensity = env.intensity;
                }
            }
        }
        // Sun: spec.sun takes precedence unless CLI --no-sun/--sun-dir
        // was used (detected by sun_direction still matching the default).
        if let Some(s) = &spec.sun {
            sun_direction = Some(Vec3::from_array(s.direction).normalize_or_zero());
            sun_intensity = s.intensity;
            if let Some(c) = s.color {
                sun_color = Vec3::from_array(c);
            }
        } else if spec.sun.is_none() {
            // Explicit null in the JSON means "no sun" — honor it
            // unless the user overrode with CLI --sun-dir.
            sun_direction = None;
        }
        if let Some(defaults) = &spec.reference_defaults {
            if let Some(s) = defaults.spp {
                if spp == 64 {
                    spp = s;
                }
            }
            if let Some(b) = defaults.bounces {
                if max_bounces == 4 {
                    max_bounces = b;
                }
            }
        }
    }

    Ok(Args {
        scene_path: scene_path.ok_or("--scene or --spec is required")?,
        out_path: out_path.ok_or("--out is required")?,
        env_path,
        env_intensity,
        sun_direction,
        sun_color,
        sun_intensity,
        camera_override,
        width,
        height,
        spp,
        max_bounces,
        seed,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let load_start = Instant::now();
    let scene = match load_scene(Path::new(&args.scene_path)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error loading {}: {}", args.scene_path, e);
            return ExitCode::from(1);
        }
    };
    println!(
        "scene: {} triangles, {} materials, {} textures ({:?})",
        scene.triangles.len(),
        scene.materials.len(),
        scene.textures.len(),
        load_start.elapsed()
    );

    let bvh_start = Instant::now();
    let bvh = build_bvh(&scene.triangles);
    println!(
        "bvh: {} nodes ({:?})",
        bvh.nodes.len(),
        bvh_start.elapsed()
    );

    let environment = match &args.env_path {
        Some(p) => {
            let start = Instant::now();
            match Environment::load_hdr(Path::new(p), args.env_intensity) {
                Ok(env) => {
                    println!("env: {}x{} ({:?})", env.width, env.height, start.elapsed());
                    env
                }
                Err(e) => {
                    eprintln!("warn: failed to load env {}: {}. Using procedural.", p, e);
                    Environment::procedural()
                }
            }
        }
        None => {
            println!("env: procedural (no --env supplied)");
            Environment::procedural()
        }
    };

    let sun = args.sun_direction.map(|dir| SunLight {
        direction_to_light: dir,
        color: args.sun_color,
        intensity: args.sun_intensity,
    });

    // Spec-driven camera if provided, otherwise auto-frame against
    // the scene's bbox so naive `--scene x.glb --out y.png` still
    // produces a reasonable view.
    let camera = if let Some(cs) = &args.camera_override {
        Camera::looking_at(
            Vec3::from_array(cs.position),
            Vec3::from_array(cs.target),
            Vec3::from_array(cs.up),
            cs.fov_y_deg,
            args.width as f32 / args.height as f32,
        )
    } else {
        let center = scene.bbox_center();
        let radius = scene.bbox_radius().max(0.1);
        let camera_distance = radius * 2.5;
        Camera::looking_at(
            center + Vec3::new(camera_distance, camera_distance * 0.4, camera_distance),
            center,
            Vec3::Y,
            45.0,
            args.width as f32 / args.height as f32,
        )
    };

    let render_start = Instant::now();
    let pixels = render(
        &scene,
        &bvh,
        &environment,
        sun,
        &camera,
        &RenderOptions {
            width: args.width,
            height: args.height,
            spp: args.spp,
            max_bounces: args.max_bounces,
            seed: args.seed,
        },
    );
    println!(
        "render: {}x{} @ {}spp, {} bounces, {:?}",
        args.width,
        args.height,
        args.spp,
        args.max_bounces,
        render_start.elapsed()
    );

    let img = image::RgbImage::from_raw(args.width, args.height, pixels)
        .expect("pixel buffer size mismatch");
    if let Err(e) = img.save(&args.out_path) {
        eprintln!("error writing {}: {}", args.out_path, e);
        return ExitCode::from(1);
    }
    println!("wrote {}", args.out_path);
    ExitCode::SUCCESS
}
