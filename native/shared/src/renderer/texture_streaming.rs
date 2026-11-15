//! Texture streaming, stage 1: a texture slot can hold part of its mip chain.
//!
//! A block-compressed model texture (a KTX2 image inside a `.glb`) used to be
//! uploaded whole and kept whole for the life of the model. A large world
//! carries far more texels than it shows: a 2048² ASTC texture is 5.6 MB, and
//! a prop two hundred metres away samples its 64² level. On a tablet the
//! textures of everything loaded were what ran the device out of memory.
//!
//! With streaming on, such a texture is created from its small levels only
//! (`FLOOR_DIM` and below, a sixty-fourth of a 2048² chain) and the renderer
//! remembers where the image's bytes are in its file. [`Renderer::
//! set_texture_resident_level`] reads the levels it needs from that file,
//! builds a larger or smaller texture and puts it in the same slot. The
//! slot index and the global texture id do not change, so nothing that
//! refers to the texture has to be told.
//!
//! Who decides the level is the next stage. This one has only a test driver:
//! `BLOOM_TEXTURE_STREAM_RAISE=1` raises a few textures to their full size
//! every frame until all are whole, which exercises the swap on real content.
//!
//! Streaming is off unless `BLOOM_TEXTURE_STREAM=1`: with nothing yet to
//! raise a texture when it comes near, turning it on leaves near surfaces
//! soft. The next stage makes it the default on phones, tablets and TV.
//!
//! Bind groups hold texture views, so a swap also rebuilds the material bind
//! groups that show the slot: those of cached models here, those of scene
//! nodes when the scene next prepares its materials. Materials with
//! transmission or layered-PBR extensions have further bind groups; their
//! textures are made whole and not streamed (`pin_special_material_textures`).

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::material_indirection::{ResidentTexture, TextureColorSpace, TextureSemantic};
use super::Renderer;
use crate::compressed_texture::CompressedImage;

/// Levels this size and smaller stay resident always.
pub(crate) const FLOOR_DIM: u32 = 256;
/// How many textures the test driver raises in one frame.
const RAISES_PER_FRAME: usize = 2;

/// Where a streamed texture's KTX2 bytes are: a byte range of a file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StreamSource {
    pub path: PathBuf,
    pub offset: u64,
    pub length: u64,
}

struct StreamedTexture {
    source: StreamSource,
    /// The largest level resident now (0 = the whole chain).
    resident_level: u32,
    /// The level the texture was created at and never goes below.
    floor_level: u32,
    is_normal: bool,
    srgb: bool,
}

/// One renderer's streamed textures. Slot indices mean nothing outside the
/// renderer that handed them out, so this is a field of it, not process state.
#[derive(Default)]
pub(crate) struct TextureStreaming {
    /// The file being imported, set by the model loader round its work.
    current_source: Option<PathBuf>,
    entries: HashMap<u32, StreamedTexture>,
    /// Slots given another texture since the scene last rebuilt its material
    /// bind groups (`take_replaced_textures`).
    replaced: Vec<u32>,
}

fn env_flag(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" => Some(true),
        "0" | "false" | "off" => Some(false),
        _ => None,
    }
}

/// Whether textures are streamed in this process.
pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("BLOOM_TEXTURE_STREAM").unwrap_or(false))
}

/// The file range of an image that is `view_length` bytes at `view_offset` in
/// the binary chunk of the GLB `file`, or `None` when `file` is not a GLB with
/// a JSON chunk followed by a binary chunk that holds the range.
pub(crate) fn glb_view_file_range(
    file: &[u8],
    view_offset: usize,
    view_length: usize,
) -> Option<(u64, u64)> {
    let word = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(file.get(at..at + 4)?.try_into().ok()?))
    };
    if file.get(0..4)? != b"glTF" || word(16)? != 0x4E4F_534A {
        return None;
    }
    let json_length = word(12)? as usize;
    let bin_header = 20usize.checked_add(json_length)?;
    if word(bin_header + 4)? != 0x004E_4942 {
        return None;
    }
    let bin_length = word(bin_header)? as usize;
    let bin_start = bin_header + 8;
    if view_offset.checked_add(view_length)? > bin_length
        || bin_start.checked_add(bin_length)? > file.len()
    {
        return None;
    }
    Some(((bin_start + view_offset) as u64, view_length as u64))
}

/// The level a texture is created at: the first whose larger side is at most
/// `floor_dim`. 0 when the image is already that small.
pub(crate) fn floor_level(width: u32, height: u32, level_count: u32, floor_dim: u32) -> u32 {
    let mut level = 0;
    while level + 1 < level_count && (width.max(height) >> level) > floor_dim {
        level += 1;
    }
    level
}

/// The nearest level at or above `level` in size (so at or below it in
/// number) whose width and height are whole blocks, which a compressed
/// texture's base level has to be. `None` when not even level 0 is: such an
/// image is not streamed.
pub(crate) fn block_aligned_level(image: &CompressedImage, level: u32) -> Option<u32> {
    let (block_width, block_height) = image.format.block_size();
    let whole = |l: u32| {
        (image.width >> l).max(1) % block_width == 0
            && (image.height >> l).max(1) % block_height == 0
    };
    if !whole(0) {
        return None; // it could never be raised to its full size
    }
    (0..=level).rev().find(|&l| whole(l))
}

/// `image` from `level` down: the same texture with its largest levels left out.
pub(crate) fn sub_image(image: &CompressedImage, level: u32) -> CompressedImage {
    let level = (level as usize).min(image.levels.len().saturating_sub(1));
    CompressedImage {
        format: image.format,
        width: (image.width >> level).max(1),
        height: (image.height >> level).max(1),
        srgb: image.srgb,
        levels: image.levels[level..].to_vec(),
    }
}

fn read_source(source: &StreamSource) -> Option<Vec<u8>> {
    let mut file = std::fs::File::open(&source.path).ok()?;
    file.seek(SeekFrom::Start(source.offset)).ok()?;
    let mut bytes = vec![0u8; usize::try_from(source.length).ok()?];
    file.read_exact(&mut bytes).ok()?;
    Some(bytes)
}

impl Renderer {
    /// Register a KTX2 image found at `file_range` (offset, length) of the
    /// file being imported. With streaming on and the image larger than the
    /// floor, only its small levels are uploaded and the rest stays on disk.
    pub(crate) fn register_compressed_streamed(
        &mut self,
        image: &CompressedImage,
        is_normal: bool,
        srgb: bool,
        file_range: Option<(u64, u64)>,
    ) -> Option<u32> {
        let source = if enabled() {
            let path = self.texture_streaming.current_source.clone();
            path.zip(file_range)
                .map(|(path, (offset, length))| StreamSource {
                    path,
                    offset,
                    length,
                })
        } else {
            None
        };
        let level = block_aligned_level(
            image,
            floor_level(
                image.width,
                image.height,
                image.levels.len() as u32,
                FLOOR_DIM,
            ),
        )
        .unwrap_or(0);
        let Some(source) = source.filter(|_| level > 0) else {
            return self.register_texture_compressed(image, is_normal, srgb);
        };
        let index = self.register_texture_compressed(&sub_image(image, level), is_normal, srgb)?;
        self.texture_streaming.entries.insert(
            index,
            StreamedTexture {
                source,
                resident_level: level,
                floor_level: level,
                is_normal,
                srgb,
            },
        );
        Some(index)
    }

    /// Give the streamed texture in slot `idx` the levels from `level` down
    /// (0 = all of them; clamped to the level it was created at). Reads the
    /// image from its file. False when the slot is not streamed, the file can
    /// no longer be read, or the texture could not be made; the slot then
    /// keeps what it has.
    pub(crate) fn set_texture_resident_level(&mut self, idx: u32, level: u32) -> bool {
        let (source, level, is_normal, srgb) = {
            let Some(entry) = self.texture_streaming.entries.get(&idx) else {
                return false;
            };
            let level = level.min(entry.floor_level);
            if level == entry.resident_level {
                return true;
            }
            (entry.source.clone(), level, entry.is_normal, entry.srgb)
        };
        let Some(image) = read_source(&source).and_then(|bytes| CompressedImage::from_ktx2(&bytes))
        else {
            return false;
        };
        let Some(level) = block_aligned_level(&image, level) else {
            return false;
        };
        if !self.replace_texture_compressed(idx, &sub_image(&image, level), is_normal, srgb) {
            return false;
        }
        if let Some(entry) = self.texture_streaming.entries.get_mut(&idx) {
            entry.resident_level = level;
        }
        true
    }

    /// Put another block-compressed texture in an existing slot. The slot's
    /// index and its global texture id stay as they are.
    fn replace_texture_compressed(
        &mut self,
        idx: u32,
        image: &CompressedImage,
        is_normal: bool,
        srgb_rgb: bool,
    ) -> bool {
        let i = idx as usize;
        if i == 0 || i >= self.textures.len() {
            return false;
        }
        let Some((texture, view, bind_group)) = self.create_compressed_texture(image) else {
            return false;
        };
        let srgb = srgb_rgb && !is_normal;
        let resident = ResidentTexture {
            view,
            width: image.width,
            height: image.height,
            mip_count: image.levels.len() as u32,
            color_space: if srgb {
                TextureColorSpace::Srgb
            } else {
                TextureColorSpace::Linear
            },
            semantic: if is_normal {
                TextureSemantic::Normal
            } else if srgb {
                TextureSemantic::BaseColor
            } else {
                TextureSemantic::MetallicRoughness
            },
            hardware_srgb_decode: false,
            global_2d: true,
        };
        let id = self.global_texture_ids[i];
        if !self
            .material_system
            .indirection
            .replace_texture(id, resident)
        {
            return false;
        }
        self.textures[i] = texture;
        self.texture_bind_groups[i] = bind_group;
        self.texture_sizes[i] = (image.width, image.height);
        self.refresh_cached_model_materials(idx);
        self.texture_streaming.replaced.push(idx);
        // The path tracer binds every texture view in one group; it is made
        // again from the slots as they are now.
        self.pt_tex_bg = None;
        true
    }

    /// The model loader names the file it is importing (and `None` when
    /// done), so images found inside it can be streamed from it later.
    pub(crate) fn set_streaming_source(&mut self, path: Option<&Path>) {
        self.texture_streaming.current_source = path.map(Path::to_path_buf);
    }

    /// The slot was unloaded: it is no longer streamed.
    pub(crate) fn forget_streamed_texture(&mut self, idx: u32) {
        self.texture_streaming.entries.remove(&idx);
    }

    /// Streamed textures, and how many of them are not whole.
    pub(crate) fn streamed_texture_counts(&self) -> (usize, usize) {
        let entries = &self.texture_streaming.entries;
        let partial = entries.values().filter(|e| e.resident_level > 0).count();
        (entries.len(), partial)
    }

    /// The slots given another texture since the last call. The scene rebuilds
    /// the material bind groups of the nodes that show one of them.
    pub(crate) fn take_replaced_textures(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.texture_streaming.replaced)
    }

    /// Cached models bind their textures through a bind group made when the
    /// model was first drawn; it holds the old views. Make those again.
    fn refresh_cached_model_materials(&mut self, idx: u32) {
        let mut rebuilt = Vec::new();
        for (&key, meshes) in &self.model_gpu_cache {
            for (at, mesh) in meshes.iter().flatten().enumerate() {
                let t = mesh.material_textures;
                if t.contains(&idx) {
                    let bg = self.create_scene_material_bg(
                        t[0],
                        t[1],
                        t[2],
                        t[3],
                        t[4],
                        &mesh._material_uniform,
                    );
                    rebuilt.push((key, at, bg));
                }
            }
        }
        for (key, at, bg) in rebuilt {
            if let Some(Some(meshes)) = self.model_gpu_cache.get_mut(&key) {
                meshes[at].material_bg = bg;
            }
        }
    }

    /// A material with transmission or layered-PBR extensions binds its
    /// textures through bind groups of its own, which a swap does not
    /// remake. Its textures are therefore not streamed: each is made whole
    /// and left alone from here on. Call it before the material's bind groups
    /// are built.
    pub(crate) fn pin_special_material_textures(
        &mut self,
        standard: [u32; 5],
        transmission: &crate::models::MaterialTransmission,
        layered: &crate::models::MaterialLayeredPbr,
    ) {
        if self.texture_streaming.entries.is_empty()
            || !(transmission.is_active() || layered.is_active())
        {
            return;
        }
        let bindings = [
            transmission.texture,
            transmission.thickness_texture,
            layered.clearcoat_texture,
            layered.clearcoat_roughness_texture,
            layered.clearcoat_normal_texture,
            layered.specular_texture,
            layered.specular_color_texture,
            layered.sheen_color_texture,
            layered.sheen_roughness_texture,
            layered.anisotropy_texture,
            layered.iridescence_texture,
            layered.iridescence_thickness_texture,
        ];
        let slots = standard.into_iter().chain(
            bindings
                .into_iter()
                .flatten()
                .filter_map(|binding| binding.runtime_texture_idx),
        );
        for slot in slots.collect::<Vec<u32>>() {
            if self.texture_streaming.entries.contains_key(&slot) {
                self.set_texture_resident_level(slot, 0);
                self.forget_streamed_texture(slot);
            }
        }
    }

    /// Once a frame. Stage 1 has only the test driver here.
    pub(crate) fn texture_streaming_tick(&mut self) {
        static RAISE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*RAISE.get_or_init(|| env_flag("BLOOM_TEXTURE_STREAM_RAISE").unwrap_or(false)) {
            return;
        }
        let pending: Vec<u32> = self
            .texture_streaming
            .entries
            .iter()
            .filter(|(_, entry)| entry.resident_level > 0)
            .map(|(&idx, _)| idx)
            .take(RAISES_PER_FRAME)
            .collect();
        let last = !pending.is_empty();
        for idx in pending {
            if !self.set_texture_resident_level(idx, 0) {
                self.forget_streamed_texture(idx); // unreadable: leave it small and stop trying
            }
        }
        let (streamed, partial) = self.streamed_texture_counts();
        if last && partial == 0 {
            crate::ffi::log_error(&format!(
                "bloom: texture streaming test driver: all {streamed} streamed textures are whole"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressed_texture::BlockFormat;

    fn image(size: u32) -> CompressedImage {
        let format = BlockFormat::Astc {
            width: 4,
            height: 4,
        };
        let mut levels = Vec::new();
        let mut dim = size;
        loop {
            levels.push(vec![levels.len() as u8; format.level_bytes(dim, dim)]);
            if dim == 1 {
                break;
            }
            dim /= 2;
        }
        CompressedImage {
            format,
            width: size,
            height: size,
            srgb: true,
            levels,
        }
    }

    #[test]
    fn the_floor_is_the_first_level_that_fits() {
        assert_eq!(floor_level(2048, 2048, 12, 256), 3);
        assert_eq!(floor_level(4096, 2048, 13, 256), 4);
        assert_eq!(floor_level(256, 256, 9, 256), 0);
        assert_eq!(floor_level(64, 64, 7, 256), 0);
        // a chain that stops early cannot go below its last level
        assert_eq!(floor_level(2048, 2048, 2, 256), 1);
    }

    #[test]
    fn a_streamed_base_level_is_whole_blocks() {
        assert_eq!(block_aligned_level(&image(2048), 3), Some(3));
        // 1032 px: level 3 would be 129 px, not whole 4 px blocks; level 1 (516) is
        let mut odd = image(1024);
        odd.width = 1032;
        odd.height = 1032;
        assert_eq!(block_aligned_level(&odd, 3), Some(1));
        // 1030 px: level 3 (128 px) is whole blocks, but level 0 is not
        odd.width = 1030;
        odd.height = 1030;
        assert_eq!(block_aligned_level(&odd, 3), None);
    }

    #[test]
    fn a_sub_image_is_the_tail_of_the_chain() {
        let whole = image(2048);
        let small = sub_image(&whole, 3);
        assert_eq!((small.width, small.height), (256, 256));
        assert_eq!(small.levels.len(), whole.levels.len() - 3);
        assert_eq!(small.levels[0], whole.levels[3]);
        // and it survives a trip through the container, as a re-read does
        let again = CompressedImage::from_ktx2(&small.to_ktx2().unwrap()).unwrap();
        assert_eq!(again.levels, small.levels);
        // the floor chain of a 2048² texture is a sixty-fourth of the whole
        let bytes = |i: &CompressedImage| i.levels.iter().map(Vec::len).sum::<usize>();
        assert!(bytes(&small) * 60 < bytes(&whole));
    }

    #[test]
    fn a_glb_view_maps_to_its_place_in_the_file() {
        let json = b"{\"a\":1}  "; // 8 bytes, as chunks are padded to 4
        let json = &json[..8];
        let bin = [7u8; 16];
        let mut file = Vec::new();
        file.extend_from_slice(b"glTF");
        file.extend_from_slice(&2u32.to_le_bytes());
        file.extend_from_slice(&((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
        file.extend_from_slice(&(json.len() as u32).to_le_bytes());
        file.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
        file.extend_from_slice(json);
        file.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        file.extend_from_slice(&0x004E_4942u32.to_le_bytes());
        file.extend_from_slice(&bin);
        assert_eq!(glb_view_file_range(&file, 4, 8), Some((40, 8)));
        assert_eq!(glb_view_file_range(&file, 12, 8), None, "past the chunk");
        assert_eq!(glb_view_file_range(b"not a glb at all....", 0, 4), None);
    }
}
