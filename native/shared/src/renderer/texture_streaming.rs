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
//! Stage 2 decides the level. Everything drawn says how sharp its textures
//! have to be: a mesh's texture coordinates per metre and its distance from
//! the camera give the texels one screen pixel covers, and so the level
//! (`needed_level`). A texture takes the sharpest level asked of it. Once a
//! frame the scheduler raises the texture furthest below what it was asked
//! for and, when the budget is full, first lowers the ones that have held
//! more than anyone asked for the longest. A texture nothing asked for goes
//! back to its floor only under that pressure, so looking away costs nothing.
//!
//! Stage 3 takes the file off the frame: the scheduler only asks, a worker
//! thread reads and parses the levels, and a later frame uploads them. Where
//! threads cannot be started (the web) the read stays on the frame.
//!
//! Streaming is off until the game turns it on with a budget
//! (`setTextureStreaming(megabytes)`, before it loads its models): how much
//! memory there is to spend is the game's knowledge, not the engine's. For
//! tests, `BLOOM_TEXTURE_STREAM=1` turns it on without the game and
//! `BLOOM_TEXTURE_STREAM_MB` sets the budget (default 384).
//! `BLOOM_TEXTURE_STREAM_RAISE=1` is a test driver: it ignores demand and
//! raises every streamed texture to full size, a few a frame.
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
/// How many reads the scheduler starts in one frame. They run on a worker.
const SCHEDULED_RAISES_PER_FRAME: usize = 2;
/// How many finished reads are uploaded in one frame.
const UPLOADS_PER_FRAME: usize = 2;
/// Reads in flight at once: more would only queue behind the disk and commit
/// the budget to demand that may have moved on.
const READS_IN_FLIGHT: usize = 4;
/// Ticks a texture has to hold more than it is asked for before the
/// scheduler may take it back (two seconds at 60 Hz).
const SURPLUS_TICKS_BEFORE_LOWERING: u32 = 120;
/// Default budget for streamed textures, in megabytes.
const DEFAULT_BUDGET_MB: u64 = 384;
/// Levels of margin: ask for one level sharper than the estimate, which
/// covers surfaces seen at a slant and the estimate's own roughness.
const LEVEL_BIAS: f32 = 1.0;

fn budget_bytes() -> u64 {
    static BUDGET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("BLOOM_TEXTURE_STREAM_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_BUDGET_MB)
            * 1024
            * 1024
    })
}

/// Texture coordinates per metre of a mesh: the square root of its area in
/// texture space over its area in model space. 0 when either is empty (no
/// texture coordinates, or a degenerate mesh): unknown.
pub(crate) fn uv_density(vertices: &[super::Vertex3D], indices: &[u32]) -> f32 {
    let (mut uv_area, mut area) = (0.0f64, 0.0f64);
    for tri in indices.chunks_exact(3) {
        let (Some(a), Some(b), Some(c)) = (
            vertices.get(tri[0] as usize),
            vertices.get(tri[1] as usize),
            vertices.get(tri[2] as usize),
        ) else {
            continue;
        };
        let e1 = [
            b.position[0] - a.position[0],
            b.position[1] - a.position[1],
            b.position[2] - a.position[2],
        ];
        let e2 = [
            c.position[0] - a.position[0],
            c.position[1] - a.position[1],
            c.position[2] - a.position[2],
        ];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        area += f64::from((cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt());
        let (u1, v1) = (b.uv[0] - a.uv[0], b.uv[1] - a.uv[1]);
        let (u2, v2) = (c.uv[0] - a.uv[0], c.uv[1] - a.uv[1]);
        uv_area += f64::from((u1 * v2 - u2 * v1).abs());
    }
    if area <= 0.0 || uv_area <= 0.0 {
        return 0.0;
    }
    (uv_area / area).sqrt() as f32
}

/// The level a texture `full_dim` texels across has to hold for a surface
/// with `density` texture coordinates per metre, `distance` metres away, when
/// one screen pixel covers `pixel_metres` metres at one metre. 0 (everything)
/// when the density is unknown.
pub(crate) fn needed_level(full_dim: u32, density: f32, distance: f32, pixel_metres: f32) -> u32 {
    if !(density > 0.0) || !(pixel_metres > 0.0) {
        return 0;
    }
    let texels_per_pixel = distance.max(0.05) * pixel_metres * density * full_dim as f32;
    (texels_per_pixel.log2() - LEVEL_BIAS).floor().max(0.0) as u32
}

/// Distance from `point` to the box `lo`..`hi` (0 inside it).
fn distance_to_box(point: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let mut sum = 0.0;
    for a in 0..3 {
        let d = (lo[a] - point[a]).max(point[a] - hi[a]).max(0.0);
        sum += d * d;
    }
    sum.sqrt()
}

/// What the scheduler does this tick: `(slot, level)` to set, lowerings first.
/// `entries`: slot, resident level, wanted level, ticks in surplus, and the
/// chain's bytes from each level down.
pub(crate) fn plan_tick(
    entries: &[(u32, u32, u32, u32, &[u64])],
    budget: u64,
    raises: usize,
) -> Vec<(u32, u32)> {
    let bytes = |chain: &[u64], level: u32| chain.get(level as usize).copied().unwrap_or(0);
    let mut resident: u64 = entries.iter().map(|e| bytes(e.4, e.1)).sum();
    // Those holding more than asked, longest first: what pressure takes back.
    let mut spare: Vec<&(u32, u32, u32, u32, &[u64])> = entries
        .iter()
        .filter(|e| e.2 > e.1 && e.3 >= SURPLUS_TICKS_BEFORE_LOWERING)
        .collect();
    spare.sort_by(|a, b| b.3.cmp(&a.3).then(a.0.cmp(&b.0)));
    let mut spare = spare.into_iter();
    // Those holding less than asked, the largest shortfall first.
    let mut short: Vec<&(u32, u32, u32, u32, &[u64])> =
        entries.iter().filter(|e| e.2 < e.1).collect();
    short.sort_by(|a, b| (b.1 - b.2).cmp(&(a.1 - a.2)).then(a.0.cmp(&b.0)));
    let mut plan = Vec::new();
    let mut lower_one = |resident: &mut u64, plan: &mut Vec<(u32, u32)>| -> bool {
        let Some(e) = spare.next() else {
            return false;
        };
        *resident -= bytes(e.4, e.1) - bytes(e.4, e.2);
        plan.push((e.0, e.2));
        true
    };
    while resident > budget {
        if !lower_one(&mut resident, &mut plan) {
            break;
        }
    }
    for e in short.into_iter().take(raises) {
        // The sharpest level the budget has room for, making room if it can.
        let mut level = e.2;
        loop {
            let cost = bytes(e.4, level) - bytes(e.4, e.1);
            if resident + cost <= budget {
                break;
            }
            if lower_one(&mut resident, &mut plan) {
                continue;
            }
            level += 1;
            if level >= e.1 {
                break;
            }
        }
        if level < e.1 {
            resident += bytes(e.4, level) - bytes(e.4, e.1);
            plan.push((e.0, level));
        }
    }
    plan
}

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
    /// Bytes of the chain from each level down (`chain_bytes[0]` = all of it).
    chain_bytes: Vec<u64>,
    /// The larger side of level 0, in texels.
    full_dim: u32,
    /// The sharpest level anything on screen asked for since the last tick;
    /// `u32::MAX` when nothing did.
    wanted: u32,
    /// What the last tick settled on for it.
    planned: u32,
    /// The level a read in flight will give it; the scheduler leaves the
    /// texture alone until that read has come back.
    pending: Option<u32>,
    /// Ticks in a row the texture has held more than was asked for.
    surplus_ticks: u32,
}

/// The bytes of `image`'s chain from each level down.
/// The bytes of each level and all smaller ones, of the image whose shape
/// `image` is (its levels may be empty): they follow from the format and size.
fn chain_bytes(image: &CompressedImage) -> Vec<u64> {
    let mut sizes: Vec<u64> = (0..image.levels.len() as u32)
        .map(|l| {
            let (w, h) = ((image.width >> l).max(1), (image.height >> l).max(1));
            image.format.level_bytes(w, h) as u64
        })
        .collect();
    for at in (0..sizes.len().saturating_sub(1)).rev() {
        sizes[at] += sizes[at + 1];
    }
    sizes
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
    /// What the game asked for (`set_texture_streaming`); `None` leaves it
    /// to the environment.
    enabled: Option<bool>,
    budget: Option<u64>,
    /// Texture coordinates per metre of each mesh seen, by the address of its
    /// vertices (shared model primitives are measured once).
    uv_density: HashMap<usize, f32>,
    /// The worker that reads levels from disk, started with the first read.
    reader: Option<LevelReader>,
}

/// A level chain read from a file on a worker thread: the read and the parse
/// are milliseconds for a large texture and do not belong on a frame.
/// Dropping it (with the renderer) closes the channel and ends the thread.
struct LevelReader {
    requests: std::sync::mpsc::Sender<(u32, StreamSource, u32)>,
    results: std::sync::mpsc::Receiver<(u32, u32, Option<CompressedImage>)>,
}

/// The texture in `source` from `level` down (moved to the nearest larger
/// level that is whole blocks), with the level it starts at.
fn load_levels(source: &StreamSource, level: u32) -> Option<(u32, CompressedImage)> {
    let mut file = std::fs::File::open(&source.path).ok()?;
    // The image's header and level index first, in a zeroed buffer of the
    // image's size (pages never written cost nothing), then only the bytes of
    // the levels kept: KTX2 stores the smallest level first, so those are one
    // range. The whole chain was read for every raise and every lowering.
    let length = usize::try_from(source.length).ok()?;
    let mut bytes = vec![0u8; length];
    let read_at = |file: &mut std::fs::File, bytes: &mut [u8], from: usize, to: usize| {
        file.seek(SeekFrom::Start(source.offset + from as u64))
            .ok()?;
        file.read_exact(bytes.get_mut(from..to)?).ok()
    };
    read_at(&mut file, &mut bytes, 0, 80.min(length))?;
    let count = u32::from_le_bytes(bytes.get(40..44)?.try_into().ok()?).max(1) as usize;
    let index_end = 80usize.checked_add(count.checked_mul(24)?)?;
    read_at(&mut file, &mut bytes, 80, index_end)?;
    let shape = CompressedImage::from_ktx2_from(&bytes, u32::MAX)?;
    let level = block_aligned_level(&shape, level)?;
    let range = |at: usize| -> Option<usize> {
        usize::try_from(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?)).ok()
    };
    let (mut from, mut to) = (usize::MAX, 0usize);
    for l in level as usize..count {
        let (offset, len) = (range(80 + l * 24)?, range(88 + l * 24)?);
        from = from.min(offset);
        to = to.max(offset.checked_add(len)?);
    }
    read_at(&mut file, &mut bytes, from, to)?;
    Some((level, CompressedImage::from_ktx2_from(&bytes, level)?))
}

impl LevelReader {
    /// `None` where threads cannot be started (the web): reads then stay on
    /// the frame.
    fn start() -> Option<Self> {
        if cfg!(target_arch = "wasm32") {
            return None;
        }
        let (requests, inbox) = std::sync::mpsc::channel::<(u32, StreamSource, u32)>();
        let (outbox, results) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("bloom-texture-stream".into())
            .spawn(move || {
                for (slot, source, level) in inbox {
                    let loaded = load_levels(&source, level);
                    let (level, image) = match loaded {
                        Some((level, image)) => (level, Some(image)),
                        None => (level, None),
                    };
                    if outbox.send((slot, level, image)).is_err() {
                        break;
                    }
                }
            })
            .ok()?;
        Some(Self { requests, results })
    }
}

fn env_flag(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" => Some(true),
        "0" | "false" | "off" => Some(false),
        _ => None,
    }
}

/// Whether textures are streamed when the game has not said: only when
/// `BLOOM_TEXTURE_STREAM=1` asks for it.
fn enabled_by_environment() -> bool {
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

/// Whether every level from 0 to `floor` is whole blocks. Only such a texture
/// is streamed: the scheduler budgets a level by its bytes, and a level that
/// had to be moved to a larger one on loading would cost more than planned.
pub(crate) fn every_level_is_whole_blocks(image: &CompressedImage, floor: u32) -> bool {
    (0..=floor).all(|level| block_aligned_level(image, level) == Some(level))
}

/// `image` from `level` down: the same texture with its largest levels left out.
#[cfg(test)]
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

impl Renderer {
    /// Register the KTX2 image `ktx2` found at `file_range` (offset, length)
    /// of the file being imported. With streaming on and the image larger
    /// than the floor, only its small levels are uploaded and the rest stays
    /// on disk: only those levels are copied out of `ktx2` (the larger ones
    /// were copied and thrown away, most of a model's bytes).
    pub(crate) fn register_compressed_streamed(
        &mut self,
        ktx2: &[u8],
        is_normal: bool,
        srgb: bool,
        file_range: Option<(u64, u64)>,
    ) -> Option<u32> {
        // the shape only (one empty level per level): no level is copied yet
        let image = &CompressedImage::from_ktx2_from(ktx2, u32::MAX)?;
        let source = if self
            .texture_streaming
            .enabled
            .unwrap_or_else(enabled_by_environment)
        {
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
        let streamable = level > 0 && every_level_is_whole_blocks(image, level);
        let Some(source) = source.filter(|_| streamable) else {
            let whole = CompressedImage::from_ktx2(ktx2)?;
            return self.register_texture_compressed(&whole, is_normal, srgb);
        };
        let resident = CompressedImage::from_ktx2_from(ktx2, level)?;
        let index = self.register_texture_compressed(&resident, is_normal, srgb)?;
        self.texture_streaming.entries.insert(
            index,
            StreamedTexture {
                source,
                resident_level: level,
                floor_level: level,
                is_normal,
                srgb,
                chain_bytes: chain_bytes(image),
                full_dim: image.width.max(image.height),
                wanted: u32::MAX,
                planned: level,
                pending: None,
                surplus_ticks: 0,
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
        let Some((level, image)) = load_levels(&source, level) else {
            return false;
        };
        if !self.replace_texture_compressed(idx, &image, is_normal, srgb) {
            return false;
        }
        if let Some(entry) = self.texture_streaming.entries.get_mut(&idx) {
            entry.resident_level = level;
            entry.pending = None;
        }
        true
    }

    /// Ask for the levels of slot `idx` from `level` down without waiting:
    /// the worker reads them and `finish_level_reads` puts them in the slot
    /// on a later frame. Falls back to reading here and now where there is
    /// no worker. False when the slot is not streamed or the read failed.
    fn request_texture_resident_level(&mut self, idx: u32, level: u32) -> bool {
        let Some(entry) = self.texture_streaming.entries.get(&idx) else {
            return false;
        };
        let level = level.min(entry.floor_level);
        if level == entry.resident_level || entry.pending.is_some() {
            return true;
        }
        let source = entry.source.clone();
        if self.texture_streaming.reader.is_none() {
            self.texture_streaming.reader = LevelReader::start();
        }
        let sent = self
            .texture_streaming
            .reader
            .as_ref()
            .is_some_and(|reader| reader.requests.send((idx, source, level)).is_ok());
        if !sent {
            return self.set_texture_resident_level(idx, level);
        }
        if let Some(entry) = self.texture_streaming.entries.get_mut(&idx) {
            entry.pending = Some(level);
        }
        true
    }

    /// Put the level chains the worker has finished reading into their slots,
    /// at most `UPLOADS_PER_FRAME` of them: the upload is on the frame.
    fn finish_level_reads(&mut self) {
        for _ in 0..UPLOADS_PER_FRAME {
            let Some(done) = self
                .texture_streaming
                .reader
                .as_ref()
                .and_then(|reader| reader.results.try_recv().ok())
            else {
                return;
            };
            let (slot, level, image) = done;
            // The slot may have been unloaded or pinned while the read ran.
            let Some(entry) = self.texture_streaming.entries.get_mut(&slot) else {
                continue;
            };
            if entry.pending.take().is_none() {
                continue;
            }
            let (is_normal, srgb) = (entry.is_normal, entry.srgb);
            let swapped = image.is_some_and(|image| {
                self.replace_texture_compressed(slot, &image, is_normal, srgb)
            });
            if !swapped {
                self.forget_streamed_texture(slot); // unreadable: leave it as it is
            } else if let Some(entry) = self.texture_streaming.entries.get_mut(&slot) {
                entry.resident_level = level;
            }
        }
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

    /// Turn streaming on with a budget of `budget_mb` megabytes for the
    /// streamed textures, or off with 0. It applies to textures loaded from
    /// here on: call it before loading models. Textures already streamed
    /// stay streamed (and keep being scheduled) when it is turned off.
    pub fn set_texture_streaming(&mut self, budget_mb: f32) {
        let on = budget_mb > 0.0;
        self.texture_streaming.enabled = Some(on);
        if on {
            self.texture_streaming.budget = Some((f64::from(budget_mb) * 1024.0 * 1024.0) as u64);
        }
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

    /// Something drawn asks for its textures: `slots` are the texture slots
    /// of its material, `density` its texture coordinates per metre in world
    /// space, `distance` how far it is from the camera.
    fn want_textures(&mut self, slots: [u32; 5], density: f32, distance: f32, pixel_metres: f32) {
        for slot in slots {
            if slot == 0 {
                continue;
            }
            if let Some(entry) = self.texture_streaming.entries.get_mut(&slot) {
                let level = needed_level(entry.full_dim, density, distance, pixel_metres);
                entry.wanted = entry.wanted.min(level);
            }
        }
    }

    /// Metres one output pixel covers at one metre from the camera, and
    /// whether the projection is orthographic: a pixel then covers that many
    /// metres at any distance.
    fn streaming_pixel_metres(&self) -> (f32, bool) {
        let focal = self.current_proj_matrix[1][1].abs().max(1e-3);
        let orthographic = self.current_proj_matrix[3][3].abs() > 0.5;
        (
            2.0 / focal / self.surface_config.height.max(1) as f32,
            orthographic,
        )
    }

    /// Every scene node on screen asks for its textures.
    pub(crate) fn texture_streaming_observe_scene(&mut self, scene: &crate::scene::SceneGraph) {
        if self.texture_streaming.entries.is_empty() {
            return;
        }
        let camera = self.current_camera_pos;
        let (pixel_metres, orthographic) = self.streaming_pixel_metres();
        for (_, node) in scene.nodes.iter() {
            if !node.shown_on_screen() || node.indices().is_empty() {
                continue;
            }
            let material = &node.material;
            let slots = [
                material.texture_idx,
                material.normal_texture_idx,
                material.metallic_roughness_texture_idx,
                material.emissive_texture_idx,
                material.occlusion_texture_idx,
            ];
            if !slots
                .iter()
                .any(|slot| self.texture_streaming.entries.contains_key(slot))
            {
                continue;
            }
            let key = node.vertices().as_ptr() as usize;
            let local = *self
                .texture_streaming
                .uv_density
                .entry(key)
                .or_insert_with(|| uv_density(node.vertices(), node.indices()));
            let world = node.world_transform();
            let scale = (0..3)
                .map(|c| (world[c][0].powi(2) + world[c][1].powi(2) + world[c][2].powi(2)).sqrt())
                .fold(0.0f32, f32::max)
                .max(1e-4);
            let distance = if orthographic {
                1.0
            } else if node.world_bounds_min[0] <= node.world_bounds_max[0] {
                distance_to_box(camera, node.world_bounds_min, node.world_bounds_max)
            } else {
                0.0
            };
            self.want_textures(slots, local / scale, distance, pixel_metres);
        }
    }

    /// A cached model drawn at `position` and `scale` asks for its textures.
    pub(crate) fn texture_streaming_observe_model(
        &mut self,
        handle_bits: u64,
        position: [f32; 3],
        scale: f32,
    ) {
        if self.texture_streaming.entries.is_empty() {
            return;
        }
        let camera = self.current_camera_pos;
        let (pixel_metres, orthographic) = self.streaming_pixel_metres();
        let asks: Vec<([u32; 5], f32, f32)> = match self.model_gpu_cache.get(&handle_bits) {
            Some(Some(meshes)) => meshes
                .iter()
                .map(|mesh| {
                    // The mesh is drawn through translate(position) * scale *
                    // its own source transform: bounds and density follow it.
                    let t = &mesh.source_transform;
                    let s = scale.abs().max(1e-4);
                    let source_scale = (0..3)
                        .map(|c| (t[c][0].powi(2) + t[c][1].powi(2) + t[c][2].powi(2)).sqrt())
                        .fold(0.0f32, f32::max)
                        .max(1e-4);
                    let distance = if orthographic {
                        1.0
                    } else if mesh.local_min[0] <= mesh.local_max[0] {
                        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
                        for corner in 0..8 {
                            let p = [0, 1, 2].map(|a| {
                                if corner >> a & 1 == 0 {
                                    mesh.local_min[a]
                                } else {
                                    mesh.local_max[a]
                                }
                            });
                            for a in 0..3 {
                                let w = position[a]
                                    + s * (t[0][a] * p[0]
                                        + t[1][a] * p[1]
                                        + t[2][a] * p[2]
                                        + t[3][a]);
                                lo[a] = lo[a].min(w);
                                hi[a] = hi[a].max(w);
                            }
                        }
                        distance_to_box(camera, lo, hi)
                    } else {
                        0.0
                    };
                    (
                        mesh.material_textures,
                        mesh.uv_density / (s * source_scale),
                        distance,
                    )
                })
                .collect(),
            _ => return,
        };
        for (slots, density, distance) in asks {
            self.want_textures(slots, density, distance, pixel_metres);
        }
    }

    /// Once a frame: turn what was asked for into at most a few swaps.
    fn texture_streaming_schedule(&mut self) {
        self.finish_level_reads();
        if self.texture_streaming.entries.is_empty() {
            return;
        }
        for entry in self.texture_streaming.entries.values_mut() {
            // Nothing on screen showed it: it could go back to its floor.
            let wanted = entry.wanted.min(entry.floor_level);
            entry.wanted = u32::MAX;
            if wanted > entry.resident_level {
                entry.surplus_ticks = entry.surplus_ticks.saturating_add(1);
            } else {
                entry.surplus_ticks = 0;
            }
            // `wanted` is read back by the plan below through this field.
            entry.planned = wanted;
        }
        let plan = {
            let view: Vec<(u32, u32, u32, u32, &[u64])> = self
                .texture_streaming
                .entries
                .iter()
                .map(|(&slot, e)| {
                    // A texture with a read in flight counts as what it will
                    // be, and asks for nothing more until the read is back.
                    let level = e.pending.unwrap_or(e.resident_level);
                    let wanted = if e.pending.is_some() {
                        level
                    } else {
                        e.planned
                    };
                    (
                        slot,
                        level,
                        wanted,
                        e.surplus_ticks,
                        e.chain_bytes.as_slice(),
                    )
                })
                .collect();
            let in_flight = view
                .iter()
                .zip(self.texture_streaming.entries.values())
                .filter(|(_, e)| e.pending.is_some())
                .count();
            let raises = SCHEDULED_RAISES_PER_FRAME.min(READS_IN_FLIGHT.saturating_sub(in_flight));
            plan_tick(
                &view,
                self.texture_streaming.budget.unwrap_or_else(budget_bytes),
                raises,
            )
        };
        // Lowerings are reads too. The plan puts each one before the raise it
        // makes room for, so cutting it short leaves a plan that still holds.
        let in_flight = self
            .texture_streaming
            .entries
            .values()
            .filter(|e| e.pending.is_some())
            .count();
        let room = READS_IN_FLIGHT.saturating_sub(in_flight);
        for (slot, level) in plan.into_iter().take(room) {
            if !self.request_texture_resident_level(slot, level) {
                self.forget_streamed_texture(slot); // unreadable: leave it as it is
            }
        }
    }

    /// Once a frame.
    pub(crate) fn texture_streaming_tick(&mut self) {
        static RAISE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*RAISE.get_or_init(|| env_flag("BLOOM_TEXTURE_STREAM_RAISE").unwrap_or(false)) {
            self.texture_streaming_schedule();
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
    fn the_needed_level_follows_distance_and_density() {
        // 2048 texels over one metre, a 1080-line screen with a 90 degree view:
        // one pixel covers 2/1080 m at 1 m. At 4 m that is 15 texels a pixel:
        // level 3 would do, and the margin of one level makes it 2.
        let pixel = 2.0 / 1080.0;
        assert_eq!(needed_level(2048, 1.0, 0.5, pixel), 0, "close: everything");
        assert_eq!(needed_level(2048, 1.0, 4.0, pixel), 2);
        assert_eq!(needed_level(2048, 1.0, 64.0, pixel), 6);
        // the same texture tiled four times a metre needs it four times as far
        assert_eq!(needed_level(2048, 4.0, 16.0, pixel), 6);
        assert_eq!(
            needed_level(2048, 0.0, 500.0, pixel),
            0,
            "unknown density: everything"
        );
    }

    #[test]
    fn uv_density_is_texture_coordinates_per_metre() {
        let v = |x: f32, y: f32, u: f32, w: f32| super::super::Vertex3D {
            position: [x, y, 0.0],
            uv: [u, w],
            ..Default::default()
        };
        // a 2 m square showing the texture once: half a coordinate per metre
        let quad = [
            v(0., 0., 0., 0.),
            v(2., 0., 1., 0.),
            v(2., 2., 1., 1.),
            v(0., 2., 0., 1.),
        ];
        let indices = [0, 1, 2, 0, 2, 3];
        assert!((uv_density(&quad, &indices) - 0.5).abs() < 1e-5);
        let flat = [
            v(0., 0., 0., 0.),
            v(2., 0., 0., 0.),
            v(2., 2., 0., 0.),
            v(0., 2., 0., 0.),
        ];
        assert_eq!(uv_density(&flat, &indices), 0.0);
    }

    #[test]
    fn the_scheduler_raises_the_worst_shortfall_and_lowers_only_under_pressure() {
        let chain: Vec<u64> = vec![1000, 250, 60, 15];
        let c = chain.as_slice();
        // slot, resident, wanted, surplus ticks, chain
        let plenty = 10_000;
        // room for everything: the larger shortfall goes first, nothing is lowered
        let entries = [(1, 3, 2, 0, c), (2, 3, 0, 0, c), (3, 0, 3, 500, c)];
        assert_eq!(plan_tick(&entries, plenty, 1), vec![(2, 0)]);
        // no room: the texture that has held too much for long is taken back first
        let entries = [(2, 3, 0, 0, c), (3, 0, 3, 500, c)];
        assert_eq!(plan_tick(&entries, 1015, 1), vec![(3, 3), (2, 0)]);
        // a surplus too recent to touch: the raise settles for what fits
        let entries = [(2, 3, 0, 0, c), (3, 0, 3, 5, c)];
        assert_eq!(plan_tick(&entries, 1300, 1), vec![(2, 1)]);
        // nothing fits at all: nothing happens
        assert_eq!(plan_tick(&entries, 1015, 1), vec![]);
        // over budget with nothing asked for: the long-held surplus goes
        let entries = [(3, 0, 3, 500, c), (4, 0, 0, 0, c)];
        assert_eq!(plan_tick(&entries, 1500, 1), vec![(3, 3)]);
    }

    #[test]
    fn the_worker_reads_a_level_chain_from_its_place_in_a_file() {
        let whole = image(1024);
        let ktx2 = whole.to_ktx2().unwrap();
        let path = std::env::temp_dir().join(format!(
            "bloom_stream_test_{}_{:?}.bin",
            std::process::id(),
            std::thread::current().id()
        ));
        // the image sits after other bytes, as inside a GLB
        let mut file = vec![0xEEu8; 37];
        file.extend_from_slice(&ktx2);
        file.extend_from_slice(&[0xEE; 11]);
        std::fs::write(&path, &file).unwrap();
        let source = StreamSource {
            path: path.clone(),
            offset: 37,
            length: ktx2.len() as u64,
        };
        let (level, tail) = load_levels(&source, 2).unwrap();
        assert_eq!((level, tail.width, tail.height), (2, 256, 256));
        assert_eq!(tail.levels[0], whole.levels[2]);

        let reader = LevelReader::start().expect("threads on native targets");
        reader.requests.send((9, source.clone(), 1)).unwrap();
        let missing = StreamSource {
            path: path.with_extension("gone"),
            ..source
        };
        reader.requests.send((10, missing, 0)).unwrap();
        let wait = std::time::Duration::from_secs(20);
        let (slot, level, image) = reader.results.recv_timeout(wait).unwrap();
        assert_eq!((slot, level, image.map(|i| i.width)), (9, 1, Some(512)));
        let (slot, _, image) = reader.results.recv_timeout(wait).unwrap();
        assert_eq!(
            (slot, image.is_none()),
            (10, true),
            "an unreadable file is reported"
        );
        let _ = std::fs::remove_file(&path);
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
    fn only_a_texture_whose_levels_are_all_whole_blocks_is_streamed() {
        assert!(every_level_is_whole_blocks(&image(2048), 3));
        // 4100 px: level 0 is whole blocks, level 1 (2050 px) is not
        let mut odd = image(4096);
        odd.width = 4100;
        odd.height = 4100;
        assert!(!every_level_is_whole_blocks(&odd, 4));
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
