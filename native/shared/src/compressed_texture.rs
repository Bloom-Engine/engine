//! Block-compressed textures that arrive ready for the GPU.
//!
//! A glTF image may be a KTX2 container holding ASTC or BC7 blocks with a
//! complete mip chain (`bloom-cook glb-compress` writes them). On a device
//! with the matching feature the blocks upload untouched: a 4096² texture
//! costs 22 MB (ASTC 4×4 / BC7) or 10 MB (ASTC 6×6) instead of the 85 MB of
//! RGBA8 plus mips, and nothing is decoded or filtered at load. Everywhere
//! else level zero is decoded on the CPU and the image takes the ordinary
//! RGBA8 path, so a compressed asset still renders on every adapter.
//!
//! Mips are authored, never derived: the cooker stores the same chains the
//! runtime builds for RGBA8 uploads (variance in normal-map alpha, sRGB-aware
//! colour averaging), so both paths shade alike.

/// KTX2 file identifier.
pub const KTX2_MAGIC: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// True when `bytes` starts with the KTX2 identifier.
pub fn is_ktx2(bytes: &[u8]) -> bool {
    bytes.len() >= KTX2_MAGIC.len() && bytes[..KTX2_MAGIC.len()] == KTX2_MAGIC
}

/// Block encodings the renderer uploads directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockFormat {
    /// ASTC LDR with the given block footprint in texels.
    Astc { width: u8, height: u8 },
    /// BC7 (4×4 blocks).
    Bc7,
}

/// Vulkan format numbers of the supported ASTC footprints (UNORM; +1 = sRGB).
const ASTC_VK_FORMATS: [(u32, u8, u8); 14] = [
    (157, 4, 4),
    (159, 5, 4),
    (161, 5, 5),
    (163, 6, 5),
    (165, 6, 6),
    (167, 8, 5),
    (169, 8, 6),
    (171, 8, 8),
    (173, 10, 5),
    (175, 10, 6),
    (177, 10, 8),
    (179, 10, 10),
    (181, 12, 10),
    (183, 12, 12),
];
const BC7_VK_FORMAT: u32 = 145;

impl BlockFormat {
    /// `(format, container says sRGB)` for a Vulkan format number.
    pub fn from_vk_format(value: u32) -> Option<(Self, bool)> {
        if value == BC7_VK_FORMAT || value == BC7_VK_FORMAT + 1 {
            return Some((Self::Bc7, value != BC7_VK_FORMAT));
        }
        ASTC_VK_FORMATS
            .iter()
            .find(|(vk, _, _)| value == *vk || value == *vk + 1)
            .map(|(vk, width, height)| {
                (
                    Self::Astc {
                        width: *width,
                        height: *height,
                    },
                    value != *vk,
                )
            })
    }

    /// The Vulkan format number, in its UNORM or sRGB flavour.
    pub fn vk_format(self, srgb: bool) -> Option<u32> {
        let base = match self {
            Self::Bc7 => BC7_VK_FORMAT,
            Self::Astc { width, height } => {
                ASTC_VK_FORMATS
                    .iter()
                    .find(|(_, w, h)| *w == width && *h == height)?
                    .0
            }
        };
        Some(base + u32::from(srgb))
    }

    /// Block footprint in texels.
    pub fn block_size(self) -> (u32, u32) {
        match self {
            Self::Astc { width, height } => (u32::from(width), u32::from(height)),
            Self::Bc7 => (4, 4),
        }
    }

    /// Bytes one mip level of `width`×`height` texels occupies (16 per block).
    pub fn level_bytes(self, width: u32, height: u32) -> usize {
        let (block_width, block_height) = self.block_size();
        width.div_ceil(block_width) as usize * height.div_ceil(block_height) as usize * 16
    }

    /// The device feature a direct upload needs.
    pub fn required_feature(self) -> wgpu::Features {
        match self {
            Self::Astc { .. } => wgpu::Features::TEXTURE_COMPRESSION_ASTC,
            Self::Bc7 => wgpu::Features::TEXTURE_COMPRESSION_BC,
        }
    }

    /// The texture format, always UNORM: game textures bind without hardware
    /// sRGB decode because the shaders apply the transfer themselves.
    pub fn texture_format(self) -> Option<wgpu::TextureFormat> {
        let block = match self {
            Self::Bc7 => return Some(wgpu::TextureFormat::Bc7RgbaUnorm),
            Self::Astc { width, height } => match (width, height) {
                (4, 4) => wgpu::AstcBlock::B4x4,
                (5, 4) => wgpu::AstcBlock::B5x4,
                (5, 5) => wgpu::AstcBlock::B5x5,
                (6, 5) => wgpu::AstcBlock::B6x5,
                (6, 6) => wgpu::AstcBlock::B6x6,
                (8, 5) => wgpu::AstcBlock::B8x5,
                (8, 6) => wgpu::AstcBlock::B8x6,
                (8, 8) => wgpu::AstcBlock::B8x8,
                (10, 5) => wgpu::AstcBlock::B10x5,
                (10, 6) => wgpu::AstcBlock::B10x6,
                (10, 8) => wgpu::AstcBlock::B10x8,
                (10, 10) => wgpu::AstcBlock::B10x10,
                (12, 10) => wgpu::AstcBlock::B12x10,
                (12, 12) => wgpu::AstcBlock::B12x12,
                _ => return None,
            },
        };
        Some(wgpu::TextureFormat::Astc {
            block,
            channel: wgpu::AstcChannel::Unorm,
        })
    }
}

/// A parsed block-compressed 2D texture: level zero first.
#[derive(Clone, Debug)]
pub struct CompressedImage {
    pub format: BlockFormat,
    pub width: u32,
    pub height: u32,
    /// What the container declares. The glTF material decides how the texture
    /// is used; this only records authoring intent.
    pub srgb: bool,
    pub levels: Vec<Vec<u8>>,
}

impl CompressedImage {
    /// Parse a KTX2 file. `None` for anything that is not a plain 2D ASTC or
    /// BC7 texture without supercompression, or whose levels are truncated.
    pub fn from_ktx2(bytes: &[u8]) -> Option<Self> {
        if !is_ktx2(bytes) {
            return None;
        }
        let word = |at: usize| -> Option<u32> {
            Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
        };
        let long = |at: usize| -> Option<usize> {
            let value = u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?);
            usize::try_from(value).ok()
        };
        let (format, srgb) = BlockFormat::from_vk_format(word(12)?)?;
        let (width, height) = (word(20)?, word(24)?);
        let (depth, layers, faces) = (word(28)?, word(32)?, word(36)?);
        let level_count = word(40)?.max(1);
        let supercompression = word(44)?;
        if width == 0 || height == 0 || depth != 0 || layers > 1 || faces != 1 {
            return None;
        }
        if supercompression != 0 || level_count > u32::BITS - width.max(height).leading_zeros() {
            return None;
        }
        let mut levels = Vec::with_capacity(level_count as usize);
        for index in 0..level_count as usize {
            let entry = 80 + index * 24;
            let (offset, length) = (long(entry)?, long(entry + 8)?);
            let level_width = (width >> index).max(1);
            let level_height = (height >> index).max(1);
            if length != format.level_bytes(level_width, level_height) {
                return None;
            }
            levels.push(bytes.get(offset..offset.checked_add(length)?)?.to_vec());
        }
        Some(Self {
            format,
            width,
            height,
            srgb,
            levels,
        })
    }

    /// Serialise as KTX2 (no supercompression, smallest level stored first as
    /// the format requires). Inverse of [`Self::from_ktx2`].
    pub fn to_ktx2(&self) -> Option<Vec<u8>> {
        let vk_format = self.format.vk_format(self.srgb)?;
        let (block_width, block_height) = self.format.block_size();
        let level_count = self.levels.len();
        if level_count == 0 {
            return None;
        }
        // Data format descriptor: one basic block with a single 128-bit sample.
        let mut dfd = Vec::with_capacity(44);
        dfd.extend_from_slice(&44u32.to_le_bytes());
        dfd.extend_from_slice(&0u32.to_le_bytes()); // Khronos vendor, basic descriptor
        dfd.extend_from_slice(&2u16.to_le_bytes()); // KDF 1.3
        dfd.extend_from_slice(&40u16.to_le_bytes()); // block size: 24 + one sample
        dfd.push(match self.format {
            BlockFormat::Astc { .. } => 162,
            BlockFormat::Bc7 => 134,
        });
        dfd.push(1); // BT.709 primaries
        dfd.push(if self.srgb { 2 } else { 1 });
        dfd.push(0); // straight alpha
        dfd.extend_from_slice(&[(block_width - 1) as u8, (block_height - 1) as u8, 0, 0]);
        dfd.extend_from_slice(&[16, 0, 0, 0, 0, 0, 0, 0]); // bytes per plane
        dfd.extend_from_slice(&0u16.to_le_bytes()); // bit offset
        dfd.push(127); // bit length - 1
        dfd.push(0); // channel: RGBA
        dfd.extend_from_slice(&[0, 0, 0, 0]); // sample position
        dfd.extend_from_slice(&0u32.to_le_bytes());
        dfd.extend_from_slice(&u32::MAX.to_le_bytes());

        let level_index_offset = 80usize;
        let dfd_offset = level_index_offset + level_count * 24;
        let mut cursor = (dfd_offset + dfd.len()).next_multiple_of(16);
        let mut offsets = vec![0usize; level_count];
        for index in (0..level_count).rev() {
            offsets[index] = cursor;
            cursor = (cursor + self.levels[index].len()).next_multiple_of(16);
        }
        let total = offsets[0] + self.levels[0].len();

        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&KTX2_MAGIC);
        for value in [
            vk_format,
            1, // type size
            self.width,
            self.height,
            0, // depth
            0, // layers
            1, // faces
            level_count as u32,
            0, // no supercompression
            dfd_offset as u32,
            dfd.len() as u32,
            0, // key/value offset
            0, // key/value length
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&0u64.to_le_bytes()); // supercompression global data
        out.extend_from_slice(&0u64.to_le_bytes());
        for index in 0..level_count {
            let length = self.levels[index].len() as u64;
            out.extend_from_slice(&(offsets[index] as u64).to_le_bytes());
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&length.to_le_bytes());
        }
        out.extend_from_slice(&dfd);
        for index in (0..level_count).rev() {
            out.resize(offsets[index], 0);
            out.extend_from_slice(&self.levels[index]);
        }
        Some(out)
    }

    /// Decode level zero to RGBA8 for adapters without the block format.
    #[cfg(feature = "image-extras")]
    pub fn decode_rgba8(&self) -> Option<Vec<u8>> {
        let level = self.levels.first()?;
        match self.format {
            BlockFormat::Bc7 => image_dds::Surface {
                width: self.width,
                height: self.height,
                depth: 1,
                layers: 1,
                mipmaps: 1,
                image_format: image_dds::ImageFormat::BC7RgbaUnorm,
                data: level.as_slice(),
            }
            .decode_rgba8()
            .ok()
            .map(|surface| surface.data),
            BlockFormat::Astc { width, height } => {
                let mut rgba = vec![0u8; self.width as usize * self.height as usize * 4];
                let row = self.width as usize * 4;
                astc_decode::astc_decode(
                    level.as_slice(),
                    self.width,
                    self.height,
                    astc_decode::Footprint::new(u32::from(width), u32::from(height)),
                    |x, y, texel| {
                        let at = y as usize * row + x as usize * 4;
                        rgba[at..at + 4].copy_from_slice(&texel);
                    },
                )
                .ok()?;
                Some(rgba)
            }
        }
    }
}

#[cfg(all(test, feature = "image-extras"))]
mod tests {
    use super::*;

    fn sample(format: BlockFormat, srgb: bool) -> CompressedImage {
        let (width, height) = (20u32, 12u32);
        let levels = (0..5)
            .map(|level| {
                let size = format.level_bytes((width >> level).max(1), (height >> level).max(1));
                (0..size).map(|byte| (byte as u8) ^ (level as u8)).collect()
            })
            .collect();
        CompressedImage {
            format,
            width,
            height,
            srgb,
            levels,
        }
    }

    #[test]
    fn ktx2_round_trip_keeps_every_level() {
        for (format, srgb) in [
            (
                BlockFormat::Astc {
                    width: 4,
                    height: 4,
                },
                true,
            ),
            (
                BlockFormat::Astc {
                    width: 6,
                    height: 6,
                },
                false,
            ),
            (
                BlockFormat::Astc {
                    width: 8,
                    height: 5,
                },
                false,
            ),
            (BlockFormat::Bc7, true),
        ] {
            let image = sample(format, srgb);
            let bytes = image.to_ktx2().expect("serialises");
            let parsed = CompressedImage::from_ktx2(&bytes).expect("parses");
            assert_eq!(parsed.format, format);
            assert_eq!(parsed.srgb, srgb);
            assert_eq!((parsed.width, parsed.height), (image.width, image.height));
            assert_eq!(parsed.levels, image.levels);
        }
    }

    #[test]
    fn truncated_or_foreign_files_are_rejected() {
        let bytes = sample(
            BlockFormat::Astc {
                width: 4,
                height: 4,
            },
            false,
        )
        .to_ktx2()
        .unwrap();
        assert!(CompressedImage::from_ktx2(&bytes[..bytes.len() - 1]).is_none());
        assert!(CompressedImage::from_ktx2(b"\x89PNG\r\n\x1a\n").is_none());
        let mut wrong_format = bytes.clone();
        wrong_format[12..16].copy_from_slice(&37u32.to_le_bytes()); // R8G8B8A8_UNORM
        assert!(CompressedImage::from_ktx2(&wrong_format).is_none());
    }

    #[test]
    fn block_formats_map_to_vulkan_numbers_both_ways() {
        for (vk, width, height) in ASTC_VK_FORMATS {
            let format = BlockFormat::Astc { width, height };
            assert_eq!(BlockFormat::from_vk_format(vk), Some((format, false)));
            assert_eq!(BlockFormat::from_vk_format(vk + 1), Some((format, true)));
            assert_eq!(format.vk_format(true), Some(vk + 1));
            assert!(format.texture_format().is_some());
        }
        assert_eq!(
            BlockFormat::from_vk_format(146),
            Some((BlockFormat::Bc7, true))
        );
        assert_eq!(BlockFormat::from_vk_format(37), None);
    }

    #[test]
    fn a_flat_astc_block_decodes_to_its_colour() {
        // A void-extent block: every texel is the stored 16-bit RGBA colour.
        let mut block = vec![0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        for channel in [0xFFFFu16, 0x8080, 0x0000, 0xFFFF] {
            block.extend_from_slice(&channel.to_le_bytes());
        }
        let image = CompressedImage {
            format: BlockFormat::Astc {
                width: 4,
                height: 4,
            },
            width: 4,
            height: 4,
            srgb: false,
            levels: vec![block],
        };
        let rgba = image.decode_rgba8().expect("decodes");
        assert_eq!(rgba.len(), 64);
        assert!(rgba.chunks(4).all(|texel| texel == [255, 128, 0, 255]));
    }
}
