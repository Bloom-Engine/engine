//! Colour grading through a lookup table.
//!
//! A grade is a display-referred 3D table: sRGB-encoded colour in, sRGB-encoded
//! colour out, the same thing every grading tool exports. The final composite
//! applies it after the tonemap and the saturation control (which is the same
//! space), blended by a strength, and converts back to linear for the
//! swapchain's encode.
//!
//! The file format is the common **strip**: an image `N * N` wide and `N` high
//! made of `N` tiles side by side, red along each tile's x, green along its y,
//! blue from tile to tile. `N` is 16, 32 or 64 (anything from 2 to 64 works).
//! To author one, write the neutral strip with [`neutral_strip`] (or
//! `writeNeutralColorGradeLut` from TypeScript), run it through the same
//! adjustments as a screenshot in any image editor, and load the result.
//!
//! With no table loaded the composite binds a 2x2x2 identity and, at strength
//! 0, never samples it.

use super::Renderer;

/// Largest table edge accepted.
pub(super) const GRADE_LUT_MAX: u32 = 64;

pub(super) struct ColorGrade {
    _texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
    /// Texels per axis of the bound table.
    pub(super) size: u32,
    /// 0 = off (the table is not sampled), 1 = the grade as authored.
    pub(super) strength: f32,
}

/// The RGBA8 texels of an identity table of edge `n`, in 3D-texture order
/// (x = red fastest, then y = green, then z = blue).
pub(super) fn identity_texels(n: u32) -> Vec<u8> {
    let last = (n.max(2) - 1) as f32;
    let mut out = Vec::with_capacity((n * n * n * 4) as usize);
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                out.push((r as f32 / last * 255.0).round() as u8);
                out.push((g as f32 / last * 255.0).round() as u8);
                out.push((b as f32 / last * 255.0).round() as u8);
                out.push(255);
            }
        }
    }
    out
}

/// A strip image (`n * n` by `n`, RGBA8, rows top to bottom) to 3D-texture
/// order. `None` if the dimensions are not a strip.
pub(super) fn strip_to_texels(rgba: &[u8], width: u32, height: u32) -> Option<(Vec<u8>, u32)> {
    let n = height;
    if !(2..=GRADE_LUT_MAX).contains(&n) || width != n * n {
        return None;
    }
    if rgba.len() != (width * height * 4) as usize {
        return None;
    }
    let mut out = Vec::with_capacity((n * n * n * 4) as usize);
    for b in 0..n {
        for g in 0..n {
            let row = (g * width + b * n) as usize * 4;
            out.extend_from_slice(&rgba[row..row + (n as usize) * 4]);
        }
    }
    Some((out, n))
}

/// The neutral strip for edge `n`: `(rgba, width, height)`. Grade this image
/// and load it back.
pub fn neutral_strip(n: u32) -> (Vec<u8>, u32, u32) {
    let n = n.clamp(2, GRADE_LUT_MAX);
    let texels = identity_texels(n);
    let (width, height) = (n * n, n);
    let mut out = vec![0u8; (width * height * 4) as usize];
    for b in 0..n {
        for g in 0..n {
            let src = ((b * n + g) * n) as usize * 4;
            let dst = (g * width + b * n) as usize * 4;
            out[dst..dst + (n as usize) * 4].copy_from_slice(&texels[src..src + (n as usize) * 4]);
        }
    }
    (out, width, height)
}

/// Write the neutral strip of edge `n` to `path` as a PNG.
pub fn write_neutral(path: &str, n: u32) -> bool {
    let (rgba, w, h) = neutral_strip(n);
    #[cfg(not(target_arch = "wasm32"))]
    {
        image::RgbaImage::from_raw(w, h, rgba)
            .is_some_and(|img| img.save_with_format(path, image::ImageFormat::Png).is_ok())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (path, rgba, w, h);
        false
    }
}

impl ColorGrade {
    fn from_texels(device: &wgpu::Device, queue: &wgpu::Queue, texels: &[u8], n: u32) -> Self {
        let size = wgpu::Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: n,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color_grade_lut"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            // Unorm, not sRGB: the table's values ARE display-encoded and the
            // shader wants them as stored.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(n * 4),
                rows_per_image: Some(n),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
            size: n,
            strength: 0.0,
        }
    }

    /// The identity, unsampled: what a renderer starts with.
    pub(super) fn identity(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_texels(device, queue, &identity_texels(2), 2)
    }
}

impl Renderer {
    /// Load a grade from an encoded strip image (PNG). Returns false, and
    /// leaves the current grade alone, if the image is not a strip.
    pub fn set_color_grade_lut(&mut self, file_data: &[u8], strength: f32) -> bool {
        let Ok(img) = image::load_from_memory(file_data) else {
            return false;
        };
        let img = img.to_rgba8();
        let Some((texels, n)) = strip_to_texels(img.as_raw(), img.width(), img.height()) else {
            return false;
        };
        let mut grade = ColorGrade::from_texels(&self.device, &self.queue, &texels, n);
        grade.strength = strength.clamp(0.0, 1.0);
        self.color_grade = grade;
        // The composite's bind groups hold the old table's view.
        self.composite_bind_group_cache = std::array::from_fn(|_| None);
        true
    }

    /// Blend of the loaded grade: 0 = off, 1 = as authored.
    pub fn set_color_grade_strength(&mut self, strength: f32) {
        self.color_grade.strength = strength.clamp(0.0, 1.0);
    }

    /// Back to no grade.
    pub fn clear_color_grade(&mut self) {
        self.color_grade = ColorGrade::identity(&self.device, &self.queue);
        self.composite_bind_group_cache = std::array::from_fn(|_| None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_neutral_strip_loads_as_the_identity() {
        for n in [2u32, 16, 32] {
            let (rgba, w, h) = neutral_strip(n);
            assert_eq!((w, h), (n * n, n));
            let (texels, size) = strip_to_texels(&rgba, w, h).expect("a strip");
            assert_eq!(size, n);
            assert_eq!(texels, identity_texels(n));
        }
    }

    #[test]
    fn the_strip_puts_blue_across_tiles_and_green_down() {
        let n = 4u32;
        let (rgba, w, _) = neutral_strip(n);
        let px = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [rgba[i], rgba[i + 1], rgba[i + 2]]
        };
        assert_eq!(px(0, 0), [0, 0, 0]);
        assert_eq!(px(n - 1, 0), [255, 0, 0]); // red along a tile
        assert_eq!(px(0, n - 1), [0, 255, 0]); // green down it
        assert_eq!(px(n * (n - 1), 0), [0, 0, 255]); // blue in the last tile
        assert_eq!(px(n * n - 1, n - 1), [255, 255, 255]);
    }

    #[test]
    fn anything_that_is_not_a_strip_is_refused() {
        assert!(strip_to_texels(&[0; 64 * 64 * 4], 64, 64).is_none());
        assert!(strip_to_texels(&[0; 4], 1, 1).is_none());
        assert!(strip_to_texels(&[0; 10], 16, 4).is_none());
    }
}
