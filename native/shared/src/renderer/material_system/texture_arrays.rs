//! Texture-array creation lives outside `material_system.rs` so the core
//! material lifecycle and dispatch code remain reviewable.

use super::{map_texture_array_format, MaterialSystem, TextureArray, MAX_TEXTURE_ARRAY_LAYERS};
use crate::renderer::material_indirection::{ResidentTexture, TextureColorSpace, TextureSemantic};

impl MaterialSystem {
    /// Create a 2D texture array from RGBA8 layers using the default sRGB
    /// format and a single mip. Returns a 1-based handle, or zero on error.
    pub fn create_texture_array(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layers: &[(&[u8], u32, u32)],
    ) -> u32 {
        self.create_texture_array_ex(device, queue, layers, 0, 1)
    }

    /// Create a 2D texture array with explicit color-space and mip control.
    ///
    /// `format` is 0 for sRGB color and 1 for linear data. `mip_levels` is
    /// 1 for a single mip; 0 or values above 1 generate the full chain.
    pub fn create_texture_array_ex(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layers: &[(&[u8], u32, u32)],
        format: u32,
        mip_levels: u32,
    ) -> u32 {
        let layer_count = (layers.len() as u32).min(MAX_TEXTURE_ARRAY_LAYERS);
        if layer_count == 0 {
            return 0;
        }
        let (_first_bytes, width, height) = layers[0];
        if width == 0 || height == 0 {
            return 0;
        }
        for (index, (_, layer_width, layer_height)) in
            layers.iter().enumerate().take(layer_count as usize)
        {
            if *layer_width != width || *layer_height != height {
                eprintln!(
                    "[texture_array] layer {} extent {}×{} does not match layer 0 ({}×{}); aborting create",
                    index, layer_width, layer_height, width, height,
                );
                return 0;
            }
        }

        let wgpu_format = map_texture_array_format(format);
        let max_mips = (width.max(height) as f32).log2().floor() as u32 + 1;
        let auto_generate = mip_levels == 0 || mip_levels > 1;
        let mip_level_count = if mip_levels == 1 { 1 } else { max_mips.max(1) };
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        if auto_generate && mip_level_count > 1 {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }

        let bytes_per_layer = (width as usize) * (height as usize) * 4;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material_texture_array"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layer_count,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage,
            view_formats: &[],
        });
        for (index, (bytes, _, _)) in layers.iter().enumerate().take(layer_count as usize) {
            if bytes.len() < bytes_per_layer {
                eprintln!(
                    "[texture_array] layer {} short: {} B < {} B (skipping)",
                    index,
                    bytes.len(),
                    bytes_per_layer,
                );
                continue;
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: index as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes[..bytes_per_layer],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Box-filtered mips, built on the CPU and uploaded per level.
        //
        // This used to be a chain of `copy_texture_to_texture` calls from the top-left
        // quarter of each level into the next ("point-copy"). That is not a downsample, it is
        // a CROP: level n held the top-left 1/2^n of the image at full texel size, so a
        // surface seen from a distance showed its texture MAGNIFIED 2^n times instead of
        // minified (a brick wall across a plaza turned into metre-long bricks, a grass scan
        // into smeared streaks). Colour is averaged in the stored encoding, which for sRGB
        // layers darkens fine high-contrast detail very slightly; correct enough for scans.
        if auto_generate && mip_level_count > 1 {
            for (index, (bytes, _, _)) in layers.iter().enumerate().take(layer_count as usize) {
                if bytes.len() < bytes_per_layer {
                    continue;
                }
                let mut prev: Vec<u8> = bytes[..bytes_per_layer].to_vec();
                let (mut pw, mut ph) = (width as usize, height as usize);
                for mip in 1..mip_level_count {
                    let nw = (pw / 2).max(1);
                    let nh = (ph / 2).max(1);
                    let mut next = vec![0u8; nw * nh * 4];
                    for y in 0..nh {
                        let y0 = (y * 2).min(ph - 1);
                        let y1 = (y * 2 + 1).min(ph - 1);
                        for x in 0..nw {
                            let x0 = (x * 2).min(pw - 1);
                            let x1 = (x * 2 + 1).min(pw - 1);
                            let o = (y * nw + x) * 4;
                            for c in 0..4 {
                                let sum = prev[(y0 * pw + x0) * 4 + c] as u32
                                    + prev[(y0 * pw + x1) * 4 + c] as u32
                                    + prev[(y1 * pw + x0) * 4 + c] as u32
                                    + prev[(y1 * pw + x1) * 4 + c] as u32;
                                next[o + c] = ((sum + 2) / 4) as u8;
                            }
                        }
                    }
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: mip,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: index as u32,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        &next,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(nw as u32 * 4),
                            rows_per_image: Some(nh as u32),
                        },
                        wgpu::Extent3d {
                            width: nw as u32,
                            height: nh as u32,
                            depth_or_array_layers: 1,
                        },
                    );
                    prev = next;
                    pw = nw;
                    ph = nh;
                }
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("material_texture_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let texture_id = self.indirection.register_texture(ResidentTexture {
            view: view.clone(),
            width,
            height,
            mip_count: mip_level_count,
            color_space: if wgpu_format.is_srgb() {
                TextureColorSpace::Srgb
            } else {
                TextureColorSpace::Linear
            },
            semantic: TextureSemantic::General,
            hardware_srgb_decode: wgpu_format.is_srgb(),
            global_2d: false,
        });
        self.texture_arrays.push(Some(TextureArray {
            texture,
            view,
            layer_count,
        }));
        self.texture_array_ids.push(texture_id);
        self.texture_arrays.len() as u32
    }
}
