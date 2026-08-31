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

        // Keep the established point-copy mip behavior. The indirection
        // record still exposes the real mip count to every capability tier.
        if auto_generate && mip_level_count > 1 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("material_texture_array_mipgen"),
            });
            for mip in 1..mip_level_count {
                let src_width = (width >> (mip - 1)).max(1);
                let src_height = (height >> (mip - 1)).max(1);
                let copy_width = (width >> mip).max(1).min(src_width);
                let copy_height = (height >> mip).max(1).min(src_height);
                for layer in 0..layer_count {
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: mip - 1,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: layer,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: mip,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: layer,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: copy_width,
                            height: copy_height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            queue.submit(std::iter::once(encoder.finish()));
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
