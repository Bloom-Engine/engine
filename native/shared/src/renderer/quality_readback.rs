//! Texture readback for the qualification captures (quality_capture.rs).

use super::Renderer;

#[derive(Clone, Copy)]
pub(super) enum ReadbackKind {
    Hdr,
    Depth,
    Rgba8,
}

pub(super) struct QualityReadback {
    pub(super) name: &'static str,
    pub(super) kind: ReadbackKind,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) padded_bytes_per_row: u32,
    pub(super) buffer: wgpu::Buffer,
}

impl Renderer {
    pub(super) fn record_quality_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        name: &'static str,
        kind: ReadbackKind,
        aspect: wgpu::TextureAspect,
        bytes_per_pixel: u32,
    ) -> QualityReadback {
        self.record_quality_texture_layer(encoder, texture, 0, name, kind, aspect, bytes_per_pixel)
    }

    /// `record_quality_texture` for one layer of an array texture.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_quality_texture_layer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        layer: u32,
        name: &'static str,
        kind: ReadbackKind,
        aspect: wgpu::TextureAspect,
        bytes_per_pixel: u32,
    ) -> QualityReadback {
        let size = texture.size();
        let unpadded_bytes_per_row = size.width * bytes_per_pixel;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom_quality_intermediate_readback"),
            size: (padded_bytes_per_row * size.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(size.height),
                },
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );
        QualityReadback {
            name,
            kind,
            width: size.width,
            height: size.height,
            padded_bytes_per_row,
            buffer,
        }
    }
}
