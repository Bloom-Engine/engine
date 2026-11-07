//! Renderer surface queries, debug readback, and custom shader registration.

use super::*;

impl Renderer {
    // ============================================================
    // Queries
    // ============================================================

    /// Logical (points / CSS px) width — what user code sees via
    /// `screenWidth` and what 2D HUD coordinates are expressed in.
    /// On HiDPI displays the underlying render target is larger (see
    /// `physical_width`).
    pub fn width(&self) -> u32 {
        self.logical_width
    }

    pub fn height(&self) -> u32 {
        self.logical_height
    }

    /// Physical pixel dimensions of the swapchain and post-process
    /// render targets. Always equal to `width`/`height` on non-HiDPI
    /// platforms; `logical * scale_factor` on Retina/Web.
    pub fn physical_width(&self) -> u32 {
        self.surface_config.width
    }

    pub fn physical_height(&self) -> u32 {
        self.surface_config.height
    }

    /// Actual size of the render-resolution targets (depth / HDR /
    /// G-buffer). Must always equal `render_extent()`: the pass chain
    /// computes viewports and dispatches from the latter and writes into
    /// the former. Pinned by `tests/render_targets.rs`.
    pub fn render_target_extent(&self) -> (u32, u32) {
        let s = self.depth_texture.size();
        (s.width, s.height)
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        // The RENDER format (sRGB view on web), not the configure format —
        // post passes and user post-pass pipelines target frame views.
        self.output_format
    }

    /// Capture the current framebuffer as RGBA pixels.
    /// Returns (width, height, rgba_data). Call after end_frame.
    /// Not available on WASM (requires synchronous GPU readback).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn capture_screenshot(&self) -> Option<(u32, u32, Vec<u8>)> {
        let width = self.surface_config.width;
        let height = self.surface_config.height;
        let bytes_per_pixel = 4u32;
        // wgpu requires rows aligned to 256 bytes
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        // Render one frame to a texture we can copy from
        let output = match self.acquire_frame() {
            Some(t) => t,
            None => return None,
        };
        let texture = self.frame_texture(&output);

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("screenshot_encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer and read pixels
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if rx.recv().ok()?.is_err() {
            return None;
        }

        let data = buffer_slice.get_mapped_range();
        // Remove row padding
        let mut rgba = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + (width * bytes_per_pixel) as usize;
            rgba.extend_from_slice(&data[start..end]);
        }
        drop(data);
        staging_buffer.unmap();
        self.present_frame(output);

        Some((width, height, rgba))
    }

    /// Dump a shadow cascade's depth texture to a grayscale PNG for debugging.
    /// Depth 0.0 (near) → white, depth 1.0 (far / clear) → black.
    /// `cascade` selects which cascade to dump (0, 1, or 2).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn dump_shadow_map(&self, path: &str) {
        self.dump_shadow_cascade(path, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn dump_shadow_cascade(&self, path: &str, cascade: usize) {
        let cascade = cascade.min(crate::shadows::NUM_CASCADES - 1);
        let size = crate::shadows::CASCADE_MAP_SIZE;
        let bytes_per_pixel = 4u32; // Depth32Float = 4 bytes
        let unpadded_bpr = size * bytes_per_pixel;
        let padded_bpr = (unpadded_bpr + 255) & !255;
        let buf_size = (padded_bpr * size) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_dump_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_dump_encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.shadow_map.depth_textures[cascade],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            // Convert f32 depth values to grayscale RGB
            let mut rgb = Vec::with_capacity((size * size * 3) as usize);
            for row in 0..size {
                let row_start = (row * padded_bpr) as usize;
                for col in 0..size {
                    let offset = row_start + (col * bytes_per_pixel) as usize;
                    let depth = f32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    // depth 0 = near (white), depth 1 = far/clear (black)
                    let gray = ((1.0 - depth.clamp(0.0, 1.0)) * 255.0) as u8;
                    rgb.push(gray);
                    rgb.push(gray);
                    rgb.push(gray);
                }
            }
            drop(data);
            if let Some(png) = encode_png_simple(size, size, &rgb) {
                let _ = std::fs::write(path, &png);
            }
        }
        staging.unmap();
    }

    /// Returns true if vsync is active (Fifo or FifoRelaxed present mode).
    pub fn vsync_active(&self) -> bool {
        matches!(
            self.surface_config.present_mode,
            wgpu::PresentMode::Fifo | wgpu::PresentMode::FifoRelaxed
        )
    }

    pub fn load_custom_shader(&mut self, wgsl_source: &str) -> usize {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("custom_shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
            });

        // Create layout matching the default 3D pipeline
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("custom_shader_bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("custom_pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("custom_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: Some("vs_main_3d"),
                    buffers: &[Vertex3D::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: Some("fs_main_3d"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.output_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        self.custom_pipelines.push(pipeline);
        self.created_pipelines(1);
        self.custom_pipelines.len() // 1-based index
    }
}
