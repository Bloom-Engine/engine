//! Game-texture storage on the Renderer: registration (with mip
//! generation), in-place updates, unload (VRAM release + white
//! placeholder), filter selection, and the model GPU-cache eviction
//! hook. Split out of renderer/mod.rs (2000-line file policy); the
//! backing fields (`textures`, `texture_bind_groups`, `texture_sizes`)
//! still live on [`Renderer`].

use super::Renderer;
use wgpu::util::DeviceExt;

impl Renderer {

    // (encode_png_simple is defined as a free function below the impl
    // block so it can be reused by other capture paths if needed.)

    pub fn register_texture(&mut self, width: u32, height: u32, data: &[u8]) -> u32 {
        self.register_texture_kind(width, height, data, false)
    }

    /// Single-mip texture for dynamically updated atlases.
    pub fn register_texture_no_mips(&mut self, width: u32, height: u32, data: &[u8]) -> u32 {
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("atlas_no_mips"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor, data,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas_bg"), layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let idx = self.texture_bind_groups.len() as u32;
        self.texture_bind_groups.push(bind_group);
        self.textures.push(texture);
        self.texture_sizes.push((width, height));
        idx
    }

    /// Replace an existing no-mips texture in-place.
    pub fn replace_texture_no_mips(&mut self, idx: u32, width: u32, height: u32, data: &[u8]) {
        let i = idx as usize;
        if i >= self.textures.len() { return; }
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("atlas_replaced"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor, data,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas_replaced_bg"), layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.textures[i] = texture;
        self.texture_bind_groups[i] = bind_group;
        self.texture_sizes[i] = (width, height);
    }

    /// Register a texture with optional normal-map preprocessing.
    ///
    /// For normal maps (is_normal_map=true), mip chain is built with
    /// vector-space averaging instead of scalar RGB averaging, and
    /// per-mip variance (1 - |vector_avg|²) is baked into the alpha
    /// channel. The shader reads alpha as a Toksvig-style σ² addition
    /// that accumulates normal-direction disagreement across the
    /// footprint the sampler ends up integrating — the simplified
    /// scalar LEADR/LEAN filter. Alpha is unused by glTF normal maps
    /// (they carry (x,y,z) in RGB) so we can safely repurpose it.
    pub fn register_texture_kind(
        &mut self,
        width: u32,
        height: u32,
        data: &[u8],
        is_normal_map: bool,
    ) -> u32 {
        let max_dim = if width > height { width } else { height };
        // On Android/Vulkan, multi-level mipmap upload can fail silently.
        // Use single mip for 2D textures; only generate mipmaps on desktop.
        #[cfg(target_os = "android")]
        let mip_count = 1u32;
        #[cfg(not(target_os = "android"))]
        let mip_count = (max_dim as f32).log2().floor() as u32 + 1;

        // Generate mip chain data
        let mut mip_data = Vec::with_capacity(data.len() * 2); // overallocate
        if is_normal_map {
            // Level 0: normalize input RGB and clear alpha to 0 (no
            // variance at the finest level — each texel is assumed unit).
            mip_data.reserve(data.len());
            for i in 0..(width as usize * height as usize) {
                let r = data[i * 4];
                let g = data[i * 4 + 1];
                let b = data[i * 4 + 2];
                mip_data.push(r);
                mip_data.push(g);
                mip_data.push(b);
                mip_data.push(0);
            }
        } else {
            mip_data.extend_from_slice(data);
        }
        let mut mip_offsets = vec![0usize]; // byte offset of each mip level
        let mut mw = width;
        let mut mh = height;
        for _ in 1..mip_count {
            let prev_offset = *mip_offsets.last().unwrap();
            let pw = mw as usize; // previous width
            let ph = mh as usize; // previous height
            mw = if mw > 1 { mw / 2 } else { 1 };
            mh = if mh > 1 { mh / 2 } else { 1 };
            mip_offsets.push(mip_data.len());
            for y in 0..mh as usize {
                for x in 0..mw as usize {
                    let sx = x * 2;
                    let sy = y * 2;
                    let sx1 = (sx + 1).min(pw - 1);
                    let sy1 = (sy + 1).min(ph - 1);
                    if is_normal_map {
                        // Decode 4 children to signed [-1, 1] vectors
                        let dec = |r: u8, g: u8, b: u8| -> [f32; 3] {
                            [
                                r as f32 * (2.0 / 255.0) - 1.0,
                                g as f32 * (2.0 / 255.0) - 1.0,
                                b as f32 * (2.0 / 255.0) - 1.0,
                            ]
                        };
                        let idx = |sx: usize, sy: usize| -> usize {
                            prev_offset + (sy * pw + sx) * 4
                        };
                        let n00 = dec(mip_data[idx(sx, sy)], mip_data[idx(sx, sy) + 1], mip_data[idx(sx, sy) + 2]);
                        let n10 = dec(mip_data[idx(sx1, sy)], mip_data[idx(sx1, sy) + 1], mip_data[idx(sx1, sy) + 2]);
                        let n01 = dec(mip_data[idx(sx, sy1)], mip_data[idx(sx, sy1) + 1], mip_data[idx(sx, sy1) + 2]);
                        let n11 = dec(mip_data[idx(sx1, sy1)], mip_data[idx(sx1, sy1) + 1], mip_data[idx(sx1, sy1) + 2]);
                        // Previous-mip baked variances
                        let v00 = mip_data[idx(sx, sy) + 3] as f32 / 255.0;
                        let v10 = mip_data[idx(sx1, sy) + 3] as f32 / 255.0;
                        let v01 = mip_data[idx(sx, sy1) + 3] as f32 / 255.0;
                        let v11 = mip_data[idx(sx1, sy1) + 3] as f32 / 255.0;
                        // Average the vectors
                        let avg_x = (n00[0] + n10[0] + n01[0] + n11[0]) * 0.25;
                        let avg_y = (n00[1] + n10[1] + n01[1] + n11[1]) * 0.25;
                        let avg_z = (n00[2] + n10[2] + n01[2] + n11[2]) * 0.25;
                        let len_sq = avg_x * avg_x + avg_y * avg_y + avg_z * avg_z;
                        let len = len_sq.sqrt().max(1e-6);
                        // Normalize direction (what the shader reads as
                        // the shading normal). Re-encode to [0, 255].
                        let encode = |v: f32| -> u8 {
                            ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
                        };
                        mip_data.push(encode(avg_x / len));
                        mip_data.push(encode(avg_y / len));
                        mip_data.push(encode(avg_z / len));
                        // Variance at this mip = disagreement among the
                        // 4 children (1 - |avg|²) PLUS the weighted mean
                        // of the children's own variances. Both live in
                        // [0, 1]; combined variance clamped.
                        let v_children_avg = (v00 + v10 + v01 + v11) * 0.25;
                        let v_local = (1.0 - len_sq).max(0.0);
                        let v_out = (v_local + v_children_avg).min(1.0);
                        mip_data.push((v_out * 255.0).round().clamp(0.0, 255.0) as u8);
                    } else {
                        for c in 0..4usize {
                            let p00 = mip_data[prev_offset + (sy * pw + sx) * 4 + c] as u32;
                            let p10 = mip_data[prev_offset + (sy * pw + sx1) * 4 + c] as u32;
                            let p01 = mip_data[prev_offset + (sy1 * pw + sx) * 4 + c] as u32;
                            let p11 = mip_data[prev_offset + (sy1 * pw + sx1) * 4 + c] as u32;
                            mip_data.push(((p00 + p10 + p01 + p11 + 2) / 4) as u8);
                        }
                    }
                }
            }
        }

        let texture = if mip_count == 1 {
            // Simple path: single mip level, use create_texture_with_data
            self.device.create_texture_with_data(
                &self.queue,
                &wgpu::TextureDescriptor {
                    label: Some("registered_texture"),
                    size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &mip_data[..((width * height * 4) as usize)],
            )
        } else {
            // Multi-mip path: create texture, upload each level
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("registered_texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let mut lw = width;
            let mut lh = height;
            for level in 0..mip_count {
                let offset = mip_offsets[level as usize];
                let level_size = (lw * lh * 4) as usize;
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex, mip_level: level,
                        origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
                    },
                    &mip_data[offset..offset + level_size],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0, bytes_per_row: Some(4 * lw), rows_per_image: Some(lh),
                    },
                    wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
                );
                lw = if lw > 1 { lw / 2 } else { 1 };
                lh = if lh > 1 { lh / 2 } else { 1 };
            }
            tex
        };

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bg"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });

        let idx = self.texture_bind_groups.len() as u32;
        self.texture_bind_groups.push(bind_group);
        self.textures.push(texture);
        self.texture_sizes.push((width, height));
        idx
    }

    pub fn update_texture(&mut self, idx: u32, width: u32, height: u32, data: &[u8]) {
        let i = idx as usize;
        if i >= self.textures.len() { return; }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.textures[i],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }

    pub fn unload_texture(&mut self, idx: u32) {
        // Replace the slot's texture with a 1×1 white placeholder so the
        // heavyweight texture drops (wgpu releases the VRAM once the
        // queue is done with it). The previous implementation only
        // zeroed the size entry and kept the texture alive forever —
        // load/unload cycles grew VRAM unboundedly.
        //
        // Slots are intentionally NOT reused: a stale bind_group_idx
        // held by a scene material renders white rather than ever
        // aliasing a texture loaded later. The retired slot costs a
        // handful of bytes, not VRAM.
        let i = idx as usize;
        if i == 0 || i >= self.textures.len() {
            return;
        }
        let white = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("unloaded_texture_placeholder"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let view = white.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unloaded_texture_placeholder_bg"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.textures[i] = white;
        self.texture_bind_groups[i] = bind_group;
        self.texture_sizes[i] = (0, 0);
    }

    /// Drop a model's cached GPU meshes. Must be called when the model is
    /// unloaded: the cache is keyed by handle bits, so without eviction
    /// the buffers leak — and a future model whose handle reuses the slot
    /// would render the stale cached geometry.
    pub fn evict_model_cache(&mut self, handle_bits: u64) {
        self.model_gpu_cache.remove(&handle_bits);
    }

    pub fn set_texture_filter(&mut self, idx: u32, nearest: bool) {
        let i = idx as usize;
        if i >= self.textures.len() { return; }
        let view = self.textures[i].create_view(&wgpu::TextureViewDescriptor::default());
        let chosen_sampler = if nearest { &self.nearest_sampler } else { &self.sampler };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bg_refiltered"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(chosen_sampler) },
            ],
        });
        self.texture_bind_groups[i] = bind_group;
    }
}
