//! Opt-in qualification readback and adapter evidence.
//!
//! Nothing in this module runs unless the qualification API requests a
//! capture. Copies are encoded into the same post-measurement command buffer
//! as the final screenshot, then converted to display PNGs on the CPU.

use std::sync::mpsc;

use super::util::encode_png_simple;
use super::weighted_transparency::WEIGHTED_TRANSPARENCY_AUTO_DRAW_THRESHOLD;
use super::Renderer;

#[derive(Clone, Copy)]
enum ReadbackKind {
    Hdr,
    Depth,
    Rgba8,
}

pub(super) struct QualityReadback {
    name: &'static str,
    kind: ReadbackKind,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    buffer: wgpu::Buffer,
}

pub(super) struct FrameReadback {
    staging: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    quality_capture_dir: Option<String>,
    quality_readbacks: Vec<QualityReadback>,
}

fn json_string(out: &mut String, value: &str) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c <= '\u{1f}' => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn enum_name(value: impl std::fmt::Debug) -> String {
    format!("{value:?}").to_ascii_lowercase().replace('_', "-")
}

fn linear_to_srgb(value: f32) -> u8 {
    let value = value.max(0.0);
    let aces = (value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14);
    let clamped = aces.clamp(0.0, 1.0);
    let srgb = if clamped <= 0.003_130_8 {
        12.92 * clamped
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5) as u8
}

fn hdr_rgb(data: &[u8], width: u32, height: u32, padded_bytes_per_row: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for row in 0..height {
        let row_start = (row * padded_bytes_per_row) as usize;
        for column in 0..width {
            let base = row_start + (column * 8) as usize;
            for channel in 0..3 {
                let offset = base + channel * 2;
                let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                rgb.push(linear_to_srgb(half::f16::from_bits(bits).to_f32()));
            }
        }
    }
    rgb
}

fn hdr_luminance_at(data: &[u8], x: u32, y: u32, padded_bytes_per_row: u32) -> Option<f32> {
    let base = (y * padded_bytes_per_row + x * 8) as usize;
    let channel = |index: usize| {
        let offset = base + index * 2;
        half::f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]])).to_f32()
    };
    let luma = 0.2126 * channel(0) + 0.7152 * channel(1) + 0.0722 * channel(2);
    luma.is_finite().then_some(luma.max(0.0))
}

fn hdr_metrics_json(data: &[u8], width: u32, height: u32, padded_bytes_per_row: u32) -> String {
    let mut luminances = Vec::with_capacity((width * height) as usize);
    let mut non_finite = 0usize;
    let mut non_finite_alpha = 0usize;
    let mut max_alpha = 0.0f32;
    let mut hit_alpha_pixels = 0usize;
    for y in 0..height {
        for x in 0..width {
            match hdr_luminance_at(data, x, y, padded_bytes_per_row) {
                Some(luma) => luminances.push(luma),
                None => non_finite += 1,
            }
            let alpha_offset = (y * padded_bytes_per_row + x * 8 + 6) as usize;
            let alpha = half::f16::from_bits(u16::from_le_bytes([
                data[alpha_offset],
                data[alpha_offset + 1],
            ]))
            .to_f32();
            if alpha.is_finite() {
                max_alpha = max_alpha.max(alpha);
                hit_alpha_pixels += usize::from(alpha > 0.1);
            } else {
                non_finite_alpha += 1;
            }
        }
    }
    luminances.sort_by(f32::total_cmp);
    let percentile = |fraction: f32| {
        let index = ((luminances.len().saturating_sub(1)) as f32 * fraction).round() as usize;
        luminances.get(index).copied().unwrap_or(0.0)
    };
    let mean = luminances
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>()
        / luminances.len().max(1) as f64;
    let mut isolated_local_outliers = 0usize;
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let Some(center) = hdr_luminance_at(data, x, y, padded_bytes_per_row) else {
                continue;
            };
            if center <= 4.0 {
                continue;
            }
            let mut neighbor_max = 0.0f32;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    neighbor_max = neighbor_max.max(
                        hdr_luminance_at(
                            data,
                            x.wrapping_add_signed(ox),
                            y.wrapping_add_signed(oy),
                            padded_bytes_per_row,
                        )
                        .unwrap_or(0.0),
                    );
                }
            }
            isolated_local_outliers += usize::from(center > neighbor_max.max(1.0) * 4.0);
        }
    }
    format!(
        "{{\n  \"width\": {width},\n  \"height\": {height},\n  \"finite_pixels\": {},\n  \
         \"non_finite_pixels\": {non_finite},\n  \
         \"non_finite_alpha\": {non_finite_alpha},\n  \"mean_luminance\": {mean:.9},\n  \
         \"max_luminance\": {:.9},\n  \"p99_luminance\": {:.9},\n  \
         \"p999_luminance\": {:.9},\n  \"max_alpha\": {max_alpha:.9},\n  \
         \"hit_alpha_pixels\": {hit_alpha_pixels},\n  \
         \"isolated_local_outliers\": {isolated_local_outliers},\n  \
         \"outlier_rule\": \"luma > 4 and > 4x every 3x3 neighbor\"\n}}\n",
        luminances.len(),
        luminances.last().copied().unwrap_or(0.0),
        percentile(0.99),
        percentile(0.999),
    )
}

fn depth_rgb(data: &[u8], width: u32, height: u32, padded_bytes_per_row: u32) -> Vec<u8> {
    let mut min_proximity = f32::INFINITY;
    let mut max_proximity = 0.0f32;
    for row in 0..height {
        let row_start = (row * padded_bytes_per_row) as usize;
        for column in 0..width {
            let offset = row_start + (column * 4) as usize;
            let depth = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            let proximity = 1.0 - depth.clamp(0.0, 1.0);
            if proximity > 1.0e-7 {
                min_proximity = min_proximity.min(proximity);
                max_proximity = max_proximity.max(proximity);
            }
        }
    }
    let range = (max_proximity - min_proximity).max(1.0e-7);
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for row in 0..height {
        let row_start = (row * padded_bytes_per_row) as usize;
        for column in 0..width {
            let offset = row_start + (column * 4) as usize;
            let depth = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            // Normalize the non-clear range per capture. This is a diagnostic
            // display transform only; it makes perspective depth and all
            // three cascade ranges legible without touching render state.
            let proximity = 1.0 - depth.clamp(0.0, 1.0);
            let normalized = if proximity <= 1.0e-7 {
                0.0
            } else {
                ((proximity - min_proximity) / range).clamp(0.0, 1.0).sqrt()
            };
            let gray = (normalized * 255.0 + 0.5) as u8;
            rgb.extend_from_slice(&[gray, gray, gray]);
        }
    }
    rgb
}

fn rgba8_rgb(data: &[u8], width: u32, height: u32, padded_bytes_per_row: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for row in 0..height {
        let row_start = (row * padded_bytes_per_row) as usize;
        for column in 0..width {
            let offset = row_start + (column * 4) as usize;
            rgb.extend_from_slice(&data[offset..offset + 3]);
        }
    }
    rgb
}

impl Renderer {
    fn record_quality_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
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
                origin: wgpu::Origin3d::ZERO,
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

    /// Resolve logical graph resource names to the renderer-owned imports and
    /// encode their copies. Unknown names fail loudly instead of silently
    /// capturing the wrong attachment after a topology refactor.
    pub(super) fn record_quality_resources_by_name(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        names: &[&'static str],
    ) -> Result<Vec<QualityReadback>, String> {
        let mut readbacks = Vec::with_capacity(names.len());
        for &name in names {
            let readback = match name {
                "hdr-scene" => self.record_quality_texture(
                    encoder,
                    &self.hdr_rt_texture,
                    name,
                    ReadbackKind::Hdr,
                    wgpu::TextureAspect::All,
                    8,
                ),
                "scene-depth" => self.record_quality_texture(
                    encoder,
                    &self.depth_texture,
                    name,
                    ReadbackKind::Depth,
                    wgpu::TextureAspect::DepthOnly,
                    4,
                ),
                "ssr" => self.record_quality_texture(
                    encoder,
                    &self.ssr_history_textures[self.ssr_history_idx],
                    name,
                    ReadbackKind::Hdr,
                    wgpu::TextureAspect::All,
                    8,
                ),
                "shadow-cascade-0" | "shadow-cascade-1" | "shadow-cascade-2" => {
                    let cascade = name
                        .as_bytes()
                        .last()
                        .copied()
                        .and_then(|value| value.checked_sub(b'0'))
                        .map(usize::from)
                        .filter(|&index| index < self.shadow_map.depth_textures.len())
                        .ok_or_else(|| format!("invalid shadow capture resource '{name}'"))?;
                    self.record_quality_texture(
                        encoder,
                        &self.shadow_map.depth_textures[cascade],
                        name,
                        ReadbackKind::Depth,
                        wgpu::TextureAspect::DepthOnly,
                        4,
                    )
                }
                _ => return Err(format!("unknown render-graph capture resource '{name}'")),
            };
            readbacks.push(readback);
        }
        Ok(readbacks)
    }

    pub(super) fn record_frame_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::Texture,
    ) -> FrameReadback {
        let size = output.size();
        let width = size.width;
        let height = size.height;
        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot_staging"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
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

        let quality_capture_dir = self.pending_quality_capture_dir.clone();
        let mut quality_readbacks = if quality_capture_dir.is_some() {
            match self.record_quality_resources_by_name(
                encoder,
                &super::graph::QUALITY_CAPTURE_RESOURCE_NAMES,
            ) {
                Ok(readbacks) => readbacks,
                Err(error) => {
                    eprintln!("bloom: graph quality capture request failed: {error}");
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        if quality_capture_dir.is_some() {
            // The graph's `ssr` name resolves to filtered history. Keep the
            // noisy march as an explicitly physical companion diagnostic.
            quality_readbacks.push(self.record_quality_texture(
                encoder,
                &self.ssr_rt_texture,
                "ssr-raw",
                ReadbackKind::Hdr,
                wgpu::TextureAspect::All,
                8,
            ));
            if let Some(textures) = self.taa_diagnostic_textures() {
                for (&name, texture) in super::temporal_diagnostics::TAA_DIAGNOSTIC_NAMES
                    .iter()
                    .zip(textures)
                {
                    quality_readbacks.push(self.record_quality_texture(
                        encoder,
                        texture,
                        name,
                        ReadbackKind::Rgba8,
                        wgpu::TextureAspect::All,
                        4,
                    ));
                }
            }
            if let Some(textures) = self.ssr_temporal_diagnostic_textures() {
                for (&name, texture) in
                    super::ssr_temporal_diagnostics::SSR_TEMPORAL_DIAGNOSTIC_NAMES
                        .iter()
                        .zip(textures)
                {
                    quality_readbacks.push(self.record_quality_texture(
                        encoder,
                        texture,
                        name,
                        ReadbackKind::Rgba8,
                        wgpu::TextureAspect::All,
                        4,
                    ));
                }
            }
        }
        FrameReadback {
            staging,
            width,
            height,
            padded_bytes_per_row,
            quality_capture_dir,
            quality_readbacks,
        }
    }

    pub(super) fn finish_frame_readback(&mut self, readback: FrameReadback) {
        let slice = readback.staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let quality_receivers = Self::begin_quality_intermediate_maps(&readback.quality_readbacks);
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        if self.surface.is_none() {
            self.headless_in_flight.clear();
        }

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let mut rgba = Vec::with_capacity((readback.width * readback.height * 4) as usize);
            for row in 0..readback.height {
                let start = (row * readback.padded_bytes_per_row) as usize;
                let end = start + (readback.width * 4) as usize;
                rgba.extend_from_slice(&data[start..end]);
            }
            drop(data);
            if let Some(path) = self.pending_screenshot_path.take() {
                let mut rgb = Vec::with_capacity((readback.width * readback.height * 3) as usize);
                for chunk in rgba.chunks_exact(4) {
                    // Native surface captures are BGRA; the PNG contract is RGB.
                    rgb.extend_from_slice(&[chunk[2], chunk[1], chunk[0]]);
                }
                match encode_png_simple(readback.width, readback.height, &rgb) {
                    Some(png) => {
                        if let Err(error) = std::fs::write(&path, png) {
                            eprintln!("bloom: screenshot write '{path}' failed: {error}");
                        }
                    }
                    None => eprintln!(
                        "bloom: screenshot PNG encode failed ({}x{})",
                        readback.width, readback.height
                    ),
                }
            }
            self.screenshot_data = Some((readback.width, readback.height, rgba));
        }
        readback.staging.unmap();

        if let Some(directory) = readback.quality_capture_dir.as_deref() {
            self.finish_quality_intermediates(
                directory,
                &readback.quality_readbacks,
                quality_receivers,
            );
        }
        self.pending_quality_capture_dir.take();
        self.release_temporal_diagnostics();
        self.release_ssr_temporal_diagnostics();
        self.screenshot_requested = false;
    }

    pub(super) fn begin_quality_intermediate_maps(
        readbacks: &[QualityReadback],
    ) -> Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> {
        readbacks
            .iter()
            .map(|readback| {
                let (tx, rx) = mpsc::channel();
                readback
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = tx.send(result);
                    });
                rx
            })
            .collect()
    }

    pub(super) fn finish_quality_intermediates(
        &self,
        directory: &str,
        readbacks: &[QualityReadback],
        receivers: Vec<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    ) {
        let directory = std::path::Path::new(directory);
        if let Err(error) = std::fs::create_dir_all(directory) {
            eprintln!("bloom: cannot create quality capture directory '{directory:?}': {error}");
            return;
        }
        for (readback, receiver) in readbacks.iter().zip(receivers) {
            if !matches!(receiver.recv(), Ok(Ok(()))) {
                eprintln!("bloom: intermediate '{}' readback failed", readback.name);
                continue;
            }
            let data = readback.buffer.slice(..).get_mapped_range();
            if matches!(readback.kind, ReadbackKind::Hdr) {
                let metrics = hdr_metrics_json(
                    &data,
                    readback.width,
                    readback.height,
                    readback.padded_bytes_per_row,
                );
                let metrics_path = directory.join(format!("{}.metrics.json", readback.name));
                if let Err(error) = std::fs::write(&metrics_path, metrics) {
                    eprintln!("bloom: HDR metrics write '{metrics_path:?}' failed: {error}");
                }
            }
            let rgb = match readback.kind {
                ReadbackKind::Hdr => hdr_rgb(
                    &data,
                    readback.width,
                    readback.height,
                    readback.padded_bytes_per_row,
                ),
                ReadbackKind::Depth => depth_rgb(
                    &data,
                    readback.width,
                    readback.height,
                    readback.padded_bytes_per_row,
                ),
                ReadbackKind::Rgba8 => rgba8_rgb(
                    &data,
                    readback.width,
                    readback.height,
                    readback.padded_bytes_per_row,
                ),
            };
            drop(data);
            readback.buffer.unmap();
            let path = directory.join(format!("{}.png", readback.name));
            match encode_png_simple(readback.width, readback.height, &rgb) {
                Some(png) => {
                    if let Err(error) = std::fs::write(&path, png) {
                        eprintln!("bloom: intermediate write '{path:?}' failed: {error}");
                    }
                }
                None => eprintln!("bloom: intermediate '{}' PNG encode failed", readback.name),
            }
        }
        for (name, width, height, rgb) in self.shadow_map.virtual_map.debug_images() {
            let path = directory.join(format!("{name}.png"));
            match encode_png_simple(width, height, &rgb) {
                Some(png) => {
                    if let Err(error) = std::fs::write(&path, png) {
                        eprintln!("bloom: VSM debug write '{path:?}' failed: {error}");
                    }
                }
                None => eprintln!("bloom: VSM debug PNG encode failed for '{name}'"),
            }
        }
    }

    pub fn quality_adapter_json(&self) -> String {
        let info = self.device.adapter_info();
        let features = self.device.features();
        let mut semantic_features = Vec::new();
        for (feature, enabled) in [
            (
                "timestamp-query",
                features.contains(wgpu::Features::TIMESTAMP_QUERY),
            ),
            (
                "ray-query",
                features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
            ),
            (
                "texture-binding-array",
                features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY),
            ),
            (
                "non-uniform-indexing",
                features.contains(
                    wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                ),
            ),
            (
                "texture-compression-bc",
                features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
            ),
        ] {
            if enabled {
                semantic_features.push(feature);
            }
        }
        let tier = if self.hw_rt_enabled && self.pt_texture_arrays_enabled {
            "high-end-rt-bindless"
        } else if self.hw_rt_enabled {
            "high-end-rt"
        } else if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            "raster-timestamp"
        } else {
            "baseline"
        };
        let mut out = String::from("{\"availability\":\"reported\",\"name\":");
        json_string(&mut out, &info.name);
        out.push_str(",\"vendor_id\":");
        out.push_str(&info.vendor.to_string());
        out.push_str(",\"device_id\":");
        out.push_str(&info.device.to_string());
        out.push_str(",\"device_type\":");
        json_string(&mut out, &enum_name(info.device_type));
        out.push_str(",\"driver\":");
        json_string(&mut out, &info.driver);
        out.push_str(",\"driver_info\":");
        json_string(&mut out, &info.driver_info);
        out.push_str(",\"backend\":");
        json_string(&mut out, &enum_name(info.backend));
        out.push_str(",\"capability_tier\":");
        json_string(&mut out, tier);
        out.push_str(",\"features\":[");
        for (index, feature) in semantic_features.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            json_string(&mut out, feature);
        }
        out.push_str("]}");
        out
    }

    pub fn quality_runtime_paths_json(&self) -> String {
        let ssgi = self.ssgi_backend_logged.unwrap_or(if self.hw_rt_enabled {
            "hw-ray-query-pending"
        } else {
            "software-fallback-pending"
        });
        let mut out = String::from("{\"ssgi_trace_backend\":");
        json_string(&mut out, ssgi);
        out.push_str(",\"path_tracing_available\":");
        out.push_str(if self.pt_pipeline.is_some() {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"temporal_history\":{");
        out.push_str("\"ssr_valid\":");
        out.push_str(if self.ssr_history_valid {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"ssr_index\":");
        out.push_str(&self.ssr_history_idx.to_string());
        out.push_str(",\"ssgi_probe_valid\":");
        out.push_str(if self.probe_history_valid {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"ssgi_probe_index\":");
        out.push_str(&self.probe_history_idx.to_string());
        out.push_str(",\"taa_valid\":");
        out.push_str(if self.taa_history_valid {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"taa_index\":");
        out.push_str(&self.taa_current_idx.to_string());
        out.push_str(",\"taa_pt_owned\":");
        out.push_str(if self.taa_history_pt_owned {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"exposure_valid\":");
        out.push_str(if self.exposure_history_valid {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"exposure_index\":");
        out.push_str(&self.exposure_current_idx.to_string());
        out.push_str(",\"pt_samples\":");
        out.push_str(&self.pt_accum_count.to_string());
        out.push_str(",\"pt_index\":");
        out.push_str(&self.pt_accum_idx.to_string());
        out.push_str(",\"ssao_frames\":");
        out.push_str(&self.ssao_history_frame.to_string());
        out.push_str(",\"ssao_index\":");
        out.push_str(&self.ssao_history_idx.to_string());
        out.push_str(",\"camera_cut_pending\":");
        out.push_str(if self.temporal_camera_cut_pending {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"camera_cut_active\":");
        out.push_str(if self.temporal_camera_cut_active {
            "true"
        } else {
            "false"
        });
        let (cached_motion_entries, cached_motion_cpu_bytes) = self.cached_model_motion_stats();
        out.push_str(",\"cached_model_motion_entries\":");
        out.push_str(&cached_motion_entries.to_string());
        out.push_str(",\"cached_model_motion_cpu_capacity_bytes\":");
        out.push_str(&cached_motion_cpu_bytes.to_string());
        out.push_str(",\"cached_model_motion_gpu_bytes\":0");
        out.push_str(",\"cached_model_motion_passes\":0");
        out.push_str(",\"diagnostic_persistent_bytes\":0");
        let diagnostic_texture_bytes = u64::from(self.surface_config.width)
            * u64::from(self.surface_config.height)
            * super::temporal_diagnostics::TAA_DIAGNOSTIC_NAMES.len() as u64
            * 4;
        let diagnostic_row_bytes = u64::from((self.surface_config.width * 4 + 255) & !255);
        let diagnostic_readback_bytes = diagnostic_row_bytes
            * u64::from(self.surface_config.height)
            * super::temporal_diagnostics::TAA_DIAGNOSTIC_NAMES.len() as u64;
        out.push_str(",\"diagnostic_capture_texture_bytes\":");
        out.push_str(&diagnostic_texture_bytes.to_string());
        out.push_str(",\"diagnostic_capture_readback_bytes\":");
        out.push_str(&diagnostic_readback_bytes.to_string());
        out.push_str(",\"diagnostic_capture_passes\":1");
        out.push_str(",\"diagnostic_resources_live\":");
        out.push_str(if self.taa_diagnostic_textures().is_some() {
            "true"
        } else {
            "false"
        });
        let ssr_size = self.ssr_rt_texture.size();
        let ssr_hdr_row_bytes = u64::from((ssr_size.width * 8 + 255) & !255);
        let ssr_rgba8_row_bytes = u64::from((ssr_size.width * 4 + 255) & !255);
        let ssr_capture_texture_bytes = u64::from(ssr_size.width)
            * u64::from(ssr_size.height)
            * super::ssr_temporal_diagnostics::SSR_TEMPORAL_DIAGNOSTIC_NAMES.len() as u64
            * 4;
        let ssr_capture_bytes = ssr_hdr_row_bytes * u64::from(ssr_size.height) * 2
            + ssr_rgba8_row_bytes
                * u64::from(ssr_size.height)
                * super::ssr_temporal_diagnostics::SSR_TEMPORAL_DIAGNOSTIC_NAMES.len() as u64;
        out.push_str(",\"ssr_diagnostic_persistent_bytes\":0");
        out.push_str(",\"ssr_diagnostic_capture_texture_bytes\":");
        out.push_str(&ssr_capture_texture_bytes.to_string());
        out.push_str(",\"ssr_diagnostic_capture_readback_bytes\":");
        out.push_str(&ssr_capture_bytes.to_string());
        out.push_str(",\"ssr_diagnostic_capture_passes\":1");
        out.push_str(",\"ssr_diagnostic_resources_live\":");
        out.push_str(if self.ssr_temporal_diagnostic_textures().is_some() {
            "true"
        } else {
            "false"
        });
        out.push('}');
        out.push_str(",\"transparent_gi\":{");
        out.push_str("\"enabled\":");
        out.push_str(if super::transparent_gi::transparent_gi_enabled() {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"active\":");
        out.push_str(if self.transparent_gi_active {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"representation\":\"one-layer-colored-continuation\"");
        out.push_str(",\"additional_persistent_bytes\":0");
        out.push_str(",\"instance_count\":");
        out.push_str(&self.transparent_gi_instance_count.to_string());
        out.push('}');
        out.push_str(",\"refractive_reflections\":{");
        out.push_str("\"enabled\":");
        out.push_str(
            if cfg!(not(fold_scene_inputs))
                && super::refractive_reflections::refractive_reflection_hierarchy_enabled()
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"active\":");
        out.push_str(if self.refractive_reflections_active {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"source\":");
        json_string(&mut out, self.refractive_reflection_source_name());
        out.push_str(",\"march_steps\":");
        out.push_str(
            &(super::refractive_reflections::REFRACTIVE_REFLECTION_STEPS as u32).to_string(),
        );
        out.push_str(",\"max_distance\":");
        out.push_str(
            &super::refractive_reflections::REFRACTIVE_REFLECTION_MAX_DISTANCE.to_string(),
        );
        out.push_str(",\"max_roughness\":");
        out.push_str(
            &super::refractive_reflections::REFRACTIVE_REFLECTION_MAX_ROUGHNESS.to_string(),
        );
        out.push_str(",\"persistent_bytes_when_initialized\":");
        out.push_str(
            &super::refractive_reflections::REFRACTIVE_REFLECTION_PERSISTENT_BYTES.to_string(),
        );
        out.push_str(",\"additional_graph_passes\":0");
        out.push_str(",\"additional_image_bytes\":0}");
        out.push_str(",\"physical_texture_uv\":{");
        out.push_str("\"supported_sets\":[0,1]");
        out.push_str(",\"uv1_pipeline_initialized\":");
        out.push_str(if self.scene_refractive_uv1_pipeline.is_some() {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"ordinary_vertex_stride_bytes\":");
        out.push_str(&std::mem::size_of::<super::Vertex3D>().to_string());
        out.push_str(",\"uv1_sidecar_stride_bytes\":");
        out.push_str(&std::mem::size_of::<[f32; 2]>().to_string());
        out.push_str(",\"additional_graph_passes\":0");
        out.push_str(",\"additional_image_bytes\":0}");
        out.push_str(",\"layered_pbr\":{");
        let default_bound_material = super::material_system::MaterialFactorsUniforms::default();
        let default_global_material = super::material_indirection::GpuMaterialRecord::default();
        out.push_str("\"material_record_version\":");
        out.push_str(&super::layered_pbr::MATERIAL_RECORD_VERSION.to_string());
        out.push_str(",\"bound_material_record_version\":");
        out.push_str(&default_bound_material.layered_pbr_version().to_string());
        out.push_str(",\"global_material_record_version\":");
        out.push_str(&default_global_material.layered_pbr_version().to_string());
        out.push_str(",\"lobe_mask_bits\":");
        out.push_str(&super::layered_pbr::MATERIAL_LOBE_MASK_BITS.to_string());
        out.push_str(",\"known_lobe_mask\":");
        out.push_str(
            &super::layered_pbr::MaterialLobeMask::KNOWN
                .bits()
                .to_string(),
        );
        out.push_str(",\"default_bound_lobe_mask\":");
        out.push_str(
            &default_bound_material
                .layered_pbr_lobe_mask()
                .bits()
                .to_string(),
        );
        out.push_str(",\"default_global_lobe_mask\":");
        out.push_str(
            &default_global_material
                .layered_pbr_lobe_mask()
                .bits()
                .to_string(),
        );
        out.push_str(",\"active_lobe_material_count\":");
        out.push_str(
            &self
                .material_system
                .indirection
                .active_layered_material_count()
                .to_string(),
        );
        out.push_str(",\"bound_material_uniform_bytes\":");
        out.push_str(
            &std::mem::size_of::<super::material_system::MaterialFactorsUniforms>().to_string(),
        );
        out.push_str(",\"global_material_record_bytes\":");
        out.push_str(
            &std::mem::size_of::<super::material_indirection::GpuMaterialRecord>().to_string(),
        );
        out.push_str(",\"scene_specialization_initialized\":");
        out.push_str(if self.scene_layered_pbr_resources.is_some() {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"combined_refraction_specialization_initialized\":");
        out.push_str(if self.scene_layered_refractive_resources.is_some() {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"sheen_lut_initialized\":");
        out.push_str(if self.scene_sheen_albedo_lut.is_some() {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"sheen_lut_bytes_when_initialized\":");
        out.push_str(&super::layered_pbr_scene::SHEEN_ALBEDO_LUT_BYTES.to_string());
        out.push_str(",\"layered_shared_sampler_count\":1");
        out.push_str(",\"path_tracing_specialization_initialized\":");
        out.push_str(if self.pt_layered.pipelines.iter().any(Option::is_some) {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"path_tracing_sheen_specialization_initialized\":");
        out.push_str(
            if self
                .pt_layered
                .pipelines
                .iter()
                .enumerate()
                .any(|(index, pipeline)| index & 1 != 0 && pipeline.is_some())
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"path_tracing_anisotropy_specialization_initialized\":");
        out.push_str(
            if self
                .pt_layered
                .pipelines
                .iter()
                .enumerate()
                .any(|(index, pipeline)| index & 2 != 0 && pipeline.is_some())
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"path_tracing_iridescence_specialization_initialized\":");
        out.push_str(
            if self
                .pt_layered
                .pipelines
                .iter()
                .enumerate()
                .any(|(index, pipeline)| index & 4 != 0 && pipeline.is_some())
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"path_tracing_texture_specialization_initialized\":");
        out.push_str(
            if self
                .pt_layered
                .pipelines
                .iter()
                .enumerate()
                .any(|(index, pipeline)| index & 8 != 0 && pipeline.is_some())
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"path_tracing_uv1_specialization_initialized\":");
        out.push_str(
            if self
                .pt_layered
                .pipelines
                .iter()
                .enumerate()
                .any(|(index, pipeline)| index & 16 != 0 && pipeline.is_some())
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"path_tracing_clearcoat_texture_specialization_initialized\":");
        out.push_str(
            if self
                .pt_layered
                .pipelines
                .iter()
                .enumerate()
                .any(|(index, pipeline)| index & 32 != 0 && pipeline.is_some())
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"path_tracing_sheen_texture_specialization_initialized\":");
        out.push_str(
            if self
                .pt_layered
                .pipelines
                .iter()
                .enumerate()
                .any(|(index, pipeline)| index & 64 != 0 && pipeline.is_some())
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"path_tracing_iridescence_texture_specialization_initialized\":");
        out.push_str(
            if self
                .pt_layered
                .pipelines
                .iter()
                .enumerate()
                .any(|(index, pipeline)| index & 128 != 0 && pipeline.is_some())
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"path_tracing_anisotropy_texture_specialization_initialized\":");
        out.push_str(
            if self
                .pt_layered
                .pipelines
                .iter()
                .enumerate()
                .any(|(index, pipeline)| index & 256 != 0 && pipeline.is_some())
            {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"path_tracing_active_instance_count\":");
        out.push_str(
            &self
                .pt_layered
                .records
                .iter()
                .filter(|record| record.active())
                .count()
                .to_string(),
        );
        out.push_str(",\"path_tracing_sidecar_record_bytes\":");
        out.push_str(
            &std::mem::size_of::<super::layered_pbr_pt::PtLayeredMaterialCpu>().to_string(),
        );
        out.push_str(",\"path_tracing_sidecar_allocated_bytes\":");
        out.push_str(
            &self
                .pt_layered
                .instance_buffer
                .as_ref()
                .map_or(0, wgpu::Buffer::size)
                .to_string(),
        );
        out.push_str(",\"path_tracing_texture_sidecar_record_bytes\":");
        out.push_str(
            &std::mem::size_of::<super::layered_pbr_pt::PtLayeredTextureCpu>().to_string(),
        );
        out.push_str(",\"path_tracing_texture_sidecar_allocated_bytes\":");
        out.push_str(
            &self
                .pt_layered
                .texture_buffer
                .as_ref()
                .map_or(0, wgpu::Buffer::size)
                .to_string(),
        );
        out.push_str(",\"path_tracing_clearcoat_texture_sidecar_record_bytes\":");
        out.push_str(
            &std::mem::size_of::<super::layered_pbr_pt::PtClearcoatTextureCpu>().to_string(),
        );
        out.push_str(",\"path_tracing_clearcoat_texture_sidecar_allocated_bytes\":");
        out.push_str(
            &self
                .pt_layered
                .clearcoat_texture_buffer
                .as_ref()
                .map_or(0, wgpu::Buffer::size)
                .to_string(),
        );
        out.push_str(",\"path_tracing_sheen_texture_sidecar_record_bytes\":");
        out.push_str(&std::mem::size_of::<super::layered_pbr_pt::PtSheenTextureCpu>().to_string());
        out.push_str(",\"path_tracing_sheen_texture_sidecar_allocated_bytes\":");
        out.push_str(
            &self
                .pt_layered
                .sheen_texture_buffer
                .as_ref()
                .map_or(0, wgpu::Buffer::size)
                .to_string(),
        );
        out.push_str(",\"path_tracing_iridescence_texture_sidecar_record_bytes\":");
        out.push_str(
            &std::mem::size_of::<super::layered_pbr_pt::PtIridescenceTextureCpu>().to_string(),
        );
        out.push_str(",\"path_tracing_iridescence_texture_sidecar_allocated_bytes\":");
        out.push_str(
            &self
                .pt_layered
                .iridescence_texture_buffer
                .as_ref()
                .map_or(0, wgpu::Buffer::size)
                .to_string(),
        );
        out.push_str(",\"path_tracing_anisotropy_texture_sidecar_record_bytes\":");
        out.push_str(
            &std::mem::size_of::<super::layered_pbr_pt::PtAnisotropyTextureCpu>().to_string(),
        );
        out.push_str(",\"path_tracing_anisotropy_texture_sidecar_allocated_bytes\":");
        out.push_str(
            &self
                .pt_layered
                .anisotropy_texture_buffer
                .as_ref()
                .map_or(0, wgpu::Buffer::size)
                .to_string(),
        );
        out.push_str(",\"path_tracing_uv1_sidecar_allocated_bytes\":");
        out.push_str(
            &self
                .pt_layered
                .uv1_buffer
                .as_ref()
                .map_or(0, wgpu::Buffer::size)
                .to_string(),
        );
        out.push_str(",\"additional_base_material_bytes\":0");
        out.push_str(",\"additional_base_material_bindings\":0");
        out.push_str(",\"additional_base_material_branches\":0");
        out.push_str(",\"additional_graph_passes\":0");
        out.push_str(",\"additional_image_bytes\":0}");
        out.push_str(",\"transparency\":{");
        out.push_str("\"preference\":");
        json_string(
            &mut out,
            match self.transparency_composition_mode_code() {
                0 => "sorted",
                2 => "weighted",
                _ => "auto",
            },
        );
        out.push_str(",\"active\":");
        json_string(
            &mut out,
            if self.active_transparency_composition_mode_code() == 1 {
                "weighted"
            } else {
                "sorted"
            },
        );
        out.push_str(",\"auto_draw_threshold\":");
        out.push_str(&WEIGHTED_TRANSPARENCY_AUTO_DRAW_THRESHOLD.to_string());
        out.push_str(",\"sorted_interleaving\":\"global-depth-source-stable-id\"");
        out.push_str(",\"sorted_interleaving_enabled\":");
        out.push_str(
            if super::sorted_transparency::sorted_interleaving_enabled() {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"reactive_custom_pipeline_count\":");
        out.push_str(
            &self
                .material_system
                .reactive_translucent_pipeline_count()
                .to_string(),
        );
        out.push_str(",\"sorted_interleaving_additional_draws\":0");
        out.push_str(",\"sorted_interleaving_additional_graph_passes\":0");
        out.push('}');
        out.push_str(",\"temporal_reactive\":{");
        out.push_str("\"enabled\":");
        out.push_str(if super::temporal_reactive::temporal_reactive_enabled() {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"active\":");
        out.push_str(if self.temporal_reactive_active {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"format\":\"r8unorm\"");
        out.push_str(",\"bytes_per_render_pixel\":1");
        out.push_str(",\"sources\":\"imported-blend-and-transmission\"}");
        out.push_str(",\"transmitted_shadows\":{");
        out.push_str("\"enabled\":");
        out.push_str(
            if super::transmitted_shadows::transmitted_shadows_enabled() {
                "true"
            } else {
                "false"
            },
        );
        out.push_str(",\"active\":");
        out.push_str(if self.transmitted_shadows_active {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"representation\":\"nearest-layer-rgb-depth\"");
        out.push_str(",\"resolution\":");
        out.push_str(&super::transmitted_shadows::TRANSMITTED_SHADOW_MAP_SIZE.to_string());
        out.push_str(",\"persistent_bytes_when_allocated\":");
        out.push_str(&super::transmitted_shadows::TRANSMITTED_SHADOW_PERSISTENT_BYTES.to_string());
        out.push_str(",\"caster_count\":");
        out.push_str(
            &self
                .transmitted_shadow_resources
                .as_ref()
                .map_or(0, |resources| resources.last_caster_count)
                .to_string(),
        );
        out.push('}');
        out.push_str(",\"masked_alpha\":{");
        out.push_str("\"coverage_mip_textures\":");
        out.push_str(&self.mask_coverage_texture_count.to_string());
        out.push_str(",\"coverage_mips_supported\":");
        out.push_str(if cfg!(target_os = "android") {
            "false"
        } else {
            "true"
        });
        out.push_str(",\"sample_count\":1");
        out.push_str(",\"alpha_to_coverage_supported\":false");
        out.push_str(",\"single_sample_fallback\":\"coverage-mips-bayer-4x4\"}");
        let graph_stats = self.render_graph_cache_stats();
        out.push_str(",\"render_graph\":{");
        out.push_str("\"compile_count\":");
        out.push_str(&graph_stats.compile_count.to_string());
        out.push_str(",\"cache_hit_count\":");
        out.push_str(&graph_stats.hit_count.to_string());
        out.push_str(",\"cached_plan_count\":");
        out.push_str(&graph_stats.plan_count.to_string());
        if let Some(plan) = self.last_frame_plan.as_ref() {
            let render_extent = self.render_extent();
            let output_extent = (self.surface_config.width, self.surface_config.height);
            out.push_str(",\"plan_id\":");
            json_string(&mut out, &format!("{:016x}", plan.plan_id));
            out.push_str(",\"pass_count\":");
            out.push_str(&plan.passes.len().to_string());
            out.push_str(",\"aliasing_enabled\":");
            out.push_str(if plan.aliasing_enabled {
                "true"
            } else {
                "false"
            });
            out.push_str(",\"transient_bytes\":");
            out.push_str(
                &plan
                    .transient_bytes(render_extent, output_extent)
                    .to_string(),
            );
            out.push_str(",\"unaliased_transient_bytes\":");
            out.push_str(
                &plan
                    .unaliased_transient_bytes(render_extent, output_extent)
                    .to_string(),
            );
            out.push_str(",\"physical_transient_slots\":");
            out.push_str(&self.transient_pool.compiled_slot_count().to_string());
        }
        out.push('}');
        out.push_str(",\"gpu_driven\":");
        out.push_str(&self.gpu_driven.report_json());
        out.push_str(",\"virtual_shadows\":");
        out.push_str(&self.shadow_map.virtual_map.report_json());
        out.push_str(",\"material_binding\":");
        out.push_str(&self.material_binding_report_json());
        out.push('}');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::hdr_metrics_json;

    #[test]
    fn hdr_metrics_detect_a_single_local_firefly() {
        let width = 3;
        let height = 3;
        let row_bytes = width * 8;
        let mut data = vec![0u8; (row_bytes * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let value = if (x, y) == (1, 1) { 10.0 } else { 1.0 };
                let bits = half::f16::from_f32(value).to_bits().to_le_bytes();
                let base = (y * row_bytes + x * 8) as usize;
                for channel in 0..3 {
                    data[base + channel * 2..base + channel * 2 + 2].copy_from_slice(&bits);
                }
                data[base + 6..base + 8]
                    .copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
            }
        }
        let metrics = hdr_metrics_json(&data, width, height, row_bytes);
        assert!(metrics.contains("\"non_finite_pixels\": 0"));
        assert!(metrics.contains("\"max_luminance\": 10.000000000"));
        assert!(metrics.contains("\"isolated_local_outliers\": 1"));
    }
}
