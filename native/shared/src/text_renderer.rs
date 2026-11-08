use crate::renderer::Renderer;
use fontdue::{Font, FontSettings};
use std::collections::HashMap;

static DEFAULT_FONT_DATA: &[u8] = include_bytes!("../assets/default_font.ttf");

struct GlyphInfo {
    atlas_x: u32,
    atlas_y: u32,
    width: u32,
    height: u32,
    advance: f32,
    x_offset: f32,
    y_offset: f32,
}

/// Generate a signed distance field from a bitmap glyph.
/// `bitmap` is the alpha channel, `w`/`h` are dimensions.
/// `spread` is the distance range in pixels.
/// Returns an RGBA SDF texture (white + SDF in alpha).
fn generate_sdf(bitmap: &[u8], w: u32, h: u32, spread: f32) -> Vec<u8> {
    let ww = w as i32;
    let hh = h as i32;
    let spread_i = spread as i32;
    let mut sdf = vec![0u8; (w * h * 4) as usize];

    for y in 0..hh {
        for x in 0..ww {
            let inside = bitmap[(y * ww + x) as usize] > 127;
            let mut min_dist = spread;

            // Search in a window around the pixel for the nearest edge
            let x0 = (x - spread_i).max(0);
            let x1 = (x + spread_i).min(ww - 1);
            let y0 = (y - spread_i).max(0);
            let y1 = (y + spread_i).min(hh - 1);

            for sy in y0..=y1 {
                for sx in x0..=x1 {
                    let other_inside = bitmap[(sy * ww + sx) as usize] > 127;
                    if other_inside != inside {
                        let dx = (sx - x) as f32;
                        let dy = (sy - y) as f32;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist < min_dist {
                            min_dist = dist;
                        }
                    }
                }
            }

            let normalized = if inside {
                0.5 + 0.5 * (min_dist / spread)
            } else {
                0.5 - 0.5 * (min_dist / spread)
            };
            let alpha = (normalized.clamp(0.0, 1.0) * 255.0) as u8;
            let idx = ((y * ww + x) * 4) as usize;
            sdf[idx] = 255;
            sdf[idx + 1] = 255;
            sdf[idx + 2] = 255;
            sdf[idx + 3] = alpha;
        }
    }
    sdf
}

/// The SDF atlas keeps its size; the bitmap atlas grows (see `end_frame`).
const SDF_ATLAS_SIZE: u32 = 1024;
/// The bitmap atlas starts at 1024 and doubles up to this when a frame fills it.
const MAX_ATLAS_SIZE: u32 = 2048;

pub struct TextRenderer {
    fonts: Vec<Font>,
    glyph_cache: HashMap<(usize, char, u32), GlyphInfo>,
    atlas_data: Vec<u8>,
    atlas_width: u32,
    atlas_height: u32,
    atlas_cursor_x: u32,
    atlas_cursor_y: u32,
    atlas_row_height: u32,
    atlas_bind_group_idx: Option<u32>,
    atlas_dirty: bool,
    /// One-shot guard for the atlas-full warning.
    atlas_overflow_warned: bool,
    /// A glyph didn't fit this frame: the atlas grows (or starts over) in `end_frame`.
    atlas_full: bool,
    /// Rows written since the last upload (`upload` sends only these), and whether the
    /// texture must be sent whole (new size, cleared).
    dirty_rows: Option<(u32, u32)>,
    atlas_resized: bool,
    /// Advances for measuring, without rasterizing (logical-size glyphs filled the atlas).
    advances: HashMap<(usize, char, u32), f32>,
    // SDF atlas (separate from bitmap atlas)
    sdf_glyph_cache: HashMap<(usize, char), GlyphInfo>,
    sdf_atlas_data: Vec<u8>,
    sdf_atlas_cursor_x: u32,
    sdf_atlas_cursor_y: u32,
    sdf_atlas_row_height: u32,
    sdf_atlas_bind_group_idx: Option<u32>,
    sdf_atlas_dirty: bool,
}

impl TextRenderer {
    /// Cheap placeholder that allocates nothing. Used as a temporary swap target
    /// when we need to split borrows on EngineState (text + renderer).
    pub fn empty() -> Self {
        Self {
            fonts: Vec::new(),
            glyph_cache: HashMap::new(),
            atlas_data: Vec::new(),
            atlas_width: 0,
            atlas_height: 0,
            atlas_cursor_x: 0,
            atlas_cursor_y: 0,
            atlas_row_height: 0,
            atlas_bind_group_idx: None,
            atlas_dirty: false,
            atlas_overflow_warned: false,
            atlas_full: false,
            dirty_rows: None,
            atlas_resized: false,
            advances: HashMap::new(),
            sdf_glyph_cache: HashMap::new(),
            sdf_atlas_data: Vec::new(),
            sdf_atlas_cursor_x: 0,
            sdf_atlas_cursor_y: 0,
            sdf_atlas_row_height: 0,
            sdf_atlas_bind_group_idx: None,
            sdf_atlas_dirty: false,
        }
    }

    pub fn new() -> Self {
        let default_font = Font::from_bytes(DEFAULT_FONT_DATA, FontSettings::default())
            .expect("Failed to load default font");

        let atlas_width = 1024;
        let atlas_height = 1024;

        Self {
            fonts: vec![default_font],
            glyph_cache: HashMap::new(),
            atlas_data: vec![0u8; (atlas_width * atlas_height * 4) as usize],
            atlas_width,
            atlas_height,
            atlas_cursor_x: 0,
            atlas_cursor_y: 0,
            atlas_row_height: 0,
            atlas_bind_group_idx: None,
            atlas_dirty: true,
            atlas_overflow_warned: false,
            atlas_full: false,
            dirty_rows: None,
            atlas_resized: false,
            advances: HashMap::new(),
            sdf_glyph_cache: HashMap::new(),
            sdf_atlas_data: vec![0u8; (atlas_width * atlas_height * 4) as usize],
            sdf_atlas_cursor_x: 0,
            sdf_atlas_cursor_y: 0,
            sdf_atlas_row_height: 0,
            sdf_atlas_bind_group_idx: None,
            sdf_atlas_dirty: true,
        }
    }

    pub fn load_font(&mut self, data: &[u8]) -> usize {
        let font = Font::from_bytes(data, FontSettings::default()).expect("Failed to load font");
        self.fonts.push(font);
        self.fonts.len() - 1
    }

    /// Rasterize a glyph into the atlas (cached). When it doesn't fit, nothing is stored this
    /// frame (the glyph isn't drawn once) and the atlas grows at the end of the frame: growing or
    /// clearing mid-frame would move glyphs the frame's earlier quads already point at.
    fn rasterize_glyph(&mut self, font_idx: usize, ch: char, size: u32) {
        let key = (font_idx, ch, size);
        if self.glyph_cache.contains_key(&key) || self.atlas_full {
            return;
        }

        let font = &self.fonts[font_idx];
        let (metrics, bitmap) = font.rasterize(ch, size as f32);

        let gw = metrics.width as u32;
        let gh = metrics.height as u32;

        let mut cx = self.atlas_cursor_x;
        let mut cy = self.atlas_cursor_y;
        let mut row = self.atlas_row_height;
        if cx + gw > self.atlas_width {
            cx = 0;
            cy += row;
            row = 0;
        }
        if cy + gh > self.atlas_height || gw > self.atlas_width {
            self.atlas_full = true;
            return;
        }
        self.atlas_cursor_x = cx;
        self.atlas_cursor_y = cy;
        self.atlas_row_height = row;

        let ax = self.atlas_cursor_x;
        let ay = self.atlas_cursor_y;

        for row in 0..gh {
            for col in 0..gw {
                let src = (row * gw + col) as usize;
                let dst = (((ay + row) * self.atlas_width + ax + col) * 4) as usize;
                if src < bitmap.len() && dst + 3 < self.atlas_data.len() {
                    let alpha = bitmap[src];
                    self.atlas_data[dst] = 255;
                    self.atlas_data[dst + 1] = 255;
                    self.atlas_data[dst + 2] = 255;
                    self.atlas_data[dst + 3] = alpha;
                }
            }
        }

        let (r0, r1) = self.dirty_rows.unwrap_or((ay, ay + gh));
        self.dirty_rows = Some((r0.min(ay), r1.max(ay + gh)));
        self.atlas_cursor_x += gw + 1;
        if gh + 1 > self.atlas_row_height {
            self.atlas_row_height = gh + 1;
        }
        self.atlas_dirty = true;

        self.glyph_cache.insert(
            key,
            GlyphInfo {
                atlas_x: ax,
                atlas_y: ay,
                width: gw,
                height: gh,
                advance: metrics.advance_width,
                x_offset: metrics.xmin as f32,
                y_offset: metrics.ymin as f32,
            },
        );
    }

    /// A glyph's advance at a pixel size, from the font's metrics (cached; nothing rasterized).
    fn advance(&mut self, font_idx: usize, ch: char, size: u32) -> f32 {
        let key = (font_idx, ch, size);
        if let Some(a) = self.advances.get(&key) {
            return *a;
        }
        if self.advances.len() > 50_000 {
            self.advances.clear();
        }
        let a = self.fonts[font_idx].metrics(ch, size as f32).advance_width;
        self.advances.insert(key, a);
        a
    }

    /// Once a frame: a glyph that didn't fit doubles the atlas (up to MAX_ATLAS_SIZE), keeping
    /// what's in it; at the largest size it starts over. The next frame rasterizes what it needs.
    pub fn end_frame(&mut self) {
        if !self.atlas_full {
            return;
        }
        self.atlas_full = false;
        self.atlas_dirty = true;
        if self.atlas_width < MAX_ATLAS_SIZE {
            let (ow, oh) = (self.atlas_width, self.atlas_height);
            let (nw, nh) = (ow * 2, oh * 2);
            let mut data = vec![0u8; (nw * nh * 4) as usize];
            for row in 0..oh {
                let src = (row * ow * 4) as usize;
                let dst = (row * nw * 4) as usize;
                data[dst..dst + (ow * 4) as usize]
                    .copy_from_slice(&self.atlas_data[src..src + (ow * 4) as usize]);
            }
            self.atlas_data = data;
            self.atlas_width = nw;
            self.atlas_height = nh;
            self.atlas_resized = true;
            eprintln!(
                "bloom text: glyph atlas grew to {nw}x{nh} ({} glyphs)",
                self.glyph_cache.len()
            );
            return;
        }
        if !self.atlas_overflow_warned {
            self.atlas_overflow_warned = true;
            eprintln!(
                "bloom text: glyph atlas ({}x{}) full; starting it over",
                self.atlas_width, self.atlas_height
            );
        }
        self.glyph_cache.clear();
        self.atlas_data.iter_mut().for_each(|b| *b = 0);
        self.atlas_cursor_x = 0;
        self.atlas_cursor_y = 0;
        self.atlas_row_height = 0;
        self.atlas_resized = true;
    }

    /// The atlas texture exists (the first draw registers it); its contents go up once a frame
    /// in `upload`, before the frame is drawn. Uploading on every draw call that added a glyph
    /// staged a full atlas copy each time, which ran an iPhone out of GPU memory.
    fn ensure_atlas_uploaded(&mut self, renderer: &mut Renderer) {
        if self.atlas_bind_group_idx.is_none() {
            let idx = renderer.register_texture_no_mips(
                self.atlas_width,
                self.atlas_height,
                &self.atlas_data,
            );
            self.atlas_bind_group_idx = Some(idx);
            self.atlas_dirty = false;
            self.dirty_rows = None;
            self.atlas_resized = false;
        }
    }

    /// Send this frame's new glyphs to the GPU: the changed rows, or the whole atlas after it
    /// grew or was cleared. Called once a frame before rendering.
    pub fn upload(&mut self, renderer: &mut Renderer) {
        let Some(idx) = self.atlas_bind_group_idx else {
            return;
        };
        if self.atlas_resized {
            renderer.replace_texture_no_mips(
                idx,
                self.atlas_width,
                self.atlas_height,
                &self.atlas_data,
            );
        } else if let Some((y0, y1)) = self.dirty_rows {
            let y1 = y1.min(self.atlas_height);
            if y1 > y0 {
                let row = (self.atlas_width * 4) as usize;
                renderer.write_texture_rows(
                    idx,
                    self.atlas_width,
                    y0,
                    y1 - y0,
                    &self.atlas_data[y0 as usize * row..y1 as usize * row],
                );
            }
        }
        self.atlas_resized = false;
        self.dirty_rows = None;
        self.atlas_dirty = false;
    }

    pub fn measure_text(&mut self, text: &str, size: u32) -> f64 {
        self.measure_text_ex(0, text, size, 0.0)
    }

    pub fn measure_text_ex(&mut self, font_idx: usize, text: &str, size: u32, spacing: f32) -> f64 {
        let idx = if font_idx < self.fonts.len() {
            font_idx
        } else {
            0
        };
        let mut width = 0.0f32;
        let mut first = true;
        for ch in text.chars() {
            if !first && spacing != 0.0 {
                width += spacing;
            }
            first = false;
            width += self.advance(idx, ch, size);
        }
        width as f64
    }

    pub fn draw_text(
        &mut self,
        renderer: &mut Renderer,
        text: &str,
        x: f64,
        y: f64,
        size: u32,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        self.draw_text_ex(renderer, 0, text, x, y, size, 0.0, r, g, b, a);
    }

    pub fn draw_text_ex(
        &mut self,
        renderer: &mut Renderer,
        font_idx: usize,
        text: &str,
        x: f64,
        y: f64,
        size: u32,
        spacing: f32,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let idx = if font_idx < self.fonts.len() {
            font_idx
        } else {
            0
        };

        // Round-2 audit F5: glyphs were rasterized at LOGICAL pixel size
        // and stretched ×dpi by the 2D projection — a soft, bilinear-
        // magnified HUD at 150% display scaling. Rasterize at PHYSICAL
        // size and divide the layout metrics back to logical. fontdue
        // advances are exactly linear in pixel size, so measureText
        // (which stays in logical units) remains consistent with the
        // drawn advances.
        let dpi = if renderer.logical_width > 0 {
            (renderer.surface_config.width as f32 / renderer.logical_width as f32).max(0.25)
        } else {
            1.0
        };
        let phys_size = (((size as f32) * dpi).round().max(1.0)) as u32;
        // Exact logical/physical ratio actually in use (phys_size is
        // rounded, so derive from it rather than from dpi).
        let inv_dpi = size as f32 / phys_size as f32;

        // Ensure all glyphs are rasterized first
        for ch in text.chars() {
            self.rasterize_glyph(idx, ch, phys_size);
        }

        // Upload atlas if dirty
        self.ensure_atlas_uploaded(renderer);

        let atlas_bg = match self.atlas_bind_group_idx {
            Some(idx) => idx,
            None => return,
        };

        // sRGB-decode to linear like every other 2D vertex color — the
        // swapchain view's hardware encode expects linear shader output
        // (see renderer::srgb_u8_to_linear).
        let color = [
            crate::renderer::srgb_u8_to_linear(r),
            crate::renderer::srgb_u8_to_linear(g),
            crate::renderer::srgb_u8_to_linear(b),
            (a / 255.0) as f32,
        ];

        let aw = self.atlas_width as f32;
        let ah = self.atlas_height as f32;

        let mut cursor_x = x as f32;
        let mut first = true;
        for ch in text.chars() {
            if !first && spacing != 0.0 {
                cursor_x += spacing;
            }
            first = false;
            let key = (idx, ch, phys_size);
            if let Some(glyph) = self.glyph_cache.get(&key) {
                // Physical-res glyph metrics, scaled back to the logical
                // coordinate space the 2D pass works in. The projection's
                // ×dpi stretch then lands the bitmap 1:1 on physical
                // pixels instead of magnifying a logical-res raster.
                let gx = cursor_x + glyph.x_offset * inv_dpi;
                let gy = y as f32 - (glyph.y_offset + glyph.height as f32) * inv_dpi + size as f32;
                let gw = glyph.width as f32 * inv_dpi;
                let gh = glyph.height as f32 * inv_dpi;

                if gw > 0.0 && gh > 0.0 {
                    let u0 = glyph.atlas_x as f32 / aw;
                    let v0 = glyph.atlas_y as f32 / ah;
                    let u1 = (glyph.atlas_x + glyph.width) as f32 / aw;
                    let v1 = (glyph.atlas_y + glyph.height) as f32 / ah;

                    renderer.draw_textured_quad(gx, gy, gw, gh, u0, v0, u1, v1, color, atlas_bg);
                }

                cursor_x += glyph.advance * inv_dpi;
            } else {
                // not in the atlas this frame (it was full): keep the spacing
                cursor_x += self.advance(idx, ch, phys_size) * inv_dpi;
            }
        }
    }

    pub fn unload_font(&mut self, font_idx: usize) {
        if font_idx > 0 && font_idx < self.fonts.len() {
            self.glyph_cache.retain(|k, _| k.0 != font_idx);
            self.advances.retain(|k, _| k.0 != font_idx);
            self.sdf_glyph_cache.retain(|k, _| k.0 != font_idx);
        }
    }

    // SDF text rendering: rasterize at a fixed large size, generate SDF, then render at any size

    const SDF_BASE_SIZE: u32 = 48;
    const SDF_SPREAD: f32 = 6.0;

    fn rasterize_sdf_glyph(&mut self, font_idx: usize, ch: char) -> &GlyphInfo {
        let key = (font_idx, ch);
        if self.sdf_glyph_cache.contains_key(&key) {
            return &self.sdf_glyph_cache[&key];
        }

        let font = &self.fonts[font_idx];
        let (metrics, bitmap) = font.rasterize(ch, Self::SDF_BASE_SIZE as f32);
        let gw = metrics.width as u32;
        let gh = metrics.height as u32;

        // Generate SDF from bitmap
        let sdf_data = if gw > 0 && gh > 0 {
            generate_sdf(&bitmap, gw, gh, Self::SDF_SPREAD)
        } else {
            Vec::new()
        };

        // Place in SDF atlas
        if self.sdf_atlas_cursor_x + gw > SDF_ATLAS_SIZE {
            self.sdf_atlas_cursor_x = 0;
            self.sdf_atlas_cursor_y += self.sdf_atlas_row_height;
            self.sdf_atlas_row_height = 0;
        }

        let ax = self.sdf_atlas_cursor_x;
        let ay = self.sdf_atlas_cursor_y;

        for row in 0..gh {
            for col in 0..gw {
                let src = ((row * gw + col) * 4) as usize;
                let dst = (((ay + row) * SDF_ATLAS_SIZE + ax + col) * 4) as usize;
                if src + 3 < sdf_data.len() && dst + 3 < self.sdf_atlas_data.len() {
                    self.sdf_atlas_data[dst] = sdf_data[src];
                    self.sdf_atlas_data[dst + 1] = sdf_data[src + 1];
                    self.sdf_atlas_data[dst + 2] = sdf_data[src + 2];
                    self.sdf_atlas_data[dst + 3] = sdf_data[src + 3];
                }
            }
        }

        self.sdf_atlas_cursor_x += gw + 1;
        if gh + 1 > self.sdf_atlas_row_height {
            self.sdf_atlas_row_height = gh + 1;
        }
        self.sdf_atlas_dirty = true;

        self.sdf_glyph_cache.insert(
            key,
            GlyphInfo {
                atlas_x: ax,
                atlas_y: ay,
                width: gw,
                height: gh,
                advance: metrics.advance_width,
                x_offset: metrics.xmin as f32,
                y_offset: metrics.ymin as f32,
            },
        );

        &self.sdf_glyph_cache[&key]
    }

    fn ensure_sdf_atlas_uploaded(&mut self, renderer: &mut Renderer) {
        if !self.sdf_atlas_dirty {
            return;
        }
        match self.sdf_atlas_bind_group_idx {
            None => {
                let idx =
                    renderer.register_texture(SDF_ATLAS_SIZE, SDF_ATLAS_SIZE, &self.sdf_atlas_data);
                self.sdf_atlas_bind_group_idx = Some(idx);
            }
            Some(idx) => {
                renderer.update_texture(idx, SDF_ATLAS_SIZE, SDF_ATLAS_SIZE, &self.sdf_atlas_data);
            }
        }
        self.sdf_atlas_dirty = false;
    }

    /// Draw text using SDF atlas. The text scales smoothly to any size.
    pub fn draw_text_sdf(
        &mut self,
        renderer: &mut Renderer,
        font_idx: usize,
        text: &str,
        x: f64,
        y: f64,
        size: u32,
        spacing: f32,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let idx = if font_idx < self.fonts.len() {
            font_idx
        } else {
            0
        };
        let scale = size as f32 / Self::SDF_BASE_SIZE as f32;

        for ch in text.chars() {
            self.rasterize_sdf_glyph(idx, ch);
        }
        self.ensure_sdf_atlas_uploaded(renderer);

        let atlas_bg = match self.sdf_atlas_bind_group_idx {
            Some(idx) => idx,
            None => return,
        };

        // sRGB-decode to linear like every other 2D vertex color — the
        // swapchain view's hardware encode expects linear shader output
        // (see renderer::srgb_u8_to_linear).
        let color = [
            crate::renderer::srgb_u8_to_linear(r),
            crate::renderer::srgb_u8_to_linear(g),
            crate::renderer::srgb_u8_to_linear(b),
            (a / 255.0) as f32,
        ];

        let aw = SDF_ATLAS_SIZE as f32;
        let ah = SDF_ATLAS_SIZE as f32;

        let mut cursor_x = x as f32;
        let mut first = true;
        for ch in text.chars() {
            if !first && spacing != 0.0 {
                cursor_x += spacing * scale;
            }
            first = false;
            let key = (idx, ch);
            if let Some(glyph) = self.sdf_glyph_cache.get(&key) {
                let gx = cursor_x + glyph.x_offset * scale;
                let gy =
                    y as f32 - glyph.y_offset * scale - glyph.height as f32 * scale + size as f32;
                let gw = glyph.width as f32 * scale;
                let gh = glyph.height as f32 * scale;

                if gw > 0.0 && gh > 0.0 {
                    let u0 = glyph.atlas_x as f32 / aw;
                    let v0 = glyph.atlas_y as f32 / ah;
                    let u1 = (glyph.atlas_x + glyph.width) as f32 / aw;
                    let v1 = (glyph.atlas_y + glyph.height) as f32 / ah;
                    renderer.draw_textured_quad(gx, gy, gw, gh, u0, v0, u1, v1, color, atlas_bg);
                }

                cursor_x += glyph.advance * scale;
            }
        }
    }
}
