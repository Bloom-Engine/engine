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
                        if dist < min_dist { min_dist = dist; }
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
        let font = Font::from_bytes(data, FontSettings::default())
            .expect("Failed to load font");
        self.fonts.push(font);
        self.fonts.len() - 1
    }

    fn rasterize_glyph(&mut self, font_idx: usize, ch: char, size: u32) -> &GlyphInfo {
        let key = (font_idx, ch, size);
        if self.glyph_cache.contains_key(&key) {
            return &self.glyph_cache[&key];
        }

        let font = &self.fonts[font_idx];
        let (metrics, bitmap) = font.rasterize(ch, size as f32);

        let gw = metrics.width as u32;
        let gh = metrics.height as u32;

        if self.atlas_cursor_x + gw > self.atlas_width {
            self.atlas_cursor_x = 0;
            self.atlas_cursor_y += self.atlas_row_height;
            self.atlas_row_height = 0;
        }

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

        self.atlas_cursor_x += gw + 1;
        if gh + 1 > self.atlas_row_height {
            self.atlas_row_height = gh + 1;
        }
        self.atlas_dirty = true;

        self.glyph_cache.insert(key, GlyphInfo {
            atlas_x: ax,
            atlas_y: ay,
            width: gw,
            height: gh,
            advance: metrics.advance_width,
            x_offset: metrics.xmin as f32,
            y_offset: metrics.ymin as f32,
        });

        &self.glyph_cache[&key]
    }

    fn ensure_atlas_uploaded(&mut self, renderer: &mut Renderer) {
        if !self.atlas_dirty { return; }

        match self.atlas_bind_group_idx {
            None => {
                let idx = renderer.register_texture(self.atlas_width, self.atlas_height, &self.atlas_data);
                self.atlas_bind_group_idx = Some(idx);
            }
            Some(idx) => {
                renderer.update_texture(idx, self.atlas_width, self.atlas_height, &self.atlas_data);
            }
        }
        self.atlas_dirty = false;
    }

    pub fn measure_text(&mut self, text: &str, size: u32) -> f64 {
        self.measure_text_ex(0, text, size, 0.0)
    }

    pub fn measure_text_ex(&mut self, font_idx: usize, text: &str, size: u32, spacing: f32) -> f64 {
        let idx = if font_idx < self.fonts.len() { font_idx } else { 0 };
        let mut width = 0.0f32;
        let mut first = true;
        for ch in text.chars() {
            if !first && spacing != 0.0 {
                width += spacing;
            }
            first = false;
            let glyph = self.rasterize_glyph(idx, ch, size);
            width += glyph.advance;
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
        r: f64, g: f64, b: f64, a: f64,
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
        r: f64, g: f64, b: f64, a: f64,
    ) {
        let idx = if font_idx < self.fonts.len() { font_idx } else { 0 };

        // Ensure all glyphs are rasterized first
        for ch in text.chars() {
            self.rasterize_glyph(idx, ch, size);
        }

        // Upload atlas if dirty
        self.ensure_atlas_uploaded(renderer);

        let atlas_bg = match self.atlas_bind_group_idx {
            Some(idx) => idx,
            None => return,
        };

        let color = [
            (r / 255.0) as f32,
            (g / 255.0) as f32,
            (b / 255.0) as f32,
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
            let key = (idx, ch, size);
            if let Some(glyph) = self.glyph_cache.get(&key) {
                let gx = cursor_x + glyph.x_offset;
                let gy = y as f32 - glyph.y_offset - glyph.height as f32 + size as f32;
                let gw = glyph.width as f32;
                let gh = glyph.height as f32;

                if gw > 0.0 && gh > 0.0 {
                    let u0 = glyph.atlas_x as f32 / aw;
                    let v0 = glyph.atlas_y as f32 / ah;
                    let u1 = (glyph.atlas_x + glyph.width) as f32 / aw;
                    let v1 = (glyph.atlas_y + glyph.height) as f32 / ah;

                    renderer.draw_textured_quad(gx, gy, gw, gh, u0, v0, u1, v1, color, atlas_bg);
                }

                cursor_x += glyph.advance;
            }
        }
    }

    pub fn unload_font(&mut self, font_idx: usize) {
        if font_idx > 0 && font_idx < self.fonts.len() {
            self.glyph_cache.retain(|k, _| k.0 != font_idx);
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
        if self.sdf_atlas_cursor_x + gw > self.atlas_width {
            self.sdf_atlas_cursor_x = 0;
            self.sdf_atlas_cursor_y += self.sdf_atlas_row_height;
            self.sdf_atlas_row_height = 0;
        }

        let ax = self.sdf_atlas_cursor_x;
        let ay = self.sdf_atlas_cursor_y;

        for row in 0..gh {
            for col in 0..gw {
                let src = ((row * gw + col) * 4) as usize;
                let dst = (((ay + row) * self.atlas_width + ax + col) * 4) as usize;
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

        self.sdf_glyph_cache.insert(key, GlyphInfo {
            atlas_x: ax,
            atlas_y: ay,
            width: gw,
            height: gh,
            advance: metrics.advance_width,
            x_offset: metrics.xmin as f32,
            y_offset: metrics.ymin as f32,
        });

        &self.sdf_glyph_cache[&key]
    }

    fn ensure_sdf_atlas_uploaded(&mut self, renderer: &mut Renderer) {
        if !self.sdf_atlas_dirty { return; }
        match self.sdf_atlas_bind_group_idx {
            None => {
                let idx = renderer.register_texture(self.atlas_width, self.atlas_height, &self.sdf_atlas_data);
                self.sdf_atlas_bind_group_idx = Some(idx);
            }
            Some(idx) => {
                renderer.update_texture(idx, self.atlas_width, self.atlas_height, &self.sdf_atlas_data);
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
        r: f64, g: f64, b: f64, a: f64,
    ) {
        let idx = if font_idx < self.fonts.len() { font_idx } else { 0 };
        let scale = size as f32 / Self::SDF_BASE_SIZE as f32;

        for ch in text.chars() {
            self.rasterize_sdf_glyph(idx, ch);
        }
        self.ensure_sdf_atlas_uploaded(renderer);

        let atlas_bg = match self.sdf_atlas_bind_group_idx {
            Some(idx) => idx,
            None => return,
        };

        let color = [
            (r / 255.0) as f32,
            (g / 255.0) as f32,
            (b / 255.0) as f32,
            (a / 255.0) as f32,
        ];

        let aw = self.atlas_width as f32;
        let ah = self.atlas_height as f32;

        let mut cursor_x = x as f32;
        let mut first = true;
        for ch in text.chars() {
            if !first && spacing != 0.0 { cursor_x += spacing * scale; }
            first = false;
            let key = (idx, ch);
            if let Some(glyph) = self.sdf_glyph_cache.get(&key) {
                let gx = cursor_x + glyph.x_offset * scale;
                let gy = y as f32 - glyph.y_offset * scale - glyph.height as f32 * scale + size as f32;
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
