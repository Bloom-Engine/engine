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
        // Replace with a dummy to avoid shifting indices.
        // The default font (index 0) cannot be unloaded.
        if font_idx > 0 && font_idx < self.fonts.len() {
            // Remove cached glyphs for this font
            self.glyph_cache.retain(|k, _| k.0 != font_idx);
        }
    }
}
