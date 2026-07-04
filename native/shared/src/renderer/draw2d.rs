//! Immediate-mode 2D shape drawing (rect/line/circle — batched quads
//! over the white texture) and this-frame output plumbing
//! (FrameTarget: swapchain frame or the headless offscreen target).
//! Split out of renderer/mod.rs (2000-line file policy).

use super::types::Vertex2D;
use super::Renderer;

impl Renderer {

    pub fn draw_rect(&mut self, x: f64, y: f64, w: f64, h: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32_srgb(r, g, b, a);
        let base = self.vertices_2d.len() as u32;
        let (x, y, w, h) = (x as f32, y as f32, w as f32, h as f32);

        self.vertices_2d.push(Vertex2D { position: [x, y], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x + w, y], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x + w, y + h], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x, y + h], uv: [0.0, 0.0], color });

        self.indices_2d.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn draw_rect_lines(&mut self, x: f64, y: f64, w: f64, h: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
        let t = thickness;
        self.draw_rect(x, y, w, t, r, g, b, a);
        self.draw_rect(x, y + h - t, w, t, r, g, b, a);
        self.draw_rect(x, y + t, t, h - 2.0 * t, r, g, b, a);
        self.draw_rect(x + w - t, y + t, t, h - 2.0 * t, r, g, b, a);
    }

    pub fn draw_line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, thickness: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32_srgb(r, g, b, a);
        let dx = (x2 - x1) as f32;
        let dy = (y2 - y1) as f32;
        let len = (dx * dx + dy * dy).sqrt();
        if len == 0.0 { return; }
        let half_t = (thickness as f32) * 0.5;
        let nx = -dy / len * half_t;
        let ny = dx / len * half_t;
        let (x1, y1, x2, y2) = (x1 as f32, y1 as f32, x2 as f32, y2 as f32);
        let base = self.vertices_2d.len() as u32;

        self.vertices_2d.push(Vertex2D { position: [x1 + nx, y1 + ny], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x1 - nx, y1 - ny], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x2 - nx, y2 - ny], uv: [0.0, 0.0], color });
        self.vertices_2d.push(Vertex2D { position: [x2 + nx, y2 + ny], uv: [0.0, 0.0], color });

        self.indices_2d.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn draw_circle(&mut self, cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
        self.ensure_draw_state(0);
        let color = Self::color_to_f32_srgb(r, g, b, a);
        let segments = 36u32;
        let base = self.vertices_2d.len() as u32;
        let (cx, cy, radius) = (cx as f32, cy as f32, radius as f32);

        self.vertices_2d.push(Vertex2D { position: [cx, cy], uv: [0.0, 0.0], color });
        for i in 0..segments {
            let angle = (i as f32) / (segments as f32) * std::f32::consts::TAU;
            self.vertices_2d.push(Vertex2D {
                position: [cx + radius * angle.cos(), cy + radius * angle.sin()],
                uv: [0.0, 0.0],
                color,
            });
        }
        for i in 0..segments {
            let next = if i + 1 < segments { i + 1 } else { 0 };
            self.indices_2d.extend_from_slice(&[base, base + 1 + i, base + 1 + next]);
        }
    }

    pub fn draw_circle_lines(&mut self, cx: f64, cy: f64, radius: f64, r: f64, g: f64, b: f64, a: f64) {
        let segments = 36;
        for i in 0..segments {
            let a1 = (i as f64) / (segments as f64) * std::f64::consts::TAU;
            let a2 = ((i + 1) as f64) / (segments as f64) * std::f64::consts::TAU;
            self.draw_line(
                cx + radius * a1.cos(), cy + radius * a1.sin(),
                cx + radius * a2.cos(), cy + radius * a2.sin(),
                1.0, r, g, b, a,
            );
        }
    }
}

/// This frame's output: a swapchain frame, or the persistent offscreen
/// texture in headless mode.
pub(crate) enum FrameTarget {
    Surface(wgpu::SurfaceTexture),
    Headless,
}

impl Renderer {
    /// Acquire the frame target. `None` means the swapchain was lost and
    /// has been reconfigured — skip this frame.
    pub(super) fn acquire_frame(&self) -> Option<FrameTarget> {
        match &self.surface {
            Some(surface) => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(FrameTarget::Surface(t)),
                _ => {
                    surface.configure(&self.device, &self.surface_config);
                    None
                }
            },
            None => Some(FrameTarget::Headless),
        }
    }

    pub(super) fn frame_texture<'a>(&'a self, target: &'a FrameTarget) -> &'a wgpu::Texture {
        match target {
            FrameTarget::Surface(t) => &t.texture,
            FrameTarget::Headless => self
                .headless_target
                .as_ref()
                .expect("headless renderer always owns a headless_target"),
        }
    }

    pub(super) fn present_frame(&self, target: FrameTarget) {
        if let FrameTarget::Surface(t) = target {
            t.present();
        }
    }
}

