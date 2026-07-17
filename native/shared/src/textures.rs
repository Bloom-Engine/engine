use crate::handles::HandleRegistry;
use crate::renderer::Renderer;

pub struct TextureData {
    pub bind_group_idx: u32,
    pub width: u32,
    pub height: u32,
}

pub struct ImageData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Placeholder for Q1 render texture data. Actual GPU implementation
/// requires creating a wgpu::Texture with RENDER_ATTACHMENT + TEXTURE_BINDING
/// usage and wiring it into the renderer's pass management. The FFI surface
/// is stable; the implementation lands in a focused GPU session.
pub struct RenderTextureData {
    pub width: u32,
    pub height: u32,
    pub texture_handle: f64,  // Points to the corresponding TextureData entry.
}

pub struct TextureManager {
    pub textures: HandleRegistry<TextureData>,
    pub images: HandleRegistry<ImageData>,
    pub render_textures: HandleRegistry<RenderTextureData>,
}

impl TextureManager {
    pub fn new() -> Self {
        Self {
            textures: HandleRegistry::new(),
            images: HandleRegistry::new(),
            render_textures: HandleRegistry::new(),
        }
    }

    /// Q1: Create a render texture. Returns a handle. The actual wgpu texture
    /// is created by the calling FFI function which has access to the Renderer.
    pub fn load_render_texture(&mut self, width: u32, height: u32) -> f64 {
        self.render_textures.alloc(RenderTextureData {
            width, height, texture_handle: 0.0,
        })
    }

    /// Set the texture handle for a render texture (called after GPU creation).
    pub fn set_render_texture_handle(&mut self, rt_handle: f64, tex_handle: f64) {
        if let Some(rt) = self.render_textures.get_mut(rt_handle) {
            rt.texture_handle = tex_handle;
        }
    }

    pub fn unload_render_texture(&mut self, handle: f64) {
        self.render_textures.free(handle);
    }

    pub fn get_render_texture_texture(&self, handle: f64) -> f64 {
        match self.render_textures.get(handle) {
            Some(rt) => rt.texture_handle,
            None => 0.0,
        }
    }

    /// Decode an in-memory image (PNG/JPEG/…, whatever `image` is built with)
    /// to raw RGBA8. Shared by the web texture-array-from-files path, which
    /// fetches file bytes in JS and needs the same decode the native
    /// from-files FFI gets from `image::open`.
    pub fn decode_rgba8(file_data: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
        let img = image::load_from_memory(file_data).ok()?.to_rgba8();
        let (w, h) = img.dimensions();
        Some((img.into_raw(), w, h))
    }

    pub fn load_texture(&mut self, renderer: &mut Renderer, file_data: &[u8]) -> f64 {
        // Cooked textures (bloom-cook BC7 DDS) take the compressed path:
        // direct mip-chain upload where the adapter has BC support, CPU
        // decode to RGBA elsewhere. Magic-sniffed so loadTexture() is the
        // one entry point for raw and cooked assets alike.
        if file_data.len() >= 4 && &file_data[..4] == b"DDS " {
            #[cfg(feature = "image-extras")]
            return self.load_texture_dds(renderer, file_data);
            #[cfg(not(feature = "image-extras"))]
            {
                crate::ffi::feature_off_warn_once("loadTexture(.dds)", "image-extras");
                return 0.0;
            }
        }
        let img = match image::load_from_memory(file_data) {
            Ok(img) => img.to_rgba8(),
            Err(_) => return 0.0,
        };
        let width = img.width();
        let height = img.height();
        let data = img.into_raw();

        let bind_group_idx = renderer.register_texture(width, height, &data);
        self.textures.alloc(TextureData { bind_group_idx, width, height })
    }

    pub fn unload_texture(&mut self, handle: f64, renderer: &mut Renderer) {
        if let Some(tex) = self.textures.free(handle) {
            renderer.unload_texture(tex.bind_group_idx);
        }
    }

    pub fn get(&self, handle: f64) -> Option<&TextureData> {
        self.textures.get(handle)
    }

    pub fn load_image(&mut self, file_data: &[u8]) -> f64 {
        let img = match image::load_from_memory(file_data) {
            Ok(img) => img.to_rgba8(),
            Err(_) => return 0.0,
        };
        let width = img.width();
        let height = img.height();
        let data = img.into_raw();

        self.images.alloc(ImageData { data, width, height })
    }

    pub fn image_resize(&mut self, handle: f64, new_w: u32, new_h: u32) {
        if let Some(img_data) = self.images.get_mut(handle) {
            let src = image::RgbaImage::from_raw(img_data.width, img_data.height, std::mem::take(&mut img_data.data));
            if let Some(src) = src {
                let resized = image::imageops::resize(&src, new_w, new_h, image::imageops::FilterType::Triangle);
                img_data.width = new_w;
                img_data.height = new_h;
                img_data.data = resized.into_raw();
            }
        }
    }

    pub fn image_crop(&mut self, handle: f64, x: u32, y: u32, w: u32, h: u32) {
        if let Some(img_data) = self.images.get_mut(handle) {
            let src = image::RgbaImage::from_raw(img_data.width, img_data.height, std::mem::take(&mut img_data.data));
            if let Some(mut src) = src {
                let cropped = image::imageops::crop(&mut src, x, y, w, h).to_image();
                img_data.width = w;
                img_data.height = h;
                img_data.data = cropped.into_raw();
            }
        }
    }

    pub fn image_flip_h(&mut self, handle: f64) {
        if let Some(img_data) = self.images.get_mut(handle) {
            let src = image::RgbaImage::from_raw(img_data.width, img_data.height, std::mem::take(&mut img_data.data));
            if let Some(src) = src {
                let flipped = image::imageops::flip_horizontal(&src);
                img_data.data = flipped.into_raw();
            }
        }
    }

    pub fn image_flip_v(&mut self, handle: f64) {
        if let Some(img_data) = self.images.get_mut(handle) {
            let src = image::RgbaImage::from_raw(img_data.width, img_data.height, std::mem::take(&mut img_data.data));
            if let Some(src) = src {
                let flipped = image::imageops::flip_vertical(&src);
                img_data.data = flipped.into_raw();
            }
        }
    }

    #[cfg(feature = "image-extras")]
    fn load_texture_dds(&mut self, renderer: &mut Renderer, file_data: &[u8]) -> f64 {
        let Ok(dds) = image_dds::ddsfile::Dds::read(std::io::Cursor::new(file_data)) else {
            return 0.0;
        };
        let (width, height) = (dds.get_width(), dds.get_height());
        if let Some(bind_group_idx) = renderer.register_texture_dds(&dds) {
            return self.textures.alloc(TextureData { bind_group_idx, width, height });
        }
        // No BC support on this adapter (mobile GL): CPU-decode the top
        // mip and feed the regular RGBA path (which regenerates mips).
        match image_dds::image_from_dds(&dds, 0) {
            Ok(rgba) => {
                let bind_group_idx = renderer.register_texture(width, height, rgba.as_raw());
                self.textures.alloc(TextureData { bind_group_idx, width, height })
            }
            Err(_) => 0.0,
        }
    }

    pub fn load_texture_from_image(&mut self, handle: f64, renderer: &mut Renderer) -> f64 {
        if let Some(img_data) = self.images.get(handle) {
            let bind_group_idx = renderer.register_texture(img_data.width, img_data.height, &img_data.data);
            self.textures.alloc(TextureData { bind_group_idx, width: img_data.width, height: img_data.height })
        } else {
            0.0
        }
    }
}
