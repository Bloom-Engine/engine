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

pub struct TextureManager {
    pub textures: HandleRegistry<TextureData>,
    pub images: HandleRegistry<ImageData>,
}

impl TextureManager {
    pub fn new() -> Self {
        Self {
            textures: HandleRegistry::new(),
            images: HandleRegistry::new(),
        }
    }

    pub fn load_texture(&mut self, renderer: &mut Renderer, file_data: &[u8]) -> f64 {
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

    pub fn load_texture_from_image(&mut self, handle: f64, renderer: &mut Renderer) -> f64 {
        if let Some(img_data) = self.images.get(handle) {
            let bind_group_idx = renderer.register_texture(img_data.width, img_data.height, &img_data.data);
            self.textures.alloc(TextureData { bind_group_idx, width: img_data.width, height: img_data.height })
        } else {
            0.0
        }
    }
}
