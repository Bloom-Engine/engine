//! Renderer: the public lighting API (ambient, directional, point and spot lights). Split from mod.rs.

#[allow(unused_imports)]
use super::*;

impl Renderer {
    // ============================================================
    // Lighting
    // ============================================================

    pub fn set_ambient_light(&mut self, r: f64, g: f64, b: f64, intensity: f64) {
        self.lighting_uniforms.ambient = [
            (r / 255.0) as f32,
            (g / 255.0) as f32,
            (b / 255.0) as f32,
            intensity as f32,
        ];
    }

    pub fn set_directional_light(
        &mut self,
        dx: f64,
        dy: f64,
        dz: f64,
        r: f64,
        g: f64,
        b: f64,
        intensity: f64,
    ) {
        // Note: the shadow cache reads `lighting_uniforms.light_dir`
        // directly at gate time, so no explicit invalidate is needed
        // here. Doing it via a setter would miss light changes that
        // happen through other paths (preset reset, etc.) anyway.
        self.lighting_uniforms.light_dir = [dx as f32, dy as f32, dz as f32, intensity as f32];
        self.lighting_uniforms.light_color = [
            (r / 255.0) as f32,
            (g / 255.0) as f32,
            (b / 255.0) as f32,
            0.0,
        ];
    }

    /// Add an additional directional light (up to MAX_DIR_LIGHTS).
    /// Color is 0-1 range (not 0-255).
    pub fn add_directional_light(
        &mut self,
        dx: f32,
        dy: f32,
        dz: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
    ) {
        let idx = self.lighting_uniforms.dir_light_count[0] as usize;
        if idx >= MAX_DIR_LIGHTS {
            return;
        }
        self.lighting_uniforms.dir_lights[idx] = DirLight {
            direction: [dx, dy, dz, intensity],
            color: [r, g, b, 0.0],
        };
        self.lighting_uniforms.dir_light_count[0] = (idx + 1) as f32;
    }

    /// Add a point light (up to MAX_POINT_LIGHTS).
    /// Color is 0-1 range.
    pub fn add_point_light(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        range: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
    ) {
        let idx = self.lighting_uniforms.point_light_count[0] as usize;
        if idx >= MAX_POINT_LIGHTS {
            return;
        }
        self.lighting_uniforms.point_lights[idx] = PointLight {
            position: [x, y, z, range],
            color: [r, g, b, intensity],
        };
        self.lighting_uniforms.point_light_count[0] = (idx + 1) as f32;
    }

    /// Submit a point light whose direct contribution must never silently
    /// fall back to an unshadowed light. Unsupported renderers reject the
    /// request before changing the live light list.
    #[allow(clippy::too_many_arguments)]
    pub fn add_shadowed_point_light(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        range: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
    ) -> bool {
        let index = self.lighting_uniforms.point_light_count[0] as usize;
        if index >= MAX_POINT_LIGHTS
            || !self.shadow_map.enabled
            // The path tracer consumes the point-light list directly and
            // does not sample raster VSM pages. Reject instead of silently
            // turning this into an unshadowed traced light.
            || self.pt_active()
            || [r, g, b].iter().any(|value| !value.is_finite())
            || !self.shadow_map.virtual_map.submit_local_request(
                index as u16,
                [x, y, z],
                range,
                intensity,
            )
        {
            return false;
        }
        self.lighting_uniforms.point_lights[index] = PointLight {
            position: [x, y, z, range],
            // Fail closed at submission. The shadow pass restores intensity
            // only for a camera-visible request admitted to the shared page
            // budget; paths that never prepare shadows cannot leak an
            // unshadowed local light.
            color: [r, g, b, 0.0],
        };
        self.lighting_uniforms.point_light_count[0] = (index + 1) as f32;
        true
    }

    /// Submit a perspective spot light whose direct contribution remains
    /// fail-closed until its single local VSM page is resident and clean.
    /// Cone angles are full vertex angles in degrees.
    #[allow(clippy::too_many_arguments)]
    pub fn add_shadowed_spot_light(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        direction_x: f32,
        direction_y: f32,
        direction_z: f32,
        range: f32,
        inner_cone_degrees: f32,
        outer_cone_degrees: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
    ) -> bool {
        let index = self.lighting_uniforms.point_light_count[0] as usize;
        if index >= MAX_POINT_LIGHTS
            || !self.shadow_map.enabled
            || self.pt_active()
            || [r, g, b].iter().any(|value| !value.is_finite())
            || !self.shadow_map.virtual_map.submit_local_spot_request(
                index as u16,
                [x, y, z],
                [direction_x, direction_y, direction_z],
                range,
                inner_cone_degrees,
                outer_cone_degrees,
                intensity,
            )
        {
            return false;
        }
        self.lighting_uniforms.point_lights[index] = PointLight {
            position: [x, y, z, range],
            color: [r, g, b, 0.0],
        };
        self.lighting_uniforms.point_light_count[0] = (index + 1) as f32;
        true
    }

    /// Clear all additional lights (called at begin_frame).
    pub fn clear_additional_lights(&mut self) {
        // dir_light_count.y carries the shadows-enabled flag for the shadow
        // samplers (core sample_shadow + material-path sample_sun_shadow).
        // Written here because this runs unconditionally every frame before
        // any lighting upload; .x stays the actual additional-light count.
        // (shadow_cascade_splits.w is NOT usable for this — it carries the
        // TSR mip-LOD bias, written by the shadow pass each frame.)
        let shadows_flag = if self.shadow_map.enabled { 1.0 } else { 0.0 };
        // dir_light_count.z carries SSR's ownership share for EN-021's
        // IBL-specular complement in fs_main_scene: strength while SSR
        // runs, 0 when disabled (full IBL specular returns).
        let ssr_share = if self.ssr_enabled {
            self.ssr_strength
        } else {
            0.0
        };
        self.lighting_uniforms.dir_light_count = [0.0, shadows_flag, ssr_share, 0.0];
        self.lighting_uniforms.point_light_count = [0.0; 4];
        self.shadow_map.virtual_map.clear_local_requests();
    }
}
