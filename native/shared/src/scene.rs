//! Retained scene graph for Bloom Engine.
//!
//! Unlike immediate-mode drawing (drawCube, drawModel), the scene graph holds
//! persistent meshes that survive across frames. Systems update geometry and
//! transforms; the renderer draws all visible nodes each frame automatically.

use wgpu::util::DeviceExt;
use crate::handles::HandleRegistry;
use crate::renderer::Vertex3D;

// ============================================================
// PBR Material
// ============================================================

#[derive(Clone, Debug)]
pub struct PbrMaterial {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metalness: f32,
    pub opacity: f32,
    pub emissive: [f32; 3],
    pub double_sided: bool,
    pub texture_idx: u32,
    /// Normal-map texture. 0 means "no normal map" — scene shader falls
    /// back to the geometric normal. Stored as a texture index rather
    /// than bind group so the renderer can build per-material bind
    /// groups lazily without SceneGraph holding GPU references.
    pub normal_texture_idx: u32,
    pub metallic_roughness_texture_idx: u32,
    pub emissive_texture_idx: u32,
    pub occlusion_texture_idx: u32,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            roughness: 0.8,
            metalness: 0.0,
            opacity: 1.0,
            emissive: [0.0, 0.0, 0.0],
            double_sided: false,
            texture_idx: 0,
            normal_texture_idx: 0,
            metallic_roughness_texture_idx: 0,
            emissive_texture_idx: 0,
            occlusion_texture_idx: 0,
        }
    }
}

// ============================================================
// Scene Node Uniforms (matches Uniforms3D in renderer)
// ============================================================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NodeUniforms {
    mvp: [[f32; 4]; 4],
    model_tint: [f32; 4],
}

// ============================================================
// Scene Node
// ============================================================

pub struct SceneNode {
    // Geometry (CPU-side, updated by systems)
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
    // Material
    pub material: PbrMaterial,
    // Transform
    pub transform: [[f32; 4]; 4],
    // Flags
    pub visible: bool,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    pub parent: f64,
    // Editor user data — an arbitrary i64 attached to the node. The editor
    // uses this to store the entity id directly on the scene node so picking
    // can return the entity id without a handle → id map lookup (Q7).
    pub user_data: i64,
    // Cached world-space AABB, recomputed when geometry changes (Q5).
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    // GPU resources (lazily created)
    pub gpu_vb: Option<wgpu::Buffer>,
    pub gpu_ib: Option<wgpu::Buffer>,
    pub gpu_index_count: u32,
    gpu_uniform_buf: Option<wgpu::Buffer>,
    gpu_uniform_bg: Option<wgpu::BindGroup>,
    /// Material bind group for the scene pipeline — holds base color,
    /// normal, metallic-roughness and emissive texture views in one
    /// group. Rebuilt whenever one of the material texture indices
    /// changes (tracked via `mat_dirty`).
    pub gpu_material_bg: Option<wgpu::BindGroup>,
    pub gpu_material_uniform_buf: Option<wgpu::Buffer>,
    pub mat_dirty: bool,
    geo_dirty: bool,
}

impl SceneNode {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            material: PbrMaterial::default(),
            transform: crate::renderer::IDENTITY_MAT4,
            visible: true,
            cast_shadow: true,
            receive_shadow: true,
            parent: 0.0,
            user_data: 0,
            bounds_min: [0.0; 3],
            bounds_max: [0.0; 3],
            gpu_vb: None,
            gpu_ib: None,
            gpu_index_count: 0,
            gpu_uniform_buf: None,
            gpu_uniform_bg: None,
            gpu_material_bg: None,
            gpu_material_uniform_buf: None,
            mat_dirty: true,
            geo_dirty: true,
        }
    }
}

// ============================================================
// Scene Graph
// ============================================================

pub struct SceneGraph {
    pub nodes: HandleRegistry<SceneNode>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: HandleRegistry::new(),
        }
    }

    pub fn create_node(&mut self) -> f64 {
        self.nodes.alloc(SceneNode::new())
    }

    pub fn destroy_node(&mut self, handle: f64) {
        self.nodes.free(handle);
    }

    pub fn set_transform(&mut self, handle: f64, matrix: [[f32; 4]; 4]) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.transform = matrix;
        }
    }

    pub fn set_visible(&mut self, handle: f64, visible: bool) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.visible = visible;
        }
    }

    pub fn set_cast_shadow(&mut self, handle: f64, cast: bool) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.cast_shadow = cast;
        }
    }

    pub fn set_receive_shadow(&mut self, handle: f64, receive: bool) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.receive_shadow = receive;
        }
    }

    pub fn set_parent(&mut self, handle: f64, parent: f64) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.parent = parent;
        }
    }

    pub fn update_geometry(&mut self, handle: f64, vertices: Vec<Vertex3D>, indices: Vec<u32>) {
        if let Some(node) = self.nodes.get_mut(handle) {
            // Recompute bounds from vertex positions (Q5).
            let mut bmin = [f32::MAX; 3];
            let mut bmax = [f32::MIN; 3];
            for v in &vertices {
                for k in 0..3 {
                    if v.position[k] < bmin[k] { bmin[k] = v.position[k]; }
                    if v.position[k] > bmax[k] { bmax[k] = v.position[k]; }
                }
            }
            if vertices.is_empty() {
                bmin = [0.0; 3];
                bmax = [0.0; 3];
            }
            node.bounds_min = bmin;
            node.bounds_max = bmax;
            node.vertices = vertices;
            node.indices = indices;
            node.geo_dirty = true;
        }
    }

    // ---- Q4: transform read-back -------------------------------------------

    /// Read back the current 4x4 transform matrix of a scene node.
    pub fn get_transform(&self, handle: f64) -> [[f32; 4]; 4] {
        match self.nodes.get(handle) {
            Some(node) => node.transform,
            None => crate::renderer::IDENTITY_MAT4,
        }
    }

    // ---- Q5: world-space bounds query --------------------------------------

    /// Return the cached AABB of a scene node's geometry (local space).
    pub fn get_bounds(&self, handle: f64) -> ([f32; 3], [f32; 3]) {
        match self.nodes.get(handle) {
            Some(node) => (node.bounds_min, node.bounds_max),
            None => ([0.0; 3], [0.0; 3]),
        }
    }

    // ---- Q7: user data -----------------------------------------------------

    pub fn set_user_data(&mut self, handle: f64, data: i64) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.user_data = data;
        }
    }

    pub fn get_user_data(&self, handle: f64) -> i64 {
        match self.nodes.get(handle) {
            Some(node) => node.user_data,
            None => 0,
        }
    }

    pub fn set_material_color(&mut self, handle: f64, r: f32, g: f32, b: f32, a: f32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.color = [r, g, b];
            node.material.opacity = a;
        }
    }

    pub fn set_material_pbr(&mut self, handle: f64, roughness: f32, metalness: f32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.roughness = roughness;
            node.material.metalness = metalness;
        }
    }

    /// Q8: Set a water-like material on a scene node. The actual animated
    /// wave shader requires a dedicated WGSL pipeline pass (deferred).
    /// For now, this sets a translucent tinted material that approximates water.
    pub fn set_material_water(&mut self, handle: f64, _wave_amp: f32, _wave_speed: f32, r: f32, g: f32, b: f32, a: f32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.color = [r, g, b];
            node.material.opacity = a;
            node.material.roughness = 0.1;
            node.material.metalness = 0.3;
        }
    }

    pub fn set_material_texture(&mut self, handle: f64, texture_idx: u32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.texture_idx = texture_idx;
            node.mat_dirty = true;
        }
    }

    pub fn set_material_normal_texture(&mut self, handle: f64, texture_idx: u32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.normal_texture_idx = texture_idx;
            node.mat_dirty = true;
        }
    }

    pub fn set_material_metallic_roughness_texture(&mut self, handle: f64, texture_idx: u32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.metallic_roughness_texture_idx = texture_idx;
            node.mat_dirty = true;
        }
    }

    pub fn set_material_emissive_texture(&mut self, handle: f64, texture_idx: u32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.emissive_texture_idx = texture_idx;
            node.mat_dirty = true;
        }
    }

    pub fn set_material_emissive_factor(&mut self, handle: f64, r: f32, g: f32, b: f32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.emissive = [r, g, b];
            node.mat_dirty = true;
        }
    }

    /// Prepare GPU resources for all visible nodes. Must be called before render().
    /// Creates/updates vertex buffers, index buffers, and uniform bind groups.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vp_matrix: &[[f32; 4]; 4],
        uniform_layout: &wgpu::BindGroupLayout,
    ) {
        for (_handle, node) in self.nodes.iter_mut() {
            if !node.visible || node.indices.is_empty() {
                continue;
            }

            // Update geometry buffers if dirty
            if node.geo_dirty || node.gpu_vb.is_none() {
                node.gpu_vb = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene_node_vb"),
                    contents: bytemuck::cast_slice(&node.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }));
                node.gpu_ib = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene_node_ib"),
                    contents: bytemuck::cast_slice(&node.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }));
                node.gpu_index_count = node.indices.len() as u32;
                node.geo_dirty = false;
            }

            // Compute MVP = VP * Model
            let mvp = mat4_mul(vp_matrix, &node.transform);
            let tint = [
                node.material.color[0],
                node.material.color[1],
                node.material.color[2],
                node.material.opacity,
            ];
            let uniforms = NodeUniforms { mvp, model_tint: tint };

            // Create or update uniform buffer
            if node.gpu_uniform_buf.is_none() {
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene_node_uniform"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("scene_node_uniform_bg"),
                    layout: uniform_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                node.gpu_uniform_buf = Some(buf);
                node.gpu_uniform_bg = Some(bg);
            } else {
                queue.write_buffer(
                    node.gpu_uniform_buf.as_ref().unwrap(),
                    0,
                    bytemuck::bytes_of(&uniforms),
                );
            }
        }
    }

    /// Build / refresh per-node material bind groups for the scene
    /// pipeline. Must be called every frame after `prepare` and before
    /// `render`. Only rebuilds when a material changed (mat_dirty).
    pub fn prepare_materials(&mut self, renderer: &crate::renderer::Renderer) {
        for (_handle, node) in self.nodes.iter_mut() {
            if !node.visible || node.indices.is_empty() {
                continue;
            }
            if node.mat_dirty || node.gpu_material_bg.is_none() {
                // Allocate or reuse the per-material uniform buffer.
                // (Could be updated in place when factors change, but
                // the current path always rebuilds together with the
                // bind group — cheap and simpler.)
                let uniform = renderer.create_scene_material_uniform(
                    node.material.metalness,
                    node.material.roughness,
                    node.material.emissive,
                );
                let bg = renderer.create_scene_material_bg(
                    node.material.texture_idx,
                    node.material.normal_texture_idx,
                    node.material.metallic_roughness_texture_idx,
                    node.material.emissive_texture_idx,
                    node.material.occlusion_texture_idx,
                    &uniform,
                );
                node.gpu_material_bg = Some(bg);
                node.gpu_material_uniform_buf = Some(uniform);
                node.mat_dirty = false;
            }
        }
    }

    /// Render all visible scene nodes into the given render pass.
    /// Must be called after prepare() and after the pipeline/lighting/joints are set.
    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        for (_handle, node) in self.nodes.iter() {
            if !node.visible || node.indices.is_empty() {
                continue;
            }
            let Some(vb) = &node.gpu_vb else { continue };
            let Some(ib) = &node.gpu_ib else { continue };
            let Some(bg) = &node.gpu_uniform_bg else { continue };
            let Some(mat_bg) = &node.gpu_material_bg else { continue };

            // Bind per-node uniforms (group 0)
            pass.set_bind_group(0, bg, &[]);

            // Bind per-node material (group 2: base color + normal map)
            pass.set_bind_group(2, mat_bg, &[]);

            // Bind vertex/index buffers and draw
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..node.gpu_index_count, 0, 0..1);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.iter().count()
    }
}

// ============================================================
// Matrix math (4x4, column-major)
// ============================================================

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            result[col][row] = a[0][row] * b[col][0]
                             + a[1][row] * b[col][1]
                             + a[2][row] * b[col][2]
                             + a[3][row] * b[col][3];
        }
    }
    result
}
