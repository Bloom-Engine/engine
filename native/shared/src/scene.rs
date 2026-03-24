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
    // GPU resources (lazily created)
    pub gpu_vb: Option<wgpu::Buffer>,
    pub gpu_ib: Option<wgpu::Buffer>,
    pub gpu_index_count: u32,
    gpu_uniform_buf: Option<wgpu::Buffer>,
    gpu_uniform_bg: Option<wgpu::BindGroup>,
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
            gpu_vb: None,
            gpu_ib: None,
            gpu_index_count: 0,
            gpu_uniform_buf: None,
            gpu_uniform_bg: None,
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
            node.vertices = vertices;
            node.indices = indices;
            node.geo_dirty = true;
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

    pub fn set_material_texture(&mut self, handle: f64, texture_idx: u32) {
        if let Some(node) = self.nodes.get_mut(handle) {
            node.material.texture_idx = texture_idx;
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

    /// Render all visible scene nodes into the given render pass.
    /// Must be called after prepare() and after the pipeline/lighting/joints are set.
    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        texture_bind_groups: &'a [wgpu::BindGroup],
    ) {
        for (_handle, node) in self.nodes.iter() {
            if !node.visible || node.indices.is_empty() {
                continue;
            }
            let Some(vb) = &node.gpu_vb else { continue };
            let Some(ib) = &node.gpu_ib else { continue };
            let Some(bg) = &node.gpu_uniform_bg else { continue };

            // Bind per-node uniforms (group 0)
            pass.set_bind_group(0, bg, &[]);

            // Bind texture (group 2)
            let tex_idx = node.material.texture_idx as usize;
            if tex_idx < texture_bind_groups.len() {
                pass.set_bind_group(2, &texture_bind_groups[tex_idx], &[]);
            } else if !texture_bind_groups.is_empty() {
                pass.set_bind_group(2, &texture_bind_groups[0], &[]);
            }

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
