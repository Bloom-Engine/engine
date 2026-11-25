//! The two GPU vertex formats of a mesh, and the pipeline pairs that draw them.
//!
//! `Vertex3D` is 96 bytes: position, normal, colour, uv, joints, weights and
//! tangent. A static (unskinned) mesh has nothing in its joints and weights,
//! so its vertex buffer holds `StaticVertex3D` instead: the same fields at the
//! same offsets without the 32 bytes of joints and weights, 64 bytes a vertex.
//! Shaders keep declaring `@location(4) joints` and `@location(5) weights`
//! (the material ABI publishes locations 0..6): a pipeline that draws a static
//! mesh reads them from a second vertex stream with `array_stride` 0 over one
//! shared 32-byte buffer of zeros, so every vertex sees exactly the zeros its
//! 96-byte vertex held. Nothing in any shader changes.
//!
//! Any mesh with a joint or weight that is not exactly +0.0 (a skinned mesh, or
//! one whose skin data a game shader could read) keeps the 96-byte layout, so
//! a 64-byte mesh's zeros are exactly what its vertices held.
//! A pass that can draw both kinds holds a pipeline for each
//! (`ByLayout`) and picks one per draw from the mesh's `VertexLayout`, which
//! is stored next to its vertex buffer. The zero stream goes in the slot after
//! the pipeline's other vertex buffers (`zero_slot`): slot 1, or slot 2 where
//! slot 1 carries instances or the second texture coordinates.
//!
//! OpenGL keeps 96 bytes everywhere: GL reads a vertex-attribute stride of 0
//! as "tightly packed", not "the same vertex for every vertex".

use super::lazy_pipeline::{LazyRenderPipeline, WarmHandle};
use super::Vertex3D;
use std::sync::atomic::{AtomicBool, Ordering};

/// A static mesh's vertex: `Vertex3D` without joints and weights.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StaticVertex3D {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

impl StaticVertex3D {
    pub fn from_vertex(v: &Vertex3D) -> Self {
        Self {
            position: v.position,
            normal: v.normal,
            color: v.color,
            uv: v.uv,
            tangent: v.tangent,
        }
    }

    /// Locations 0, 1, 2, 3 and 6, as in `Vertex3D::desc()`.
    pub fn static_desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 40,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32x4,
            },
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

/// Bytes of the shared zero buffer: one `vec4` of joints and one of weights.
pub(crate) const SKIN_ZERO_BYTES: u64 = 32;

/// Locations 4 (joints) and 5 (weights) from the zero buffer, the same 32
/// bytes for every vertex.
pub(crate) fn skin_zero_desc() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 4,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];
    wgpu::VertexBufferLayout {
        array_stride: 0,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }
}

/// The shared zero joints and weights every static draw binds.
pub(crate) fn create_skin_zero_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("static_vertex_skin_zero"),
        contents: &[0u8; SKIN_ZERO_BYTES as usize],
        usage: wgpu::BufferUsages::VERTEX,
    })
}

/// Set once a renderer is made on a backend that cannot use the 64-byte format.
static STATIC_LAYOUT_OFF: AtomicBool = AtomicBool::new(false);

/// Called when a renderer is made: OpenGL keeps every mesh at 96 bytes.
pub(crate) fn configure_for_backend(backend: wgpu::Backend) {
    if backend == wgpu::Backend::Gl {
        STATIC_LAYOUT_OFF.store(true, Ordering::Relaxed);
    }
}

fn static_layout_enabled() -> bool {
    !STATIC_LAYOUT_OFF.load(Ordering::Relaxed)
}

/// True when a vertex's joints and weights are all exactly +0.0: the zero
/// stream then gives a shader exactly what the vertex holds.
fn skin_is_zero(joints: [f32; 4], weights: [f32; 4]) -> bool {
    joints
        .iter()
        .chain(weights.iter())
        .all(|v| v.to_bits() == 0)
}

/// Which of the two formats a mesh's vertex buffer holds.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum VertexLayout {
    /// `Vertex3D`, 96 bytes: skinned meshes, immediate-mode geometry, OpenGL.
    #[default]
    Full96,
    /// `StaticVertex3D`, 64 bytes, joints and weights from the zero stream.
    Static64,
}

impl VertexLayout {
    pub const fn stride(self) -> u64 {
        match self {
            VertexLayout::Full96 => 96,
            VertexLayout::Static64 => 64,
        }
    }

    /// 32-bit words a vertex takes (storage readers of vertex buffers).
    pub const fn words(self) -> u32 {
        (self.stride() / 4) as u32
    }

    /// The layout of the mesh's own vertex stream (slot 0).
    pub fn primary_desc(self) -> wgpu::VertexBufferLayout<'static> {
        match self {
            VertexLayout::Full96 => Vertex3D::desc(),
            VertexLayout::Static64 => StaticVertex3D::static_desc(),
        }
    }

    /// The format these vertices go to the GPU in: 64 bytes when every joint
    /// and weight is exactly zero, else 96.
    pub fn for_vertices(vertices: &[Vertex3D]) -> Self {
        if !static_layout_enabled() || !vertices.iter().all(|v| skin_is_zero(v.joints, v.weights)) {
            VertexLayout::Full96
        } else {
            VertexLayout::Static64
        }
    }

    /// As `for_vertices`, for a shared model primitive's compact store.
    pub fn for_store(store: &crate::models::VertexStore) -> Self {
        if !static_layout_enabled()
            || !(0..store.len()).all(|i| skin_is_zero(store.joint(i), store.weight(i)))
        {
            VertexLayout::Full96
        } else {
            VertexLayout::Static64
        }
    }

    /// The format of the shared static-geometry arena, which holds only
    /// unskinned meshes.
    pub fn arena() -> Self {
        if static_layout_enabled() {
            VertexLayout::Static64
        } else {
            VertexLayout::Full96
        }
    }

    /// The bytes of `vertices` in this format.
    pub fn encode(self, vertices: &[Vertex3D]) -> std::borrow::Cow<'_, [u8]> {
        match self {
            VertexLayout::Full96 => std::borrow::Cow::Borrowed(bytemuck::cast_slice(vertices)),
            VertexLayout::Static64 => {
                let mut bytes = Vec::with_capacity(vertices.len() * 64);
                for v in vertices {
                    bytes.extend_from_slice(bytemuck::bytes_of(&StaticVertex3D::from_vertex(v)));
                }
                std::borrow::Cow::Owned(bytes)
            }
        }
    }
}

/// The vertex buffers of a pipeline drawing this layout, from the 96-byte
/// list `full` (`Vertex3D::desc()` first, then the pipeline's other
/// streams): the primary stream swapped and the zero stream appended.
pub(crate) fn layout_buffers<'a>(
    layout: VertexLayout,
    full: &[wgpu::VertexBufferLayout<'a>],
) -> Vec<wgpu::VertexBufferLayout<'a>> {
    debug_assert_eq!(
        full.first().map(|b| b.array_stride),
        Some(std::mem::size_of::<Vertex3D>() as u64)
    );
    let mut buffers = full.to_vec();
    if layout == VertexLayout::Static64 {
        buffers[0] = StaticVertex3D::static_desc();
        buffers.push(skin_zero_desc());
    }
    buffers
}

/// The slot of the zero stream in a pipeline with `streams` vertex buffers
/// besides the mesh's own (0, or 1 with instances or texture coordinates 1).
pub(crate) const fn zero_slot(streams: u32) -> u32 {
    1 + streams
}

/// Binds the zero joints and weights for a static draw (nothing for a
/// 96-byte one).
pub(crate) fn bind_skin_zero<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    layout: VertexLayout,
    slot: u32,
    zero: &'a wgpu::Buffer,
) {
    if layout == VertexLayout::Static64 {
        pass.set_vertex_buffer(slot, zero.slice(..));
    }
}

/// One of something for each vertex layout: typically a pipeline.
pub struct ByLayout<T> {
    pub(crate) full: T,
    pub(crate) compact: T,
}

impl<T> ByLayout<T> {
    pub(crate) fn new(mut make: impl FnMut(VertexLayout) -> T) -> Self {
        Self {
            full: make(VertexLayout::Full96),
            compact: make(VertexLayout::Static64),
        }
    }

    pub fn get(&self, layout: VertexLayout) -> &T {
        match layout {
            VertexLayout::Full96 => &self.full,
            VertexLayout::Static64 => &self.compact,
        }
    }
}

impl<T: Clone> Clone for ByLayout<T> {
    fn clone(&self) -> Self {
        Self {
            full: self.full.clone(),
            compact: self.compact.clone(),
        }
    }
}

/// A lazy pipeline for each layout of the same descriptor.
pub type MeshPipelines = ByLayout<LazyRenderPipeline>;

impl MeshPipelines {
    /// Both layouts of `desc`, whose vertex buffers are the 96-byte list
    /// (see `layout_buffers`). The 64-byte one is labelled `<label>_static64`.
    pub(crate) fn lazy(device: &wgpu::Device, desc: &wgpu::RenderPipelineDescriptor<'_>) -> Self {
        ByLayout::new(|layout| {
            let buffers = layout_buffers(layout, desc.vertex.buffers);
            let label = match layout {
                VertexLayout::Full96 => desc.label.map(str::to_owned),
                VertexLayout::Static64 => desc.label.map(|l| format!("{l}_static64")),
            };
            let mut variant = desc.clone();
            variant.label = label.as_deref();
            variant.vertex.buffers = &buffers;
            LazyRenderPipeline::new(device, &variant)
        })
    }

    /// Prewarm handles: both pipelines, or only the 96-byte one where the
    /// 64-byte format is off.
    pub(crate) fn warm_handles(&self) -> Vec<WarmHandle> {
        let mut handles = vec![self.full.warm_handle()];
        if static_layout_enabled() {
            handles.push(self.compact.warm_handle());
        }
        handles
    }
}

/// A pipeline for each layout of a descriptor that is created now, in the
/// caller's error scope (a game material's, whose errors the caller reports),
/// for the layout most meshes take (`VertexLayout::arena()`), and on first
/// use for the other. The two differ only in their vertex buffers, so the
/// second validates whenever the first does.
pub struct MeshPipelinePair {
    eager_layout: VertexLayout,
    eager: wgpu::RenderPipeline,
    lazy: LazyRenderPipeline,
}

impl MeshPipelinePair {
    /// `desc`'s vertex buffers are the 96-byte list (see `layout_buffers`).
    pub(crate) fn new(device: &wgpu::Device, desc: &wgpu::RenderPipelineDescriptor<'_>) -> Self {
        let eager_layout = VertexLayout::arena();
        let lazy_layout = match eager_layout {
            VertexLayout::Full96 => VertexLayout::Static64,
            VertexLayout::Static64 => VertexLayout::Full96,
        };
        let variant = |layout: VertexLayout| {
            let buffers = layout_buffers(layout, desc.vertex.buffers);
            let label = match layout {
                VertexLayout::Full96 => desc.label.map(str::to_owned),
                VertexLayout::Static64 => desc.label.map(|l| format!("{l}_static64")),
            };
            (buffers, label)
        };
        let (eager_buffers, eager_label) = variant(eager_layout);
        let mut eager_desc = desc.clone();
        eager_desc.label = eager_label.as_deref();
        eager_desc.vertex.buffers = &eager_buffers;
        let eager = device.create_render_pipeline(&eager_desc);
        let (lazy_buffers, lazy_label) = variant(lazy_layout);
        let mut lazy_desc = desc.clone();
        lazy_desc.label = lazy_label.as_deref();
        lazy_desc.vertex.buffers = &lazy_buffers;
        let lazy = LazyRenderPipeline::new(device, &lazy_desc);
        Self {
            eager_layout,
            eager,
            lazy,
        }
    }

    pub fn get(&self, layout: VertexLayout) -> &wgpu::RenderPipeline {
        if layout == self.eager_layout {
            &self.eager
        } else {
            &self.lazy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(i: usize) -> Vertex3D {
        let f = i as f32;
        Vertex3D {
            position: [f + 0.1, f + 0.2, f + 0.3],
            normal: [0.0, 1.0, -0.0],
            color: [0.25, 0.5, 0.75, 1.0],
            uv: [f * 0.5, -f],
            joints: [0.0; 4],
            weights: [0.0; 4],
            tangent: [1.0, 0.0, 0.0, -1.0],
        }
    }

    #[test]
    fn static_vertex_packs_the_same_bytes_as_the_vertex3d_fields() {
        assert_eq!(std::mem::size_of::<StaticVertex3D>(), 64);
        let vertices: Vec<Vertex3D> = (0..5).map(vertex).collect();
        let full: &[u8] = bytemuck::cast_slice(&vertices);
        let packed = VertexLayout::Static64.encode(&vertices);
        assert_eq!(packed.len(), vertices.len() * 64);
        // Each attribute of the 64-byte layout holds the bytes the same
        // location holds in the 96-byte one.
        let full_desc = Vertex3D::desc();
        let static_desc = StaticVertex3D::static_desc();
        for a in static_desc.attributes {
            let f = full_desc
                .attributes
                .iter()
                .find(|b| b.shader_location == a.shader_location)
                .unwrap();
            assert_eq!(a.format, f.format);
            let size = a.format.size() as usize;
            for i in 0..vertices.len() {
                let s = i * 64 + a.offset as usize;
                let d = i * 96 + f.offset as usize;
                assert_eq!(&packed[s..s + size], &full[d..d + size]);
            }
        }
        // What the 64-byte layout leaves out is exactly joints and weights,
        // and those were zero: the zero stream gives the same bytes.
        let mut locations: Vec<u32> = static_desc
            .attributes
            .iter()
            .chain(skin_zero_desc().attributes)
            .map(|a| a.shader_location)
            .collect();
        locations.sort();
        assert_eq!(locations, vec![0, 1, 2, 3, 4, 5, 6]);
        for v in &vertices {
            let mut joints_weights = bytemuck::bytes_of(&v.joints).to_vec();
            joints_weights.extend_from_slice(bytemuck::bytes_of(&v.weights));
            assert_eq!(joints_weights, vec![0u8; SKIN_ZERO_BYTES as usize]);
        }
        for a in skin_zero_desc().attributes {
            assert!(a.offset + a.format.size() <= SKIN_ZERO_BYTES);
        }
    }

    #[test]
    fn skinned_vertices_keep_96_bytes() {
        let mut vertices: Vec<Vertex3D> = (0..3).map(vertex).collect();
        assert_eq!(VertexLayout::for_vertices(&vertices), VertexLayout::arena());
        vertices[1].weights = [0.5, 0.5, 0.0, 0.0];
        assert_eq!(VertexLayout::for_vertices(&vertices), VertexLayout::Full96);
        assert_eq!(
            VertexLayout::Full96.encode(&vertices).len(),
            vertices.len() * 96
        );
        let store = crate::models::VertexStore::from_vertices(&vertices);
        assert_eq!(VertexLayout::for_store(&store), VertexLayout::Full96);
    }

    #[test]
    fn any_skin_data_a_shader_could_read_keeps_96_bytes() {
        let base: Vec<Vertex3D> = (0..3).map(vertex).collect();
        for (joints, weights) in [
            ([7.0, 0.0, 0.0, 0.0], [0.0; 4]),   // joints, no weights
            ([0.0; 4], [0.005, 0.0, 0.0, 0.0]), // under the shaders' skin test
            ([-0.0, 0.0, 0.0, 0.0], [0.0; 4]),  // negative zero is not +0.0
        ] {
            let mut vertices = base.clone();
            vertices[2].joints = joints;
            vertices[2].weights = weights;
            assert_eq!(VertexLayout::for_vertices(&vertices), VertexLayout::Full96);
            let store = crate::models::VertexStore::from_vertices(&vertices);
            assert_eq!(VertexLayout::for_store(&store), VertexLayout::Full96);
        }
    }

    #[test]
    fn layouts_put_the_zero_stream_after_the_other_streams() {
        let full = [Vertex3D::desc(), super::super::types::secondary_uv_desc()];
        let compact = layout_buffers(VertexLayout::Static64, &full);
        assert_eq!(compact.len(), 3);
        assert_eq!(compact[0].array_stride, 64);
        assert_eq!(compact[1].array_stride, full[1].array_stride);
        assert_eq!(compact[2].array_stride, 0);
        assert_eq!(zero_slot(1), 2);
        assert_eq!(layout_buffers(VertexLayout::Full96, &full).len(), 2);
        assert_eq!(VertexLayout::Static64.words(), 16);
        assert_eq!(VertexLayout::Full96.words(), 24);
    }
}
