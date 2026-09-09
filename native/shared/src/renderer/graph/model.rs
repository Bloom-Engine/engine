//! Declarative render-graph model.
//!
//! This layer intentionally contains no `wgpu::Texture`, `wgpu::Buffer`, or
//! encoder references. A frame plan can therefore be built, validated, cached,
//! and dumped without touching the GPU. Runtime code binds execution closures
//! to the compiled pass identifiers later.

use std::fmt;

/// Stable identifier of a logical graph resource.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceId(pub(crate) u32);

/// Monotonically increasing version of a logical resource.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceVersion(pub(crate) u32);

/// Stable identifier of a declared pass.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PassId(pub(crate) u32);

/// A typed logical texture handle. Writes return a handle with a new version.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct TextureHandle {
    pub(crate) resource: ResourceId,
    pub(crate) version: ResourceVersion,
}

impl TextureHandle {
    pub fn resource(self) -> ResourceId {
        self.resource
    }
    pub fn version(self) -> ResourceVersion {
        self.version
    }
}

/// A typed logical buffer handle. Writes return a handle with a new version.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct BufferHandle {
    pub(crate) resource: ResourceId,
    pub(crate) version: ResourceVersion,
}

impl BufferHandle {
    pub fn resource(self) -> ResourceId {
        self.resource
    }
    pub fn version(self) -> ResourceVersion {
        self.version
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum AnyHandle {
    Texture(TextureHandle),
    Buffer(BufferHandle),
}

impl AnyHandle {
    pub(crate) fn resource(self) -> ResourceId {
        match self {
            Self::Texture(handle) => handle.resource,
            Self::Buffer(handle) => handle.resource,
        }
    }

    pub(crate) fn version(self) -> ResourceVersion {
        match self {
            Self::Texture(handle) => handle.version,
            Self::Buffer(handle) => handle.version,
        }
    }
}

/// Graph-owned texture usage contract. This is deliberately independent of
/// backend barrier APIs; `wgpu` still performs the actual transitions.
#[derive(Copy, Clone, Eq, Hash, PartialEq)]
pub struct TextureUsage(pub(crate) u32);

impl TextureUsage {
    pub const NONE: Self = Self(0);
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const SAMPLED: Self = Self(1 << 2);
    pub const STORAGE_READ: Self = Self(1 << 3);
    pub const STORAGE_WRITE: Self = Self(1 << 4);
    pub const COLOR_ATTACHMENT: Self = Self(1 << 5);
    pub const DEPTH_ATTACHMENT_READ: Self = Self(1 << 6);
    pub const DEPTH_ATTACHMENT_WRITE: Self = Self(1 << 7);
    pub const PRESENT: Self = Self(1 << 8);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Debug for TextureUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextureUsage({:#x})", self.0)
    }
}

/// Graph-owned buffer usage contract.
#[derive(Copy, Clone, Eq, Hash, PartialEq)]
pub struct BufferUsage(pub(crate) u32);

impl BufferUsage {
    pub const NONE: Self = Self(0);
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const UNIFORM: Self = Self(1 << 2);
    pub const STORAGE_READ: Self = Self(1 << 3);
    pub const STORAGE_WRITE: Self = Self(1 << 4);
    pub const VERTEX: Self = Self(1 << 5);
    pub const INDEX: Self = Self(1 << 6);
    pub const INDIRECT: Self = Self(1 << 7);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Debug for BufferUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BufferUsage({:#x})", self.0)
    }
}

/// Usage of a resource at one pass boundary.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Usage {
    Texture(TextureUsage),
    Buffer(BufferUsage),
}

impl Usage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Texture(TextureUsage::COPY_SRC) => "copy-src",
            Self::Texture(TextureUsage::COPY_DST) => "copy-dst",
            Self::Texture(TextureUsage::SAMPLED) => "sampled",
            Self::Texture(TextureUsage::STORAGE_READ) => "storage-read",
            Self::Texture(TextureUsage::STORAGE_WRITE) => "storage-write",
            Self::Texture(TextureUsage::COLOR_ATTACHMENT) => "color-attachment",
            Self::Texture(TextureUsage::DEPTH_ATTACHMENT_READ) => "depth-read",
            Self::Texture(TextureUsage::DEPTH_ATTACHMENT_WRITE) => "depth-write",
            Self::Texture(TextureUsage::PRESENT) => "present",
            Self::Texture(_) => "texture-mixed",
            Self::Buffer(BufferUsage::COPY_SRC) => "copy-src",
            Self::Buffer(BufferUsage::COPY_DST) => "copy-dst",
            Self::Buffer(BufferUsage::UNIFORM) => "uniform",
            Self::Buffer(BufferUsage::STORAGE_READ) => "storage-read",
            Self::Buffer(BufferUsage::STORAGE_WRITE) => "storage-write",
            Self::Buffer(BufferUsage::VERTEX) => "vertex",
            Self::Buffer(BufferUsage::INDEX) => "index",
            Self::Buffer(BufferUsage::INDIRECT) => "indirect",
            Self::Buffer(_) => "buffer-mixed",
        }
    }
}

/// Texture extent resolved when physical allocations are prepared.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Extent {
    /// Exact dimensions, independent of the presentation target.
    Fixed {
        width: u32,
        height: u32,
        layers: u32,
    },
    /// Rational scale of the renderer's internal render extent.
    RenderRelative {
        numerator: u16,
        denominator: u16,
        layers: u32,
    },
    /// Rational scale of the final presentation extent.
    OutputRelative {
        numerator: u16,
        denominator: u16,
        layers: u32,
    },
}

impl Extent {
    pub fn resolve(self, render: (u32, u32), output: (u32, u32)) -> (u32, u32, u32) {
        match self {
            Self::Fixed {
                width,
                height,
                layers,
            } => (width.max(1), height.max(1), layers.max(1)),
            Self::RenderRelative {
                numerator,
                denominator,
                layers,
            } => {
                let denominator = u32::from(denominator.max(1));
                (
                    (render.0.saturating_mul(u32::from(numerator)) / denominator).max(1),
                    (render.1.saturating_mul(u32::from(numerator)) / denominator).max(1),
                    layers.max(1),
                )
            }
            Self::OutputRelative {
                numerator,
                denominator,
                layers,
            } => {
                let denominator = u32::from(denominator.max(1));
                (
                    (output.0.saturating_mul(u32::from(numerator)) / denominator).max(1),
                    (output.1.saturating_mul(u32::from(numerator)) / denominator).max(1),
                    layers.max(1),
                )
            }
        }
    }
}

/// Physical texture dimensionality.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

/// Clear value stored as IEEE-754 bits so descriptors remain `Eq`.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct ClearColor([u32; 4]);

impl ClearColor {
    pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self([r.to_bits(), g.to_bits(), b.to_bits(), a.to_bits()])
    }

    pub fn channels(self) -> [f32; 4] {
        self.0.map(f32::from_bits)
    }
}

/// Load policy is part of the declaration contract, not the alias key.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum LoadPolicy {
    Discard,
    Load,
    ClearColor(ClearColor),
    ClearDepth(u32),
}

impl LoadPolicy {
    pub fn clear_depth(value: f32) -> Self {
        Self::ClearDepth(value.to_bits())
    }
}

/// Conservative aliasing partition. Resources in different classes never
/// share physical storage even if every other descriptor field matches.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum AliasClass {
    Color,
    Depth,
    Storage,
    Readback,
    Never,
    Custom(u16),
}

/// Complete logical texture descriptor used by the compiler.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TextureDesc {
    pub format: wgpu::TextureFormat,
    pub extent: Extent,
    pub dimension: TextureDimension,
    pub mip_count: u32,
    pub sample_count: u32,
    pub allowed_usage: TextureUsage,
    pub load: LoadPolicy,
    pub alias_class: AliasClass,
}

impl TextureDesc {
    pub fn color(format: wgpu::TextureFormat, extent: Extent, allowed_usage: TextureUsage) -> Self {
        Self {
            format,
            extent,
            dimension: TextureDimension::D2,
            mip_count: 1,
            sample_count: 1,
            allowed_usage,
            load: LoadPolicy::Discard,
            alias_class: AliasClass::Color,
        }
    }
}

/// Complete logical buffer descriptor used by the compiler.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BufferDesc {
    pub size: u64,
    pub allowed_usage: BufferUsage,
    pub alias_class: AliasClass,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ResourceDesc {
    Texture(TextureDesc),
    Buffer(BufferDesc),
}

impl ResourceDesc {
    pub(crate) fn permits(&self, usage: Usage) -> bool {
        match (self, usage) {
            (Self::Texture(desc), Usage::Texture(usage)) => desc.allowed_usage.contains(usage),
            (Self::Buffer(desc), Usage::Buffer(usage)) => desc.allowed_usage.contains(usage),
            _ => false,
        }
    }
}

/// Ownership of an imported resource at the frame boundary.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Ownership {
    Graph,
    External,
}

/// Lifetime class of a logical resource.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum ResourceOrigin {
    Transient,
    Persistent {
        initial_usage: Usage,
        final_usage: Usage,
        ownership: Ownership,
    },
    External {
        initial_usage: Usage,
        final_usage: Usage,
        ownership: Ownership,
    },
}

impl ResourceOrigin {
    pub fn is_transient(self) -> bool {
        matches!(self, Self::Transient)
    }

    pub(crate) fn initial_usage(self) -> Option<Usage> {
        match self {
            Self::Transient => None,
            Self::Persistent { initial_usage, .. } | Self::External { initial_usage, .. } => {
                Some(initial_usage)
            }
        }
    }

    pub(crate) fn final_usage(self) -> Option<Usage> {
        match self {
            Self::Transient => None,
            Self::Persistent { final_usage, .. } | Self::External { final_usage, .. } => {
                Some(final_usage)
            }
        }
    }
}

/// Queue declaration retained in diagnostics even though all currently
/// supported backends execute a single serial graphics queue.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum QueueClass {
    Graphics,
    ComputeCapable,
    CopyCapable,
}

/// Observable pass behavior not represented by resource dependencies.
#[derive(Copy, Clone, Eq, Hash, PartialEq)]
pub struct SideEffects(pub(crate) u32);

impl SideEffects {
    pub const NONE: Self = Self(0);
    pub const PRESENT: Self = Self(1 << 0);
    pub const READBACK: Self = Self(1 << 1);
    pub const TIMESTAMP: Self = Self(1 << 2);
    pub const EXTERNAL_STATE: Self = Self(1 << 3);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl fmt::Debug for SideEffects {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SideEffects({:#x})", self.0)
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum AccessKind {
    Read,
    Write,
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct Access {
    pub handle: AnyHandle,
    pub usage: Usage,
    pub kind: AccessKind,
}

#[derive(Clone, Debug)]
pub(crate) struct PassDeclaration {
    pub id: PassId,
    pub name: String,
    pub queue: QueueClass,
    pub side_effects: SideEffects,
    pub accesses: Vec<Access>,
    pub after: Vec<PassId>,
    pub before: Vec<PassId>,
}

#[derive(Clone, Debug)]
pub(crate) struct VersionDeclaration {
    pub producer: Option<PassId>,
    pub parent: Option<ResourceVersion>,
    pub initialized: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ResourceDeclaration {
    pub id: ResourceId,
    pub name: String,
    pub desc: ResourceDesc,
    pub origin: ResourceOrigin,
    pub versions: Vec<VersionDeclaration>,
}

/// Pure declaration builder. Errors caused by graph relationships are
/// intentionally deferred to compilation so callers get deterministic,
/// aggregate validation rather than order-dependent panics.
#[derive(Clone, Debug)]
pub struct GraphBuilder {
    pub(crate) label: String,
    pub(crate) passes: Vec<PassDeclaration>,
    pub(crate) resources: Vec<ResourceDeclaration>,
}

impl GraphBuilder {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            passes: Vec::new(),
            resources: Vec::new(),
        }
    }

    pub fn add_pass(&mut self, name: impl Into<String>) -> PassId {
        let id = PassId(self.passes.len() as u32);
        self.passes.push(PassDeclaration {
            id,
            name: name.into(),
            queue: QueueClass::Graphics,
            side_effects: SideEffects::NONE,
            accesses: Vec::new(),
            after: Vec::new(),
            before: Vec::new(),
        });
        id
    }

    pub fn add_optional_pass(&mut self, enabled: bool, name: impl Into<String>) -> Option<PassId> {
        enabled.then(|| self.add_pass(name))
    }

    pub fn set_queue(&mut self, pass: PassId, queue: QueueClass) {
        self.pass_mut(pass).queue = queue;
    }

    pub fn set_side_effects(&mut self, pass: PassId, side_effects: SideEffects) {
        self.pass_mut(pass).side_effects = side_effects;
    }

    pub fn after(&mut self, pass: PassId, predecessor: PassId) {
        self.pass_mut(pass).after.push(predecessor);
    }

    pub fn before(&mut self, pass: PassId, successor: PassId) {
        self.pass_mut(pass).before.push(successor);
    }

    pub fn create_texture(&mut self, name: impl Into<String>, desc: TextureDesc) -> TextureHandle {
        let resource =
            self.push_resource(name, ResourceDesc::Texture(desc), ResourceOrigin::Transient);
        TextureHandle {
            resource,
            version: ResourceVersion(0),
        }
    }

    pub fn create_buffer(&mut self, name: impl Into<String>, desc: BufferDesc) -> BufferHandle {
        let resource =
            self.push_resource(name, ResourceDesc::Buffer(desc), ResourceOrigin::Transient);
        BufferHandle {
            resource,
            version: ResourceVersion(0),
        }
    }

    pub fn import_texture(
        &mut self,
        name: impl Into<String>,
        desc: TextureDesc,
        origin: ResourceOrigin,
    ) -> TextureHandle {
        assert!(
            !origin.is_transient(),
            "imports must be persistent or external"
        );
        let resource = self.push_resource(name, ResourceDesc::Texture(desc), origin);
        self.resources[resource.0 as usize].versions[0].initialized = true;
        TextureHandle {
            resource,
            version: ResourceVersion(0),
        }
    }

    pub fn import_buffer(
        &mut self,
        name: impl Into<String>,
        desc: BufferDesc,
        origin: ResourceOrigin,
    ) -> BufferHandle {
        assert!(
            !origin.is_transient(),
            "imports must be persistent or external"
        );
        let resource = self.push_resource(name, ResourceDesc::Buffer(desc), origin);
        self.resources[resource.0 as usize].versions[0].initialized = true;
        BufferHandle {
            resource,
            version: ResourceVersion(0),
        }
    }

    pub fn read_texture(&mut self, pass: PassId, handle: TextureHandle, usage: TextureUsage) {
        self.push_access(
            pass,
            AnyHandle::Texture(handle),
            Usage::Texture(usage),
            AccessKind::Read,
        );
    }

    pub fn read_buffer(&mut self, pass: PassId, handle: BufferHandle, usage: BufferUsage) {
        self.push_access(
            pass,
            AnyHandle::Buffer(handle),
            Usage::Buffer(usage),
            AccessKind::Read,
        );
    }

    pub fn write_texture(
        &mut self,
        pass: PassId,
        previous: TextureHandle,
        usage: TextureUsage,
    ) -> TextureHandle {
        let version = self.push_version(previous.resource, pass, previous.version);
        let handle = TextureHandle {
            resource: previous.resource,
            version,
        };
        self.push_access(
            pass,
            AnyHandle::Texture(handle),
            Usage::Texture(usage),
            AccessKind::Write,
        );
        handle
    }

    pub fn write_buffer(
        &mut self,
        pass: PassId,
        previous: BufferHandle,
        usage: BufferUsage,
    ) -> BufferHandle {
        let version = self.push_version(previous.resource, pass, previous.version);
        let handle = BufferHandle {
            resource: previous.resource,
            version,
        };
        self.push_access(
            pass,
            AnyHandle::Buffer(handle),
            Usage::Buffer(usage),
            AccessKind::Write,
        );
        handle
    }

    pub fn read_write_texture(
        &mut self,
        pass: PassId,
        previous: TextureHandle,
        read_usage: TextureUsage,
        write_usage: TextureUsage,
    ) -> TextureHandle {
        self.read_texture(pass, previous, read_usage);
        self.write_texture(pass, previous, write_usage)
    }

    pub fn read_write_buffer(
        &mut self,
        pass: PassId,
        previous: BufferHandle,
        read_usage: BufferUsage,
        write_usage: BufferUsage,
    ) -> BufferHandle {
        self.read_buffer(pass, previous, read_usage);
        self.write_buffer(pass, previous, write_usage)
    }

    pub fn pass_name(&self, pass: PassId) -> Option<&str> {
        self.passes
            .get(pass.0 as usize)
            .map(|pass| pass.name.as_str())
    }

    fn push_resource(
        &mut self,
        name: impl Into<String>,
        desc: ResourceDesc,
        origin: ResourceOrigin,
    ) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceDeclaration {
            id,
            name: name.into(),
            desc,
            origin,
            versions: vec![VersionDeclaration {
                producer: None,
                parent: None,
                initialized: false,
            }],
        });
        id
    }

    fn push_version(
        &mut self,
        resource: ResourceId,
        producer: PassId,
        parent: ResourceVersion,
    ) -> ResourceVersion {
        let resource = self.resource_mut(resource);
        let version = ResourceVersion(resource.versions.len() as u32);
        resource.versions.push(VersionDeclaration {
            producer: Some(producer),
            parent: Some(parent),
            initialized: true,
        });
        version
    }

    fn push_access(&mut self, pass: PassId, handle: AnyHandle, usage: Usage, kind: AccessKind) {
        self.pass_mut(pass).accesses.push(Access {
            handle,
            usage,
            kind,
        });
    }

    fn pass_mut(&mut self, pass: PassId) -> &mut PassDeclaration {
        self.passes
            .get_mut(pass.0 as usize)
            .unwrap_or_else(|| panic!("unknown graph pass {}", pass.0))
    }

    fn resource_mut(&mut self, resource: ResourceId) -> &mut ResourceDeclaration {
        self.resources
            .get_mut(resource.0 as usize)
            .unwrap_or_else(|| panic!("unknown graph resource {}", resource.0))
    }
}
