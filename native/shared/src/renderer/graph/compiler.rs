//! Validation, deterministic scheduling, lifetime analysis, and conservative
//! physical allocation planning for [`GraphBuilder`](super::GraphBuilder).

use super::model::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CompileOptions {
    /// When false, every transient receives a unique physical allocation.
    /// This is the migration/parity mode.
    pub aliasing: bool,
}

impl CompileOptions {
    pub const NO_ALIASING: Self = Self { aliasing: false };
    pub const CONSERVATIVE_ALIASING: Self = Self { aliasing: true };
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    DuplicatePassName(String),
    DuplicateResourceName(String),
    UnknownPass(PassId),
    UnknownResource(ResourceId),
    UnknownVersion {
        resource: String,
        version: ResourceVersion,
    },
    ResourceTypeMismatch {
        resource: String,
    },
    MissingProducer {
        pass: String,
        resource: String,
        version: ResourceVersion,
    },
    MultipleWriters {
        resource: String,
        version: ResourceVersion,
        writers: Vec<String>,
    },
    UsageNotDeclared {
        pass: String,
        resource: String,
        usage: Usage,
    },
    SelfReadOfWrite {
        pass: String,
        resource: String,
        version: ResourceVersion,
    },
    Cycle(Vec<String>),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePassName(name) => write!(f, "duplicate pass name '{name}'"),
            Self::DuplicateResourceName(name) => write!(f, "duplicate resource name '{name}'"),
            Self::UnknownPass(pass) => write!(f, "unknown pass {}", pass.0),
            Self::UnknownResource(resource) => write!(f, "unknown resource {}", resource.0),
            Self::UnknownVersion { resource, version } => {
                write!(f, "resource '{resource}' has no version {}", version.0)
            }
            Self::ResourceTypeMismatch { resource } => {
                write!(f, "typed handle does not match resource '{resource}'")
            }
            Self::MissingProducer {
                pass,
                resource,
                version,
            } => write!(
                f,
                "pass '{pass}' reads '{resource}' v{} before any producer",
                version.0
            ),
            Self::MultipleWriters {
                resource,
                version,
                writers,
            } => write!(
                f,
                "'{resource}' v{} has divergent writers: {}",
                version.0,
                writers.join(", ")
            ),
            Self::UsageNotDeclared {
                pass,
                resource,
                usage,
            } => write!(
                f,
                "pass '{pass}' uses '{resource}' as '{}' without descriptor permission",
                usage.name()
            ),
            Self::SelfReadOfWrite {
                pass,
                resource,
                version,
            } => write!(
                f,
                "pass '{pass}' reads its own produced '{resource}' v{}",
                version.0
            ),
            Self::Cycle(names) => write!(f, "render graph cycle: {}", names.join(" -> ")),
        }
    }
}

impl std::error::Error for CompileError {}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum CompiledAccessKind {
    Read,
    Write,
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct CompiledAccess {
    pub resource: ResourceId,
    pub version: ResourceVersion,
    pub usage: Usage,
    pub kind: CompiledAccessKind,
}

#[derive(Clone, Debug)]
pub struct CompiledPass {
    pub id: PassId,
    pub name: String,
    pub queue: QueueClass,
    pub side_effects: SideEffects,
    pub dependencies: Vec<PassId>,
    pub accesses: Vec<CompiledAccess>,
}

#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PhysicalAllocationId(pub u32);

#[derive(Clone, Debug)]
pub struct CompiledResource {
    pub id: ResourceId,
    pub name: String,
    pub desc: ResourceDesc,
    pub origin: ResourceOrigin,
    /// Inclusive pass positions in the compiled schedule.
    pub first_use: Option<usize>,
    pub last_use: Option<usize>,
    pub physical: Option<PhysicalAllocationId>,
}

#[derive(Clone, Debug)]
pub struct PhysicalAllocation {
    pub id: PhysicalAllocationId,
    pub desc: ResourceDesc,
    pub resources: Vec<ResourceId>,
    pub first_use: usize,
    pub last_use: usize,
}

#[derive(Clone, Debug)]
pub struct UsageTransition {
    pub resource: ResourceId,
    pub pass: Option<PassId>,
    pub before: Option<Usage>,
    pub after: Usage,
    pub from_queue: Option<QueueClass>,
    pub to_queue: QueueClass,
}

#[derive(Clone, Debug)]
pub struct CompiledGraph {
    pub label: String,
    /// Passes in deterministic execution order.
    pub passes: Vec<CompiledPass>,
    pub resources: Vec<CompiledResource>,
    pub allocations: Vec<PhysicalAllocation>,
    pub transitions: Vec<UsageTransition>,
    pub aliasing_enabled: bool,
    pub plan_id: u64,
    pass_by_name: HashMap<String, PassId>,
    pass_position: HashMap<PassId, usize>,
    resource_by_name: HashMap<String, ResourceId>,
}

impl CompiledGraph {
    pub fn pass(&self, name: &str) -> Option<&CompiledPass> {
        let id = self.pass_by_name.get(name)?;
        self.pass_by_id(*id)
    }

    pub fn pass_by_id(&self, id: PassId) -> Option<&CompiledPass> {
        let position = *self.pass_position.get(&id)?;
        self.passes.get(position)
    }

    pub fn pass_position(&self, id: PassId) -> Option<usize> {
        self.pass_position.get(&id).copied()
    }

    pub fn resource(&self, name: &str) -> Option<&CompiledResource> {
        let id = *self.resource_by_name.get(name)?;
        self.resources.get(id.0 as usize)
    }

    pub fn resource_by_id(&self, id: ResourceId) -> Option<&CompiledResource> {
        self.resources.get(id.0 as usize)
    }

    pub fn transient_bytes(&self, render_extent: (u32, u32), output_extent: (u32, u32)) -> u64 {
        self.allocations
            .iter()
            .map(|allocation| descriptor_size_bytes(&allocation.desc, render_extent, output_extent))
            .sum()
    }

    pub fn unaliased_transient_bytes(
        &self,
        render_extent: (u32, u32),
        output_extent: (u32, u32),
    ) -> u64 {
        self.resources
            .iter()
            .filter(|resource| resource.origin.is_transient() && resource.first_use.is_some())
            .map(|resource| descriptor_size_bytes(&resource.desc, render_extent, output_extent))
            .sum()
    }

    pub fn alias_savings_percent(
        &self,
        render_extent: (u32, u32),
        output_extent: (u32, u32),
    ) -> f64 {
        let unaliased = self.unaliased_transient_bytes(render_extent, output_extent);
        if unaliased == 0 {
            return 0.0;
        }
        let allocated = self.transient_bytes(render_extent, output_extent);
        100.0 * (unaliased.saturating_sub(allocated) as f64) / unaliased as f64
    }
}

impl GraphBuilder {
    pub fn compile(self, options: CompileOptions) -> Result<CompiledGraph, CompileError> {
        compile(self, options)
    }
}

fn compile(builder: GraphBuilder, options: CompileOptions) -> Result<CompiledGraph, CompileError> {
    validate_unique_names(&builder)?;

    let pass_count = builder.passes.len();
    let mut predecessors: Vec<HashSet<usize>> = vec![HashSet::new(); pass_count];
    let mut readers: Vec<Vec<Vec<PassId>>> = builder
        .resources
        .iter()
        .map(|resource| vec![Vec::new(); resource.versions.len()])
        .collect();

    // Explicit ordering constraints.
    for pass in &builder.passes {
        let pass_index = pass.id.0 as usize;
        if pass_index >= pass_count {
            return Err(CompileError::UnknownPass(pass.id));
        }
        for predecessor in &pass.after {
            let predecessor_index = predecessor.0 as usize;
            if predecessor_index >= pass_count {
                return Err(CompileError::UnknownPass(*predecessor));
            }
            if predecessor_index != pass_index {
                predecessors[pass_index].insert(predecessor_index);
            }
        }
        for successor in &pass.before {
            let successor_index = successor.0 as usize;
            if successor_index >= pass_count {
                return Err(CompileError::UnknownPass(*successor));
            }
            if successor_index != pass_index {
                predecessors[successor_index].insert(pass_index);
            }
        }
    }

    // Validate every access and add producer -> reader dependencies.
    for pass in &builder.passes {
        let pass_index = pass.id.0 as usize;
        let mut produced_earlier_in_pass = HashSet::new();
        for access in &pass.accesses {
            let resource_id = access.handle.resource();
            let resource = builder
                .resources
                .get(resource_id.0 as usize)
                .ok_or(CompileError::UnknownResource(resource_id))?;
            validate_handle_type(resource, access.handle)?;
            let version = access.handle.version();
            let declaration = resource.versions.get(version.0 as usize).ok_or_else(|| {
                CompileError::UnknownVersion {
                    resource: resource.name.clone(),
                    version,
                }
            })?;
            if !resource.desc.permits(access.usage) {
                return Err(CompileError::UsageNotDeclared {
                    pass: pass.name.clone(),
                    resource: resource.name.clone(),
                    usage: access.usage,
                });
            }
            match access.kind {
                AccessKind::Read => {
                    if let Some(producer) = declaration.producer {
                        if producer == pass.id {
                            // A pass closure is serial. It may copy/write a
                            // resource and consume that produced version later
                            // in the same closure (for example the translucent
                            // scene snapshots), but it may not read the version
                            // before its declared write.
                            if !produced_earlier_in_pass.contains(&access.handle) {
                                return Err(CompileError::SelfReadOfWrite {
                                    pass: pass.name.clone(),
                                    resource: resource.name.clone(),
                                    version,
                                });
                            }
                        } else {
                            predecessors[pass_index].insert(producer.0 as usize);
                        }
                    } else if !declaration.initialized {
                        return Err(CompileError::MissingProducer {
                            pass: pass.name.clone(),
                            resource: resource.name.clone(),
                            version,
                        });
                    }
                    readers[resource_id.0 as usize][version.0 as usize].push(pass.id);
                }
                AccessKind::Write => {
                    if declaration.producer != Some(pass.id) {
                        let writers = declaration
                            .producer
                            .into_iter()
                            .chain(std::iter::once(pass.id))
                            .filter_map(|id| builder.passes.get(id.0 as usize))
                            .map(|writer| writer.name.clone())
                            .collect();
                        return Err(CompileError::MultipleWriters {
                            resource: resource.name.clone(),
                            version,
                            writers,
                        });
                    }
                    produced_earlier_in_pass.insert(access.handle);
                }
            }
        }
    }

    // A write from one source version may have only one successor. This catches
    // the classic "two passes both overwrite v0" error while still making every
    // successful write a distinct version.
    for resource in &builder.resources {
        let mut children: HashMap<ResourceVersion, Vec<&VersionDeclaration>> = HashMap::new();
        for version in resource.versions.iter().skip(1) {
            if let Some(parent) = version.parent {
                children.entry(parent).or_default().push(version);
            }
        }
        for (parent, versions) in children {
            if versions.len() > 1 {
                let mut writers: Vec<String> = versions
                    .iter()
                    .filter_map(|version| version.producer)
                    .filter_map(|pass| builder.passes.get(pass.0 as usize))
                    .map(|pass| pass.name.clone())
                    .collect();
                writers.sort();
                writers.dedup();
                return Err(CompileError::MultipleWriters {
                    resource: resource.name.clone(),
                    version: parent,
                    writers,
                });
            }
        }
    }

    // In-place logical versions share physical storage. Preserve WAW and WAR
    // hazards between a parent version and its successor.
    for resource in &builder.resources {
        let resource_index = resource.id.0 as usize;
        for version in resource.versions.iter().skip(1) {
            let Some(writer) = version.producer else {
                continue;
            };
            let writer_index = writer.0 as usize;
            let Some(parent) = version.parent else {
                continue;
            };
            let parent_declaration = resource.versions.get(parent.0 as usize).ok_or_else(|| {
                CompileError::UnknownVersion {
                    resource: resource.name.clone(),
                    version: parent,
                }
            })?;
            if let Some(previous_writer) = parent_declaration.producer {
                if previous_writer != writer {
                    predecessors[writer_index].insert(previous_writer.0 as usize);
                }
            }
            for reader in &readers[resource_index][parent.0 as usize] {
                if *reader != writer {
                    predecessors[writer_index].insert(reader.0 as usize);
                }
            }
        }
    }

    let order = deterministic_topological_sort(&builder, &predecessors)?;
    let schedule_position: HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(position, pass_index)| (*pass_index, position))
        .collect();

    let mut compiled_resources: Vec<CompiledResource> = builder
        .resources
        .iter()
        .map(|resource| CompiledResource {
            id: resource.id,
            name: resource.name.clone(),
            desc: resource.desc.clone(),
            origin: resource.origin,
            first_use: None,
            last_use: None,
            physical: None,
        })
        .collect();

    for pass in &builder.passes {
        let position = schedule_position[&(pass.id.0 as usize)];
        for access in &pass.accesses {
            let resource = &mut compiled_resources[access.handle.resource().0 as usize];
            resource.first_use = Some(resource.first_use.map_or(position, |old| old.min(position)));
            resource.last_use = Some(resource.last_use.map_or(position, |old| old.max(position)));
        }
    }

    let allocations = plan_allocations(&mut compiled_resources, options.aliasing);
    let transitions = plan_transitions(&builder, &order);

    let mut compiled_passes = Vec::with_capacity(pass_count);
    let mut pass_position = HashMap::with_capacity(pass_count);
    for (position, pass_index) in order.iter().copied().enumerate() {
        let pass = &builder.passes[pass_index];
        let mut dependencies: Vec<PassId> = predecessors[pass_index]
            .iter()
            .copied()
            .map(|index| builder.passes[index].id)
            .collect();
        dependencies.sort_by_key(|dependency| schedule_position[&(dependency.0 as usize)]);
        let accesses = pass
            .accesses
            .iter()
            .map(|access| CompiledAccess {
                resource: access.handle.resource(),
                version: access.handle.version(),
                usage: access.usage,
                kind: match access.kind {
                    AccessKind::Read => CompiledAccessKind::Read,
                    AccessKind::Write => CompiledAccessKind::Write,
                },
            })
            .collect();
        compiled_passes.push(CompiledPass {
            id: pass.id,
            name: pass.name.clone(),
            queue: pass.queue,
            side_effects: pass.side_effects,
            dependencies,
            accesses,
        });
        pass_position.insert(pass.id, position);
    }

    let pass_by_name = compiled_passes
        .iter()
        .map(|pass| (pass.name.clone(), pass.id))
        .collect();
    let resource_by_name = compiled_resources
        .iter()
        .map(|resource| (resource.name.clone(), resource.id))
        .collect();
    let plan_id = stable_plan_id(
        &builder.label,
        &compiled_passes,
        &compiled_resources,
        options.aliasing,
    );

    Ok(CompiledGraph {
        label: builder.label,
        passes: compiled_passes,
        resources: compiled_resources,
        allocations,
        transitions,
        aliasing_enabled: options.aliasing,
        plan_id,
        pass_by_name,
        pass_position,
        resource_by_name,
    })
}

fn validate_unique_names(builder: &GraphBuilder) -> Result<(), CompileError> {
    let mut pass_names = HashSet::new();
    for pass in &builder.passes {
        if !pass_names.insert(pass.name.as_str()) {
            return Err(CompileError::DuplicatePassName(pass.name.clone()));
        }
    }
    let mut resource_names = HashSet::new();
    for resource in &builder.resources {
        if !resource_names.insert(resource.name.as_str()) {
            return Err(CompileError::DuplicateResourceName(resource.name.clone()));
        }
    }
    Ok(())
}

fn validate_handle_type(
    resource: &ResourceDeclaration,
    handle: AnyHandle,
) -> Result<(), CompileError> {
    let matches = matches!(
        (&resource.desc, handle),
        (ResourceDesc::Texture(_), AnyHandle::Texture(_))
            | (ResourceDesc::Buffer(_), AnyHandle::Buffer(_))
    );
    if matches {
        Ok(())
    } else {
        Err(CompileError::ResourceTypeMismatch {
            resource: resource.name.clone(),
        })
    }
}

fn deterministic_topological_sort(
    builder: &GraphBuilder,
    predecessors: &[HashSet<usize>],
) -> Result<Vec<usize>, CompileError> {
    let pass_count = builder.passes.len();
    let mut successors = vec![Vec::new(); pass_count];
    let mut in_degree: Vec<usize> = predecessors.iter().map(HashSet::len).collect();
    for (successor, pass_predecessors) in predecessors.iter().enumerate() {
        for predecessor in pass_predecessors {
            successors[*predecessor].push(successor);
        }
    }
    for list in &mut successors {
        list.sort_unstable();
        list.dedup();
    }

    // PassId is declaration order, which is deterministic for a given plan.
    let mut ready: BTreeSet<usize> = (0..pass_count)
        .filter(|index| in_degree[*index] == 0)
        .collect();
    let mut order = Vec::with_capacity(pass_count);
    while let Some(index) = ready.pop_first() {
        order.push(index);
        for successor in &successors[index] {
            in_degree[*successor] -= 1;
            if in_degree[*successor] == 0 {
                ready.insert(*successor);
            }
        }
    }
    if order.len() != pass_count {
        let mut cycle: Vec<String> = (0..pass_count)
            .filter(|index| in_degree[*index] != 0)
            .map(|index| builder.passes[index].name.clone())
            .collect();
        cycle.sort();
        return Err(CompileError::Cycle(cycle));
    }
    Ok(order)
}

fn plan_allocations(resources: &mut [CompiledResource], aliasing: bool) -> Vec<PhysicalAllocation> {
    let mut candidates: Vec<usize> = resources
        .iter()
        .enumerate()
        .filter(|(_, resource)| resource.origin.is_transient() && resource.first_use.is_some())
        .map(|(index, _)| index)
        .collect();
    candidates.sort_by_key(|index| {
        (
            resources[*index].first_use.unwrap_or(usize::MAX),
            resources[*index].id,
        )
    });

    let mut allocations: Vec<PhysicalAllocation> = Vec::new();
    for resource_index in candidates {
        let resource = &resources[resource_index];
        let first_use = resource.first_use.expect("candidate is used");
        let last_use = resource.last_use.expect("candidate is used");
        let compatible = aliasing
            .then(|| {
                allocations.iter().position(|allocation| {
                    allocation.last_use < first_use
                        && descriptors_alias_compatible(&allocation.desc, &resource.desc)
                })
            })
            .flatten();
        let allocation_index = if let Some(index) = compatible {
            let allocation = &mut allocations[index];
            allocation.resources.push(resource.id);
            allocation.last_use = last_use;
            index
        } else {
            let index = allocations.len();
            allocations.push(PhysicalAllocation {
                id: PhysicalAllocationId(index as u32),
                desc: resource.desc.clone(),
                resources: vec![resource.id],
                first_use,
                last_use,
            });
            index
        };
        resources[resource_index].physical = Some(allocations[allocation_index].id);
    }
    allocations
}

fn descriptors_alias_compatible(a: &ResourceDesc, b: &ResourceDesc) -> bool {
    match (a, b) {
        (ResourceDesc::Texture(a), ResourceDesc::Texture(b)) => {
            a.alias_class != AliasClass::Never
                && a.alias_class == b.alias_class
                && a.format == b.format
                && a.extent == b.extent
                && a.dimension == b.dimension
                && a.mip_count == b.mip_count
                && a.sample_count == b.sample_count
                && a.allowed_usage == b.allowed_usage
        }
        (ResourceDesc::Buffer(a), ResourceDesc::Buffer(b)) => {
            a.alias_class != AliasClass::Never
                && a.alias_class == b.alias_class
                && a.size == b.size
                && a.allowed_usage == b.allowed_usage
        }
        _ => false,
    }
}

fn plan_transitions(builder: &GraphBuilder, order: &[usize]) -> Vec<UsageTransition> {
    let mut transitions = Vec::new();
    let mut usage: Vec<Option<Usage>> = builder
        .resources
        .iter()
        .map(|resource| resource.origin.initial_usage())
        .collect();
    let mut queue: Vec<Option<QueueClass>> = vec![None; builder.resources.len()];

    for pass_index in order {
        let pass = &builder.passes[*pass_index];
        for access in &pass.accesses {
            let resource = access.handle.resource();
            let index = resource.0 as usize;
            if usage[index] != Some(access.usage) || queue[index] != Some(pass.queue) {
                transitions.push(UsageTransition {
                    resource,
                    pass: Some(pass.id),
                    before: usage[index],
                    after: access.usage,
                    from_queue: queue[index],
                    to_queue: pass.queue,
                });
            }
            usage[index] = Some(access.usage);
            queue[index] = Some(pass.queue);
        }
    }

    for resource in &builder.resources {
        let Some(final_usage) = resource.origin.final_usage() else {
            continue;
        };
        let index = resource.id.0 as usize;
        if usage[index] != Some(final_usage) {
            transitions.push(UsageTransition {
                resource: resource.id,
                pass: None,
                before: usage[index],
                after: final_usage,
                from_queue: queue[index],
                to_queue: queue[index].unwrap_or(QueueClass::Graphics),
            });
        }
    }
    transitions
}

fn descriptor_size_bytes(
    desc: &ResourceDesc,
    render_extent: (u32, u32),
    output_extent: (u32, u32),
) -> u64 {
    match desc {
        ResourceDesc::Buffer(desc) => desc.size,
        ResourceDesc::Texture(desc) => {
            let (mut width, mut height, mut layers) =
                desc.extent.resolve(render_extent, output_extent);
            if desc.dimension == TextureDimension::D3 {
                layers = layers.max(1);
            }
            let (block_width, block_height) = desc.format.block_dimensions();
            let block_bytes = desc
                .format
                .block_copy_size(Some(wgpu::TextureAspect::All))
                .or_else(|| desc.format.block_copy_size(None))
                .unwrap_or(4) as u64;
            let mut total = 0u64;
            for _ in 0..desc.mip_count.max(1) {
                let blocks_w = width.div_ceil(block_width.max(1)) as u64;
                let blocks_h = height.div_ceil(block_height.max(1)) as u64;
                total = total.saturating_add(
                    blocks_w
                        .saturating_mul(blocks_h)
                        .saturating_mul(layers as u64)
                        .saturating_mul(block_bytes),
                );
                width = (width / 2).max(1);
                height = (height / 2).max(1);
                if desc.dimension == TextureDimension::D3 {
                    layers = (layers / 2).max(1);
                }
            }
            total.saturating_mul(desc.sample_count.max(1) as u64)
        }
    }
}

fn stable_plan_id(
    label: &str,
    passes: &[CompiledPass],
    resources: &[CompiledResource],
    aliasing: bool,
) -> u64 {
    // FNV-1a is intentionally simple and specified. `DefaultHasher` is not a
    // stable persistence format across Rust releases.
    let mut hash = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    feed(label.as_bytes());
    feed(&[u8::from(aliasing)]);
    for pass in passes {
        feed(pass.name.as_bytes());
        feed(format!("{:?}{:?}", pass.queue, pass.side_effects).as_bytes());
        for dependency in &pass.dependencies {
            feed(&dependency.0.to_le_bytes());
        }
        for access in &pass.accesses {
            feed(&access.resource.0.to_le_bytes());
            feed(&access.version.0.to_le_bytes());
            feed(format!("{:?}{:?}", access.kind, access.usage).as_bytes());
        }
    }
    for resource in resources {
        feed(resource.name.as_bytes());
        feed(format!("{:?}{:?}", resource.desc, resource.origin).as_bytes());
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color_desc(format: wgpu::TextureFormat) -> TextureDesc {
        TextureDesc::color(
            format,
            Extent::RenderRelative {
                numerator: 1,
                denominator: 1,
                layers: 1,
            },
            TextureUsage::COLOR_ATTACHMENT
                .union(TextureUsage::SAMPLED)
                .union(TextureUsage::COPY_SRC)
                .union(TextureUsage::COPY_DST),
        )
    }

    fn imported_origin() -> ResourceOrigin {
        ResourceOrigin::Persistent {
            initial_usage: Usage::Texture(TextureUsage::SAMPLED),
            final_usage: Usage::Texture(TextureUsage::SAMPLED),
            ownership: Ownership::Graph,
        }
    }

    #[test]
    fn missing_producer_is_rejected() {
        let mut graph = GraphBuilder::new("missing-producer");
        let texture = graph.create_texture(
            "uninitialized",
            color_desc(wgpu::TextureFormat::Rgba16Float),
        );
        let pass = graph.add_pass("reader");
        graph.read_texture(pass, texture, TextureUsage::SAMPLED);
        assert!(matches!(
            graph.compile(CompileOptions::NO_ALIASING),
            Err(CompileError::MissingProducer { .. })
        ));
    }

    #[test]
    fn divergent_writers_of_one_version_are_rejected() {
        let mut graph = GraphBuilder::new("writers");
        let texture = graph.create_texture("color", color_desc(wgpu::TextureFormat::Rgba16Float));
        let first = graph.add_pass("first");
        let second = graph.add_pass("second");
        let _first_version = graph.write_texture(first, texture, TextureUsage::COLOR_ATTACHMENT);
        let _second_version = graph.write_texture(second, texture, TextureUsage::COLOR_ATTACHMENT);
        match graph.compile(CompileOptions::NO_ALIASING) {
            Err(CompileError::MultipleWriters {
                resource,
                version,
                writers,
            }) => {
                assert_eq!(resource, "color");
                assert_eq!(version, ResourceVersion(0));
                assert_eq!(writers, vec!["first", "second"]);
            }
            other => panic!("expected MultipleWriters, got {other:?}"),
        }
    }

    #[test]
    fn explicit_cycle_is_rejected_deterministically() {
        let mut graph = GraphBuilder::new("cycle");
        let a = graph.add_pass("a");
        let b = graph.add_pass("b");
        graph.after(a, b);
        graph.after(b, a);
        match graph.compile(CompileOptions::NO_ALIASING) {
            Err(CompileError::Cycle(names)) => {
                assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
            }
            other => panic!("expected Cycle, got {other:?}"),
        }
    }

    #[test]
    fn persistent_import_is_initialized_and_gets_final_transition() {
        let mut graph = GraphBuilder::new("persistent");
        let history = graph.import_texture(
            "taa-history",
            color_desc(wgpu::TextureFormat::Rgba16Float),
            imported_origin(),
        );
        let pass = graph.add_pass("taa");
        graph.read_texture(pass, history, TextureUsage::SAMPLED);
        let _output = graph.write_texture(pass, history, TextureUsage::COLOR_ATTACHMENT);
        let compiled = graph.compile(CompileOptions::NO_ALIASING).unwrap();
        assert!(compiled.resource("taa-history").unwrap().physical.is_none());
        assert!(compiled.transitions.iter().any(|transition| {
            transition.resource == history.resource()
                && transition.pass.is_none()
                && transition.after == Usage::Texture(TextureUsage::SAMPLED)
        }));
    }

    #[test]
    fn optional_pass_is_selected_before_compilation() {
        fn build(enabled: bool) -> CompiledGraph {
            let mut graph = GraphBuilder::new("optional");
            graph.add_pass("always");
            graph.add_optional_pass(enabled, "optional");
            graph.compile(CompileOptions::NO_ALIASING).unwrap()
        }
        assert_eq!(build(false).passes.len(), 1);
        assert_eq!(build(true).passes.len(), 2);
    }

    #[test]
    fn compatible_non_overlapping_textures_alias() {
        let mut graph = GraphBuilder::new("alias");
        let desc = color_desc(wgpu::TextureFormat::Rgba16Float);
        let first_texture = graph.create_texture("first-texture", desc.clone());
        let first_write = graph.add_pass("first-write");
        let first_texture =
            graph.write_texture(first_write, first_texture, TextureUsage::COLOR_ATTACHMENT);
        let first_read = graph.add_pass("first-read");
        graph.read_texture(first_read, first_texture, TextureUsage::SAMPLED);

        let second_texture = graph.create_texture("second-texture", desc);
        let second_write = graph.add_pass("second-write");
        let second_texture =
            graph.write_texture(second_write, second_texture, TextureUsage::COLOR_ATTACHMENT);
        let second_read = graph.add_pass("second-read");
        graph.read_texture(second_read, second_texture, TextureUsage::SAMPLED);

        let compiled = graph
            .compile(CompileOptions::CONSERVATIVE_ALIASING)
            .unwrap();
        assert_eq!(compiled.allocations.len(), 1);
        assert_eq!(compiled.allocations[0].resources.len(), 2);
        assert_eq!(
            compiled.transient_bytes((1920, 1080), (1920, 1080)),
            1920 * 1080 * 8
        );
        assert_eq!(
            compiled.unaliased_transient_bytes((1920, 1080), (1920, 1080)),
            2 * 1920 * 1080 * 8
        );
        assert_eq!(
            compiled.alias_savings_percent((1920, 1080), (1920, 1080)),
            50.0
        );
    }

    #[test]
    fn overlapping_or_incompatible_textures_do_not_alias() {
        let mut graph = GraphBuilder::new("no-alias");
        let first = graph.create_texture("rgba16", color_desc(wgpu::TextureFormat::Rgba16Float));
        let second = graph.create_texture("rgba8", color_desc(wgpu::TextureFormat::Rgba8Unorm));
        let write_first = graph.add_pass("write-first");
        let first = graph.write_texture(write_first, first, TextureUsage::COLOR_ATTACHMENT);
        let write_second = graph.add_pass("write-second");
        let second = graph.write_texture(write_second, second, TextureUsage::COLOR_ATTACHMENT);
        let read_both = graph.add_pass("read-both");
        graph.read_texture(read_both, first, TextureUsage::SAMPLED);
        graph.read_texture(read_both, second, TextureUsage::SAMPLED);
        let compiled = graph
            .compile(CompileOptions::CONSERVATIVE_ALIASING)
            .unwrap();
        assert_eq!(compiled.allocations.len(), 2);
    }

    #[test]
    fn buffer_and_texture_handles_cannot_cross_usage_contracts() {
        let mut graph = GraphBuilder::new("typed");
        let buffer = graph.create_buffer(
            "indirect",
            BufferDesc {
                size: 256,
                allowed_usage: BufferUsage::STORAGE_WRITE.union(BufferUsage::INDIRECT),
                alias_class: AliasClass::Storage,
            },
        );
        let produce = graph.add_pass("produce");
        let buffer = graph.write_buffer(produce, buffer, BufferUsage::STORAGE_WRITE);
        let draw = graph.add_pass("draw");
        graph.read_buffer(draw, buffer, BufferUsage::INDIRECT);
        let compiled = graph.compile(CompileOptions::NO_ALIASING).unwrap();
        assert_eq!(compiled.passes[0].name, "produce");
        assert_eq!(compiled.passes[1].name, "draw");
    }

    #[test]
    fn usage_outside_descriptor_is_rejected_before_wgpu_validation() {
        let mut graph = GraphBuilder::new("usage");
        let texture = graph.create_texture(
            "sample-only",
            TextureDesc::color(
                wgpu::TextureFormat::Rgba8Unorm,
                Extent::Fixed {
                    width: 1,
                    height: 1,
                    layers: 1,
                },
                TextureUsage::SAMPLED,
            ),
        );
        let pass = graph.add_pass("bad-write");
        let _ = graph.write_texture(pass, texture, TextureUsage::COLOR_ATTACHMENT);
        assert!(matches!(
            graph.compile(CompileOptions::NO_ALIASING),
            Err(CompileError::UsageNotDeclared { .. })
        ));
    }

    #[test]
    fn sorting_and_plan_identity_are_deterministic() {
        fn build() -> CompiledGraph {
            let mut graph = GraphBuilder::new("deterministic");
            let texture =
                graph.create_texture("color", color_desc(wgpu::TextureFormat::Rgba16Float));
            let late = graph.add_pass("late");
            let root = graph.add_pass("root");
            let texture = graph.write_texture(root, texture, TextureUsage::COLOR_ATTACHMENT);
            graph.read_texture(late, texture, TextureUsage::SAMPLED);
            graph.add_pass("independent");
            graph.compile(CompileOptions::NO_ALIASING).unwrap()
        }
        let first = build();
        let second = build();
        let names = |graph: &CompiledGraph| {
            graph
                .passes
                .iter()
                .map(|pass| pass.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&first), names(&second));
        assert_eq!(first.plan_id, second.plan_id);
        assert_eq!(first.to_json(), second.to_json());
        assert_eq!(first.to_dot(), second.to_dot());
    }
}
