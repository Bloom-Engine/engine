//! Deterministic machine- and human-readable render-graph diagnostics.

use super::{CompiledAccessKind, CompiledGraph, ResourceDesc};
use std::fmt::Write;

impl CompiledGraph {
    /// JSON dump containing topology, lifetimes, aliases, physical allocation
    /// identifiers, declared usages, and queue intent.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(4096);
        let _ = write!(
            out,
            "{{\"schema_version\":1,\"label\":\"{}\",\"plan_id\":\"{:016x}\",\
             \"aliasing_enabled\":{},\"passes\":[",
            escape_json(&self.label),
            self.plan_id,
            self.aliasing_enabled,
        );
        for (index, pass) in self.passes.iter().enumerate() {
            if index != 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{{\"id\":{},\"name\":\"{}\",\"queue\":\"{:?}\",\
                 \"side_effects\":{},\"dependencies\":[",
                pass.id.0,
                escape_json(&pass.name),
                pass.queue,
                pass.side_effects.0,
            );
            for (dependency_index, dependency) in pass.dependencies.iter().enumerate() {
                if dependency_index != 0 {
                    out.push(',');
                }
                let _ = write!(out, "{}", dependency.0);
            }
            out.push_str("],\"accesses\":[");
            for (access_index, access) in pass.accesses.iter().enumerate() {
                if access_index != 0 {
                    out.push(',');
                }
                let resource = &self.resources[access.resource.0 as usize];
                let _ = write!(
                    out,
                    "{{\"resource\":{},\"resource_name\":\"{}\",\"version\":{},\
                     \"kind\":\"{}\",\"usage\":\"{}\"}}",
                    access.resource.0,
                    escape_json(&resource.name),
                    access.version.0,
                    match access.kind {
                        CompiledAccessKind::Read => "read",
                        CompiledAccessKind::Write => "write",
                    },
                    access.usage.name(),
                );
            }
            out.push_str("]}");
        }
        out.push_str("],\"resources\":[");
        for (index, resource) in self.resources.iter().enumerate() {
            if index != 0 {
                out.push(',');
            }
            let (kind, descriptor) = descriptor_json(&resource.desc);
            let _ = write!(
                out,
                "{{\"id\":{},\"name\":\"{}\",\"kind\":\"{}\",\"origin\":\"{:?}\",\
                 \"first_use\":{},\"last_use\":{},\"physical\":{},\"descriptor\":{}}}",
                resource.id.0,
                escape_json(&resource.name),
                kind,
                resource.origin,
                optional_usize(resource.first_use),
                optional_usize(resource.last_use),
                resource
                    .physical
                    .map_or_else(|| "null".to_string(), |id| id.0.to_string()),
                descriptor,
            );
        }
        out.push_str("],\"allocations\":[");
        for (index, allocation) in self.allocations.iter().enumerate() {
            if index != 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{{\"id\":{},\"first_use\":{},\"last_use\":{},\"resources\":[",
                allocation.id.0, allocation.first_use, allocation.last_use,
            );
            for (resource_index, resource) in allocation.resources.iter().enumerate() {
                if resource_index != 0 {
                    out.push(',');
                }
                let _ = write!(out, "{}", resource.0);
            }
            out.push_str("]}");
        }
        out.push_str("],\"transitions\":[");
        for (index, transition) in self.transitions.iter().enumerate() {
            if index != 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{{\"resource\":{},\"pass\":{},\"before\":{},\"after\":\"{}\",\
                 \"from_queue\":{},\"to_queue\":\"{:?}\"}}",
                transition.resource.0,
                transition
                    .pass
                    .map_or_else(|| "null".to_string(), |pass| pass.0.to_string()),
                transition.before.map_or_else(
                    || "null".to_string(),
                    |usage| format!("\"{}\"", usage.name())
                ),
                transition.after.name(),
                transition
                    .from_queue
                    .map_or_else(|| "null".to_string(), |queue| format!("\"{:?}\"", queue)),
                transition.to_queue,
            );
        }
        out.push_str("]}");
        out
    }

    /// Graphviz DOT dump. Pass nodes are boxes, resources are ellipses, and
    /// allocation identifiers are included in resource labels.
    pub fn to_dot(&self) -> String {
        let mut out = String::with_capacity(4096);
        let _ = writeln!(out, "digraph \"{}\" {{", escape_dot(&self.label));
        out.push_str("  rankdir=LR;\n");
        for pass in &self.passes {
            let _ = writeln!(
                out,
                "  p{} [shape=box,label=\"{}\\n{:?}\"];",
                pass.id.0,
                escape_dot(&pass.name),
                pass.queue,
            );
        }
        for resource in &self.resources {
            let physical = resource
                .physical
                .map_or_else(|| "external".to_string(), |id| format!("physical {}", id.0));
            let lifetime = match (resource.first_use, resource.last_use) {
                (Some(first), Some(last)) => format!("{first}..{last}"),
                _ => "unused".to_string(),
            };
            let _ = writeln!(
                out,
                "  r{} [shape=ellipse,label=\"{}\\n{}\\nlife {}\"];",
                resource.id.0,
                escape_dot(&resource.name),
                physical,
                lifetime,
            );
        }
        for pass in &self.passes {
            for dependency in &pass.dependencies {
                let _ = writeln!(out, "  p{} -> p{} [color=gray];", dependency.0, pass.id.0);
            }
            for access in &pass.accesses {
                match access.kind {
                    CompiledAccessKind::Read => {
                        let _ = writeln!(
                            out,
                            "  r{} -> p{} [label=\"v{} {}\"];",
                            access.resource.0,
                            pass.id.0,
                            access.version.0,
                            access.usage.name(),
                        );
                    }
                    CompiledAccessKind::Write => {
                        let _ = writeln!(
                            out,
                            "  p{} -> r{} [label=\"v{} {}\"];",
                            pass.id.0,
                            access.resource.0,
                            access.version.0,
                            access.usage.name(),
                        );
                    }
                }
            }
        }
        out.push_str("}\n");
        out
    }
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn descriptor_json(desc: &ResourceDesc) -> (&'static str, String) {
    match desc {
        ResourceDesc::Texture(desc) => (
            "texture",
            format!(
                "{{\"format\":\"{:?}\",\"extent\":\"{:?}\",\"dimension\":\"{:?}\",\
                 \"mips\":{},\"samples\":{},\"usage\":{},\"load\":\"{:?}\",\
                 \"alias_class\":\"{:?}\"}}",
                desc.format,
                desc.extent,
                desc.dimension,
                desc.mip_count,
                desc.sample_count,
                desc.allowed_usage.0,
                desc.load,
                desc.alias_class,
            ),
        ),
        ResourceDesc::Buffer(desc) => (
            "buffer",
            format!(
                "{{\"size\":{},\"usage\":{},\"alias_class\":\"{:?}\"}}",
                desc.size, desc.allowed_usage.0, desc.alias_class,
            ),
        ),
    }
}

fn escape_json(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                let _ = write!(escaped, "\\u{:04x}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn escape_dot(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use crate::renderer::graph::{
        CompileOptions, Extent, GraphBuilder, Ownership, ResourceOrigin, TextureDesc, TextureUsage,
        Usage,
    };

    #[test]
    fn json_and_dot_escape_names_and_include_physical_ids() {
        let usage = TextureUsage::SAMPLED.union(TextureUsage::COLOR_ATTACHMENT);
        let desc = TextureDesc::color(
            wgpu::TextureFormat::Rgba16Float,
            Extent::RenderRelative {
                numerator: 1,
                denominator: 1,
                layers: 1,
            },
            usage,
        );
        let mut graph = GraphBuilder::new("dump \"test\"");
        let imported = graph.import_texture(
            "history",
            desc.clone(),
            ResourceOrigin::Persistent {
                initial_usage: Usage::Texture(TextureUsage::SAMPLED),
                final_usage: Usage::Texture(TextureUsage::SAMPLED),
                ownership: Ownership::Graph,
            },
        );
        let transient = graph.create_texture("scratch", desc);
        let pass = graph.add_pass("shade");
        graph.read_texture(pass, imported, TextureUsage::SAMPLED);
        let _ = graph.write_texture(pass, transient, TextureUsage::COLOR_ATTACHMENT);
        let compiled = graph.compile(CompileOptions::NO_ALIASING).unwrap();

        let json = compiled.to_json();
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("\"physical\":0"));
        assert!(json.contains("dump \\\"test\\\""));
        assert!(compiled.to_dot().contains("physical 0"));
    }
}
