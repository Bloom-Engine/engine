// Render graph — Phase 2 of RFC 0001.
//
// Declarative pass scheduler. Each `PassNode` declares what it reads,
// what it writes, and a `run` closure that encodes its rendering work
// against an owning `Renderer`. The scheduler topologically sorts the
// nodes so a node that reads a resource always runs after the node
// that writes it; explicit `after` / `before` hints break ties.
//
// This module landed in two steps: Phase 2 brought the types, the
// scheduler, and ordering unit tests; Phase 2b (complete — see the
// frame-graph construction in `mod.rs`) ported the existing passes, so
// the graph now drives the real frame in `end_frame_with_scene`.

use std::collections::{HashMap, HashSet};

// =====================================================================
// Inputs / Outputs — the resource kinds the graph schedules around
// =====================================================================

/// Resources a pass may read. `Transient` is the open extension point
/// for Phase 3's resource pool; the named variants cover the
/// well-known render targets that the engine already owns.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PassInput {
    /// Snapshot of HDR colour before the current pass ran. Triggers
    /// a `CopyToSample` synthetic node if no prior pass explicitly
    /// produced it.
    SceneColor,
    /// Linearised scene depth, sampled. Same synthetic-copy semantics.
    SceneDepth,
    /// Cascaded shadow map depth at cascade index.
    Shadow(u8),
    EnvCubemap,
    MotionVectors,
    /// Phase 7 — world-space impulse texture (decals, splashes, wakes).
    Impulse,
    /// Phase-3-defined transient texture identified by the pool's
    /// handle type. Represented as a u32 here so the graph module
    /// doesn't depend on the pool implementation.
    Transient(u32),
}

/// Resources a pass may write. The variants mirror `PassInput` where
/// it makes sense, plus some write-only targets (MRT G-buffer slots,
/// the shadow map, the swapchain).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PassOutput {
    HdrColor,
    Depth,
    MaterialRt,
    VelocityRt,
    AlbedoRt,
    Shadow(u8),
    Transient(u32),
    /// Terminal node — the final surface presentation target. Every
    /// frame's graph must end with exactly one node that writes
    /// Swapchain.
    Swapchain,
}

// =====================================================================
// PassNode — what each scheduler unit looks like
// =====================================================================

/// A single schedulable unit of GPU work. `run` is deliberately left
/// as a boxed closure so pass bodies own whatever captured state they
/// need; the scheduler doesn't know or care.
///
/// `RunFn` takes a single untyped context argument because this module
/// must not depend on `Renderer` (would create a module cycle). The
/// `'a` lifetime lets closures borrow frame-scoped data (render
/// targets, uniform buffers) from the renderer — dropping the `Send`
/// bound is deliberate, since graph execution is single-threaded from
/// the caller's perspective.
pub type RunFn<'a, Ctx> = Box<dyn FnOnce(&mut Ctx) + 'a>;

pub struct PassNode<'a, Ctx> {
    pub name:   &'static str,
    pub reads:  Vec<PassInput>,
    pub writes: Vec<PassOutput>,
    /// Hard "run this node after X" hints — used for ordering that the
    /// data dependencies alone can't express (e.g. two passes that
    /// share a read target but where one conceptually follows the
    /// other for timing reasons).
    pub after:  Vec<&'static str>,
    /// Hard "run this node before X" hints. Validator rejects cycles.
    pub before: Vec<&'static str>,
    pub run:    RunFn<'a, Ctx>,
}

impl<'a, Ctx> PassNode<'a, Ctx> {
    pub fn new(name: &'static str, run: RunFn<'a, Ctx>) -> Self {
        Self {
            name, reads: Vec::new(), writes: Vec::new(),
            after: Vec::new(), before: Vec::new(),
            run,
        }
    }
    pub fn with_reads(mut self, reads: &[PassInput]) -> Self {
        self.reads = reads.to_vec(); self
    }
    pub fn with_writes(mut self, writes: &[PassOutput]) -> Self {
        self.writes = writes.to_vec(); self
    }
    pub fn with_after(mut self, after: &[&'static str]) -> Self {
        self.after = after.to_vec(); self
    }
    pub fn with_before(mut self, before: &[&'static str]) -> Self {
        self.before = before.to_vec(); self
    }
}

// =====================================================================
// Graph — a container of nodes + schedule + execute
// =====================================================================

pub struct Graph<'a, Ctx> {
    pub nodes: Vec<PassNode<'a, Ctx>>,
}

#[derive(Debug)]
pub enum GraphError {
    /// `after` / `before` referenced a name that isn't in the graph.
    UnknownNode { node: String, referenced: String },
    /// Data dependencies + hints imply a cycle that can't be resolved.
    Cycle(Vec<String>),
}

impl<'a, Ctx> Graph<'a, Ctx> {
    pub fn new() -> Self { Self { nodes: Vec::new() } }
    pub fn push(&mut self, node: PassNode<'a, Ctx>) -> &mut Self {
        self.nodes.push(node); self
    }

    /// Compute execution order. Returns indices into `self.nodes`.
    pub fn schedule(&self) -> Result<Vec<usize>, GraphError> {
        schedule(&self.nodes)
    }

    /// Schedule + run all nodes in order. Consumes the graph because
    /// each node's `run` closure is `FnOnce`.
    pub fn execute(self, ctx: &mut Ctx) -> Result<(), GraphError> {
        let order = schedule(&self.nodes)?;
        // Drain in order. Using Option<PassNode> trick so we can take()
        // nodes out by index without panicking the Vec's geometry.
        let mut slots: Vec<Option<PassNode<'a, Ctx>>> = self.nodes.into_iter().map(Some).collect();
        for i in order {
            let node = slots[i].take().expect("each node scheduled exactly once");
            (node.run)(ctx);
        }
        Ok(())
    }
}

impl<'a, Ctx> Default for Graph<'a, Ctx> {
    fn default() -> Self { Self::new() }
}

// =====================================================================
// Scheduler — topological sort (Kahn's algorithm) with hint tie-breaks
// =====================================================================

fn schedule<'a, Ctx>(nodes: &[PassNode<'a, Ctx>]) -> Result<Vec<usize>, GraphError> {
    let n = nodes.len();
    if n == 0 { return Ok(Vec::new()); }

    // Name → index lookup for `after` / `before` hint resolution.
    let mut name_to_idx: HashMap<&'static str, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        name_to_idx.insert(node.name, i);
    }

    // Build the predecessor set for each node.
    // Edge A → B means A must run before B.
    let mut preds: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    // 1) Data-dependency edges: if B reads what A writes, A → B.
    //    Approximate "writes" as "any write matches any read" — the
    //    enum comparison treats PassOutput::HdrColor as matching
    //    PassInput::SceneColor via the `matches_write` helper below.
    for b in 0..n {
        for read in &nodes[b].reads {
            for a in 0..n {
                if a == b { continue; }
                for write in &nodes[a].writes {
                    if input_matches_write(read, write) {
                        preds[b].insert(a);
                    }
                }
            }
        }
    }
    // 2) Explicit `after` hints: node B.after = [A] → A → B.
    for (b, node) in nodes.iter().enumerate() {
        for name in &node.after {
            let a = name_to_idx.get(name).ok_or_else(|| GraphError::UnknownNode {
                node: node.name.to_string(),
                referenced: name.to_string(),
            })?;
            if *a != b { preds[b].insert(*a); }
        }
    }
    // 3) Explicit `before` hints: node A.before = [B] → A → B.
    for (a, node) in nodes.iter().enumerate() {
        for name in &node.before {
            let b = name_to_idx.get(name).ok_or_else(|| GraphError::UnknownNode {
                node: node.name.to_string(),
                referenced: name.to_string(),
            })?;
            if *b != a { preds[*b].insert(a); }
        }
    }

    // Kahn's algorithm: repeatedly pick a node whose `preds` is empty
    // (or whose unsatisfied preds are already in the output), add it,
    // strip it from every other preds set. Break ties by declaration
    // order to keep the schedule deterministic.
    let mut in_degree: Vec<usize> = preds.iter().map(|p| p.len()).collect();
    let mut out: Vec<usize> = Vec::with_capacity(n);
    loop {
        let next = (0..n).find(|&i| !out.contains(&i) && in_degree[i] == 0);
        match next {
            Some(i) => {
                out.push(i);
                for j in 0..n {
                    if preds[j].remove(&i) { in_degree[j] -= 1; }
                }
            }
            None => break,
        }
    }

    if out.len() != n {
        // Anything still in `preds` is part of a cycle.
        let cycle: Vec<String> = (0..n)
            .filter(|i| !out.contains(i))
            .map(|i| nodes[i].name.to_string())
            .collect();
        return Err(GraphError::Cycle(cycle));
    }
    Ok(out)
}

/// A read-of-output match: SceneColor reads from HdrColor,
/// SceneDepth reads from Depth, etc. The synthetic copy-to-sample
/// node that Phase 2b introduces is the one that actually produces
/// the sampled snapshot; for scheduling purposes we treat them as
/// the same resource.
fn input_matches_write(input: &PassInput, output: &PassOutput) -> bool {
    match (input, output) {
        (PassInput::SceneColor,       PassOutput::HdrColor)    => true,
        (PassInput::SceneDepth,       PassOutput::Depth)       => true,
        (PassInput::Shadow(i),        PassOutput::Shadow(j))   => i == j,
        (PassInput::Transient(i),     PassOutput::Transient(j)) => i == j,
        // EnvCubemap, MotionVectors, Impulse are produced outside
        // the render graph for now (env is loaded at startup; motion
        // vectors are written as a G-buffer target but not individually
        // tracked). No-op match.
        _ => false,
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Test context that records the order of pass executions.
    type TestCtx = Vec<&'static str>;

    fn record(name: &'static str) -> RunFn<'static, TestCtx> {
        Box::new(move |ctx: &mut TestCtx| ctx.push(name))
    }

    #[test]
    fn empty_graph_runs_clean() {
        let graph: Graph<'_, TestCtx> = Graph::new();
        let mut ctx = Vec::new();
        graph.execute(&mut ctx).unwrap();
        assert!(ctx.is_empty());
    }

    #[test]
    fn single_node_runs() {
        let mut graph = Graph::new();
        graph.push(PassNode::new("a", record("a")));
        let mut ctx = Vec::new();
        graph.execute(&mut ctx).unwrap();
        assert_eq!(ctx, vec!["a"]);
    }

    #[test]
    fn data_dependency_orders_reads_after_writes() {
        // A writes HdrColor, B reads SceneColor. B must run after A.
        let mut graph = Graph::new();
        graph.push(PassNode::new("B_reads_scene", record("B"))
            .with_reads(&[PassInput::SceneColor]));
        graph.push(PassNode::new("A_writes_hdr", record("A"))
            .with_writes(&[PassOutput::HdrColor]));
        let mut ctx = Vec::new();
        graph.execute(&mut ctx).unwrap();
        assert_eq!(ctx, vec!["A", "B"]);
    }

    #[test]
    fn explicit_after_hint_orders_nodes() {
        let mut graph = Graph::new();
        graph.push(PassNode::new("later", record("later"))
            .with_after(&["earlier"]));
        graph.push(PassNode::new("earlier", record("earlier")));
        let mut ctx = Vec::new();
        graph.execute(&mut ctx).unwrap();
        assert_eq!(ctx, vec!["earlier", "later"]);
    }

    #[test]
    fn explicit_before_hint_orders_nodes() {
        let mut graph = Graph::new();
        graph.push(PassNode::new("second", record("second")));
        graph.push(PassNode::new("first", record("first"))
            .with_before(&["second"]));
        let mut ctx = Vec::new();
        graph.execute(&mut ctx).unwrap();
        assert_eq!(ctx, vec!["first", "second"]);
    }

    #[test]
    fn chain_of_three() {
        // A -> B -> C via data dependencies.
        let mut graph = Graph::new();
        graph.push(PassNode::new("C", record("C"))
            .with_reads(&[PassInput::Transient(1)]));
        graph.push(PassNode::new("A", record("A"))
            .with_writes(&[PassOutput::HdrColor]));
        graph.push(PassNode::new("B", record("B"))
            .with_reads(&[PassInput::SceneColor])
            .with_writes(&[PassOutput::Transient(1)]));
        let mut ctx = Vec::new();
        graph.execute(&mut ctx).unwrap();
        assert_eq!(ctx, vec!["A", "B", "C"]);
    }

    #[test]
    fn parallel_branches_deterministic_by_declaration_order() {
        // Both X and Y depend on ROOT, neither depends on the other.
        // Order should be declaration order among the tied candidates.
        let mut graph = Graph::new();
        graph.push(PassNode::new("X", record("X"))
            .with_reads(&[PassInput::SceneColor]));
        graph.push(PassNode::new("Y", record("Y"))
            .with_reads(&[PassInput::SceneColor]));
        graph.push(PassNode::new("ROOT", record("ROOT"))
            .with_writes(&[PassOutput::HdrColor]));
        let mut ctx = Vec::new();
        graph.execute(&mut ctx).unwrap();
        assert_eq!(ctx, vec!["ROOT", "X", "Y"]);
    }

    #[test]
    fn unknown_after_is_reported() {
        let mut graph: Graph<'_, TestCtx> = Graph::new();
        graph.push(PassNode::new("a", record("a"))
            .with_after(&["does_not_exist"]));
        let err = graph.schedule().unwrap_err();
        match err {
            GraphError::UnknownNode { node, referenced } => {
                assert_eq!(node, "a");
                assert_eq!(referenced, "does_not_exist");
            }
            _ => panic!("expected UnknownNode"),
        }
    }

    #[test]
    fn cycle_via_explicit_hints_is_rejected() {
        let mut graph: Graph<'_, TestCtx> = Graph::new();
        graph.push(PassNode::new("a", record("a"))
            .with_after(&["b"]));
        graph.push(PassNode::new("b", record("b"))
            .with_after(&["a"]));
        match graph.schedule() {
            Err(GraphError::Cycle(names)) => {
                assert!(names.contains(&"a".to_string()));
                assert!(names.contains(&"b".to_string()));
            }
            other => panic!("expected Cycle, got {:?}", other),
        }
    }

    #[test]
    fn concrete_frame_graph_sketches_correctly() {
        // A rough sketch of the Phase-2b target frame graph — shadow
        // → main_hdr → ssao → translucent → composite → swapchain —
        // proves the scheduler produces the expected order.
        let mut graph: Graph<'_, TestCtx> = Graph::new();
        graph.push(PassNode::new("composite", record("composite"))
            .with_reads(&[PassInput::Transient(10)])
            // Composite is the terminal pass — explicit `after` on
            // every predecessor so tie-breaking (declaration order)
            // doesn't float composite past a sibling that hasn't run.
            .with_after(&["bloom", "translucent", "ssao"])
            .with_writes(&[PassOutput::Swapchain]));
        graph.push(PassNode::new("bloom", record("bloom"))
            .with_reads(&[PassInput::SceneColor])
            .with_writes(&[PassOutput::Transient(10)]));
        graph.push(PassNode::new("translucent", record("translucent"))
            .with_reads(&[PassInput::SceneColor, PassInput::SceneDepth])
            .with_writes(&[PassOutput::HdrColor])
            .with_after(&["main_hdr"]));   // explicit so SceneColor is
                                           // a snapshot, not a write-
                                           // then-read loop
        graph.push(PassNode::new("ssao", record("ssao"))
            .with_reads(&[PassInput::SceneDepth])
            .with_writes(&[PassOutput::Transient(20)]));
        graph.push(PassNode::new("main_hdr", record("main_hdr"))
            .with_reads(&[PassInput::Shadow(0)])
            .with_writes(&[PassOutput::HdrColor, PassOutput::Depth,
                           PassOutput::MaterialRt, PassOutput::VelocityRt,
                           PassOutput::AlbedoRt]));
        graph.push(PassNode::new("shadow_0", record("shadow_0"))
            .with_writes(&[PassOutput::Shadow(0)]));

        let mut ctx = Vec::new();
        graph.execute(&mut ctx).unwrap();
        // Invariants — strict total order isn't guaranteed, but these
        // data / explicit-hint dependencies must hold:
        let pos = |name: &str| ctx.iter().position(|&n| n == name).unwrap();
        assert!(pos("shadow_0")   < pos("main_hdr"));
        assert!(pos("main_hdr")   < pos("ssao"));
        assert!(pos("main_hdr")   < pos("translucent"));
        assert!(pos("main_hdr")   < pos("bloom"));
        assert!(pos("bloom")      < pos("composite"));
        assert!(pos("translucent") < pos("composite"));
        assert!(pos("ssao")        < pos("composite"));
        assert_eq!(pos("composite"), ctx.len() - 1);
    }

    #[test]
    fn run_fn_captures_mutable_state() {
        // Proves the RunFn / context model supports shared mutable
        // data between passes — the real engine use case is writing
        // to the encoder or the profiler.
        let counter = Arc::new(Mutex::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let mut graph: Graph<'_, TestCtx> = Graph::new();
        graph.push(PassNode::new("a", Box::new(move |_ctx| {
            *c1.lock().unwrap() += 1;
        })));
        graph.push(PassNode::new("b", Box::new(move |_ctx| {
            *c2.lock().unwrap() += 10;
        })));
        let mut ctx = Vec::new();
        graph.execute(&mut ctx).unwrap();
        assert_eq!(*counter.lock().unwrap(), 11);
    }
}
