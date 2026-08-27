//! Independent positional error bound for cooked hierarchy simplification.

use crate::meshlet::StaticVertex;

const LEAF_TRIANGLES: usize = 8;

#[derive(Clone, Copy)]
struct Triangle {
    points: [[f32; 3]; 3],
    minimum: [f32; 3],
    maximum: [f32; 3],
    centroid: [f32; 3],
}

#[derive(Clone, Copy)]
struct Bounds {
    minimum: [f32; 3],
    maximum: [f32; 3],
}

#[derive(Clone, Copy)]
enum NodeKind {
    Leaf { start: usize, end: usize },
    Branch { left: usize, right: usize },
}

#[derive(Clone, Copy)]
struct Node {
    bounds: Bounds,
    kind: NodeKind,
}

struct TriangleBvh {
    triangles: Vec<Triangle>,
    order: Vec<usize>,
    nodes: Vec<Node>,
    root: usize,
}

/// Maximum one-sided distance from every source vertex to the simplified
/// triangle surface. The hierarchy accumulates this independently measured
/// positional bound instead of treating the simplifier's quadric metric as a
/// Hausdorff-style guarantee.
pub fn maximum_vertex_deviation(
    vertices: &[StaticVertex],
    simplified_indices: &[u32],
) -> Result<f32, String> {
    let bvh = TriangleBvh::new(vertices, simplified_indices)?;
    Ok(vertices
        .iter()
        .map(|vertex| bvh.nearest_squared(vertex.position).sqrt())
        .fold(0.0, f32::max))
}

impl TriangleBvh {
    fn new(vertices: &[StaticVertex], indices: &[u32]) -> Result<Self, String> {
        if indices.is_empty() || !indices.len().is_multiple_of(3) {
            return Err("simplified geometry is not a non-empty triangle list".to_string());
        }
        let mut triangles = Vec::with_capacity(indices.len() / 3);
        for (triangle_index, indices) in indices.as_chunks::<3>().0.iter().enumerate() {
            let points = indices.map(|index| {
                vertices
                    .get(index as usize)
                    .map(|vertex| vertex.position)
                    .ok_or_else(|| {
                        format!(
                            "simplified triangle {triangle_index} index {index} exceeds {} vertices",
                            vertices.len()
                        )
                    })
            });
            let [a, b, c] = points;
            let [a, b, c] = [a?, b?, c?];
            let minimum = component_min(component_min(a, b), c);
            let maximum = component_max(component_max(a, b), c);
            triangles.push(Triangle {
                points: [a, b, c],
                minimum,
                maximum,
                centroid: mul3(add3(add3(a, b), c), 1.0 / 3.0),
            });
        }
        let mut bvh = Self {
            order: (0..triangles.len()).collect(),
            triangles,
            nodes: Vec::new(),
            root: 0,
        };
        bvh.root = bvh.build(0, bvh.order.len());
        Ok(bvh)
    }

    fn build(&mut self, start: usize, end: usize) -> usize {
        let bounds = self.bounds(start, end);
        let node_index = self.nodes.len();
        self.nodes.push(Node {
            bounds,
            kind: NodeKind::Leaf { start, end },
        });
        if end - start <= LEAF_TRIANGLES {
            return node_index;
        }
        let centroid_bounds = self.centroid_bounds(start, end);
        let extent = sub3(centroid_bounds.maximum, centroid_bounds.minimum);
        let axis = if extent[1] > extent[0] && extent[1] >= extent[2] {
            1
        } else if extent[2] > extent[0] {
            2
        } else {
            0
        };
        let triangles = &self.triangles;
        self.order[start..end]
            .sort_by(|a, b| triangles[*a].centroid[axis].total_cmp(&triangles[*b].centroid[axis]));
        let middle = start + (end - start) / 2;
        let left = self.build(start, middle);
        let right = self.build(middle, end);
        self.nodes[node_index].kind = NodeKind::Branch { left, right };
        node_index
    }

    fn bounds(&self, start: usize, end: usize) -> Bounds {
        self.order[start..end]
            .iter()
            .fold(empty_bounds(), |bounds, triangle| Bounds {
                minimum: component_min(bounds.minimum, self.triangles[*triangle].minimum),
                maximum: component_max(bounds.maximum, self.triangles[*triangle].maximum),
            })
    }

    fn centroid_bounds(&self, start: usize, end: usize) -> Bounds {
        self.order[start..end]
            .iter()
            .fold(empty_bounds(), |bounds, triangle| {
                let centroid = self.triangles[*triangle].centroid;
                Bounds {
                    minimum: component_min(bounds.minimum, centroid),
                    maximum: component_max(bounds.maximum, centroid),
                }
            })
    }

    fn nearest_squared(&self, point: [f32; 3]) -> f32 {
        self.nearest_node(point, self.root, f32::INFINITY)
    }

    fn nearest_node(&self, point: [f32; 3], node_index: usize, mut best: f32) -> f32 {
        let node = self.nodes[node_index];
        if point_bounds_distance_squared(point, node.bounds) >= best {
            return best;
        }
        match node.kind {
            NodeKind::Leaf { start, end } => {
                for triangle in &self.order[start..end] {
                    best = best.min(point_triangle_distance_squared(
                        point,
                        self.triangles[*triangle].points,
                    ));
                }
                best
            }
            NodeKind::Branch { left, right } => {
                let left_distance = point_bounds_distance_squared(point, self.nodes[left].bounds);
                let right_distance = point_bounds_distance_squared(point, self.nodes[right].bounds);
                let (near, far) = if left_distance <= right_distance {
                    (left, right)
                } else {
                    (right, left)
                };
                best = self.nearest_node(point, near, best);
                self.nearest_node(point, far, best)
            }
        }
    }
}

fn point_bounds_distance_squared(point: [f32; 3], bounds: Bounds) -> f32 {
    (0..3)
        .map(|axis| {
            let distance = if point[axis] < bounds.minimum[axis] {
                bounds.minimum[axis] - point[axis]
            } else if point[axis] > bounds.maximum[axis] {
                point[axis] - bounds.maximum[axis]
            } else {
                0.0
            };
            distance * distance
        })
        .sum()
}

fn point_triangle_distance_squared(point: [f32; 3], [a, b, c]: [[f32; 3]; 3]) -> f32 {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(point, a);
    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return dot3(ap, ap);
    }
    let bp = sub3(point, b);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dot3(bp, bp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return distance_squared(point, add3(a, mul3(ab, d1 / (d1 - d3))));
    }
    let cp = sub3(point, c);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dot3(cp, cp);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return distance_squared(point, add3(a, mul3(ac, d2 / (d2 - d6))));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let edge = sub3(c, b);
        let weight = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return distance_squared(point, add3(b, mul3(edge, weight)));
    }
    let sum = va + vb + vc;
    if sum <= 1.0e-30 {
        return point_segment_distance_squared(point, a, b)
            .min(point_segment_distance_squared(point, b, c))
            .min(point_segment_distance_squared(point, c, a));
    }
    let denominator = sum.recip();
    let closest = add3(
        a,
        add3(mul3(ab, vb * denominator), mul3(ac, vc * denominator)),
    );
    distance_squared(point, closest)
}

fn point_segment_distance_squared(point: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let edge = sub3(b, a);
    let length_squared = dot3(edge, edge);
    if length_squared <= 1.0e-30 {
        return distance_squared(point, a);
    }
    let weight = (dot3(sub3(point, a), edge) / length_squared).clamp(0.0, 1.0);
    distance_squared(point, add3(a, mul3(edge, weight)))
}

fn empty_bounds() -> Bounds {
    Bounds {
        minimum: [f32::INFINITY; 3],
        maximum: [f32::NEG_INFINITY; 3],
    }
}

fn component_min(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
}

fn component_max(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul3(value: [f32; 3], factor: f32) -> [f32; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    dot3(sub3(a, b), sub3(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(position: [f32; 3]) -> StaticVertex {
        StaticVertex {
            position,
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv0: [0.0; 2],
            uv1: [0.0; 2],
            color: [1.0; 4],
        }
    }

    #[test]
    fn bvh_measures_one_sided_vertex_deviation() {
        let vertices = [
            vertex([0.0, 0.0, 0.0]),
            vertex([1.0, 0.0, 0.0]),
            vertex([0.0, 1.0, 0.0]),
            vertex([0.25, 0.25, 0.5]),
        ];
        let error = maximum_vertex_deviation(&vertices, &[0, 1, 2]).unwrap();
        assert!((error - 0.5).abs() <= 1.0e-6);
    }
}
