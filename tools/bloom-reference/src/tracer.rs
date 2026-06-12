//! Path-tracing core for bloom-reference: rays, BVH, camera, RNG,
//! GGX/Burley BRDF, environment sampling, punctual lights, and the
//! integrator. Split from main.rs (2000-line file policy).

use crate::*;

// ============================================================
// Ray
// ============================================================

pub(crate) struct Ray {
    pub(crate) origin: Vec3,
    pub(crate) direction: Vec3,     // unit
    pub(crate) inv_direction: Vec3, // 1.0 / direction, cached for AABB tests
}

impl Ray {
    pub(crate) fn new(origin: Vec3, direction: Vec3) -> Self {
        let d = direction.normalize();
        Self {
            origin,
            direction: d,
            inv_direction: Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z),
        }
    }
}

pub(crate) struct Hit {
    pub(crate) t: f32,
    pub(crate) barycentric: Vec2, // (u, v); w = 1 - u - v
    pub(crate) triangle_index: u32,
}

pub(crate) fn intersect_triangle(ray: &Ray, tri: &Triangle, max_t: f32) -> Option<(f32, Vec2)> {
    const EPS: f32 = 1.0e-6;
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = ray.direction.cross(edge2);
    let a = edge1.dot(h);
    if a.abs() < EPS {
        return None;
    }
    let f = 1.0 / a;
    let s = ray.origin - tri.v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = s.cross(edge1);
    let v = f * ray.direction.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * edge2.dot(q);
    if t > EPS && t < max_t {
        Some((t, Vec2::new(u, v)))
    } else {
        None
    }
}

/// Slab-based ray vs AABB. Returns `Some((t_near, t_far))` when the ray
/// intersects the box in front of the origin; used by the BVH walk to
/// decide which children to descend into.
pub(crate) fn intersect_aabb(ray: &Ray, bounds_min: Vec3, bounds_max: Vec3, max_t: f32) -> Option<f32> {
    let t1 = (bounds_min - ray.origin) * ray.inv_direction;
    let t2 = (bounds_max - ray.origin) * ray.inv_direction;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);
    let near = t_min.x.max(t_min.y).max(t_min.z);
    let far = t_max.x.min(t_max.y).min(t_max.z);
    if far >= near.max(0.0) && near < max_t {
        Some(near.max(0.0))
    } else {
        None
    }
}

// ============================================================
// BVH
// ============================================================

/// Flat BVH stored as a Vec<BvhNode>. Internal nodes use
/// `first_triangle` as the index of their LEFT child in the same Vec;
/// leaves set `tri_count > 0` and use `first_triangle` as the starting
/// index into the triangle-index array. This is the layout used by most
/// production renderers because it's cache-friendly and pointer-free.
#[derive(Clone)]
pub(crate) struct BvhNode {
    pub(crate) bounds_min: Vec3,
    pub(crate) bounds_max: Vec3,
    pub(crate) first_triangle: u32, // leaf: start in triangle_indices; internal: left child
    pub(crate) tri_count: u32,      // 0 = internal node
}

pub(crate) struct Bvh {
    pub(crate) nodes: Vec<BvhNode>,
    /// Remapped triangle ordering so each leaf's triangles are
    /// contiguous in the scene's triangle buffer. Indexed into `triangles`.
    pub(crate) triangle_indices: Vec<u32>,
}

/// Median-split BVH builder. Not as tight as SAH but 3-5× faster to
/// build, and for our scene sizes (tens to low hundreds of thousands
/// of triangles) the intersection cost difference is negligible. We
/// can revisit this if Phase 4 needs tighter trees.
pub(crate) fn build_bvh_recursive(
    items: &mut [BvhItem],
    offset: usize,
    nodes: &mut Vec<BvhNode>,
    order: &mut Vec<u32>,
    node_index: usize,
) {
    // Compute combined bounds for this subtree.
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    for it in items.iter() {
        bmin = bmin.min(it.bounds_min);
        bmax = bmax.max(it.bounds_max);
    }

    const LEAF_THRESHOLD: usize = 4;
    if items.len() <= LEAF_THRESHOLD {
        let first = order.len() as u32;
        for it in items.iter() {
            order.push(it.triangle_index);
        }
        nodes[node_index] = BvhNode {
            bounds_min: bmin,
            bounds_max: bmax,
            first_triangle: first,
            tri_count: items.len() as u32,
        };
        return;
    }

    // Split on the longest axis of the centroid bounds (not the
    // triangle bounds — the centroid bounds give the meaningful split
    // range even when triangles are large).
    let mut cmin = Vec3::splat(f32::INFINITY);
    let mut cmax = Vec3::splat(f32::NEG_INFINITY);
    for it in items.iter() {
        cmin = cmin.min(it.centroid);
        cmax = cmax.max(it.centroid);
    }
    let extent = cmax - cmin;
    let axis = if extent.x > extent.y && extent.x > extent.z {
        0
    } else if extent.y > extent.z {
        1
    } else {
        2
    };

    // Median split: partial sort so the middle item has the actual median.
    // select_nth_unstable is O(N); a full sort would be O(N log N).
    let mid = items.len() / 2;
    items.select_nth_unstable_by(mid, |a, b| {
        let av = match axis {
            0 => a.centroid.x,
            1 => a.centroid.y,
            _ => a.centroid.z,
        };
        let bv = match axis {
            0 => b.centroid.x,
            1 => b.centroid.y,
            _ => b.centroid.z,
        };
        av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reserve two child nodes. We fill them in recursively; the left
    // child is at `nodes.len()`, the right at `nodes.len() + 1`.
    let left_idx = nodes.len();
    nodes.push(BvhNode {
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::ZERO,
        first_triangle: 0,
        tri_count: 0,
    });
    nodes.push(BvhNode {
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::ZERO,
        first_triangle: 0,
        tri_count: 0,
    });

    let (left, right) = items.split_at_mut(mid);
    build_bvh_recursive(left, offset, nodes, order, left_idx);
    build_bvh_recursive(right, offset + mid, nodes, order, left_idx + 1);

    nodes[node_index] = BvhNode {
        bounds_min: bmin,
        bounds_max: bmax,
        first_triangle: left_idx as u32,
        tri_count: 0, // 0 marks internal
    };
}

#[derive(Clone)]
pub(crate) struct BvhItem {
    pub(crate) bounds_min: Vec3,
    pub(crate) bounds_max: Vec3,
    pub(crate) centroid: Vec3,
    pub(crate) triangle_index: u32,
}

pub(crate) fn build_bvh(triangles: &[Triangle]) -> Bvh {
    let items: Vec<BvhItem> = triangles
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let bmin = t.v0.min(t.v1).min(t.v2);
            let bmax = t.v0.max(t.v1).max(t.v2);
            BvhItem {
                bounds_min: bmin,
                bounds_max: bmax,
                centroid: (bmin + bmax) * 0.5,
                triangle_index: i as u32,
            }
        })
        .collect();

    let mut items_mut = items;
    let mut nodes: Vec<BvhNode> = Vec::new();
    let mut order: Vec<u32> = Vec::with_capacity(triangles.len());
    nodes.push(BvhNode {
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::ZERO,
        first_triangle: 0,
        tri_count: 0,
    });
    build_bvh_recursive(&mut items_mut, 0, &mut nodes, &mut order, 0);
    Bvh {
        nodes,
        triangle_indices: order,
    }
}

pub(crate) fn intersect_bvh(ray: &Ray, scene: &Scene, bvh: &Bvh) -> Option<Hit> {
    let mut closest: Option<Hit> = None;
    let mut max_t = f32::INFINITY;

    // Fixed-size stack of node indices to visit. Depth 64 supports
    // BVHs up to 2^64 triangles — way past any scene we'll see. Using
    // a stack rather than recursion both avoids the call overhead and
    // makes the traversal order (near-child-first) explicit.
    let mut stack = [0u32; 64];
    let mut sp: usize = 1;
    stack[0] = 0;

    while sp > 0 {
        sp -= 1;
        let node = &bvh.nodes[stack[sp] as usize];
        if intersect_aabb(ray, node.bounds_min, node.bounds_max, max_t).is_none() {
            continue;
        }
        if node.tri_count > 0 {
            // Leaf: test each triangle.
            for i in 0..node.tri_count as usize {
                let tri_idx = bvh.triangle_indices[node.first_triangle as usize + i] as usize;
                let tri = &scene.triangles[tri_idx];
                if let Some((t, bary)) = intersect_triangle(ray, tri, max_t) {
                    max_t = t;
                    closest = Some(Hit {
                        t,
                        barycentric: bary,
                        triangle_index: tri_idx as u32,
                    });
                }
            }
        } else {
            // Internal: visit both children, near first. Pushing the
            // far child first means the near is popped first.
            let left = node.first_triangle;
            let right = left + 1;
            let ln = &bvh.nodes[left as usize];
            let rn = &bvh.nodes[right as usize];
            let lt = intersect_aabb(ray, ln.bounds_min, ln.bounds_max, max_t);
            let rt = intersect_aabb(ray, rn.bounds_min, rn.bounds_max, max_t);
            match (lt, rt) {
                (Some(ld), Some(rd)) => {
                    if ld < rd {
                        if sp < 63 {
                            stack[sp] = right;
                            sp += 1;
                        }
                        if sp < 63 {
                            stack[sp] = left;
                            sp += 1;
                        }
                    } else {
                        if sp < 63 {
                            stack[sp] = left;
                            sp += 1;
                        }
                        if sp < 63 {
                            stack[sp] = right;
                            sp += 1;
                        }
                    }
                }
                (Some(_), None) => {
                    if sp < 63 {
                        stack[sp] = left;
                        sp += 1;
                    }
                }
                (None, Some(_)) => {
                    if sp < 63 {
                        stack[sp] = right;
                        sp += 1;
                    }
                }
                (None, None) => {}
            }
        }
    }

    closest
}

// ============================================================
// Camera
// ============================================================

pub(crate) struct Camera {
    pub(crate) position: Vec3,
    pub(crate) forward: Vec3,
    pub(crate) right: Vec3,
    pub(crate) up: Vec3,
    pub(crate) fov_y_radians: f32,
    pub(crate) aspect: f32,
}

impl Camera {
    pub(crate) fn looking_at(position: Vec3, target: Vec3, up_hint: Vec3, fov_y_degrees: f32, aspect: f32) -> Self {
        let forward = (target - position).normalize();
        let right = forward.cross(up_hint).normalize();
        let up = right.cross(forward).normalize();
        Self {
            position,
            forward,
            right,
            up,
            fov_y_radians: fov_y_degrees.to_radians(),
            aspect,
        }
    }

    /// Sub-pixel-jittered ray. `jitter` is in [0,1)² and samples
    /// different points within the pixel for multi-sample AA. Phase 1
    /// always used (0.5, 0.5); Phase 2 lets the caller supply random
    /// offsets per sample.
    pub(crate) fn ray_for_pixel_jittered(
        &self,
        pixel: UVec2,
        image_size: UVec2,
        jitter: Vec2,
    ) -> Ray {
        let ndc_x = (2.0 * ((pixel.x as f32 + jitter.x) / image_size.x as f32) - 1.0) * self.aspect;
        let ndc_y = 1.0 - 2.0 * ((pixel.y as f32 + jitter.y) / image_size.y as f32);
        let scale = (self.fov_y_radians * 0.5).tan();
        let dir = self.forward + self.right * ndc_x * scale + self.up * ndc_y * scale;
        Ray::new(self.position, dir)
    }
}

// ============================================================
// RNG (PCG-like; per-pixel seeded so results are deterministic)
// ============================================================

/// Lightweight PCG-style RNG. One state per ray chain; advances
/// predictably so renders are reproducible given a fixed seed.
pub(crate) struct Rng {
    pub(crate) state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut s = Self { state: seed };
        // Warm up so the first value isn't degenerate.
        s.next_u32();
        s
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        // xoshiro256-ish mixed with a xorshift — not cryptographically
        // good, but passes statistical tests well enough for Monte
        // Carlo. The constants are the standard PCG multiplier + inc.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let xorshifted = (((self.state >> 18) ^ self.state) >> 27) as u32;
        let rot = (self.state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1).
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub(crate) fn next_vec2(&mut self) -> Vec2 {
        Vec2::new(self.next_f32(), self.next_f32())
    }
}

/// Seed derived from pixel coords + sample index + global seed so each
/// (pixel, sample) pair gets an independent RNG stream without needing
/// to allocate or synchronize anything. The 0x9E3779B97F4A7C15 golden-
/// ratio constant decorrelates nearby pixels.
pub(crate) fn seed_for(pixel: UVec2, sample: u32, global_seed: u64) -> u64 {
    let mut x = global_seed;
    x ^= (pixel.x as u64).wrapping_mul(0x9E3779B97F4A7C15);
    x ^= (pixel.y as u64).wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= (sample as u64).wrapping_mul(0x94D049BB133111EB);
    x
}

// ============================================================
// BRDF (GGX + Burley diffuse, metalness-aware)
// ============================================================

/// Schlick's Fresnel approximation. `f0` is reflectance at normal
/// incidence; `cos_theta` is dot(surface_normal, view_dir).
pub(crate) fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    f0 + (Vec3::ONE - f0) * (m * m * m * m * m)
}

/// GGX (Trowbridge-Reitz) normal distribution.
pub(crate) fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let nh2 = n_dot_h * n_dot_h;
    let denom = nh2 * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * denom * denom)
}

/// Smith visibility term for GGX with height-correlated formulation.
/// Closer to the ground truth than the separable G1*G2, and only
/// marginally more expensive.
pub(crate) fn v_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggx_v = n_dot_l * ((n_dot_v * (1.0 - a2) + a2) * n_dot_v).sqrt();
    let ggx_l = n_dot_v * ((n_dot_l * (1.0 - a2) + a2) * n_dot_l).sqrt();
    0.5 / (ggx_v + ggx_l + 1e-6)
}

/// Burley (Disney) diffuse term. Tracks the view-dependent darkening
/// near grazing angles for rough dielectrics that Lambert gets wrong.
pub(crate) fn burley_diffuse(n_dot_l: f32, n_dot_v: f32, l_dot_h: f32, roughness: f32) -> f32 {
    let fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * roughness;
    let light = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_l).powi(5);
    let view = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_v).powi(5);
    light * view / std::f32::consts::PI
}

#[derive(Clone, Copy)]
pub(crate) struct SurfaceSample {
    pub(crate) position: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) base_color: Vec3,
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    pub(crate) emissive: Vec3,
    /// Ambient/indirect lighting attenuation in [0, 1]. Applied only
    /// to throughput from BRDF-sampled bounces (indirect light); direct
    /// NEE samples compute their own visibility via shadow rays.
    pub(crate) occlusion: f32,
}

/// Build the per-pixel surface sample from a BVH hit. Interpolates
/// barycentric attributes, samples material textures, and perturbs the
/// shading normal via the tangent-space normal map (if present).
pub(crate) fn surface_from_hit(scene: &Scene, ray: &Ray, hit: &Hit) -> SurfaceSample {
    let tri = &scene.triangles[hit.triangle_index as usize];
    let u = hit.barycentric.x;
    let v = hit.barycentric.y;
    let w = 1.0 - u - v;

    let geom_normal = (tri.n0 * w + tri.n1 * u + tri.n2 * v).normalize_or_zero();
    let uv = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;
    let position = ray.origin + ray.direction * hit.t;

    let material = &scene.materials[tri.material_index as usize];
    let base_color = scene.sample_base_color(material, uv);
    let (metallic, roughness) = scene.sample_metallic_roughness(material, uv);
    let emissive = scene.sample_emissive(material, uv);
    let occlusion = scene.sample_occlusion(material, uv);

    // Build the per-hit TBN from the interpolated tangent+normal. The
    // bitangent sign comes from the glTF tangent.w (±1) — if the mesh
    // has no tangents (length 0), skip normal mapping entirely.
    let tangent_interp = tri.t0 * w + tri.t1 * u + tri.t2 * v;
    let tangent_xyz = Vec3::new(tangent_interp.x, tangent_interp.y, tangent_interp.z);
    let shading_normal = if material.normal_texture.is_some()
        && tangent_xyz.length_squared() > 1e-8
    {
        let t = tangent_xyz.normalize();
        // Re-orthogonalize the tangent against the normal (Gram-Schmidt)
        // so numerical drift from interpolation doesn't skew the basis.
        let t = (t - geom_normal * geom_normal.dot(t)).normalize_or_zero();
        let bitangent_sign = tangent_interp.w.signum().max(-1.0).min(1.0);
        let b = geom_normal.cross(t) * bitangent_sign;

        let n_tangent = scene.sample_tangent_normal(material, uv);
        // Compose tangent-space normal into world space.
        (t * n_tangent.x + b * n_tangent.y + geom_normal * n_tangent.z).normalize_or_zero()
    } else {
        geom_normal
    };

    SurfaceSample {
        position,
        normal: shading_normal,
        base_color,
        metallic,
        roughness: roughness.max(0.04), // clamp to avoid div-by-zero
        emissive,
        occlusion,
    }
}

/// Build an orthonormal basis (tangent, bitangent) around `n`.
/// Branchless method from Frisvad 2012 — more stable than cross(n, up)
/// when n is close to the up vector.
pub(crate) fn build_tbn(n: Vec3) -> (Vec3, Vec3) {
    if n.z < -0.9999999 {
        return (Vec3::new(0.0, -1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
    }
    let a = 1.0 / (1.0 + n.z);
    let b = -n.x * n.y * a;
    let t = Vec3::new(1.0 - n.x * n.x * a, b, -n.x);
    let bt = Vec3::new(b, 1.0 - n.y * n.y * a, -n.y);
    (t, bt)
}

/// Cosine-weighted hemisphere sample, used for diffuse direction
/// sampling. `rand` is two uniform samples in [0, 1)².
pub(crate) fn sample_cosine_hemisphere(rand: Vec2) -> Vec3 {
    let r = rand.x.sqrt();
    let theta = 2.0 * std::f32::consts::PI * rand.y;
    let x = r * theta.cos();
    let y = r * theta.sin();
    let z = (1.0 - rand.x).max(0.0).sqrt();
    Vec3::new(x, y, z)
}

/// GGX VNDF sample — Heitz 2018 "Sampling the GGX Distribution of
/// Visible Normals". Gives better low-variance estimates than sampling
/// the full distribution because we sample only the normals actually
/// visible from the view direction.
pub(crate) fn sample_ggx_vndf(view_tangent: Vec3, alpha: f32, rand: Vec2) -> Vec3 {
    // Transform view to hemisphere of ellipsoid.
    let vh = Vec3::new(alpha * view_tangent.x, alpha * view_tangent.y, view_tangent.z).normalize();
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 0.0 {
        Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t2 = vh.cross(t1);

    let r = rand.x.sqrt();
    let phi = 2.0 * std::f32::consts::PI * rand.y;
    let t1v = r * phi.cos();
    let mut t2v = r * phi.sin();
    let s = 0.5 * (1.0 + vh.z);
    t2v = (1.0 - s) * (1.0 - t1v * t1v).max(0.0).sqrt() + s * t2v;

    // Build the normal, transform back to world ellipsoid.
    let nh = t1v * t1 + t2v * t2 + (1.0 - t1v * t1v - t2v * t2v).max(0.0).sqrt() * vh;
    Vec3::new(alpha * nh.x, alpha * nh.y, nh.z.max(0.0)).normalize()
}

pub(crate) struct BrdfSample {
    pub(crate) direction_world: Vec3, // outgoing/light direction in world space
    /// Throughput multiplier for this bounce: BRDF * cos / PDF.
    /// Already includes the cosine, so the integrator just multiplies.
    pub(crate) throughput: Vec3,
    /// True when the sampled direction goes into the surface — the
    /// caller should terminate this ray (no transmissive support yet).
    pub(crate) terminated: bool,
}

/// Sample an outgoing direction from the BRDF at a surface point.
/// Uses a simple metal/dielectric split: metals are pure specular,
/// dielectrics pick diffuse vs specular by Fresnel-at-normal weight.
pub(crate) fn sample_brdf(surface: &SurfaceSample, view_world: Vec3, rng: &mut Rng) -> BrdfSample {
    let n = surface.normal;
    let alpha = surface.roughness * surface.roughness;
    let (t, bt) = build_tbn(n);

    // View in the tangent frame (z-up).
    let v_tangent = Vec3::new(view_world.dot(t), view_world.dot(bt), view_world.dot(n));

    // f0: dielectrics use 0.04; metals use the base color as f0.
    let f0 = Vec3::splat(0.04).lerp(surface.base_color, surface.metallic);

    // Decide diffuse vs specular lobe. Weighting by luminance of the
    // Fresnel-at-normal-incidence gives a reasonable importance
    // distribution without a second sample. Pure metals have ~zero
    // diffuse so this collapses naturally.
    let spec_weight = (f0.x + f0.y + f0.z) / 3.0;
    let diff_weight = (1.0 - spec_weight) * (1.0 - surface.metallic);
    let total = spec_weight + diff_weight + 1e-6;
    let p_spec = spec_weight / total;
    let pick_spec = rng.next_f32() < p_spec;

    let rand = rng.next_vec2();

    if pick_spec {
        // Sample a microfacet normal via VNDF, then reflect the view
        // direction across it.
        let h_tangent = sample_ggx_vndf(v_tangent, alpha, rand);
        let l_tangent = reflect(-v_tangent, h_tangent);
        if l_tangent.z <= 0.0 {
            return BrdfSample {
                direction_world: Vec3::Z,
                throughput: Vec3::ZERO,
                terminated: true,
            };
        }
        let _h_world = t * h_tangent.x + bt * h_tangent.y + n * h_tangent.z;
        let l_world = t * l_tangent.x + bt * l_tangent.y + n * l_tangent.z;
        let n_dot_l = l_tangent.z;
        let n_dot_v = v_tangent.z.max(1e-4);
        let n_dot_h = h_tangent.z.max(1e-4);
        let v_dot_h = v_tangent.dot(h_tangent).max(1e-4);

        // VNDF sampling PDF: D * G1 * max(0, V·H) / N·V.
        // The full BRDF is F * D * V_smith, so the throughput reduces
        // to F * G2/G1 * (V·H / (N·V * N·H))... but with height-
        // correlated V_smith the clean form is:
        //   throughput = F * G2_height_correlated / G1(V)
        // which we can write more stably as below.
        let f = fresnel_schlick(v_dot_h, f0);
        let g2 = v_smith(n_dot_v, n_dot_l, alpha) * 4.0 * n_dot_v * n_dot_l;
        // For the VNDF sampler the combined BRDF*cos/PDF simplifies
        // essentially to F * G2/G1. We approximate with the ratio of
        // the correlated V term to the monodir G1 — functionally
        // equivalent and numerically well-behaved.
        let g1_v = smith_g1(n_dot_v, alpha);
        let weight = if g1_v > 0.0 {
            f * g2 / (g1_v + 1e-6)
        } else {
            Vec3::ZERO
        };
        // Divide out the lobe-pick probability so the estimator is
        // unbiased.
        let _ = n_dot_h; // (fold into VNDF PDF accounting — kept for clarity)
        BrdfSample {
            direction_world: l_world,
            throughput: weight / p_spec,
            terminated: false,
        }
    } else {
        // Diffuse lobe: cosine-weighted hemisphere sample.
        let l_tangent = sample_cosine_hemisphere(rand);
        let l_world = t * l_tangent.x + bt * l_tangent.y + n * l_tangent.z;
        let n_dot_l = l_tangent.z.max(0.0);
        let n_dot_v = v_tangent.z.max(0.0);
        let h_tangent = (v_tangent + l_tangent).normalize_or_zero();
        let l_dot_h = l_tangent.dot(h_tangent).max(0.0);

        // Diffuse color is zero for metals; for dielectrics we scale
        // base color by (1 - Fresnel-at-normal) so total reflectance
        // stays ≤ 1.
        let diffuse_albedo = surface.base_color * (1.0 - surface.metallic) * (Vec3::ONE - f0);
        let fd = burley_diffuse(n_dot_l, n_dot_v, l_dot_h, surface.roughness);
        // BRDF · cos / PDF, with PDF = cos / pi, collapses to
        // BRDF · pi. Burley already divides by pi, so multiply here.
        let weight = diffuse_albedo * fd * std::f32::consts::PI;
        let p_diff = 1.0 - p_spec;
        BrdfSample {
            direction_world: l_world,
            throughput: weight / p_diff.max(1e-6),
            terminated: false,
        }
    }
}

pub(crate) fn reflect(incoming: Vec3, normal: Vec3) -> Vec3 {
    incoming - normal * (2.0 * incoming.dot(normal))
}

pub(crate) fn smith_g1(n_dot_x: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let inner = ((1.0 - a2) * n_dot_x * n_dot_x + a2).sqrt();
    2.0 * n_dot_x / (n_dot_x + inner + 1e-6)
}

// ============================================================
// Environment (HDR equirectangular map + procedural fallback)
// ============================================================

/// Equirectangular HDR environment map. Stored as linear-space RGB in
/// row-major order. Bilinear-sampled with wrap-around on U and clamp
/// on V (matching all major renderers' equirect conventions).
pub(crate) struct Environment {
    /// One Vec3 per pixel, linear-space.
    pub(crate) pixels: Vec<Vec3>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Multiplier applied on every sample — useful for scaling down
    /// bright HDR maps so the scene isn't blown out.
    pub(crate) intensity: f32,
    /// Piecewise-constant 2D distribution over the env map, built from
    /// pixel luminance × sin(theta) jacobian. Lets NEE draw samples
    /// toward bright lights (the sun, lamps) with low variance instead
    /// of firing uniformly into the hemisphere.
    pub(crate) distribution: EnvDistribution,
}

/// 2D inverse-CDF sampler over an equirectangular env map. The
/// marginal CDF picks a row; the per-row conditional CDF picks a
/// column within it. Classic PBRT Ch. 13 construction.
pub(crate) struct EnvDistribution {
    /// Marginal CDF over rows: `marginal[y+1]` is the cumulative
    /// weight of rows 0..=y. First entry is 0.
    pub(crate) marginal: Vec<f32>,
    /// Conditional CDFs over columns, one per row, concatenated.
    /// Row y occupies indices `y*(width+1) .. (y+1)*(width+1)`.
    pub(crate) conditional: Vec<f32>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// sum of all pixel weights (luminance * sin(theta)). Used to
    /// convert per-pixel probability to per-solid-angle PDF.
    pub(crate) total_weight: f32,
}

impl EnvDistribution {
    pub(crate) fn build(pixels: &[Vec3], width: u32, height: u32) -> Self {
        let w = width as usize;
        let h = height as usize;

        // Per-pixel weight = luminance × sin(theta). The sin factor
        // is the equirect jacobian — pixels near the poles cover less
        // solid angle and should be sampled proportionally less often.
        let mut pixel_weight = vec![0f32; w * h];
        for y in 0..h {
            let v = (y as f32 + 0.5) / h as f32;
            let theta = v * std::f32::consts::PI;
            let sin_theta = theta.sin();
            for x in 0..w {
                let p = pixels[y * w + x];
                let lum = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
                pixel_weight[y * w + x] = lum.max(0.0) * sin_theta;
            }
        }

        // Conditional CDF per row: prefix sum of pixel weights in
        // the row, normalized by the row's total weight.
        let mut conditional = vec![0f32; h * (w + 1)];
        let mut row_sum = vec![0f32; h];
        for y in 0..h {
            let base = y * (w + 1);
            conditional[base] = 0.0;
            for x in 0..w {
                conditional[base + x + 1] = conditional[base + x] + pixel_weight[y * w + x];
            }
            row_sum[y] = conditional[base + w];
            // Normalize so the final value is exactly 1. Skip rows
            // with zero total weight (e.g. pure-black band at a pole).
            if row_sum[y] > 0.0 {
                for x in 0..=w {
                    conditional[base + x] /= row_sum[y];
                }
            }
        }

        // Marginal CDF over rows: prefix sum of row sums, normalized.
        let mut marginal = vec![0f32; h + 1];
        for y in 0..h {
            marginal[y + 1] = marginal[y] + row_sum[y];
        }
        let total_weight = marginal[h];
        if total_weight > 0.0 {
            for y in 0..=h {
                marginal[y] /= total_weight;
            }
        }

        Self {
            marginal,
            conditional,
            width,
            height,
            total_weight,
        }
    }

    /// Inverse-CDF sample: map (u, v) in [0,1]² to a pixel plus the
    /// per-pixel selection probability. The caller converts pixel ↔
    /// direction and applies the solid-angle jacobian.
    pub(crate) fn sample_pixel(&self, u1: f32, u2: f32) -> (u32, u32, f32) {
        // Marginal → row.
        let y = upper_bound_index(&self.marginal, u1);
        let y = y.min(self.height as usize - 1);
        let row_base = y * (self.width as usize + 1);
        let row = &self.conditional[row_base..row_base + self.width as usize + 1];
        // Conditional → column.
        let x = upper_bound_index(row, u2).min(self.width as usize - 1);

        // p_pixel = (row weight / total) × (pixel weight / row weight)
        //         = pixel weight / total
        // But we only have CDFs, so reconstruct the pixel weight from
        // consecutive CDF deltas.
        let row_total = self.marginal[y + 1] - self.marginal[y];
        let px_p_given_row = row[x + 1] - row[x];
        let p_pixel = row_total * px_p_given_row;
        (x as u32, y as u32, p_pixel)
    }

    /// Probability (per pixel, NOT per solid angle) that this pixel
    /// would be chosen by `sample_pixel`. For MIS we need this in a
    /// form we can compare with the BRDF's PDF, which is per solid
    /// angle — the Environment wrapper handles the conversion.
    pub(crate) fn pixel_pdf(&self, x: u32, y: u32) -> f32 {
        if self.total_weight <= 0.0 {
            return 0.0;
        }
        let w = self.width as usize;
        let h = self.height as usize;
        if x as usize >= w || y as usize >= h {
            return 0.0;
        }
        // Reconstruct pixel weight / total.
        let row_base = y as usize * (w + 1);
        let px_p_given_row = self.conditional[row_base + x as usize + 1]
            - self.conditional[row_base + x as usize];
        let row_total = self.marginal[y as usize + 1] - self.marginal[y as usize];
        row_total * px_p_given_row
    }
}

/// Inverse-CDF inversion: returns the largest index i such that
/// `cdf[i] <= u`. Used for both marginal and conditional sampling.
pub(crate) fn upper_bound_index(cdf: &[f32], u: f32) -> usize {
    // Binary search for the last index whose CDF value is <= u.
    let mut lo = 0usize;
    let mut hi = cdf.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cdf[mid] <= u {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        0
    } else {
        lo - 1
    }
}

impl Environment {
    pub(crate) fn load_hdr(path: &Path, intensity: f32) -> Result<Self, String> {
        use image::ImageDecoder;
        use std::fs::File;
        use std::io::BufReader;
        let file = File::open(path).map_err(|e| format!("open {:?}: {e}", path))?;
        let decoder = image::codecs::hdr::HdrDecoder::new(BufReader::new(file))
            .map_err(|e| format!("hdr decode: {e}"))?;
        let (width, height) = decoder.dimensions();
        // Rgb32F — 3 f32s per pixel. Allocate the byte buffer and
        // reinterpret after decoding.
        let byte_len = (width as usize) * (height as usize) * 3 * 4;
        let mut buf = vec![0u8; byte_len];
        decoder
            .read_image(&mut buf)
            .map_err(|e| format!("hdr read: {e}"))?;
        let pixels: Vec<Vec3> = buf
            .chunks_exact(12)
            .map(|c| {
                let r = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let g = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                let b = f32::from_le_bytes([c[8], c[9], c[10], c[11]]);
                Vec3::new(r, g, b)
            })
            .collect();
        let distribution = EnvDistribution::build(&pixels, width, height);
        Ok(Self {
            pixels,
            width,
            height,
            intensity,
            distribution,
        })
    }

    /// Procedural fallback when no HDR is supplied. Warm horizon, cool
    /// zenith, faint sun — not physically correct, but better than a
    /// black background.
    pub(crate) fn procedural() -> Self {
        // Build a tiny 256×128 equirect from a procedural formula so
        // the same sampling path works whether or not the user
        // supplied an .hdr.
        let w = 256u32;
        let h = 128u32;
        let sun_dir = Vec3::new(0.4, 0.8, 0.3).normalize();
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let v = (y as f32 + 0.5) / h as f32;
                let theta = v * std::f32::consts::PI; // 0 at +Y, PI at -Y
                let phi = (u - 0.5) * 2.0 * std::f32::consts::PI;
                let dir = Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
                let horizon = Vec3::new(0.95, 0.85, 0.7);
                let zenith = Vec3::new(0.4, 0.55, 0.85);
                let sky = horizon.lerp(zenith, (0.5 * (dir.y + 1.0)).clamp(0.0, 1.0));
                let sun = dir.dot(sun_dir).max(0.0).powf(512.0) * 8.0;
                pixels.push(sky * 1.2 + Vec3::splat(sun));
            }
        }
        let distribution = EnvDistribution::build(&pixels, w, h);
        Self {
            pixels,
            width: w,
            height: h,
            intensity: 1.0,
            distribution,
        }
    }

    /// Sample the environment in a given direction. Uses the standard
    /// equirectangular convention: theta=0 at +Y (top), phi=0 at +Z,
    /// phi rotating toward +X. Bilinear-filtered.
    pub(crate) fn sample(&self, direction: Vec3) -> Vec3 {
        let d = direction.normalize_or_zero();
        let theta = d.y.clamp(-1.0, 1.0).acos();
        let phi = d.z.atan2(d.x);
        let u = (phi / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
        let v = (theta / std::f32::consts::PI).clamp(0.0, 1.0);

        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let w = self.width as i32;
        let h = self.height as i32;
        let fetch = |x: i32, y: i32| -> Vec3 {
            let xc = x.rem_euclid(w) as usize;
            let yc = y.clamp(0, h - 1) as usize;
            self.pixels[yc * self.width as usize + xc]
        };
        let c00 = fetch(x0, y0);
        let c10 = fetch(x0 + 1, y0);
        let c01 = fetch(x0, y0 + 1);
        let c11 = fetch(x0 + 1, y0 + 1);
        c00.lerp(c10, tx).lerp(c01.lerp(c11, tx), ty) * self.intensity
    }

    /// Convert a pixel (column, row) to a world-space direction. The
    /// sample lands at the pixel center. Returns the direction
    /// together with sin(theta) — callers need that for the PDF
    /// conversion between pixel and solid angle.
    pub(crate) fn pixel_to_direction(&self, x: u32, y: u32) -> (Vec3, f32) {
        let u = (x as f32 + 0.5) / self.width as f32;
        let v = (y as f32 + 0.5) / self.height as f32;
        let theta = v * std::f32::consts::PI;
        let phi = (u - 0.5) * 2.0 * std::f32::consts::PI;
        let sin_theta = theta.sin();
        let dir = Vec3::new(sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin());
        (dir.normalize(), sin_theta)
    }

    /// Inverse — project a direction to the pixel that would have
    /// produced it. Used in `pdf` to evaluate BRDF-sample directions
    /// against the environment distribution.
    pub(crate) fn direction_to_pixel(&self, direction: Vec3) -> (u32, u32, f32) {
        let d = direction.normalize_or_zero();
        let theta = d.y.clamp(-1.0, 1.0).acos();
        let phi = d.z.atan2(d.x);
        let u = (phi / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
        let v = (theta / std::f32::consts::PI).clamp(0.0, 1.0);
        let x = ((u * self.width as f32) as u32).min(self.width - 1);
        let y = ((v * self.height as f32) as u32).min(self.height - 1);
        let sin_theta = theta.sin();
        (x, y, sin_theta)
    }

    /// Draw a direction from the env's luminance distribution along
    /// with the per-solid-angle PDF of that choice. The PDF is what
    /// MIS needs when weighting this sample against a BRDF sample.
    pub(crate) fn sample_importance(&self, rng: &mut Rng) -> (Vec3, Vec3, f32) {
        let (x, y, pixel_p) = self
            .distribution
            .sample_pixel(rng.next_f32(), rng.next_f32());
        let (direction, sin_theta) = self.pixel_to_direction(x, y);
        let radiance = self.sample(direction);
        // Convert pixel probability → solid-angle PDF.
        //   pdf_omega = pixel_p / (omega_per_pixel)
        //   omega_per_pixel = (2π/W)(π/H)·sin(θ)
        let omega_per_pixel = (2.0 * std::f32::consts::PI / self.width as f32)
            * (std::f32::consts::PI / self.height as f32)
            * sin_theta.max(1e-6);
        let pdf = pixel_p / omega_per_pixel;
        (direction, radiance, pdf)
    }

    /// PDF (per solid angle) of the env distribution for a given
    /// direction. Evaluates the pixel the direction maps to and
    /// converts back to solid-angle measure.
    pub(crate) fn pdf(&self, direction: Vec3) -> f32 {
        let (x, y, sin_theta) = self.direction_to_pixel(direction);
        if sin_theta <= 0.0 {
            return 0.0;
        }
        let pixel_p = self.distribution.pixel_pdf(x, y);
        let omega_per_pixel = (2.0 * std::f32::consts::PI / self.width as f32)
            * (std::f32::consts::PI / self.height as f32)
            * sin_theta;
        pixel_p / omega_per_pixel
    }
}

// ============================================================
// Direct lights (glTF-style punctual lights, simplified)
// ============================================================

/// A directional "sun" light. Has no position, just a direction *from
/// the light toward the scene* and a radiant intensity. Modeled as a
/// delta function (infinitesimally small angular extent), which makes
/// shadows crisp and sampling trivial.
#[derive(Clone, Copy)]
pub(crate) struct SunLight {
    /// Direction the light is *coming from* (i.e., pointing TOWARD the
    /// sun from the surface). We keep the same convention as the
    /// Bloom realtime shader, so swapping the realtime light in and
    /// out doesn't require remapping conventions.
    pub(crate) direction_to_light: Vec3,
    pub(crate) color: Vec3,
    pub(crate) intensity: f32,
}

/// Test whether a point can see a given direction without being
/// occluded. Returns true if no triangle blocks the ray from
/// `origin` going toward `direction_to_light` within `max_distance`.
/// The origin should already be offset along the surface normal to
/// avoid self-intersection.
pub(crate) fn visible(ray_origin: Vec3, direction_to_light: Vec3, max_distance: f32, scene: &Scene, bvh: &Bvh) -> bool {
    let ray = Ray::new(ray_origin, direction_to_light);
    // We only need to know if ANY triangle is hit before max_distance;
    // a specialized "any-hit" traversal would be faster than
    // closest-hit, but the closest-hit cost is acceptable for Phase 3.
    match intersect_bvh(&ray, scene, bvh) {
        Some(hit) => hit.t >= max_distance,
        None => true,
    }
}

// ============================================================
// Path tracer
// ============================================================

pub(crate) struct Scenario<'a> {
    pub(crate) scene: &'a Scene,
    pub(crate) bvh: &'a Bvh,
    pub(crate) environment: &'a Environment,
    pub(crate) sun: Option<SunLight>,
}

/// Evaluate the BRDF at a surface for a given outgoing (toward-eye)
/// and incoming (toward-light) direction pair. Returns (brdf * N·L)
/// which is what the light-transport equation actually needs, plus
/// the PDF the BRDF sampler would have chosen for this incoming
/// direction — used by MIS to weight NEE vs BRDF samples.
pub(crate) fn evaluate_brdf(surface: &SurfaceSample, view: Vec3, light: Vec3) -> (Vec3, f32) {
    let n = surface.normal;
    let n_dot_v = n.dot(view).max(0.0);
    let n_dot_l = n.dot(light).max(0.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return (Vec3::ZERO, 0.0);
    }
    let h = (view + light).normalize_or_zero();
    let n_dot_h = n.dot(h).max(0.0);
    let v_dot_h = view.dot(h).max(0.0);

    let f0 = Vec3::splat(0.04).lerp(surface.base_color, surface.metallic);
    let alpha = surface.roughness * surface.roughness;

    // Specular: F * D * Vsmith. Vsmith is the height-correlated form
    // that already includes the 1/(4·N·V·N·L) term.
    let f = fresnel_schlick(v_dot_h, f0);
    let d = d_ggx(n_dot_h, alpha);
    let vsmith = v_smith(n_dot_v, n_dot_l, alpha);
    let specular = f * d * vsmith;

    // Diffuse: Burley (already 1/pi-normalized). Scale by (1 - F) and
    // (1 - metallic) for energy conservation.
    let fd = burley_diffuse(n_dot_l, n_dot_v, v_dot_h, surface.roughness);
    let diffuse_albedo = surface.base_color * (1.0 - surface.metallic) * (Vec3::ONE - f);
    let diffuse = diffuse_albedo * fd;

    let brdf_cos = (specular + diffuse) * n_dot_l;

    // Rough approximation of the BRDF sampler's PDF for MIS. Uses the
    // same spec/diff split heuristic as `sample_brdf`.
    let spec_weight = (f0.x + f0.y + f0.z) / 3.0;
    let diff_weight = (1.0 - spec_weight) * (1.0 - surface.metallic);
    let total = spec_weight + diff_weight + 1e-6;
    let p_spec = spec_weight / total;
    let p_diff = 1.0 - p_spec;

    // Spec PDF (GGX VNDF): D * G1(V) * max(0, V·H) / (4 * N·V * V·H).
    // We just approximate D·cos/(4·V·H) since we only need a
    // reasonable ratio for MIS — exact matching isn't required.
    let pdf_spec = d * n_dot_h / (4.0 * v_dot_h + 1e-6);
    let pdf_diff = n_dot_l / std::f32::consts::PI;
    let pdf = p_spec * pdf_spec + p_diff * pdf_diff;

    (brdf_cos, pdf.max(0.0))
}

/// Balance heuristic for MIS — weights a sample from strategy A by
/// p_a / (p_a + p_b). Standard in PBRT / Eric Veach's thesis.
pub(crate) fn mis_balance(pdf_this: f32, pdf_other: f32) -> f32 {
    pdf_this / (pdf_this + pdf_other + 1e-6)
}

pub(crate) fn trace_path(
    scenario: &Scenario,
    primary: Ray,
    max_bounces: u32,
    rng: &mut Rng,
) -> Vec3 {
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    let mut ray = primary;

    // Tracks the BRDF PDF used to generate the CURRENT ray — needed
    // to apply MIS when the ray escapes to the environment. On the
    // primary ray nothing sampled it via BRDF, so we mark it with a
    // sentinel of None (meaning "full weight, no MIS").
    let mut last_brdf_pdf: Option<f32> = None;

    for bounce in 0..max_bounces {
        let hit = match intersect_bvh(&ray, scenario.scene, scenario.bvh) {
            Some(h) => h,
            None => {
                // BRDF-sampled ray escaped into the environment. Weight
                // with MIS against the env-importance-sampler we'd use
                // at the previous hit. The primary ray gets full weight.
                let env_radiance = scenario.environment.sample(ray.direction);
                let weight = match last_brdf_pdf {
                    Some(brdf_pdf) => {
                        let env_pdf = scenario.environment.pdf(ray.direction);
                        mis_balance(brdf_pdf, env_pdf)
                    }
                    None => 1.0,
                };
                radiance += throughput * env_radiance * weight;
                break;
            }
        };

        let mut surface = surface_from_hit(scenario.scene, &ray, &hit);
        let view = -ray.direction;
        if surface.normal.dot(view) < 0.0 {
            surface.normal = -surface.normal;
        }

        // Emissive surfaces contribute directly. No NEE toward glTF
        // emissive surfaces yet — we treat them as diffuse light
        // that's only hit by BRDF paths. Phase 5 can add emissive
        // surface importance sampling if the reference needs it for
        // small area lights.
        radiance += throughput * surface.emissive;

        let shadow_origin = surface.position + surface.normal * 1e-4;

        // --- NEE A: delta sun light. MIS weight is 1.0 because no
        //     continuous sampler can hit a zero-extent light; the
        //     BRDF sampler cannot compete with a delta direction.
        if let Some(sun) = scenario.sun {
            let l = sun.direction_to_light;
            let n_dot_l = surface.normal.dot(l);
            if n_dot_l > 0.0
                && visible(shadow_origin, l, f32::INFINITY, scenario.scene, scenario.bvh)
            {
                let (brdf_cos, _pdf_brdf) = evaluate_brdf(&surface, view, l);
                radiance += throughput * brdf_cos * sun.color * sun.intensity;
            }
        }

        // --- NEE B: env-map importance sample. Pick a direction from
        //     the env luminance CDF, test visibility, add (brdf·cos·L)
        //     / env_pdf with the MIS balance weight vs the BRDF PDF.
        let (env_dir, env_radiance, env_pdf) =
            scenario.environment.sample_importance(rng);
        if env_pdf > 0.0 {
            let n_dot_l = surface.normal.dot(env_dir);
            if n_dot_l > 0.0
                && visible(
                    shadow_origin,
                    env_dir,
                    f32::INFINITY,
                    scenario.scene,
                    scenario.bvh,
                )
            {
                let (brdf_cos, brdf_pdf) = evaluate_brdf(&surface, view, env_dir);
                let mis_w = mis_balance(env_pdf, brdf_pdf);
                radiance += throughput * brdf_cos * env_radiance * mis_w / env_pdf;
            }
        }

        // BRDF sample for the next bounce. Throughput picks up the
        // BRDF/pdf scale here; the env radiance at the end applies
        // the MIS weight.
        let sample = sample_brdf(&surface, view, rng);
        if sample.terminated || !sample.throughput.is_finite() {
            break;
        }
        // Apply occlusion to indirect lighting only (per glTF spec).
        // Direct NEE samples already compute their own visibility and
        // don't need the statistical AO attenuation.
        let occluded_throughput = sample.throughput * surface.occlusion;
        throughput *= occluded_throughput;

        // Approximate the BRDF-sampler's PDF for the chosen direction
        // so the next environment miss can do MIS. The evaluate_brdf
        // helper produces the same PDF estimate sample_brdf would.
        let (_, approx_pdf) = evaluate_brdf(&surface, view, sample.direction_world);
        last_brdf_pdf = Some(approx_pdf.max(1e-6));

        // Russian roulette after bounce 1.
        if bounce > 1 {
            let p = throughput.max_element().clamp(0.05, 0.95);
            if rng.next_f32() > p {
                break;
            }
            throughput /= p;
        }

        ray = Ray::new(shadow_origin, sample.direction_world);
    }

    if !radiance.is_finite() {
        return Vec3::ZERO;
    }
    radiance.clamp(Vec3::ZERO, Vec3::splat(50.0))
}

