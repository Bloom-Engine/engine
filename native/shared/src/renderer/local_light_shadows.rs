//! Shadows from a few local (point) lights, for material-path receivers.
//!
//! A game's own material shaders (terrain, buildings) light themselves from
//! `view.point_lights`, and until this module nothing in that loop could be
//! shadowed: a lamp lit the ground through a wall and nothing standing in
//! front of it threw a shadow.
//!
//! What this is: up to [`MAX_LIGHTS`] point lights, chosen among those the game
//! submits with `add_point_light_with_shadow` (near the camera and in view
//! first, see [`select`]), each get a cube of six perspective depth maps
//! (layers of one array texture). The material ABI exposes the array and one
//! uniform block at group 1 bindings 13 and 14, and `common/local_shadows.wgsl`
//! gives shaders one function, `local_light_shadow(light_index, world_pos,
//! normal)`.
//!
//! What keeps it cheap: a lamp does not move, so everything static around it is
//! drawn into a second, cached array and kept. A caster the camera newly sees
//! is added to that drawing; nothing is redrawn because the camera turned
//! (see [`static_work`]). Per frame, a face is touched only while something
//! that moves (a skinned character, a mover) is inside it: the cached layer is
//! copied over the live one and the movers are drawn on top, exactly as the
//! sun's cascades do. A new light's six faces are spread over three frames and
//! its shadow then fades in, so nothing pops and no frame pays for a whole
//! cube.
//!
//! What it is not: the virtual-shadow-map local lights in
//! `virtual_shadows.rs` (`add_shadowed_point_light`). Those need `BLOOM_VSM=1`,
//! replace the sun's shadow path as well, are sampled only by the built-in
//! PBR shader, and are fail-closed (a light goes dark while a page of it is
//! being redrawn, which with an animated character in range is every frame).
//! This path is independent of them and fails OPEN: a light without a slot is
//! an ordinary unshadowed light.
//!
//! The engine's own scene shader (skinned and static models: characters,
//! props, weapons) samples the same maps through two bindings of its lighting
//! group; see [`scene_receivers`] and [`scene_shader`].
//!
//! Off by default and absent on mobile and web: see [`supported`].

use super::shadow_pass::ShadowDrawEntry;
use super::*;

/// Lights that can hold a shadow cube at once.
pub const MAX_LIGHTS: usize = 4;
const FACES: usize = 6;
const MAX_REQUESTS: usize = 256;
/// Per-frame draw slots in the uniform buffer (256 bytes each).
const DRAW_SLOTS: usize = 2048;
/// Static (cached) faces drawn per frame: a new light is ready in three frames.
const STATIC_FACES_PER_FRAME: usize = 2;
/// Draws spent per frame on adding newly seen static casters to cached faces.
const STATIC_ADDED_DRAWS_PER_FRAME: usize = 64;
/// Frames between two rebuilds of a cached face whose casters have gone.
const REBUILD_INTERVAL: u64 = 30;
/// Frames a new slot's shadow takes to reach full strength.
const FADE_FRAMES: f32 = 12.0;
/// Near plane of every face, in metres. Closer geometry (the lamp's own
/// housing, when the light sits inside it) does not cast.
const NEAR: f32 = 0.1;
/// A moving caster further from the light than this share of its range is
/// not drawn: the light has fallen to a few percent of its strength there and
/// the shadow could not be seen, while the caster (a 40,000-triangle pickup
/// that spins all day) would be redrawn for ever.
const MOVER_REACH: f32 = 0.75;
const DEFAULT_RESOLUTION: u32 = 512;
/// Extra field of view per face, in texels: the filter taps of a receiver at
/// a face's edge then still land on depth that was drawn for that face.
const GUARD_TEXELS: f32 = 2.5;

pub(crate) const MATERIAL_TEXTURE_BINDING: u32 = 13;
pub(crate) const MATERIAL_PARAMS_BINDING: u32 = 14;

/// The sampling half, substituted for the stub `common/local_shadows.wgsl`
/// when [`supported`].
pub(crate) const MATERIAL_HELPER: &str =
    include_str!("../../shaders/local_shadows/material_helper.wgsl");
pub(crate) const MATERIAL_HELPER_INCLUDE: &str = "common/local_shadows.wgsl";

/// Whether this build and device carry the two extra material bindings.
///
/// Decided once, before the material layouts are created. Desktop targets on
/// the high capability tier only: mobile and web fold bind groups and have
/// tight per-stage limits, and nothing here was run on them, so their layouts
/// and shader sources stay byte-for-byte what they were.
/// `BLOOM_LOCAL_LIGHT_SHADOWS=0` removes it on desktop too.
pub(crate) fn supported() -> bool {
    static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        cfg!(all(
            not(fold_scene_inputs),
            any(
                target_os = "macos",
                target_os = "windows",
                target_os = "linux"
            )
        )) && crate::virtual_shadows::capability_tier_is_high_end()
            && std::env::var("BLOOM_LOCAL_LIGHT_SHADOWS").map_or(true, |value| value.trim() != "0")
    })
}

pub(crate) fn material_layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: MATERIAL_TEXTURE_BINDING,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: MATERIAL_PARAMS_BINDING,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(PARAMS_BYTES),
            },
            count: None,
        },
    ]
}

/// A zeroed parameter block ("no light is shadowed") for bind groups built
/// before the renderer's own resources exist.
pub(crate) fn stub_params_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("local_light_shadow_params_stub"),
        size: PARAMS_BYTES,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    })
}

/// The two material bind-group entries, when [`supported`].
pub(crate) fn push_material_bind_entries<'a>(
    entries: &mut Vec<wgpu::BindGroupEntry<'a>>,
    maps: &'a wgpu::TextureView,
    params: &'a wgpu::Buffer,
) {
    if !supported() {
        return;
    }
    entries.push(wgpu::BindGroupEntry {
        binding: MATERIAL_TEXTURE_BINDING,
        resource: wgpu::BindingResource::TextureView(maps),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: MATERIAL_PARAMS_BINDING,
        resource: params.as_entire_binding(),
    });
}

/// The scene shader's lighting group (group 1) carries the same two resources
/// here. 10 to 12 are the froxel lists and 13 to 15 the virtual shadow maps.
pub(crate) const SCENE_TEXTURE_BINDING: u32 = 16;
pub(crate) const SCENE_PARAMS_BINDING: u32 = 17;

/// Sampled textures per fragment stage the device must grant before the scene
/// shader takes one more: the most any scene pipeline asks for without the
/// virtual shadow maps is 28 (layered materials with refraction, see
/// `layered_pbr_refraction.rs`), and every gate in front of those pipelines
/// counts exactly. With 30 granted (what the high tier requests) one more
/// fits under all of them without touching a single gate.
const SCENE_SAMPLED_TEXTURES_NEEDED: u32 = 30;

static SCENE_RECEIVERS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Decide, once per process and before the lighting layout is created,
/// whether the engine's own scene shader samples the local-light shadow maps.
///
/// As [`supported`], and also: not together with the virtual shadow maps
/// (`BLOOM_VSM=1`), which shadow their own local lights in the same loop and
/// with which the largest layered pipeline already sits at the limit of
/// sampled textures; and only where the device grants
/// [`SCENE_SAMPLED_TEXTURES_NEEDED`]. Where this is false the lighting layout,
/// its bind group and every scene shader source are what they were.
/// `BLOOM_LOCAL_LIGHT_SHADOWS=material` leaves the scene shader out and keeps
/// the material receivers, for comparing the two.
pub(crate) fn decide_scene_receivers(device: &wgpu::Device) -> bool {
    *SCENE_RECEIVERS.get_or_init(|| {
        supported()
            && !crate::virtual_shadows::virtual_shadows_requested()
            && device.limits().max_sampled_textures_per_shader_stage
                >= SCENE_SAMPLED_TEXTURES_NEEDED
            && std::env::var("BLOOM_LOCAL_LIGHT_SHADOWS")
                .map_or(true, |value| value.trim() != "material")
    })
}

/// What [`decide_scene_receivers`] decided; false before any renderer exists
/// (shader-source unit tests then see the canonical sources).
pub(crate) fn scene_receivers() -> bool {
    SCENE_RECEIVERS.get().copied().unwrap_or(false)
}

pub(crate) fn scene_layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    let mut entries = material_layout_entries();
    entries[0].binding = SCENE_TEXTURE_BINDING;
    entries[1].binding = SCENE_PARAMS_BINDING;
    entries
}

/// The two entries of the lighting bind group, when [`scene_receivers`].
pub(crate) fn scene_bind_entries(lights: &LocalLightShadows) -> Vec<wgpu::BindGroupEntry<'_>> {
    let Some((maps, params)) = lights.material_resources().filter(|_| scene_receivers()) else {
        return Vec::new();
    };
    vec![
        wgpu::BindGroupEntry {
            binding: SCENE_TEXTURE_BINDING,
            resource: wgpu::BindingResource::TextureView(maps),
        },
        wgpu::BindGroupEntry {
            binding: SCENE_PARAMS_BINDING,
            resource: params.as_entire_binding(),
        },
    ]
}

/// The point-light term every variant of the scene shader's light loop ends
/// in (the plain loop, the froxel one, the layered-material ones).
const SCENE_POINT_INTENSITY: &str = "pl.color.w * atten2";

/// The sampling helper as the scene shader needs it: the material helper with
/// the scene's lighting block and binding numbers. One source, so the two
/// receivers cannot drift apart.
fn scene_helper() -> String {
    MATERIAL_HELPER
        .replace("view.point_lights", "lighting.point_lights")
        .replace(
            "@group(1) @binding(13)",
            &format!("@group(1) @binding({SCENE_TEXTURE_BINDING})"),
        )
        .replace(
            "@group(1) @binding(14)",
            &format!("@group(1) @binding({SCENE_PARAMS_BINDING})"),
        )
}

/// The scene shader with its point lights shadowed: the helper is put in
/// front and every point light's intensity is multiplied by
/// `local_light_shadow`. `source` is the canonical shader or any variant of it
/// that still holds the light loop; the froxel variant names the light's index
/// in `lighting.point_lights` `light_index`, the plain loop `i`. A source
/// without the loop is returned as it is.
pub(crate) fn scene_shader(source: &str) -> String {
    if !source.contains(SCENE_POINT_INTENSITY) {
        return source.to_owned();
    }
    let light_index = if source.contains("let light_index = cluster_indices[") {
        "light_index"
    } else {
        "i"
    };
    let shadowed = format!(
        "{SCENE_POINT_INTENSITY} * local_light_shadow({light_index}, in.world_pos, in.normal)"
    );
    let mut output = scene_helper();
    output.push_str(&source.replace(SCENE_POINT_INTENSITY, &shadowed));
    output
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct SlotParams {
    face_vps: [[[f32; 4]; 4]; FACES],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct SamplingParams {
    words: [u32; 4],
    light_slot: [[u32; 4]; MAX_REQUESTS],
    slots: [SlotParams; MAX_LIGHTS],
}

const PARAMS_BYTES: u64 = std::mem::size_of::<SamplingParams>() as u64;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Request {
    pub light_index: u16,
    pub position: [f32; 3],
    pub range: f32,
}

/// Field-of-view scale of a face's projection: a little wider than 90 degrees.
fn projection_scale(resolution: u32) -> f32 {
    1.0 / (1.0 + 2.0 * GUARD_TEXELS / resolution as f32)
}

/// The six face view-projections of a light, in the order the shader's
/// `local_light_shadow_face` picks them: +X, -X, +Y, -Y, +Z, -Z. Right-handed,
/// depth 0 at the near plane and 1 at `range`.
fn face_vps(position: [f32; 3], range: f32, resolution: u32) -> [[[f32; 4]; 4]; FACES] {
    let near = NEAR.min(range * 0.5);
    let far = range.max(near + 1.0e-3);
    let reciprocal_depth = 1.0 / (near - far);
    let scale = projection_scale(resolution);
    let projection = [
        [scale, 0.0, 0.0, 0.0],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, far * reciprocal_depth, -1.0],
        [0.0, 0.0, near * far * reciprocal_depth, 0.0],
    ];
    let directions_and_up = [
        ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    ];
    std::array::from_fn(|face| {
        let (direction, up) = directions_and_up[face];
        let center = [
            position[0] + direction[0],
            position[1] + direction[1],
            position[2] + direction[2],
        ];
        mat4_multiply(projection, mat4_look_at(position, center, up))
    })
}

fn light_bounds(request: &Request) -> ([f32; 3], [f32; 3]) {
    let (p, r) = (request.position, request.range);
    (
        [p[0] - r, p[1] - r, p[2] - r],
        [p[0] + r, p[1] + r, p[2] + r],
    )
}

/// How much nearer (in metres of reach) a light in view counts than one out
/// of view, and one that holds a shadow cube than one that does not.
const OUT_OF_VIEW_PENALTY: f32 = 10.0;
const HELD_BONUS: f32 = 4.0;

/// The requests that get a shadow cube this frame, at most `budget`, each with
/// whether its reach touches the camera's view.
///
/// Nearest reach first, but a light out of view counts as ten metres further
/// and a light that already holds a cube (`held`) as four metres nearer. The
/// first keeps cubes for what is on screen; the second, with the first, keeps
/// a turning camera from handing a cube back and forth between two lamps,
/// each exchange costing six faces to draw. Ties end at the submission order.
fn select(
    requests: &[Request],
    camera: [f32; 3],
    camera_planes: &[[f32; 4]; 6],
    budget: usize,
    held: impl Fn(&Request) -> bool,
) -> Vec<(Request, bool)> {
    let mut ranked: Vec<(f32, Request, bool)> = requests
        .iter()
        .map(|request| {
            let (minimum, maximum) = light_bounds(request);
            let in_view = !crate::scene::aabb_outside_frustum(camera_planes, minimum, maximum);
            let d = [
                request.position[0] - camera[0],
                request.position[1] - camera[1],
                request.position[2] - camera[2],
            ];
            let mut rank = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - request.range;
            if !in_view {
                rank += OUT_OF_VIEW_PENALTY;
            }
            if held(request) {
                rank -= HELD_BONUS;
            }
            (rank, *request, in_view)
        })
        .collect();
    ranked.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then(left.1.light_index.cmp(&right.1.light_index))
    });
    ranked.truncate(budget);
    ranked
        .into_iter()
        .map(|(_, request, in_view)| (request, in_view))
        .collect()
}

fn same_light(slot: &Slot, request: &Request) -> bool {
    slot.range == request.range
        && (0..3).all(|axis| (slot.position[axis] - request.position[axis]).abs() < 0.01)
}

/// What to do with a face's cached drawing of the static casters this frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum StaticWork {
    Nothing,
    /// Clear and draw every static caster now in the face.
    Build,
    /// Draw the casters that are new on top of what is there. A depth map
    /// only ever gets nearer, so adding needs no clear.
    Add,
}

/// The cache is sticky on purpose. A game submits what its camera can see,
/// so the caster list changes whenever the camera turns; redrawing a face each
/// time would cost more than everything else here together, and would drop
/// the shadow of a caster the moment it left the screen. So: a caster that
/// appears is ADDED; one that disappears stays in the map unless its place is
/// in view (then it really is gone, not culled), and even then faces are
/// rebuilt one at a time, [`REBUILD_INTERVAL`] frames apart.
fn static_work(ready: bool, has_new: bool, gone_in_view: bool, rebuild_due: bool) -> StaticWork {
    if !ready || (gone_in_view && rebuild_due) {
        StaticWork::Build
    } else if has_new {
        StaticWork::Add
    } else {
        StaticWork::Nothing
    }
}

type Bounds = ([f32; 3], [f32; 3]);

struct Slot {
    position: [f32; 3],
    range: f32,
    face_vps: [[[f32; 4]; 4]; FACES],
    face_planes: [[[f32; 4]; 6]; FACES],
    /// The cached layer holds a drawing of the static casters.
    static_ready: [bool; FACES],
    /// The static casters in that drawing, by content signature, with the
    /// bounds they had.
    drawn: [std::collections::HashMap<u64, Bounds>; FACES],
    /// Movers were drawn into the live layer last time it was touched, so it
    /// must be refreshed from the cache even if none are left.
    live_stale: [bool; FACES],
    last_selected_frame: u64,
    /// The light's reach touches the camera's view this frame.
    in_view: bool,
    light_index: u16,
    strength: f32,
}

struct Maps {
    resolution: u32,
    lights: usize,
    live: wgpu::Texture,
    live_array_view: wgpu::TextureView,
    live_face_views: Vec<wgpu::TextureView>,
    cache: wgpu::Texture,
    cache_face_views: Vec<wgpu::TextureView>,
}

impl Maps {
    fn new(device: &wgpu::Device, resolution: u32, lights: usize) -> Self {
        let layers = (lights * FACES) as u32;
        let make = |label: &str, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage,
                view_formats: &[],
            })
        };
        let face_views = |texture: &wgpu::Texture| -> Vec<wgpu::TextureView> {
            (0..layers)
                .map(|layer| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("local_light_shadow_face"),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect()
        };
        let live = make(
            "local_light_shadow_maps",
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );
        let cache = make(
            "local_light_shadow_static_cache",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let live_array_view = live.create_view(&wgpu::TextureViewDescriptor {
            label: Some("local_light_shadow_maps"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            resolution,
            lights,
            live_face_views: face_views(&live),
            cache_face_views: face_views(&cache),
            live,
            live_array_view,
            cache,
        }
    }
}

struct Gpu {
    maps: Maps,
    params_buffer: wgpu::Buffer,
    draw_buffer: wgpu::Buffer,
    draw_bind_group: wgpu::BindGroup,
}

/// What the shadow pass hands over: this frame's casters and the three depth
/// pipelines they are drawn with.
pub(crate) struct PassInputs<'a> {
    pub queue: &'a wgpu::Queue,
    pub nodes: &'a [ShadowDrawEntry],
    pub vbs: &'a [&'a wgpu::Buffer],
    pub ibs: &'a [&'a wgpu::Buffer],
    pub cutout_bgs: &'a [&'a wgpu::BindGroup],
    /// Opaque and cutout casters per vertex format (vertex_formats.rs).
    pub pipeline: &'a super::vertex_formats::MeshPipelines,
    pub pipeline_cutout: &'a super::vertex_formats::MeshPipelines,
    pub pipeline_skinned: &'a wgpu::RenderPipeline,
    /// The zero joints/weights stream of 64-byte casters.
    pub skin_zero: &'a wgpu::Buffer,
    pub joint_bind_group: &'a wgpu::BindGroup,
    pub camera_pos: [f32; 3],
    pub camera_planes: [[f32; 4]; 6],
    pub wind: [f32; 4],
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameStats {
    pub requested: u32,
    pub shadowed: u32,
    pub static_faces: u32,
    pub added_faces: u32,
    pub dynamic_faces: u32,
    pub draws: u32,
}

pub(crate) struct LocalLightShadows {
    gpu: Option<Gpu>,
    budget: usize,
    requests: Vec<Request>,
    slots: [Option<Slot>; MAX_LIGHTS],
    frame: u64,
    next_rebuild_frame: u64,
    uploaded: Option<Box<SamplingParams>>,
    stats: FrameStats,
    log: bool,
    log_casters: bool,
    logged: FrameStats,
}

impl LocalLightShadows {
    /// `uniform_layout` is the shadow pass's per-draw uniform layout
    /// (`ShadowMap::uniform_layout`); the depth pipelines are bound against it.
    pub(crate) fn new(device: &wgpu::Device, uniform_layout: &wgpu::BindGroupLayout) -> Self {
        let gpu = supported().then(|| {
            let draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("local_light_shadow_draw_uniforms"),
                size: crate::shadows::SHADOW_UNIFORM_STRIDE as u64 * DRAW_SLOTS as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("local_light_shadow_draw_uniforms"),
                layout: uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &draw_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(std::mem::size_of::<
                            crate::shadows::ShadowUniforms,
                        >() as u64),
                    }),
                }],
            });
            Gpu {
                // One texel until a game turns the feature on: the bindings
                // exist, nothing is allocated for games that never ask.
                maps: Maps::new(device, 1, 1),
                params_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("local_light_shadow_params"),
                    size: PARAMS_BYTES,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                draw_buffer,
                draw_bind_group,
            }
        });
        Self {
            gpu,
            budget: 0,
            requests: Vec::new(),
            slots: std::array::from_fn(|_| None),
            frame: 0,
            next_rebuild_frame: 0,
            uploaded: None,
            stats: FrameStats::default(),
            // 1 = a line of counts whenever they change; 2 = also the moving
            // casters in each light's reach, every 240 frames.
            log: std::env::var("BLOOM_LOCAL_LIGHT_SHADOW_LOG").is_ok(),
            log_casters: std::env::var("BLOOM_LOCAL_LIGHT_SHADOW_LOG")
                .is_ok_and(|value| value == "2"),
            logged: FrameStats::default(),
        }
    }

    /// The array view and parameter buffer for the material bind groups.
    pub(crate) fn material_resources(&self) -> Option<(&wgpu::TextureView, &wgpu::Buffer)> {
        self.gpu
            .as_ref()
            .map(|gpu| (&gpu.maps.live_array_view, &gpu.params_buffer))
    }

    pub(crate) fn active(&self) -> bool {
        self.gpu.is_some() && self.budget > 0
    }

    /// Set how many lights may be shadowed (0 = off) and the face resolution.
    /// Returns true when the maps were reallocated: every bind group holding
    /// the old array view must then be rebuilt.
    pub(crate) fn configure(
        &mut self,
        device: &wgpu::Device,
        max_lights: usize,
        resolution: u32,
    ) -> bool {
        let Some(gpu) = self.gpu.as_mut() else {
            return false;
        };
        let budget = max_lights.min(MAX_LIGHTS);
        let resolution = if resolution == 0 {
            DEFAULT_RESOLUTION
        } else {
            resolution.clamp(128, 1024)
        };
        self.budget = budget;
        if budget == 0 {
            // Keep the maps: turning the feature back on is then free. The
            // next pass uploads "nothing is shadowed".
            return false;
        }
        if gpu.maps.resolution == resolution && gpu.maps.lights >= budget {
            return false;
        }
        gpu.maps = Maps::new(device, resolution, budget.max(gpu.maps.lights));
        self.slots = std::array::from_fn(|_| None);
        true
    }

    pub(crate) fn clear_requests(&mut self) {
        self.requests.clear();
    }

    pub(crate) fn request(&mut self, light_index: usize, position: [f32; 3], range: f32) {
        if !self.active()
            || light_index >= MAX_REQUESTS
            || self.requests.len() >= MAX_REQUESTS
            || !range.is_finite()
            || range <= 0.0
            || position.iter().any(|value| !value.is_finite())
        {
            return;
        }
        self.requests.push(Request {
            light_index: light_index as u16,
            position,
            range,
        });
    }

    /// Give every selected request a slot: its own from an earlier frame if it
    /// has one, else a free one, else the one unused for longest.
    fn assign_slots(&mut self, selected: &[(Request, bool)], resolution: u32) {
        let frame = self.frame;
        let mut pending = Vec::new();
        for (request, in_view) in selected {
            match self
                .slots
                .iter_mut()
                .flatten()
                .find(|slot| same_light(slot, request))
            {
                Some(slot) => {
                    if slot.last_selected_frame + 1 != frame {
                        // Back in view: whatever moved meanwhile is not in
                        // the live layers.
                        slot.live_stale = [true; FACES];
                    }
                    slot.last_selected_frame = frame;
                    slot.in_view = *in_view;
                    slot.light_index = request.light_index;
                }
                None => pending.push((*request, *in_view)),
            }
        }
        for (request, in_view) in pending {
            let index = self
                .slots
                .iter()
                .take(self.budget)
                .position(Option::is_none)
                .or_else(|| {
                    self.slots
                        .iter()
                        .take(self.budget)
                        .enumerate()
                        .filter_map(|(index, slot)| slot.as_ref().map(|slot| (index, slot)))
                        .filter(|(_, slot)| slot.last_selected_frame != frame)
                        .min_by_key(|(_, slot)| slot.last_selected_frame)
                        .map(|(index, _)| index)
                });
            let Some(index) = index else { continue };
            let vps = face_vps(request.position, request.range, resolution);
            self.slots[index] = Some(Slot {
                position: request.position,
                range: request.range,
                face_vps: vps,
                face_planes: std::array::from_fn(|face| {
                    crate::scene::extract_frustum_planes(&vps[face])
                }),
                static_ready: [false; FACES],
                drawn: std::array::from_fn(|_| std::collections::HashMap::new()),
                live_stale: [false; FACES],
                last_selected_frame: frame,
                in_view,
                light_index: request.light_index,
                strength: 0.0,
            });
        }
    }

    pub(crate) fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        inputs: &PassInputs<'_>,
    ) {
        if self.gpu.is_none() {
            return;
        }
        self.frame += 1;
        self.stats = FrameStats {
            requested: self.requests.len() as u32,
            ..FrameStats::default()
        };
        let resolution = self.gpu.as_ref().map_or(1, |gpu| gpu.maps.resolution);
        let selected = if self.budget > 0 {
            let slots = &self.slots;
            select(
                &self.requests,
                inputs.camera_pos,
                &inputs.camera_planes,
                self.budget,
                |request| slots.iter().flatten().any(|slot| same_light(slot, request)),
            )
        } else {
            Vec::new()
        };
        if !selected.is_empty() {
            profiler.begin("local_light_shadows");
            self.assign_slots(&selected, resolution);
            self.draw_faces(encoder, profiler, inputs);
            profiler.end("local_light_shadows");
        }
        self.upload_params(inputs.queue, resolution);
        if self.log {
            // Totals over 120 frames, so work that happens on few frames
            // (a static face redrawn) cannot hide between samples.
            self.logged.static_faces += self.stats.static_faces;
            self.logged.added_faces += self.stats.added_faces;
            self.logged.dynamic_faces += self.stats.dynamic_faces;
            self.logged.draws += self.stats.draws;
            if self.frame % 120 == 0 {
                eprintln!(
                    "bloom: local light shadows, 120 frames: {} requested, {} shadowed now; {} faces built, {} added to, {} with movers redrawn, {} draws",
                    self.stats.requested,
                    self.stats.shadowed,
                    self.logged.static_faces,
                    self.logged.added_faces,
                    self.logged.dynamic_faces,
                    self.logged.draws,
                );
                self.logged = FrameStats::default();
            }
        }
    }

    /// For a frame without a shadow pass (sun shadows switched off): no light
    /// is shadowed, and the shaders are told so rather than left with the
    /// last frame's slots.
    pub(crate) fn skip_frame(&mut self, queue: &wgpu::Queue) {
        if self.gpu.is_none() {
            return;
        }
        self.frame += 1;
        self.stats = FrameStats::default();
        self.upload_params(queue, 1);
    }

    fn draw_faces(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::Profiler,
        inputs: &PassInputs<'_>,
    ) {
        let frame = self.frame;
        let gpu = self.gpu.as_ref().expect("checked by record");
        let stride = crate::shadows::SHADOW_UNIFORM_STRIDE as usize;
        let draw_cursor = std::cell::Cell::new(0usize);
        let mut build_budget = STATIC_FACES_PER_FRAME;
        let mut added_draws = 0usize;
        let mut next_rebuild_frame = self.next_rebuild_frame;
        let mut near: Vec<usize> = Vec::new();
        let mut static_entries: Vec<usize> = Vec::new();
        let mut new_entries: Vec<usize> = Vec::new();
        let mut current: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut dynamic_entries: Vec<usize> = Vec::new();
        let mut uniform_data: Vec<u8> = Vec::new();

        // Draw `entries` into one face. False (nothing recorded) when the
        // frame's uniform slots are used up.
        let mut draw = |encoder: &mut wgpu::CommandEncoder,
                        profiler: &mut crate::profiler::Profiler,
                        view: &wgpu::TextureView,
                        clear: bool,
                        face_vp: [[f32; 4]; 4],
                        entries: &[usize],
                        stats: &mut FrameStats|
         -> bool {
            if draw_cursor.get() + entries.len() > DRAW_SLOTS {
                return false;
            }
            uniform_data.clear();
            uniform_data.resize(stride * entries.len(), 0);
            for (slot, &entry_index) in entries.iter().enumerate() {
                let entry = &inputs.nodes[entry_index];
                let uniforms = crate::shadows::ShadowUniforms {
                    light_vp: face_vp,
                    model: entry.transform,
                    misc: [entry.joint_offset, 0.0, entry.foliage, 0.0],
                    wind: inputs.wind,
                };
                let offset = slot * stride;
                uniform_data
                    [offset..offset + std::mem::size_of::<crate::shadows::ShadowUniforms>()]
                    .copy_from_slice(bytemuck::bytes_of(&uniforms));
            }
            let base = draw_cursor.get() * stride;
            if !entries.is_empty() {
                inputs
                    .queue
                    .write_buffer(&gpu.draw_buffer, base as u64, &uniform_data);
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("local_light_shadow_face"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: if clear {
                            wgpu::LoadOp::Clear(1.0)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: profiler.pass_timestamp_writes("local_light_shadows"),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut current_kind = None;
            for (slot, &entry_index) in entries.iter().enumerate() {
                let entry = &inputs.nodes[entry_index];
                let kind = if entry.skinned {
                    2
                } else if entry.cutout_idx >= 0 {
                    1
                } else {
                    0
                };
                if current_kind != Some((kind, entry.layout)) {
                    pass.set_pipeline(match kind {
                        1 => inputs.pipeline_cutout.get(entry.layout),
                        2 => inputs.pipeline_skinned,
                        _ => inputs.pipeline.get(entry.layout),
                    });
                    current_kind = Some((kind, entry.layout));
                }
                pass.set_bind_group(0, &gpu.draw_bind_group, &[(base + slot * stride) as u32]);
                if kind == 1 {
                    pass.set_bind_group(1, inputs.cutout_bgs[entry.cutout_idx as usize], &[]);
                } else if kind == 2 {
                    pass.set_bind_group(1, inputs.joint_bind_group, &[]);
                }
                pass.set_vertex_buffer(0, inputs.vbs[entry.vb_idx].slice(..));
                super::vertex_formats::bind_skin_zero(&mut pass, entry.layout, 1, inputs.skin_zero);
                pass.set_index_buffer(
                    inputs.ibs[entry.ib_idx].slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(
                    entry.index_start..entry.index_start + entry.index_count,
                    entry.base_vertex,
                    0..1,
                );
            }
            draw_cursor.set(draw_cursor.get() + entries.len());
            stats.draws += entries.len() as u32;
            true
        };

        for (slot_index, slot) in self.slots.iter_mut().enumerate() {
            let Some(slot) = slot.as_mut() else { continue };
            if slot.last_selected_frame != frame {
                continue;
            }
            // Casters within the light's reach; a caster without bounds
            // (min > max) cannot be ruled out and is kept.
            near.clear();
            near.extend(
                inputs
                    .nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(index, entry)| {
                        if entry.wmin[0] > entry.wmax[0] {
                            return Some(index);
                        }
                        // Squared distance from the light to the caster's box.
                        let mut distance_sq = 0.0f32;
                        for axis in 0..3 {
                            let outside = (entry.wmin[axis] - slot.position[axis])
                                .max(slot.position[axis] - entry.wmax[axis])
                                .max(0.0);
                            distance_sq += outside * outside;
                        }
                        let reach = if entry.dynamic {
                            slot.range * MOVER_REACH
                        } else {
                            slot.range
                        };
                        (distance_sq <= reach * reach).then_some(index)
                    }),
            );
            if self.log_casters && frame % 240 == 0 {
                // What moves in the light's reach, and what sits within a
                // metre of the light itself (a fixture that swallows it).
                for &index in &near {
                    let entry = &inputs.nodes[index];
                    let mut distance_sq = 0.0f32;
                    for axis in 0..3 {
                        let outside = (entry.wmin[axis] - slot.position[axis])
                            .max(slot.position[axis] - entry.wmax[axis])
                            .max(0.0);
                        distance_sq += outside * outside;
                    }
                    if entry.dynamic || distance_sq < 1.0 {
                        eprintln!(
                            "bloom: local light at {:?}: caster {:.2} m away, min {:?} max {:?}, {} indices, moving {}, skinned {}",
                            slot.position,
                            distance_sq.sqrt(),
                            entry.wmin,
                            entry.wmax,
                            entry.index_count,
                            entry.dynamic,
                            entry.skinned,
                        );
                    }
                }
            }
            for face in 0..FACES {
                let layer = slot_index * FACES + face;
                let planes = &slot.face_planes[face];
                static_entries.clear();
                new_entries.clear();
                dynamic_entries.clear();
                current.clear();
                for &index in &near {
                    let entry = &inputs.nodes[index];
                    if entry.wmin[0] <= entry.wmax[0]
                        && crate::scene::aabb_outside_frustum(planes, entry.wmin, entry.wmax)
                    {
                        continue;
                    }
                    if entry.dynamic {
                        dynamic_entries.push(index);
                    } else {
                        static_entries.push(index);
                        current.insert(entry.sig);
                        if !slot.drawn[face].contains_key(&entry.sig) {
                            new_entries.push(index);
                        }
                    }
                }
                // A drawn caster that is no longer submitted although its
                // place is on screen has really gone (see `static_work`).
                let gone_in_view = slot.drawn[face].len() + new_entries.len() > current.len()
                    && slot.drawn[face].iter().any(|(signature, bounds)| {
                        !current.contains(signature)
                            && bounds.0[0] <= bounds.1[0]
                            && !crate::scene::aabb_outside_frustum(
                                &inputs.camera_planes,
                                bounds.0,
                                bounds.1,
                            )
                    });
                let mut refreshed = false;
                match static_work(
                    slot.static_ready[face],
                    !new_entries.is_empty(),
                    gone_in_view,
                    frame >= next_rebuild_frame,
                ) {
                    StaticWork::Build
                        if build_budget > 0
                            && draw(
                                encoder,
                                profiler,
                                &gpu.maps.cache_face_views[layer],
                                true,
                                slot.face_vps[face],
                                &static_entries,
                                &mut self.stats,
                            ) =>
                    {
                        build_budget -= 1;
                        if slot.static_ready[face] {
                            next_rebuild_frame = frame + REBUILD_INTERVAL;
                        }
                        slot.static_ready[face] = true;
                        slot.drawn[face].clear();
                        for &index in &static_entries {
                            let entry = &inputs.nodes[index];
                            slot.drawn[face].insert(entry.sig, (entry.wmin, entry.wmax));
                        }
                        self.stats.static_faces += 1;
                        refreshed = true;
                    }
                    StaticWork::Add
                        if added_draws < STATIC_ADDED_DRAWS_PER_FRAME
                            && draw(
                                encoder,
                                profiler,
                                &gpu.maps.cache_face_views[layer],
                                false,
                                slot.face_vps[face],
                                &new_entries,
                                &mut self.stats,
                            ) =>
                    {
                        added_draws += new_entries.len();
                        for &index in &new_entries {
                            let entry = &inputs.nodes[index];
                            slot.drawn[face].insert(entry.sig, (entry.wmin, entry.wmax));
                        }
                        self.stats.added_faces += 1;
                        refreshed = true;
                    }
                    _ => {}
                }
                if !slot.static_ready[face] {
                    continue;
                }
                if !slot.in_view {
                    // Nothing this light reaches is on screen: keep its
                    // cache warm (above) and leave the movers for its return.
                    slot.live_stale[face] = true;
                    continue;
                }
                if !(refreshed || slot.live_stale[face] || !dynamic_entries.is_empty()) {
                    continue;
                }
                // Rigid movers alone (a spinning pickup) are redrawn every
                // other frame; a face with a skinned character in it, every
                // frame. Faces alternate so the work is spread evenly.
                if !refreshed
                    && !dynamic_entries.is_empty()
                    && (frame + layer as u64) % 2 == 1
                    && !dynamic_entries
                        .iter()
                        .any(|&index| inputs.nodes[index].skinned)
                {
                    continue;
                }
                if draw_cursor.get() + dynamic_entries.len() > DRAW_SLOTS {
                    // Out of slots: the live layer keeps last frame's movers.
                    continue;
                }
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu.maps.cache,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu.maps.live,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: gpu.maps.resolution,
                        height: gpu.maps.resolution,
                        depth_or_array_layers: 1,
                    },
                );
                slot.live_stale[face] = !dynamic_entries.is_empty();
                if !dynamic_entries.is_empty() {
                    draw(
                        encoder,
                        profiler,
                        &gpu.maps.live_face_views[layer],
                        false,
                        slot.face_vps[face],
                        &dynamic_entries,
                        &mut self.stats,
                    );
                    self.stats.dynamic_faces += 1;
                }
            }
            if slot.static_ready.iter().all(|ready| *ready) {
                slot.strength = (slot.strength + 1.0 / FADE_FRAMES).min(1.0);
            }
        }
        self.next_rebuild_frame = next_rebuild_frame;
    }

    fn upload_params(&mut self, queue: &wgpu::Queue, resolution: u32) {
        if self.budget == 0
            && self
                .uploaded
                .as_ref()
                .is_some_and(|params| params.words[0] == 0)
        {
            // Switched off and the shaders already know: nothing to do, for
            // every game that never asks for this.
            return;
        }
        let mut params = Box::new(<SamplingParams as bytemuck::Zeroable>::zeroed());
        if self.budget > 0 {
            for (slot_index, slot) in self.slots.iter().enumerate() {
                let Some(slot) = slot else { continue };
                if slot.last_selected_frame != self.frame || !slot.in_view || slot.strength <= 0.0 {
                    continue;
                }
                params.words[0] += 1;
                params.light_slot[slot.light_index as usize][0] = slot_index as u32 + 1;
                params.slots[slot_index] = SlotParams {
                    face_vps: slot.face_vps,
                    params: [
                        2.0 / (resolution as f32 * projection_scale(resolution)),
                        slot.strength,
                        (slot_index * FACES) as f32,
                        1.0 / resolution as f32,
                    ],
                };
            }
        }
        self.stats.shadowed = params.words[0];
        if self.uploaded.as_deref() == Some(&*params) {
            return;
        }
        if let Some(gpu) = self.gpu.as_ref() {
            queue.write_buffer(&gpu.params_buffer, 0, bytemuck::bytes_of(&*params));
        }
        self.uploaded = Some(params);
    }
}

impl Renderer {
    /// Turn shadows from local lights on (`max_lights` 1 to 4) or off (0), with
    /// `resolution` texels per cube face (0 = 512; 128 to 1024). False when
    /// this build or device does not carry the feature: nothing changes then.
    pub fn set_local_light_shadows(&mut self, max_lights: u32, resolution: u32) -> bool {
        let lights = &mut self.shadow_map.local_lights;
        if lights.configure(&self.device, max_lights as usize, resolution) {
            // The material per-view bind groups hold the old array view.
            self.material_per_view_bg_live = false;
            for bind_group in self.planar_probe_view_bgs.iter_mut() {
                *bind_group = None;
            }
            // So does the scene shader's lighting bind group.
            if scene_receivers() {
                if self.lighting_bg_is_procedural {
                    self.swap_lighting_bg_to_procedural();
                } else {
                    self.swap_lighting_bg_to_panorama();
                }
            }
        }
        self.shadow_map.local_lights.active()
    }

    /// Whether the engine's own lit models (everything the scene shader
    /// draws) receive these shadows too, not only material-path receivers.
    pub fn local_light_shadows_on_scene_models(&self) -> bool {
        scene_receivers()
    }

    /// `add_point_light`, plus a request that this light cast shadows on
    /// material-path receivers and on the engine's own lit models. Never fewer lights than `add_point_light`
    /// would give: without a shadow cube to spare, or with the feature off,
    /// the light is simply unshadowed. True when the light was submitted and
    /// shadows from local lights are switched on (it may still be one of the
    /// lights left without a cube this frame).
    #[allow(clippy::too_many_arguments)]
    pub fn add_point_light_with_shadow(
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
        self.add_point_light(x, y, z, range, r, g, b, intensity);
        if self.lighting_uniforms.point_light_count[0] as usize == index {
            return false;
        }
        self.shadow_map
            .local_lights
            .request(index, [x, y, z], range);
        self.shadow_map.local_lights.active()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_planes() -> [[f32; 4]; 6] {
        [
            [1.0, 0.0, 0.0, 100.0],
            [-1.0, 0.0, 0.0, 100.0],
            [0.0, 1.0, 0.0, 100.0],
            [0.0, -1.0, 0.0, 100.0],
            [0.0, 0.0, 1.0, 100.0],
            [0.0, 0.0, -1.0, 100.0],
        ]
    }

    fn request(index: u16, x: f32) -> Request {
        Request {
            light_index: index,
            position: [x, 2.0, 0.0],
            range: 12.0,
        }
    }

    fn indices(chosen: &[(Request, bool)]) -> Vec<u16> {
        chosen
            .iter()
            .map(|(request, _)| request.light_index)
            .collect()
    }

    #[test]
    fn nearest_lights_win_the_budget_and_ties_keep_submission_order() {
        let requests = [
            request(3, 40.0),
            request(1, 5.0),
            request(2, 5.0),
            request(0, 90.0),
        ];
        let chosen = select(&requests, [0.0, 2.0, 0.0], &open_planes(), 3, |_| false);
        assert_eq!(indices(&chosen), [1, 2, 3]);
        assert!(chosen.iter().all(|(_, in_view)| *in_view));
    }

    #[test]
    fn a_light_in_view_is_preferred_to_a_nearer_one_out_of_view() {
        let mut planes = open_planes();
        // Only x >= 30 is in view.
        planes[0] = [1.0, 0.0, 0.0, -30.0];
        // From x = 20: light 0 is 20 m away and out of view, light 1 is 25 m
        // away and in view.
        let requests = [request(0, 0.0), request(1, 45.0)];
        let chosen = select(&requests, [20.0, 2.0, 0.0], &planes, 1, |_| false);
        assert_eq!(indices(&chosen), [1]);
        // With room for both, the one out of view keeps a cube and is flagged.
        let chosen = select(&requests, [20.0, 2.0, 0.0], &planes, 2, |_| false);
        assert_eq!(indices(&chosen), [1, 0]);
        assert_eq!(chosen[1].1, false);
    }

    #[test]
    fn a_light_that_holds_a_cube_is_not_displaced_by_one_slightly_nearer() {
        // Light 0 is 22 m away, light 1 is 20 m away; one cube.
        let requests = [request(0, 22.0), request(1, -20.0)];
        let camera = [0.0, 2.0, 0.0];
        let fresh = select(&requests, camera, &open_planes(), 1, |_| false);
        assert_eq!(indices(&fresh), [1]);
        let holding = select(&requests, camera, &open_planes(), 1, |request| {
            request.light_index == 0
        });
        assert_eq!(indices(&holding), [0]);
    }

    fn ndc(vp: &[[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
        let clip = mat4_mul_vec4(vp, &[point[0], point[1], point[2], 1.0]);
        [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
    }

    /// The shader's face choice (major axis, in +X -X +Y -Y +Z -Z order).
    fn shader_face(direction: [f32; 3]) -> usize {
        let a = direction.map(f32::abs);
        if a[0] >= a[1] && a[0] >= a[2] {
            usize::from(direction[0] < 0.0)
        } else if a[1] >= a[2] {
            2 + usize::from(direction[1] < 0.0)
        } else {
            4 + usize::from(direction[2] < 0.0)
        }
    }

    #[test]
    fn every_direction_lands_inside_the_face_the_shader_picks() {
        let position = [3.0, 2.0, -7.0];
        let resolution = 512;
        let vps = face_vps(position, 12.0, resolution);
        let guard = 1.0 - projection_scale(resolution);
        let mut checked = 0;
        for xi in -4..=4 {
            for yi in -4..=4 {
                for zi in -4..=4 {
                    let direction = [xi as f32, yi as f32, zi as f32];
                    if direction == [0.0; 3] {
                        continue;
                    }
                    let point = [
                        position[0] + direction[0],
                        position[1] + direction[1],
                        position[2] + direction[2],
                    ];
                    let projected = ndc(&vps[shader_face(direction)], point);
                    // Inside the face, and never in the guard band.
                    assert!(projected[0].abs() <= 1.0 - guard + 1.0e-4, "{direction:?}");
                    assert!(projected[1].abs() <= 1.0 - guard + 1.0e-4, "{direction:?}");
                    assert!((0.0..=1.0).contains(&projected[2]), "{direction:?}");
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 9 * 9 * 9 - 1);
    }

    #[test]
    fn depth_runs_from_the_near_plane_to_the_range() {
        let vps = face_vps([0.0; 3], 12.0, 256);
        assert!(ndc(&vps[0], [NEAR, 0.0, 0.0])[2].abs() < 1.0e-4);
        assert!((ndc(&vps[0], [12.0, 0.0, 0.0])[2] - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn the_static_cache_adds_what_appears_and_rebuilds_only_for_what_left_in_view() {
        // Nothing drawn yet: build, whatever else is true.
        assert_eq!(static_work(false, false, false, false), StaticWork::Build);
        // A caster came into the camera's view: add it, keep the rest.
        assert_eq!(static_work(true, true, false, true), StaticWork::Add);
        // A caster left the list but its place is off screen: it was culled
        // by the game, not removed. Keep its shadow.
        assert_eq!(static_work(true, false, false, true), StaticWork::Nothing);
        // It left the list and its place is on screen: rebuild, when due.
        assert_eq!(static_work(true, false, true, true), StaticWork::Build);
        assert_eq!(static_work(true, false, true, false), StaticWork::Nothing);
        // Not due yet, but something new arrived: still add it meanwhile.
        assert_eq!(static_work(true, true, true, false), StaticWork::Add);
    }

    #[test]
    fn the_scene_variant_shadows_both_light_loops_and_still_parses() {
        let canonical = crate::renderer::shaders::SCENE_SHADER;
        let clustered = crate::renderer::froxel::clustered_scene_shader(canonical);
        for (source, index) in [(canonical, "i"), (clustered.as_str(), "light_index")] {
            let variant = scene_shader(source);
            let call = format!(
                "pl.color.w * atten2 * local_light_shadow({index}, in.world_pos, in.normal)"
            );
            assert_eq!(variant.matches(&call).count(), 1, "{index}");
            assert_eq!(variant.matches(SCENE_POINT_INTENSITY).count(), 1);
            assert!(variant.contains("@group(1) @binding(16) var local_shadow_maps"));
            assert!(variant.contains("@group(1) @binding(17) var<uniform> local_shadows"));
            assert!(!variant.contains("view.point_lights"));
            assert!(!variant.contains("@binding(13)") || source.contains("@binding(13)"));
            wgpu::naga::front::wgsl::parse_str(&variant)
                .unwrap_or_else(|error| panic!("{index}: {}", error.emit_to_string(&variant)));
        }
        // Nothing to shadow, nothing changed.
        assert_eq!(scene_shader("fn f() {}"), "fn f() {}");
        // The canonical source never carries the feature.
        assert!(!canonical.contains("local_light_shadow"));
    }

    #[test]
    fn the_scene_bindings_are_free_in_the_lighting_group() {
        // 0 to 9: lighting, 10 to 12: froxels, 13 to 15: virtual shadow maps.
        assert!(SCENE_TEXTURE_BINDING > 15 && SCENE_PARAMS_BINDING > SCENE_TEXTURE_BINDING);
        let entries = scene_layout_entries();
        assert_eq!(entries[0].binding, SCENE_TEXTURE_BINDING);
        assert_eq!(entries[1].binding, SCENE_PARAMS_BINDING);
    }

    #[test]
    fn the_parameter_block_matches_the_wgsl_layout() {
        // words 16 + light_slot 256 * 16 + 4 slots of (6 * 64 + 16).
        assert_eq!(PARAMS_BYTES, 16 + 4096 + 4 * 400);
        assert!(MATERIAL_HELPER.contains("array<vec4<u32>, 256>"));
        assert!(MATERIAL_HELPER.contains("array<LocalLightShadowSlot, 4>"));
        assert!(MATERIAL_HELPER.contains("@group(1) @binding(13)"));
        assert!(MATERIAL_HELPER.contains("@group(1) @binding(14)"));
    }
}
