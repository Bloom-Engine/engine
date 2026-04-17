"""
Cycles path-traced reference render of the Bloom Bistro scene.

Run with:
    blender -b -P render.py -- [--samples N] [--out PATH] [--device CPU|METAL]

This script is designed to be run headlessly from Blender (5.0+) and produce
a "ground truth" image that can be visually diffed against Bloom's realtime
output. The scene parameters (camera, sun, HDR, resolution) are kept in sync
with `examples/bistro/main.ts`.

Coordinate system notes
-----------------------
Bloom uses a right-handed Y-up coordinate system, with glTF imported as-is.
Blender is internally Z-up, but when glTF is imported with `--import-convert`
(the default) the importer re-orients meshes so that +Y (glTF) maps to +Z
(Blender). To keep the rendered image consistent with how Bloom sees the
scene, we:

  1. Import the glTF without axis conversion so Bloom's Y-up coordinates
     apply directly (we pass `bpy_extras.io_utils` flags that disable the
     conversion, falling back to post-import rotation if that's not
     possible).
  2. Express the camera position and sun direction in the *same* Y-up space
     that main.ts uses, then convert to Blender's Z-up space by swapping
     axes (y_blender = -z_bloom, z_blender = y_bloom). This gives identical
     framing regardless of the importer's axis-convert setting.

FOV: Bloom's `fovy: 60` is vertical field of view. Blender's camera uses
`angle_y` for vertical FOV when `lens_unit = 'FOV'` and `sensor_fit`
enforces vertical.
"""

import bpy
import json
import math
import os
import shutil
import sys
from mathutils import Vector, Matrix

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
BISTRO_GLTF = os.path.abspath(os.path.join(
    HERE, "..", "..", "examples", "bistro", "assets", "bistro.gltf"
))
OUTDOOR_HDR = os.path.abspath(os.path.join(
    HERE, "..", "..", "examples", "bistro", "assets", "outdoor.hdr"
))

# Defaults (tune via CLI args after `--`)
DEFAULT_OUT = "/tmp/bistro_cycles.png"
DEFAULT_SAMPLES = 128
DEFAULT_DEVICE = "METAL"   # macOS: METAL; fall back to CPU if unavailable
RES_X = 3456
RES_Y = 1944   # 16:9, matches Bloom's default Bistro window aspect

# Scene parameters — mirror examples/bistro/main.ts
CAM_POS_BLOOM = (-26.43, 3.16, 11.17)   # Y-up, Bloom convention
CAM_YAW       = -1.17                    # radians — synced with bistro/main.ts
CAM_PITCH     = 0.0
CAM_FOVY_DEG  = 60.0

SUN_DIR_BLOOM   = (-0.5, 0.75, 0.4)      # Y-up, *toward* the light source
SUN_COLOR       = (1.0, 240.0/255.0, 220.0/255.0)
SUN_STRENGTH    = 5.0                     # Cycles W/m^2 — roughly matches Bloom intensity 1.8

FILL_DIR_BLOOM  = (0.0, -1.0, 0.0)
FILL_COLOR      = (0.55, 0.55, 0.7)
FILL_STRENGTH   = 1.2

ENV_INTENSITY   = 1.2                     # matches setEnvIntensity(1.2)

# ---------------------------------------------------------------------------
# Arg parsing (Blender drops anything after `--` into argv; scan manually)
# ---------------------------------------------------------------------------

def parse_args():
    out = DEFAULT_OUT
    samples = DEFAULT_SAMPLES
    device = DEFAULT_DEVICE
    view = "Standard"
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out" and i + 1 < len(argv):
            out = argv[i + 1]; i += 2; continue
        if a == "--samples" and i + 1 < len(argv):
            samples = int(argv[i + 1]); i += 2; continue
        if a == "--device" and i + 1 < len(argv):
            device = argv[i + 1].upper(); i += 2; continue
        if a == "--view" and i + 1 < len(argv):
            view = argv[i + 1]; i += 2; continue
        i += 1
    return out, samples, device, view


# ---------------------------------------------------------------------------
# Coordinate conversion: Bloom Y-up -> Blender Z-up
# ---------------------------------------------------------------------------

def bloom_to_blender(v):
    """Convert a Bloom (X, Y_up, Z) vector to Blender's Z-up space.

    Y-up right-handed (X right, Y up, Z toward viewer) maps to Z-up
    right-handed (X right, Y into screen, Z up) via:
        x_b =  x_y
        y_b = -z_y
        z_b =  y_y
    """
    return (v[0], -v[2], v[1])


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def sanitize_gltf_for_blender(src_path):
    """Produce a Blender-friendly copy of the bistro glTF.

    Bloom's bistro.gltf lists every texture twice: once as `foo.png` (the
    original source from FBX2glTF) and once as `foo.dds` via the
    `MSFT_texture_dds` extension. The `.png` files were removed after
    `etcpak.sh` converted them to BC7 DDS, so Blender's glTF importer
    crashes trying to pack the missing PNGs.

    Blender 5.0 can load DDS natively, so we generate a patched copy with
    every missing `.png` URI rewritten to its existing `.dds` sibling and
    the MSFT_texture_dds extension stripped from textures. The patched
    file is cached next to the original so we only do this once.
    """
    base_dir = os.path.dirname(src_path)
    out_path = os.path.join(base_dir, "bistro_blender.gltf")

    # Regenerate if missing or older than source
    if (os.path.isfile(out_path)
            and os.path.getmtime(out_path) >= os.path.getmtime(src_path)):
        return out_path

    with open(src_path, "r") as f:
        g = json.load(f)

    patched = 0
    for im in g.get("images", []):
        uri = im.get("uri", "")
        if not uri:
            continue
        full = os.path.join(base_dir, uri)
        if os.path.exists(full):
            continue
        if uri.lower().endswith(".png"):
            dds = uri[:-4] + ".dds"
            if os.path.exists(os.path.join(base_dir, dds)):
                im["uri"] = dds
                patched += 1

    # Strip MSFT_texture_dds extension from textures — each texture now
    # resolves directly to the DDS source (or an untouched real PNG).
    for tex in g.get("textures", []):
        exts = tex.get("extensions")
        if exts and "MSFT_texture_dds" in exts:
            del exts["MSFT_texture_dds"]
            if not exts:
                del tex["extensions"]
    used = g.get("extensionsUsed", [])
    if "MSFT_texture_dds" in used:
        used.remove("MSFT_texture_dds")

    with open(out_path, "w") as f:
        json.dump(g, f)
    print(f"[cycles_reference] wrote sanitized gltf ({patched} png->dds URIs): {out_path}")
    return out_path


def import_bistro():
    if not os.path.isfile(BISTRO_GLTF):
        raise FileNotFoundError(f"bistro.gltf not found at {BISTRO_GLTF}")
    path = sanitize_gltf_for_blender(BISTRO_GLTF)
    # Import glTF. The Blender importer re-maps +Y -> +Z by default; we
    # *want* that because we'll convert our own camera/sun vectors into
    # Blender's Z-up space below, and everything stays consistent.
    bpy.ops.import_scene.gltf(filepath=path)


def setup_world_hdr():
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out  = nt.nodes.new("ShaderNodeOutputWorld")
    bg   = nt.nodes.new("ShaderNodeBackground")
    env  = nt.nodes.new("ShaderNodeTexEnvironment")
    mapn = nt.nodes.new("ShaderNodeMapping")
    coord = nt.nodes.new("ShaderNodeTexCoord")

    if os.path.isfile(OUTDOOR_HDR):
        env.image = bpy.data.images.load(OUTDOOR_HDR, check_existing=True)
    else:
        print(f"[cycles_reference] WARNING: HDR not found at {OUTDOOR_HDR}", file=sys.stderr)

    bg.inputs["Strength"].default_value = ENV_INTENSITY

    nt.links.new(coord.outputs["Generated"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_sun(name, dir_bloom, color, strength):
    """Add a Sun light aimed along the given direction (in Bloom Y-up).

    In Bloom, `setDirectionalLight` direction is the vector *toward* the
    light source. Blender's Sun shines along the light's local -Z axis.
    We orient the Sun so its -Z points in the opposite of dir_bloom
    (i.e. the direction the light travels).
    """
    light_data = bpy.data.lights.new(name=name, type='SUN')
    light_data.color = color
    light_data.energy = strength
    light_data.angle = math.radians(0.53)  # ~solar disc size; soft shadow

    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    # The direction the light *travels* is -dir_bloom
    travel_bloom = (-dir_bloom[0], -dir_bloom[1], -dir_bloom[2])
    travel_b = Vector(bloom_to_blender(travel_bloom)).normalized()

    # Orient so local -Z aligns with travel_b.
    # Track_quat returns a rotation mapping the chosen axis onto the vector.
    # We want -Z to look along travel_b, which is equivalent to asking the
    # quat that maps '-Z' onto travel_b.
    rot = travel_b.to_track_quat('-Z', 'Y')
    light_obj.rotation_mode = 'QUATERNION'
    light_obj.rotation_quaternion = rot


def setup_camera():
    cam_data = bpy.data.cameras.new("ReferenceCamera")
    cam_data.sensor_fit = 'VERTICAL'
    cam_data.lens_unit = 'FOV'
    cam_data.angle_y = math.radians(CAM_FOVY_DEG)
    cam_data.clip_start = 0.05
    cam_data.clip_end = 5000.0

    cam_obj = bpy.data.objects.new("ReferenceCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Position in Bloom space, convert to Blender
    pos_b = Vector(bloom_to_blender(CAM_POS_BLOOM))
    cam_obj.location = pos_b

    # Forward vector per main.ts (Y-up): fwd = (-sin(yaw), sin(pitch), -cos(yaw))
    # with the separate Y component derived from pitch. main.ts computes the
    # look target as camera + (cos(pitch)*fwdX*100, sin(pitch)*100, cos(pitch)*fwdZ*100).
    fwd_x = -math.sin(CAM_YAW)
    fwd_z = -math.cos(CAM_YAW)
    look_bloom = (
        CAM_POS_BLOOM[0] + math.cos(CAM_PITCH) * fwd_x * 100.0,
        CAM_POS_BLOOM[1] + math.sin(CAM_PITCH) * 100.0,
        CAM_POS_BLOOM[2] + math.cos(CAM_PITCH) * fwd_z * 100.0,
    )
    tgt_b = Vector(bloom_to_blender(look_bloom))
    up_b  = Vector(bloom_to_blender((0.0, 1.0, 0.0)))  # world up in Blender space

    # Build a look-at rotation: camera's local -Z points at target, local +Y
    # aligns roughly with up.
    direction = (tgt_b - pos_b).normalized()
    rot = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = rot

    print(f"[cycles_reference] camera pos (Blender)   = {tuple(pos_b)}")
    print(f"[cycles_reference] camera target (Blender)= {tuple(tgt_b)}")
    print(f"[cycles_reference] requested angle_y (deg)= {CAM_FOVY_DEG}")
    print(f"[cycles_reference] actual  cam.angle_y (deg)= {math.degrees(cam_data.angle_y)}")
    print(f"[cycles_reference] actual  cam.angle_x (deg)= {math.degrees(cam_data.angle_x)}")
    print(f"[cycles_reference] sensor_fit={cam_data.sensor_fit} sensor_w={cam_data.sensor_width} sensor_h={cam_data.sensor_height} lens={cam_data.lens}")


def setup_render(out_path, samples, device, view):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.filepath = out_path

    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass

    # Film / color management. 'Standard' = pass-through (clip to [0,1]),
    # 'AgX' = Blender 4+ default DRT (soft sigmoid, preserves highlights).
    # Pass --view AgX to compare against Bloom's AgX tonemap.
    scene.view_settings.view_transform = view

    # GPU if requested and available
    if device != "CPU":
        prefs = bpy.context.preferences.addons['cycles'].preferences
        try:
            prefs.compute_device_type = device  # METAL on macOS, CUDA/OPTIX on Linux/Win
            prefs.get_devices()
            for d in prefs.devices:
                d.use = True
            scene.cycles.device = 'GPU'
            print(f"[cycles_reference] using GPU device type: {device}")
        except Exception as e:
            print(f"[cycles_reference] GPU setup failed ({e}); falling back to CPU", file=sys.stderr)
            scene.cycles.device = 'CPU'
    else:
        scene.cycles.device = 'CPU'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_path, samples, device, view = parse_args()
    print(f"[cycles_reference] out={out_path} samples={samples} device={device} view={view}")

    clear_scene()
    import_bistro()
    setup_world_hdr()
    setup_camera()
    add_sun("Sun_Key",  SUN_DIR_BLOOM,  SUN_COLOR,  SUN_STRENGTH)
    add_sun("Sun_Fill", FILL_DIR_BLOOM, FILL_COLOR, FILL_STRENGTH)
    setup_render(out_path, samples, device, view)

    print("[cycles_reference] rendering…")
    bpy.ops.render.render(write_still=True)
    print(f"[cycles_reference] wrote {out_path}")


if __name__ == "__main__":
    main()
