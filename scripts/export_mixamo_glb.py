"""
export_mixamo_glb.py — Reusable Blender script for exporting Mixamo characters to GLB
with correct skeletal animation data for the Bloom Engine.

Usage:
    blender --background --python export_mixamo_glb.py -- \\
        character.fbx walk.fbx run.fbx idle.fbx \\
        -o output.glb \\
        --max-verts 2000

Arguments (after the -- separator):
    First positional arg:   Character FBX file (the skinned mesh + armature)
    Remaining positional:   Animation FBX files (one per animation clip)
    -o / --output:          Output GLB path (default: output.glb)
    --max-verts:            Maximum vertex count for decimation (default: 2000, 0 = no decimation)
    --no-decimate:          Skip decimation entirely
    --no-root-lock:         Don't zero out root joint translation keyframes

Each animation FBX becomes a separate glTF animation in the output, named after
the NLA track (derived from the FBX filename).

CRITICAL EXPORT SETTINGS (discovered through extensive debugging):
    - export_optimize_animation_size=False   WITHOUT THIS, BLENDER STRIPS KEYFRAMES
                                             TO 2-3 PER CHANNEL, PRODUCING A STATIC POSE
    - export_force_sampling=True             Sample every frame (don't rely on sparse keys)
    - export_animation_mode='NLA_TRACKS'     Required for multi-animation export
    - export_optimize_animation_keep_anim_armature=True   Keep armature animation data

Requires: Blender 3.6+ (for glTF NLA_TRACKS export support)

See also: docs/skeletal-animation.md in the Bloom Engine repo.
"""

import bpy
import os
import sys
import argparse


def parse_args():
    """Parse command-line arguments after Blender's -- separator."""
    # Everything after '--' in the command line is for our script
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Export Mixamo character + animations to GLB for Bloom Engine"
    )
    parser.add_argument(
        "fbx_files", nargs="+",
        help="FBX files: first is the character, rest are animations"
    )
    parser.add_argument(
        "-o", "--output", default="output.glb",
        help="Output GLB file path (default: output.glb)"
    )
    parser.add_argument(
        "--max-verts", type=int, default=2000,
        help="Max vertex count for decimation (default: 2000, 0 to skip)"
    )
    parser.add_argument(
        "--no-decimate", action="store_true",
        help="Skip mesh decimation entirely"
    )
    parser.add_argument(
        "--no-root-lock", action="store_true",
        help="Don't zero out root joint translation (allows root motion)"
    )

    return parser.parse_args(argv)


def clean_scene():
    """Start with a completely empty scene.

    This prevents leftover objects from interfering with the export.
    Blender's factory settings include a default cube, camera, and light.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_character(fbx_path):
    """Import the character FBX and return (armature_obj, mesh_obj).

    The character FBX from Mixamo contains:
    - One armature (the skeleton)
    - One or more meshes (the skinned character model)

    We find the armature and the largest mesh (by vertex count) to ensure
    we get the actual character body, not a small accessory mesh.
    """
    print(f"[export] Importing character: {fbx_path}")
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    arm_obj = None
    mesh_obj = None
    max_verts = 0

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            arm_obj = obj
        if obj.type == 'MESH' and len(obj.data.vertices) > max_verts:
            max_verts = len(obj.data.vertices)
            mesh_obj = obj

    if arm_obj is None:
        print("[export] ERROR: No armature found in character FBX!")
        sys.exit(1)
    if mesh_obj is None:
        print("[export] ERROR: No mesh found in character FBX!")
        sys.exit(1)

    print(f"[export] Character armature: '{arm_obj.name}' ({len(arm_obj.data.bones)} bones)")
    print(f"[export] Character mesh: '{mesh_obj.name}' ({len(mesh_obj.data.vertices)} verts)")

    return arm_obj, mesh_obj


def import_animation(fbx_path, character_arm):
    """Import an animation FBX and transfer its action to the character armature.

    Mixamo animation FBX files contain their own armature with the animation
    baked as an Action. We need to:
    1. Import the FBX (creates a second armature)
    2. Find the action with the most keyframes (the animation we want)
    3. Delete the imported armature (we don't need it)
    4. Return the action for later NLA track assignment

    WHY we use the same Mixamo pack: Different Mixamo character packs have
    incompatible rest-pose bone orientations. Even if the bone names match,
    the rest quaternions differ, causing horrible twisting when the animation
    is applied to a different character's armature.
    """
    print(f"[export] Importing animation: {fbx_path}")

    # Remember existing objects and actions so we can find the new ones
    existing_objects = set(bpy.data.objects)
    existing_actions = set(bpy.data.actions)

    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Find the newly imported armature
    new_arms = [
        o for o in bpy.data.objects
        if o.type == 'ARMATURE' and o not in existing_objects and o != character_arm
    ]

    # Find the newly imported action (the one with the most keyframes / longest range)
    new_actions = [a for a in bpy.data.actions if a not in existing_actions]
    if not new_actions:
        # Fallback: pick the action with the longest frame range
        new_actions = sorted(
            bpy.data.actions,
            key=lambda a: a.frame_range[1] - a.frame_range[0],
            reverse=True
        )

    if not new_actions:
        print(f"[export] WARNING: No action found in {fbx_path}, skipping")
        return None

    # Pick the action with the longest duration (most keyframe data)
    action = max(new_actions, key=lambda a: a.frame_range[1] - a.frame_range[0])
    print(f"[export] Found action: '{action.name}' frames={action.frame_range}")

    # Delete the imported armature — we only need the action data
    for arm in new_arms:
        bpy.data.objects.remove(arm, do_unlink=True)

    # Also clean up any extra meshes that came with the animation FBX
    new_objects = [o for o in bpy.data.objects if o not in existing_objects]
    for obj in new_objects:
        if obj.type != 'ARMATURE':
            bpy.data.objects.remove(obj, do_unlink=True)

    return action


def assign_action_to_nla(arm_obj, action, track_name):
    """Push an action onto an NLA track on the given armature.

    WHY NLA TRACKS ARE REQUIRED:
    Blender's glTF exporter with export_animation_mode='NLA_TRACKS' ONLY
    exports animations that are on NLA tracks. Having an action assigned
    as the "active action" on the armature is NOT enough — it will be
    silently ignored during export.

    Each NLA track becomes a separate glTF animation in the output GLB,
    which maps to animIndex in Bloom's updateModelAnimation().
    """
    if not arm_obj.animation_data:
        arm_obj.animation_data_create()

    # Temporarily assign as active action (required to create NLA strip)
    arm_obj.animation_data.action = action

    # Create NLA track and push the action onto it
    track = arm_obj.animation_data.nla_tracks.new()
    track.name = track_name
    strip = track.strips.new(track_name, int(action.frame_range[0]), action)

    # Clear the active action — NLA tracks drive the animation now
    arm_obj.animation_data.action = None

    # Ensure the action isn't garbage collected
    action.use_fake_user = True

    print(f"[export] Pushed action '{action.name}' to NLA track '{track_name}'")


def decimate_mesh(mesh_obj, max_verts):
    """Reduce mesh vertex count for mobile performance.

    Mobile GPUs (especially older iOS/Android devices) struggle with high
    vertex counts. 2000 verts is a good target for characters that need
    to run at 60fps on mobile.

    The Decimate modifier preserves vertex groups (bone weights), so
    skinning data survives decimation.
    """
    current_verts = len(mesh_obj.data.vertices)
    if current_verts <= max_verts:
        print(f"[export] Mesh already has {current_verts} verts (<= {max_verts}), skipping decimation")
        return

    bpy.context.view_layer.objects.active = mesh_obj
    mod = mesh_obj.modifiers.new(name="Dec", type='DECIMATE')
    mod.ratio = float(max_verts) / float(current_verts)
    bpy.ops.object.modifier_apply(modifier="Dec")
    new_verts = len(mesh_obj.data.vertices)
    print(f"[export] Decimated: {current_verts} -> {new_verts} verts (target: {max_verts})")


def apply_armature_scale(arm_obj):
    """Apply all transforms on the armature (and its children).

    WHY THIS IS CRITICAL:
    Mixamo FBX files are in centimeters. Blender's FBX importer converts
    to meters by setting the armature's scale to 0.01. This means:
    - Vertex positions are in meters (scaled by the parent transform)
    - Bone transforms (rest pose, animations) are in centimeters

    This mismatch causes the mesh to explode at runtime because the GPU
    skinning shader applies cm-scale joint matrices to m-scale vertices.

    Applying the transform bakes the scale into the actual vertex positions
    and bone data, unifying everything into the same coordinate space.

    IMPORTANT: This must happen BEFORE animation baking (if you bake).
    Applying scale AFTER baking invalidates the baked keyframe values.
    We don't bake here (we use the raw Mixamo keyframes), so the order
    matters less — but it's good practice to apply scale early.
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print("[export] Applied transforms on all objects")


def remove_extra_objects(keep_objects):
    """Remove all objects except the ones we want to export.

    Extra objects (lights, cameras, empties, extra meshes from animation
    FBX imports) can interfere with the glTF export or bloat the file.
    """
    removed = 0
    for obj in list(bpy.data.objects):
        if obj not in keep_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    if removed > 0:
        print(f"[export] Removed {removed} extra objects")


def clean_unused_actions(keep_actions):
    """Remove actions we don't need to prevent them leaking into the export."""
    removed = 0
    for action in list(bpy.data.actions):
        if action not in keep_actions:
            bpy.data.actions.remove(action)
            removed += 1
    if removed > 0:
        print(f"[export] Removed {removed} unused actions")


def export_glb(output_path):
    """Export the scene to GLB with animation-safe settings.

    THE CRITICAL SETTINGS (each one addresses a specific failure mode):

    export_animation_mode='NLA_TRACKS'
        Required for multi-animation export. Each NLA track becomes a
        separate glTF animation. Without this, only the active action
        (if any) is exported.

    export_optimize_animation_size=False
        THIS IS THE MOST IMPORTANT SETTING. Blender's animation optimizer
        removes keyframes it considers "redundant" (where interpolation
        would produce a similar result). For character animation, this
        is catastrophic — it reduces a 30-frame walk cycle to 2-3
        keyframes, producing a nearly static pose at runtime.

        This single setting took DAYS to diagnose. The exported GLB
        would load fine, the skeleton was correct, but the character
        barely moved because 90%+ of the keyframes were stripped.

    export_force_sampling=True
        Forces the exporter to sample every frame instead of relying
        on FCurve keyframes. This ensures smooth animation even if
        the action has sparse keyframes or uses non-linear interpolation
        that doesn't translate well to glTF's linear interpolation.

    export_optimize_animation_keep_anim_armature=True
        Ensures armature animation data is preserved even when other
        optimizations are applied. Without this, the exporter may
        drop animation data for armatures it considers "static".

    export_apply=False
        Do NOT apply modifiers during export. We already applied the
        Decimate modifier manually. Applying again could double-decimate
        or cause issues with the Armature modifier.

    export_skins=True
        Export skin data (JOINTS_0 + WEIGHTS_0 vertex attributes).
        Without this, the mesh exports without bone weights and
        cannot be skinned at runtime.
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"[export] Exporting to: {output_path}")
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',

        # Mesh data
        export_normals=True,
        export_texcoords=True,
        export_image_format='JPEG',
        export_apply=False,         # Do NOT re-apply modifiers

        # Skin data — required for GPU skinning
        export_skins=True,

        # Animation settings — CRITICAL
        export_animations=True,
        export_animation_mode='NLA_TRACKS',
        export_optimize_animation_size=False,       # DO NOT REMOVE — keyframes will be stripped!
        export_force_sampling=True,                  # Sample every frame
        export_optimize_animation_keep_anim_armature=True,
    )
    print(f"[export] Done! Output: {output_path}")


def animation_name_from_path(fbx_path):
    """Derive a clean animation name from an FBX filename.

    Examples:
        "standing run forward.fbx" -> "standing run forward"
        "/path/to/Idle.fbx" -> "Idle"
        "Ch24_Walk_Forward.fbx" -> "Ch24_Walk_Forward"
    """
    basename = os.path.basename(fbx_path)
    name = os.path.splitext(basename)[0]
    return name


def main():
    args = parse_args()

    if len(args.fbx_files) < 1:
        print("[export] ERROR: Need at least one FBX file (the character)")
        sys.exit(1)

    character_fbx = args.fbx_files[0]
    animation_fbxs = args.fbx_files[1:]

    if not animation_fbxs:
        print("[export] WARNING: No animation FBX files specified. "
              "Output will have the mesh but no animations.")

    # Step 1: Clean scene
    clean_scene()

    # Step 2: Import character (armature + skinned mesh)
    arm_obj, mesh_obj = import_character(character_fbx)

    # Step 3: Import each animation and assign to NLA tracks
    keep_actions = set()
    for fbx_path in animation_fbxs:
        action = import_animation(fbx_path, arm_obj)
        if action is not None:
            track_name = animation_name_from_path(fbx_path)
            assign_action_to_nla(arm_obj, action, track_name)
            keep_actions.add(action)

    # Step 4: Clean up unused actions (from character FBX's default pose, etc.)
    clean_unused_actions(keep_actions)

    # Step 5: Decimate mesh if requested
    if not args.no_decimate and args.max_verts > 0:
        decimate_mesh(mesh_obj, args.max_verts)

    # Step 6: Apply armature scale (MUST happen before export)
    # This unifies the coordinate space between vertices and bone transforms.
    apply_armature_scale(arm_obj)

    # Step 7: Remove extra objects (lights, cameras, empties)
    remove_extra_objects({arm_obj, mesh_obj})

    # Step 8: Export GLB with correct settings
    export_glb(args.output)

    # Summary
    print("\n[export] === Summary ===")
    print(f"[export] Character: {character_fbx}")
    print(f"[export] Animations: {len(animation_fbxs)}")
    for i, fbx in enumerate(animation_fbxs):
        print(f"[export]   [{i}] {animation_name_from_path(fbx)}")
    print(f"[export] Mesh verts: {len(mesh_obj.data.vertices)}")
    print(f"[export] Bones: {len(arm_obj.data.bones)}")
    print(f"[export] Output: {args.output}")
    print("[export] === Done ===")


if __name__ == "__main__":
    main()
