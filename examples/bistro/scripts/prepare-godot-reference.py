#!/usr/bin/env python3
"""Build a Bloom-loadable glTF from Jamsers/Bistro-Demo-Tweaked.

The Godot project keeps render geometry in many material-free GLBs and assigns
its authored materials through each GLB's `.import` sidecar.  Bloom cannot load
Godot's `.tscn`/`.tres` resources directly, so this script performs a lossless
geometry merge and translates the small StandardMaterial3D subset used by the
project to glTF metallic/roughness materials.

The generated glTF references the source project's textures in place.  Only a
single merged geometry buffer is added, avoiding another copy of ~670 MB of
textures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any


PACKED_SCENE_RE = re.compile(
    r'\[ext_resource type="PackedScene"[^\]]*path="res://([^"]+)" id="([^"]+)"\]'
)
NODE_HEADER_RE = re.compile(r"\[node[^\]]+\]")
NODE_BLOCK_RE = re.compile(r"(\[node[^\]]+\])\n(.*?)(?=\n\[|\Z)", re.DOTALL)
SUBRESOURCE_BLOCK_RE = re.compile(
    r'(\[sub_resource type="([^"]+)" id="([^"]+)"\])\n(.*?)(?=\n\[|\Z)',
    re.DOTALL,
)
EXT_RESOURCE_RE = re.compile(
    r'\[ext_resource type="([^"]+)"[^\]]*path="res://([^"]+)" id="([^"]+)"\]'
)
MATERIAL_REMAP_RE = re.compile(
    r'"@MATERIAL:(\d+)"\s*:\s*\{.*?'
    r'"use_external/path"\s*:\s*"res://([^"]+)"',
    re.DOTALL,
)
PROPERTY_RE = re.compile(r"^([A-Za-z0-9_/]+)\s*=\s*(.+)$", re.MULTILINE)


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as handle:
        magic, version, total_size = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"{path}: expected a glTF 2 GLB")
        json_size, json_kind = struct.unpack("<I4s", handle.read(8))
        if json_kind != b"JSON":
            raise ValueError(f"{path}: JSON must be the first GLB chunk")
        document = json.loads(handle.read(json_size))
        binary = b""
        while handle.tell() < total_size:
            chunk_size, chunk_kind = struct.unpack("<I4s", handle.read(8))
            chunk = handle.read(chunk_size)
            if chunk_kind == b"BIN\0":
                binary = chunk
        if not binary:
            raise ValueError(f"{path}: missing binary chunk")
        return document, binary


def parse_number(value: str, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except ValueError:
        return default


def parse_color(value: str, default: list[float]) -> list[float]:
    match = re.fullmatch(r"Color\(([^)]+)\)", value.strip())
    if not match:
        return default
    parts = [parse_number(item.strip(), 0.0) for item in match.group(1).split(",")]
    if len(parts) == 3:
        parts.append(1.0)
    return parts[:4] if len(parts) >= 4 else default


def parse_ext_resource(value: str) -> str | None:
    match = re.fullmatch(r'ExtResource\("([^"]+)"\)', value.strip())
    return match.group(1) if match else None


def primary_scene_paths(root: Path) -> list[str]:
    text = (root / "MainScene.tscn").read_text()
    resources = {match.group(2): match.group(1) for match in PACKED_SCENE_RE.finditer(text)}
    result: list[str] = []
    for match in NODE_HEADER_RE.finditer(text):
        header = match.group(0)
        instance = re.search(r'instance=ExtResource\("([^"]+)"\)', header)
        parent = re.search(r'parent="([^"]*)"', header)
        if not instance or not parent:
            continue
        scene_path = resources.get(instance.group(1))
        if not scene_path or not scene_path.startswith("Scenes/"):
            continue
        if not (
            parent.group(1).startswith("Level Geometry")
            or parent.group(1).startswith("Props")
        ):
            continue
        if scene_path not in result:
            result.append(scene_path)
    return result


def scene_glb_path(root: Path, scene_path: str) -> Path:
    relative = Path(scene_path).relative_to("Scenes").with_suffix(".glb")
    direct = root / "Meshes" / relative
    if direct.is_file():
        return direct

    # Ordinary static scenes name their render GLB explicitly.  This fallback
    # also makes the converter tolerate future project layout changes.
    text = (root / scene_path).read_text()
    candidates = [
        path
        for path, _ in PACKED_SCENE_RE.findall(text)
        if path.endswith(".glb") and "CollisionOcclusion" not in path
    ]
    if candidates:
        return root / candidates[0]
    nested = [path for path, _ in PACKED_SCENE_RE.findall(text) if path.endswith(".tscn")]
    if nested:
        return scene_glb_path(root, nested[0])
    raise FileNotFoundError(f"no render GLB found for {scene_path}")


def parse_transform(value: str | None) -> list[float]:
    if value:
        match = re.fullmatch(r"Transform3D\(([^)]+)\)", value.strip())
        if match:
            values = [parse_number(item.strip(), 0.0) for item in match.group(1).split(",")]
            if len(values) == 12:
                # Godot serializes the Basis scalar constructor row-major;
                # glTF matrices are column-major. Transpose the 3x3 basis
                # while preserving the shared column-vector transform.
                return [
                    values[0], values[3], values[6], 0.0,
                    values[1], values[4], values[7], 0.0,
                    values[2], values[5], values[8], 0.0,
                    values[9], values[10], values[11], 1.0,
                ]
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


def multiply_matrix(a: list[float], b: list[float]) -> list[float]:
    # Matrices have already been normalized to glTF column-major storage.
    result = [0.0] * 16
    for column in range(4):
        for row in range(4):
            result[column * 4 + row] = sum(
                a[k * 4 + row] * b[column * 4 + k] for k in range(4)
            )
    return result


def node_blocks(text: str) -> list[tuple[str, str, dict[str, str]]]:
    result: list[tuple[str, str, dict[str, str]]] = []
    for match in NODE_BLOCK_RE.finditer(text):
        header, body = match.groups()
        name_match = re.search(r'name="([^"]+)"', header)
        if not name_match:
            continue
        result.append(
            (
                header,
                name_match.group(1),
                {item.group(1): item.group(2).strip() for item in PROPERTY_RE.finditer(body)},
            )
        )
    return result


def fillout_instances(root: Path) -> list[tuple[str, list[float]]]:
    text = (root / "MainScene.tscn").read_text()
    resources = {match.group(2): match.group(1) for match in PACKED_SCENE_RE.finditer(text)}
    world_by_path: dict[str, list[float]] = {".": parse_transform(None)}
    result: list[tuple[str, list[float]]] = []
    for header, name, properties in node_blocks(text):
        parent_match = re.search(r'parent="([^"]*)"', header)
        parent = parent_match.group(1) if parent_match else "."
        path = name if parent in ("", ".") else f"{parent}/{name}"
        parent_world = world_by_path.get(parent, parse_transform(None))
        world = multiply_matrix(parent_world, parse_transform(properties.get("transform")))
        world_by_path[path] = world
        instance = re.search(r'instance=ExtResource\("([^"]+)"\)', header)
        scene_path = resources.get(instance.group(1)) if instance else None
        if scene_path and scene_path.startswith("Scenes/FillOut/"):
            result.append((scene_path, world))
    return result


def scene_overrides(
    root: Path, scene_path: str
) -> tuple[set[str], dict[str, list[float]], set[str]]:
    text = (root / scene_path).read_text()
    hidden: set[str] = set()
    transforms: dict[str, list[float]] = {}
    no_shadow: set[str] = set()
    for _, name, properties in node_blocks(text):
        normalized = name.replace(".", "_")
        if properties.get("visible") == "false":
            hidden.add(normalized)
        if "transform" in properties:
            transforms[normalized] = parse_transform(properties["transform"])
        if properties.get("cast_shadow") == "0":
            no_shadow.add(normalized)
    return hidden, transforms, no_shadow


def annotate_shadow_overrides(
    output: dict[str, Any], source_index: int, no_shadow: set[str]
) -> int:
    """Preserve Godot GeometryInstance3D shadow intent in glTF extras."""
    node = output["nodes"][source_index]
    count = 0
    normalized = node.get("name", "").replace(".", "_")
    if normalized in no_shadow:
        extras = node.setdefault("extras", {})
        extras["BLOOM_cast_shadow"] = False
        count += 1
    for child in node.get("children", []):
        count += annotate_shadow_overrides(output, child, no_shadow)
    return count


class MaterialLibrary:
    def __init__(self, root: Path, output_dir: Path) -> None:
        self.root = root
        self.output_dir = output_dir
        self.material_indices: dict[str, int] = {}
        self.texture_indices: dict[str, int] = {}
        self.images: list[dict[str, Any]] = []
        self.textures: list[dict[str, Any]] = []
        self.materials: list[dict[str, Any]] = []

    def _texture(self, source_path: str) -> int:
        existing = self.texture_indices.get(source_path)
        if existing is not None:
            return existing
        absolute = self.root / source_path
        if not absolute.is_file():
            raise FileNotFoundError(f"material texture is missing: {absolute}")
        uri = os.path.relpath(absolute, self.output_dir).replace(os.sep, "/")
        image_index = len(self.images)
        self.images.append({"uri": uri})
        texture_index = len(self.textures)
        self.textures.append({"source": image_index, "sampler": 0})
        self.texture_indices[source_path] = texture_index
        return texture_index

    def _texture_from_property(
        self,
        properties: dict[str, str],
        external: dict[str, str],
        name: str,
    ) -> int | None:
        resource_id = parse_ext_resource(properties.get(name, ""))
        path = external.get(resource_id or "")
        return self._texture(path) if path else None

    def material(self, material_path: str) -> int:
        existing = self.material_indices.get(material_path)
        if existing is not None:
            return existing

        path = self.root / material_path
        text = path.read_text()
        header = re.search(r'^\[gd_resource type="([^"]+)"', text, re.MULTILINE)
        if not header or header.group(1) != "StandardMaterial3D":
            raise ValueError(f"unsupported Bistro material type in {path}")
        external = {
            match.group(3): match.group(2)
            for match in EXT_RESOURCE_RE.finditer(text)
            if match.group(1) == "Texture2D"
        }
        resource_text = text.split("[resource]", 1)[-1]
        properties = {match.group(1): match.group(2).strip() for match in PROPERTY_RE.finditer(resource_text)}

        base_color = parse_color(properties.get("albedo_color", ""), [1.0, 1.0, 1.0, 1.0])
        metallic = parse_number(properties.get("metallic", "0"), 0.0)
        roughness = parse_number(properties.get("roughness", "1"), 1.0)
        pbr: dict[str, Any] = {
            "baseColorFactor": base_color,
            "metallicFactor": max(0.0, min(1.0, metallic)),
            "roughnessFactor": max(0.0, min(1.0, roughness)),
        }
        base_texture = self._texture_from_property(properties, external, "albedo_texture")
        if base_texture is not None:
            pbr["baseColorTexture"] = {"index": base_texture}

        metallic_texture = self._texture_from_property(properties, external, "metallic_texture")
        roughness_texture = self._texture_from_property(properties, external, "roughness_texture")
        if metallic_texture is not None and roughness_texture is not None:
            if metallic_texture != roughness_texture:
                raise ValueError(f"{path}: separate metallic and roughness textures need repacking")
            # The Bistro project uses Godot channel 2 for metallic and channel
            # 1 for roughness, exactly glTF's B/G packing.
            metallic_channel = int(parse_number(properties.get("metallic_texture_channel", "2"), 2))
            roughness_channel = int(parse_number(properties.get("roughness_texture_channel", "1"), 1))
            if (metallic_channel, roughness_channel) != (2, 1):
                raise ValueError(f"{path}: unsupported metallic/roughness channel packing")
            pbr["metallicRoughnessTexture"] = {"index": metallic_texture}
        elif metallic_texture is not None or roughness_texture is not None:
            # glTF only exposes the combined B/G texture.  Three Bistro
            # materials deliberately texture just one response channel; keep
            # their authored scalar response instead of accidentally feeding
            # an unrelated source channel into the other property.  This is a
            # closer representation until Bloom exposes independent response
            # slots (and avoids generating another texture copy).
            print(f"warning: using scalar response for unpaired texture in {material_path}")

        material: dict[str, Any] = {
            "name": material_path.removeprefix("Materials/").removesuffix(".tres"),
            "pbrMetallicRoughness": pbr,
        }
        normal_texture = self._texture_from_property(properties, external, "normal_texture")
        if normal_texture is not None and properties.get("normal_enabled", "true") != "false":
            material["normalTexture"] = {"index": normal_texture}

        transparency = int(parse_number(properties.get("transparency", "0"), 0))
        if transparency in (2, 3):
            cutoff = parse_number(properties.get("alpha_scissor_threshold", "0.5"), 0.5)
            material["alphaMode"] = "MASK"
            material["alphaCutoff"] = max(0.0, min(1.0, cutoff))
        elif transparency in (1, 4):
            # Godot alpha blending and alpha-depth-prepass both retain
            # fractional coverage. glTF BLEND is the closest portable
            # representation; mapping these modes to MASK made thin signs
            # and glass edges pop between fully opaque and fully absent.
            material["alphaMode"] = "BLEND"
        if int(parse_number(properties.get("cull_mode", "0"), 0)) == 2:
            material["doubleSided"] = True

        # Afternoon is the reference app's launch state.  Its controller turns
        # the night/emissive material list off, so intentionally omit those
        # emission terms here instead of baking a night-only glow into the
        # daylight parity scene.
        index = len(self.materials)
        self.materials.append(material)
        self.material_indices[material_path] = index
        return index


def merge_scene(
    output: dict[str, Any],
    binary: bytearray,
    glb_path: Path,
    materials: MaterialLibrary,
    append_roots: bool = True,
) -> tuple[int, int, list[int]]:
    document, chunk = read_glb(glb_path)
    while len(binary) % 4:
        binary.append(0)
    binary_base = len(binary)
    binary.extend(chunk)

    view_base = len(output["bufferViews"])
    accessor_base = len(output["accessors"])
    mesh_base = len(output["meshes"])
    node_base = len(output["nodes"])

    for view in document.get("bufferViews", []):
        copied = dict(view)
        copied["buffer"] = 0
        copied["byteOffset"] = copied.get("byteOffset", 0) + binary_base
        output["bufferViews"].append(copied)
    for accessor in document.get("accessors", []):
        copied = dict(accessor)
        if "bufferView" in copied:
            copied["bufferView"] += view_base
        if "sparse" in copied:
            copied["sparse"] = json.loads(json.dumps(copied["sparse"]))
            copied["sparse"]["indices"]["bufferView"] += view_base
            copied["sparse"]["values"]["bufferView"] += view_base
        output["accessors"].append(copied)

    import_text = glb_path.with_suffix(glb_path.suffix + ".import").read_text()
    material_map = {
        int(local_index): materials.material(path)
        for local_index, path in MATERIAL_REMAP_RE.findall(import_text)
    }
    for local_mesh_index, mesh in enumerate(document.get("meshes", [])):
        copied = json.loads(json.dumps(mesh))
        material_index = material_map.get(local_mesh_index)
        for primitive in copied.get("primitives", []):
            primitive["attributes"] = {
                semantic: accessor + accessor_base
                for semantic, accessor in primitive.get("attributes", {}).items()
            }
            if "indices" in primitive:
                primitive["indices"] += accessor_base
            for target in primitive.get("targets", []):
                for semantic in list(target):
                    target[semantic] += accessor_base
            if material_index is not None:
                primitive["material"] = material_index
        output["meshes"].append(copied)

    for node in document.get("nodes", []):
        copied = json.loads(json.dumps(node))
        if "mesh" in copied:
            copied["mesh"] += mesh_base
        if "children" in copied:
            copied["children"] = [child + node_base for child in copied["children"]]
        output["nodes"].append(copied)
    roots = [
        root_node + node_base
        for root_node in document.get("scenes", [{}])[document.get("scene", 0)].get("nodes", [])
    ]
    if append_roots:
        output["scenes"][0]["nodes"].extend(roots)
    return len(document.get("meshes", [])), len(material_map), roots


def clone_node_tree(
    output: dict[str, Any],
    source_index: int,
    hidden: set[str],
    transforms: dict[str, list[float]],
    no_shadow: set[str],
) -> int | None:
    source = output["nodes"][source_index]
    normalized = source.get("name", "").replace(".", "_")
    if normalized in hidden:
        return None
    copied = json.loads(json.dumps(source))
    children = []
    for child in source.get("children", []):
        cloned = clone_node_tree(output, child, hidden, transforms, no_shadow)
        if cloned is not None:
            children.append(cloned)
    if children:
        copied["children"] = children
    else:
        copied.pop("children", None)
    transform = transforms.get(normalized)
    if transform is not None:
        copied["matrix"] = transform
        copied.pop("translation", None)
        copied.pop("rotation", None)
        copied.pop("scale", None)
    if normalized in no_shadow:
        extras = copied.setdefault("extras", {})
        extras["BLOOM_cast_shadow"] = False
    index = len(output["nodes"])
    output["nodes"].append(copied)
    return index


def append_window_panes(
    root: Path,
    output: dict[str, Any],
    binary: bytearray,
    materials: MaterialLibrary,
) -> int:
    """Translate MainScene's authored QuadMesh window patches to glTF."""
    text = (root / "MainScene.tscn").read_text()
    sizes: dict[str, tuple[float, float]] = {}
    for _, resource_type, resource_id, body in SUBRESOURCE_BLOCK_RE.findall(text):
        if resource_type != "QuadMesh":
            continue
        size = re.search(r"^size\s*=\s*Vector2\(([^)]+)\)", body, re.MULTILINE)
        if not size:
            continue
        values = [parse_number(item.strip(), 1.0) for item in size.group(1).split(",")]
        if len(values) == 2:
            sizes[resource_id] = (values[0], values[1])

    panes: list[tuple[str, list[float], tuple[float, float]]] = []
    for header, name, properties in node_blocks(text):
        parent = re.search(r'parent="([^"]*)"', header)
        if not parent or not parent.group(1).startswith("Patches/Window Panes/"):
            continue
        mesh = re.fullmatch(r'SubResource\("([^"]+)"\)', properties.get("mesh", ""))
        if not mesh or mesh.group(1) not in sizes:
            continue
        panes.append((name, parse_transform(properties.get("transform")), sizes[mesh.group(1)]))
    if not panes:
        return 0

    while len(binary) % 4:
        binary.append(0)
    vertex_offset = len(binary)
    vertices = (
        (-0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        (0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        (-0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
    )
    for vertex in vertices:
        binary.extend(struct.pack("<8f", *vertex))
    index_offset = len(binary)
    binary.extend(struct.pack("<6H", 0, 1, 2, 2, 1, 3))

    vertex_view = len(output["bufferViews"])
    output["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": vertex_offset,
            "byteLength": len(vertices) * 32,
            "byteStride": 32,
            "target": 34962,
        }
    )
    index_view = len(output["bufferViews"])
    output["bufferViews"].append(
        {"buffer": 0, "byteOffset": index_offset, "byteLength": 12, "target": 34963}
    )
    position_accessor = len(output["accessors"])
    output["accessors"].append(
        {
            "bufferView": vertex_view,
            "byteOffset": 0,
            "componentType": 5126,
            "count": 4,
            "type": "VEC3",
            "min": [-0.5, -0.5, 0.0],
            "max": [0.5, 0.5, 0.0],
        }
    )
    normal_accessor = len(output["accessors"])
    output["accessors"].append(
        {
            "bufferView": vertex_view,
            "byteOffset": 12,
            "componentType": 5126,
            "count": 4,
            "type": "VEC3",
        }
    )
    uv_accessor = len(output["accessors"])
    output["accessors"].append(
        {
            "bufferView": vertex_view,
            "byteOffset": 24,
            "componentType": 5126,
            "count": 4,
            "type": "VEC2",
        }
    )
    index_accessor = len(output["accessors"])
    output["accessors"].append(
        {
            "bufferView": index_view,
            "componentType": 5123,
            "count": 6,
            "type": "SCALAR",
            "min": [0],
            "max": [3],
        }
    )
    mesh_index = len(output["meshes"])
    output["meshes"].append(
        {
            "name": "Godot authored window pane",
            "primitives": [
                {
                    "attributes": {
                        "POSITION": position_accessor,
                        "NORMAL": normal_accessor,
                        "TEXCOORD_0": uv_accessor,
                    },
                    "indices": index_accessor,
                    "material": materials.material("Materials/Glass/Windows_Glass.tres"),
                }
            ],
        }
    )
    for name, transform, (width, height) in panes:
        scale = parse_transform(None)
        scale[0] = width
        scale[5] = height
        node_index = len(output["nodes"])
        output["nodes"].append(
            {"name": name, "mesh": mesh_index, "matrix": multiply_matrix(transform, scale)}
        )
        output["scenes"][0]["nodes"].append(node_index)
    return len(panes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Bistro-Demo-Tweaked source directory")
    parser.add_argument("--output", type=Path, help="output .gltf path (default: SOURCE/.bloom/BistroReference.gltf)")
    args = parser.parse_args()
    root = args.source.expanduser().resolve()
    if not (root / "project.godot").is_file() or not (root / "MainScene.tscn").is_file():
        parser.error(f"{root} is not a Bistro-Demo-Tweaked source checkout")
    output_path = (args.output or root / ".bloom" / "BistroReference.gltf").expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path = output_path.with_suffix(".bin")

    output: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "Bloom Godot Bistro reference converter"},
        "scene": 0,
        "scenes": [{"name": "Bistro-Demo-Tweaked primary geometry", "nodes": []}],
        "nodes": [],
        "meshes": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": [{"uri": binary_path.name, "byteLength": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
    }
    binary = bytearray()
    library = MaterialLibrary(root, output_path.parent)
    mesh_count = 0
    mapping_count = 0
    scene_paths = primary_scene_paths(root)
    source_roots: dict[Path, list[int]] = {}
    for scene_path in scene_paths:
        glb_path = scene_glb_path(root, scene_path)
        meshes, mappings, roots = merge_scene(output, binary, glb_path, library)
        _, _, no_shadow = scene_overrides(root, scene_path)
        for source_root in roots:
            annotate_shadow_overrides(output, source_root, no_shadow)
        source_roots[glb_path.resolve()] = roots
        mesh_count += meshes
        mapping_count += mappings
        print(f"merged {glb_path.relative_to(root)}: {meshes} meshes, {mappings} material mappings")

    fillout_count = 0
    for scene_path, world_transform in fillout_instances(root):
        glb_path = scene_glb_path(root, scene_path).resolve()
        roots = source_roots.get(glb_path)
        if roots is None:
            meshes, mappings, roots = merge_scene(
                output, binary, glb_path, library, append_roots=False
            )
            source_roots[glb_path] = roots
            mesh_count += meshes
            mapping_count += mappings
            print(
                f"merged fill-out source {glb_path.relative_to(root)}: "
                f"{meshes} meshes, {mappings} material mappings"
            )
        hidden, transforms, no_shadow = scene_overrides(root, scene_path)
        cloned_roots = [
            clone_node_tree(output, source_root, hidden, transforms, no_shadow)
            for source_root in roots
        ]
        children = [node for node in cloned_roots if node is not None]
        if not children:
            raise ValueError(f"{scene_path}: fill-out instance contains no visible render nodes")
        wrapper = len(output["nodes"])
        output["nodes"].append(
            {
                "name": f"{Path(scene_path).stem} fill-out {fillout_count + 1}",
                "matrix": world_transform,
                "children": children,
            }
        )
        output["scenes"][0]["nodes"].append(wrapper)
        fillout_count += 1

    window_count = append_window_panes(root, output, binary, library)
    no_shadow_count = sum(
        1
        for node in output["nodes"]
        if node.get("extras", {}).get("BLOOM_cast_shadow") is False
    )

    while len(binary) % 4:
        binary.append(0)
    output["buffers"][0]["byteLength"] = len(binary)
    output["materials"] = library.materials
    output["textures"] = library.textures
    output["images"] = library.images
    output_path.write_text(json.dumps(output, separators=(",", ":")))
    binary_path.write_bytes(binary)
    print(
        f"wrote {output_path} and {binary_path}: {len(scene_paths)} scenes, "
        f"{mesh_count} source meshes, {mapping_count} mappings, "
        f"{fillout_count} fill-out instances, {window_count} window panes, "
        f"{no_shadow_count} no-shadow placements, "
        f"{len(library.materials)} materials, {len(library.images)} textures"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
