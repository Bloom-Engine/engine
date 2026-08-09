#!/usr/bin/env python3
"""Build a bounded, deterministic Bistro qualification glTF.

The source asset references 551 unique meshes from 2,909 mesh nodes. Bloom's
current ModelData ABI owns vertices per mesh entry rather than an instance
table, so loading the source literally expands repeated geometry to ~19 GB.
This preparer retains the first authored world transform for every unique
mesh. It preserves the complete material/texture set and exterior scale while
bounding geometry to one copy per source mesh until native mesh instancing
lands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


def multiply(a: list[float], b: list[float]) -> list[float]:
    """Column-major 4x4 multiplication matching the glTF matrix layout."""
    out = [0.0] * 16
    for column in range(4):
        for row in range(4):
            out[column * 4 + row] = sum(
                a[k * 4 + row] * b[column * 4 + k] for k in range(4)
            )
    return out


def local_matrix(node: dict[str, Any]) -> list[float]:
    matrix = node.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 16:
        return [float(value) for value in matrix]
    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    x, y, z, w = (float(value) for value in rotation)
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length > 0.0:
        x, y, z, w = x / length, y / length, z / length, w / length
    sx, sy, sz = (float(value) for value in scale)
    tx, ty, tz = (float(value) for value in translation)
    return [
        (1.0 - 2.0 * (y * y + z * z)) * sx,
        (2.0 * (x * y + z * w)) * sx,
        (2.0 * (x * z - y * w)) * sx,
        0.0,
        (2.0 * (x * y - z * w)) * sy,
        (1.0 - 2.0 * (x * x + z * z)) * sy,
        (2.0 * (y * z + x * w)) * sy,
        0.0,
        (2.0 * (x * z + y * w)) * sz,
        (2.0 * (y * z - x * w)) * sz,
        (1.0 - 2.0 * (x * x + y * y)) * sz,
        0.0,
        tx,
        ty,
        tz,
        1.0,
    ]


IDENTITY = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def first_mesh_transforms(document: dict[str, Any]) -> dict[int, list[float]]:
    nodes = document["nodes"]
    first: dict[int, list[float]] = {}

    def visit(index: int, parent: list[float], stack: set[int]) -> None:
        if index in stack:
            raise ValueError(f"cycle in glTF node graph at node {index}")
        node = nodes[index]
        world = multiply(parent, local_matrix(node))
        mesh = node.get("mesh")
        if isinstance(mesh, int) and mesh not in first:
            first[mesh] = world
        next_stack = set(stack)
        next_stack.add(index)
        for child in node.get("children", []):
            visit(int(child), world, next_stack)

    for scene in document.get("scenes", []):
        for root in scene.get("nodes", []):
            visit(int(root), IDENTITY, set())
    return first


def mesh_bounds(
    document: dict[str, Any],
    mesh: dict[str, Any],
    transform: list[float],
) -> tuple[list[float], float]:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for primitive in mesh.get("primitives", []):
        position = primitive.get("attributes", {}).get("POSITION")
        if not isinstance(position, int):
            continue
        accessor = document["accessors"][position]
        if "min" not in accessor or "max" not in accessor:
            continue
        for axis in range(3):
            minimum[axis] = min(minimum[axis], float(accessor["min"][axis]))
            maximum[axis] = max(maximum[axis], float(accessor["max"][axis]))
    if not math.isfinite(minimum[0]):
        return [transform[12], transform[13], transform[14]], 0.0
    corners: list[list[float]] = []
    for mask in range(8):
        point = [
            maximum[axis] if mask & (1 << axis) else minimum[axis]
            for axis in range(3)
        ]
        corners.append(
            [
                sum(transform[column * 4 + row] * point[column] for column in range(3))
                + transform[12 + row]
                for row in range(3)
            ]
        )
    world_min = [min(point[axis] for point in corners) for axis in range(3)]
    world_max = [max(point[axis] for point in corners) for axis in range(3)]
    center = [(world_min[axis] + world_max[axis]) * 0.5 for axis in range(3)]
    radius = math.sqrt(
        sum(((world_max[axis] - world_min[axis]) * 0.5) ** 2 for axis in range(3))
    )
    return center, radius


def qualification_meshes(
    document: dict[str, Any],
    transforms: dict[int, list[float]],
    maximum: int,
) -> list[int]:
    # Use the authored glTF camera. Its -Z axis is camera-forward.
    camera_node = next(
        (node for node in document["nodes"] if isinstance(node.get("camera"), int)),
        None,
    )
    if camera_node is None:
        return sorted(transforms)[:maximum]
    camera = local_matrix(camera_node)
    position = camera[12:15]
    forward = [-camera[8], -camera[9], -camera[10]]
    right = camera[0:3]
    up = camera[4:7]
    ranked: list[tuple[float, int]] = []
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        center, radius = mesh_bounds(document, mesh, transforms[mesh_index])
        relative = [center[axis] - position[axis] for axis in range(3)]
        depth = sum(relative[axis] * forward[axis] for axis in range(3))
        horizontal = sum(relative[axis] * right[axis] for axis in range(3))
        vertical = sum(relative[axis] * up[axis] for axis in range(3))
        # Generous frustum around the source camera. Bounding-sphere overlap
        # retains large architecture whose center lies outside the view.
        visible = (
            depth + radius > 0.0
            and depth - radius < 200.0
            and abs(horizontal) < depth * 1.3 + radius
            and abs(vertical) < depth * 0.8 + radius
        )
        if visible:
            ranked.append((radius / max(depth, 0.1), mesh_index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(mesh_index for _, mesh_index in ranked[:maximum])
    if not selected:
        raise ValueError("Bistro camera selection produced no meshes")
    return selected


def texture_indices(value: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "index" and isinstance(child, int):
                found.add(child)
            else:
                found.update(texture_indices(child))
    elif isinstance(value, list):
        for child in value:
            found.update(texture_indices(child))
    return found


def remap_texture_indices(value: Any, mapping: dict[int, int]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "index" and isinstance(child, int):
                value[key] = mapping[child]
            else:
                remap_texture_indices(child, mapping)
    elif isinstance(value, list):
        for child in value:
            remap_texture_indices(child, mapping)


def rewrite_uri(uri: str, source_dir: Path, output_dir: Path) -> str:
    if uri.startswith("data:"):
        return uri
    source = (source_dir / uri).resolve()
    return os.path.relpath(source, output_dir).replace(os.sep, "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--metadata")
    parser.add_argument("--max-meshes", type=int, default=96)
    parser.add_argument(
        "--selection",
        choices=("camera", "all-unique"),
        default="camera",
        help=(
            "camera-ranked qualification subset (bounded by --max-meshes) "
            "or every unique source mesh"
        ),
    )
    parser.add_argument("--revision")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    transforms = first_mesh_transforms(document)
    source_mesh_count = len(document.get("meshes", []))
    if len(transforms) != source_mesh_count:
        missing = sorted(set(range(source_mesh_count)) - set(transforms))
        raise SystemExit(f"source scene does not reference every mesh: {missing[:16]}")

    selected_meshes = (
        sorted(transforms)
        if args.selection == "all-unique"
        else qualification_meshes(document, transforms, max(1, args.max_meshes))
    )
    source_meshes = document["meshes"]
    document["meshes"] = [source_meshes[index] for index in selected_meshes]
    mesh_mapping = {
        source_index: derived_index
        for derived_index, source_index in enumerate(selected_meshes)
    }
    used_materials = sorted(
        {
            primitive["material"]
            for mesh in document["meshes"]
            for primitive in mesh.get("primitives", [])
            if isinstance(primitive.get("material"), int)
        }
    )
    material_mapping = {
        source_index: derived_index
        for derived_index, source_index in enumerate(used_materials)
    }
    source_materials = document.get("materials", [])
    document["materials"] = [source_materials[index] for index in used_materials]
    for mesh in document["meshes"]:
        for primitive in mesh.get("primitives", []):
            if isinstance(primitive.get("material"), int):
                primitive["material"] = material_mapping[primitive["material"]]
    used_textures = sorted(texture_indices(document["materials"]))
    texture_mapping = {
        source_index: derived_index
        for derived_index, source_index in enumerate(used_textures)
    }
    remap_texture_indices(document["materials"], texture_mapping)
    source_textures = document.get("textures", [])
    document["textures"] = [source_textures[index] for index in used_textures]
    used_images = sorted(
        {
            texture["source"]
            for texture in document["textures"]
            if isinstance(texture.get("source"), int)
        }
    )
    image_mapping = {
        source_index: derived_index
        for derived_index, source_index in enumerate(used_images)
    }
    for texture in document["textures"]:
        if isinstance(texture.get("source"), int):
            texture["source"] = image_mapping[texture["source"]]
        extensions = texture.get("extensions", {})
        for extension in extensions.values():
            if isinstance(extension, dict) and isinstance(extension.get("source"), int):
                source_index = extension["source"]
                if source_index in image_mapping:
                    extension["source"] = image_mapping[source_index]
                else:
                    extension.pop("source")
    source_images = document.get("images", [])
    document["images"] = [source_images[index] for index in used_images]

    mesh_count = len(selected_meshes)
    document["nodes"] = [
        {
            "name": f"quality_source_mesh_{source_index}",
            "mesh": mesh_mapping[source_index],
            "matrix": transforms[source_index],
        }
        for source_index in selected_meshes
    ]
    document["scenes"] = [
        {
            "name": "Bloom deterministic unique-mesh Bistro subset",
            "nodes": list(range(mesh_count)),
        }
    ]
    document["scene"] = 0
    document.pop("animations", None)
    document.pop("skins", None)
    document.pop("cameras", None)
    source_dir = source.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    pinned_buffers: list[dict[str, str]] = []
    for buffer_index, buffer in enumerate(document.get("buffers", [])):
        uri = buffer.get("uri")
        if isinstance(uri, str):
            if args.revision and not uri.startswith("data:"):
                try:
                    payload = subprocess.check_output(
                        ["git", "-C", str(source_dir), "show", f"{args.revision}:{uri}"]
                    )
                except subprocess.CalledProcessError as exc:
                    raise SystemExit(
                        f"cannot read pinned Bistro buffer {args.revision}:{uri}"
                    ) from exc
                pinned_name = f"pinned-buffer-{buffer_index}{Path(uri).suffix}"
                pinned_path = output.parent / pinned_name
                pinned_path.write_bytes(payload)
                buffer["uri"] = pinned_name
                pinned_buffers.append(
                    {
                        "source": f"{args.revision}:{uri}",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            else:
                buffer["uri"] = rewrite_uri(uri, source_dir, output.parent)
    for image in document.get("images", []):
        uri = image.get("uri")
        if isinstance(uri, str):
            image["uri"] = rewrite_uri(uri, source_dir, output.parent)
    document.setdefault("asset", {})["generator"] = (
        "Bloom tools/quality/prepare_bistro.py unique-mesh qualification subset"
    )
    output.write_text(
        json.dumps(document, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metadata_path = Path(args.metadata).resolve() if args.metadata else output.with_suffix(".json")
    metadata = {
        "schema": "bloom-quality-derived-asset-v1",
        "source": str(source),
        "source_sha256": sha256(source),
        "source_revision": args.revision,
        "pinned_buffers": pinned_buffers,
        "output": str(output),
        "output_sha256": sha256(output),
        "source_mesh_nodes": sum(
            1 for node in json.loads(source.read_text(encoding="utf-8"))["nodes"]
            if "mesh" in node
        ),
        "derived_mesh_nodes": mesh_count,
        "derived_materials": len(document["materials"]),
        "derived_textures": len(document["textures"]),
        "derived_images": len(document["images"]),
        "selection": args.selection,
        "policy": (
            "all authored unique meshes, first world transform per unique mesh"
            if args.selection == "all-unique"
            else "largest projected authored meshes in source camera, first world "
            "transform per unique mesh"
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"prepared Bistro subset: {metadata['source_mesh_nodes']} -> "
        f"{metadata['derived_mesh_nodes']} mesh nodes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
