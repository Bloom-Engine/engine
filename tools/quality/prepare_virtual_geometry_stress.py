#!/usr/bin/env python3
"""Generate a deterministic static glTF for virtual-geometry stress tests.

The default scene contains 100 translated meshes. Every mesh references the
same 100,000-triangle grid payload, so the glTF source closure stays compact
while the cooker and runtime still process 10,000,000 authored source
triangles and 100 independently placed hierarchy instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_stress_scene(
    output: Path,
    mesh_count: int = 100,
    quads_x: int = 250,
    quads_z: int = 200,
    tile_size: float = 10.0,
) -> dict[str, Any]:
    if mesh_count < 1:
        raise ValueError("mesh_count must be positive")
    if quads_x < 1 or quads_z < 1:
        raise ValueError("grid dimensions must be positive")
    if not 0.0 < tile_size < 1.0e6:
        raise ValueError("tile_size must be finite and positive")

    output = output.resolve()
    if output.suffix.lower() != ".gltf":
        raise ValueError("output must use the .gltf extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    binary_path = output.with_suffix(".bin")
    temporary_binary = binary_path.with_name(binary_path.name + ".tmp")
    temporary_gltf = output.with_name(output.name + ".tmp")

    vertex_count = (quads_x + 1) * (quads_z + 1)
    triangles_per_mesh = quads_x * quads_z * 2
    index_count = triangles_per_mesh * 3
    position_bytes = vertex_count * 12
    index_bytes = index_count * 4

    with temporary_binary.open("wb") as binary:
        for z in range(quads_z + 1):
            position_z = tile_size * z / quads_z
            for x in range(quads_x + 1):
                position_x = tile_size * x / quads_x
                # A shallow deterministic wave prevents the hierarchy stress
                # from degenerating into a single perfectly planar polygon.
                height = ((x * 17 + z * 29) % 31) * (tile_size / 3100.0)
                binary.write(struct.pack("<3f", position_x, height, position_z))
        for z in range(quads_z):
            row = quads_x + 1
            for x in range(quads_x):
                a = z * row + x
                b = a + 1
                c = a + row
                d = c + 1
                binary.write(struct.pack("<6I", a, c, b, b, c, d))
    if temporary_binary.stat().st_size != position_bytes + index_bytes:
        raise RuntimeError("generated stress binary has an unexpected size")

    columns = max(1, int(mesh_count**0.5))
    stride = tile_size * 1.25
    meshes = []
    nodes = []
    for mesh_index in range(mesh_count):
        meshes.append(
            {
                "name": f"stress-tile-{mesh_index:03d}",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                    }
                ],
            }
        )
        nodes.append(
            {
                "mesh": mesh_index,
                "translation": [
                    float(mesh_index % columns) * stride,
                    0.0,
                    float(mesh_index // columns) * stride,
                ],
            }
        )

    document = {
        "asset": {"generator": "Bloom virtual geometry stress v1", "version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(mesh_count))}],
        "nodes": nodes,
        "meshes": meshes,
        "buffers": [{"uri": binary_path.name, "byteLength": position_bytes + index_bytes}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": position_bytes},
            {"buffer": 0, "byteOffset": position_bytes, "byteLength": index_bytes},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [tile_size, tile_size * 0.01, tile_size],
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": index_count,
                "type": "SCALAR",
                "min": [0],
                "max": [vertex_count - 1],
            },
        ],
    }
    temporary_gltf.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_binary, binary_path)
    os.replace(temporary_gltf, output)

    return {
        "schema": "bloom-virtual-geometry-stress-source-v1",
        "gltf": str(output),
        "binary": str(binary_path),
        "mesh_count": mesh_count,
        "placements": mesh_count,
        "quads_x": quads_x,
        "quads_z": quads_z,
        "vertices_per_mesh": vertex_count,
        "triangles_per_mesh": triangles_per_mesh,
        "source_triangles": mesh_count * triangles_per_mesh,
        "gltf_bytes": output.stat().st_size,
        "binary_bytes": binary_path.stat().st_size,
        "gltf_sha256": _sha256(output),
        "binary_sha256": _sha256(binary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--mesh-count", type=int, default=100)
    parser.add_argument("--quads-x", type=int, default=250)
    parser.add_argument("--quads-z", type=int, default=200)
    parser.add_argument("--tile-size", type=float, default=10.0)
    arguments = parser.parse_args()
    report = build_stress_scene(
        arguments.output,
        mesh_count=arguments.mesh_count,
        quads_x=arguments.quads_x,
        quads_z=arguments.quads_z,
        tile_size=arguments.tile_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
