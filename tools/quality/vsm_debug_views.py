#!/usr/bin/env python3
"""Validate VSM debug captures, their legend, and per-light cost telemetry."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

try:
    from tools.quality.khronos_materials import QualificationError, png_rgb
except ModuleNotFoundError:
    from khronos_materials import QualificationError, png_rgb


SCHEMA = "bloom-vsm-debug-views-v1"
PALETTE = {
    (8, 8, 8): "free",
    (255, 180, 35): "miss-unrendered",
    (255, 55, 190): "invalidated",
    (70, 210, 110): "clip-level-0",
    (70, 150, 255): "clip-level-1",
    (190, 100, 255): "clip-level-2",
}
ORDER = tuple(PALETTE.values())
COLORS = tuple(PALETTE)
VIRTUAL_AXIS = 32
CLIP_LEVELS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_from(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "renderer_paths" in document:
            state = document["renderer_paths"]["virtual_shadows"]
        else:
            state = document
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise QualificationError(f"invalid VSM telemetry {path}: {error}") from error
    if not isinstance(state, dict):
        raise QualificationError(f"invalid VSM telemetry state in {path}")
    return state


def integer(mapping: dict[str, object], field: str, source: str) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationError(f"{source} field {field!r} must be an integer")
    return value


def image_summary(
    path: Path, expected_width_cells: int, expected_height_cells: int
) -> dict[str, object]:
    width, height, pixels = png_rgb(path)
    if width < expected_width_cells or width % expected_width_cells:
        raise QualificationError(f"{path}: invalid debug-view width {width}")
    scale = width // expected_width_cells
    if height != expected_height_cells * scale:
        raise QualificationError(
            f"{path}: expected {expected_width_cells}:{expected_height_cells} cell geometry"
        )
    counts = Counter(PALETTE.get(pixel, "unknown") for pixel in pixels)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "scale": scale,
        "counts": {name: counts[name] for name in (*ORDER, "unknown")},
    }


def legend_summary(path: Path) -> dict[str, object]:
    width, height, pixels = png_rgb(path)
    if height < 1 or width != len(ORDER) * height:
        raise QualificationError(f"{path}: expected a six-cell horizontal legend")
    for index, expected in enumerate(COLORS):
        for y in range(height):
            start = y * width + index * height
            if any(pixel != expected for pixel in pixels[start : start + height]):
                raise QualificationError(
                    f"{path}: legend cell {index} is not {ORDER[index]!r}"
                )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "order": list(ORDER),
    }


def physical_geometry(capacity: int) -> tuple[int, int]:
    columns = min(16, max(1, capacity))
    return columns, max(1, math.ceil(capacity / columns))


def normalized_counts(summary: dict[str, object]) -> dict[str, int]:
    scale = summary["scale"]
    counts = summary["counts"]
    if not isinstance(scale, int) or not isinstance(counts, dict):
        raise QualificationError("invalid internal debug-view summary")
    area = scale * scale
    result = {}
    for name in ORDER:
        value = counts.get(name)
        if not isinstance(value, int) or value % area:
            raise QualificationError(f"debug color {name!r} does not align to whole cells")
        result[name] = value // area
    return result


def evaluate(
    virtual: dict[str, object],
    physical: dict[str, object],
    state: dict[str, object],
    require_miss: bool,
    require_invalidation: bool,
) -> list[str]:
    failures = []
    if state.get("active") is not True:
        failures.append("capture did not activate VSM")
    debug_views = state.get("debug_views")
    if not isinstance(debug_views, dict):
        raise QualificationError("telemetry has no debug_views object")
    if debug_views.get("available") is not True:
        failures.append("telemetry reports debug views unavailable")
    if debug_views.get("capture_only") is not True:
        failures.append("debug views are not marked capture-only")
    if debug_views.get("legend_order") != list(ORDER):
        failures.append("telemetry legend order differs from the image contract")
    if debug_views.get("colors") != [
        "#" + "".join(f"{channel:02x}" for channel in color) for color in COLORS
    ]:
        failures.append("telemetry palette differs from the image contract")
    expected_files = {
        "virtual_pages": "virtual-shadow-pages.png",
        "physical_occupancy": "virtual-shadow-physical.png",
        "legend": "virtual-shadow-legend.png",
        "report": "virtual-shadow-report.json",
    }
    for field, filename in expected_files.items():
        if debug_views.get(field) != filename:
            failures.append(f"debug-view {field} filename differs from the contract")

    capacity = integer(state, "physical_capacity", "VSM telemetry")
    expected_columns, expected_rows = physical_geometry(capacity)
    scale = physical.get("scale")
    if not isinstance(scale, int):
        raise QualificationError("invalid physical debug-view scale")
    if (physical.get("width"), physical.get("height")) != (
        expected_columns * scale,
        expected_rows * scale,
    ):
        failures.append("physical debug view does not match the reported pool capacity")

    virtual_counts = normalized_counts(virtual)
    physical_counts = normalized_counts(physical)
    raw_virtual_counts = virtual.get("counts")
    raw_physical_counts = physical.get("counts")
    if not isinstance(raw_virtual_counts, dict) or not isinstance(
        raw_physical_counts, dict
    ):
        raise QualificationError("invalid debug-view color counts")
    if raw_virtual_counts.get("unknown") != 0:
        failures.append("virtual debug view contains an unknown color")
    if raw_physical_counts.get("unknown") != 0:
        failures.append("physical debug view contains an unknown color")
    for name in ORDER[1:]:
        if virtual_counts[name] != physical_counts[name]:
            failures.append(f"virtual/physical page counts disagree for {name}")

    dirty = (
        virtual_counts["miss-unrendered"] + virtual_counts["invalidated"]
    )
    resident = sum(virtual_counts[name] for name in ORDER[1:])
    if resident != integer(state, "resident", "VSM telemetry"):
        failures.append("debug occupancy does not match resident-page telemetry")
    if dirty != integer(state, "dirty", "VSM telemetry"):
        failures.append("debug dirty states do not match dirty-page telemetry")
    levels = state.get("levels")
    if not isinstance(levels, list) or len(levels) != CLIP_LEVELS:
        raise QualificationError("telemetry must contain three clip levels")
    for index, level in enumerate(levels):
        if not isinstance(level, dict):
            raise QualificationError(f"telemetry clip level {index} is malformed")
        clean = integer(level, "resident", f"clip level {index}") - integer(
            level, "dirty", f"clip level {index}"
        )
        if virtual_counts[f"clip-level-{index}"] != clean:
            failures.append(f"clip-level-{index} clean-page count disagrees with telemetry")

    if require_miss and virtual_counts["miss-unrendered"] < 1:
        failures.append("capture contains no never-rendered cache miss")
    if require_invalidation and virtual_counts["invalidated"] < 1:
        failures.append("capture contains no previously-rendered invalidation")

    rows = state.get("per_light_cost")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise QualificationError("telemetry must contain one directional per-light cost row")
    row = rows[0]
    if row.get("light") != 0 or row.get("kind") != "directional":
        failures.append("per-light cost row does not identify directional light 0")
    mappings = {
        "requested_pages": "requested_pages",
        "cache_hits": "cache_hits",
        "cache_misses": "cache_misses",
        "invalidated_pages": "invalidated",
        "rendered_pages": "rendered",
        "resident_pages": "resident",
        "dirty_pages": "dirty",
        "clipmap_level_rebases": "clipmap_level_rebases",
        "dynamic_overlay_draws": "dynamic_overlay_draws",
        "shared_pool_bytes": "physical_bytes",
        "shared_metadata_staging_bytes": "gpu_overhead_bytes",
        "render_budget_pages": "render_budget",
    }
    for row_field, state_field in mappings.items():
        if integer(row, row_field, "per-light cost") != integer(
            state, state_field, "VSM telemetry"
        ):
            failures.append(f"per-light {row_field} disagrees with {state_field}")
    owned = integer(row, "physical_depth_bytes_owned", "per-light cost")
    physical_bytes = integer(state, "physical_bytes", "VSM telemetry")
    expected_owned = (
        physical_bytes * integer(state, "resident", "VSM telemetry") // capacity
        if capacity
        else 0
    )
    if owned != expected_owned:
        failures.append("per-light owned depth bytes disagree with resident occupancy")
    return failures


def compare(
    virtual_path: Path,
    physical_path: Path,
    legend_path: Path,
    telemetry_path: Path,
    require_miss: bool,
    require_invalidation: bool,
) -> dict[str, object]:
    state = state_from(telemetry_path)
    capacity = integer(state, "physical_capacity", "VSM telemetry")
    columns, rows = physical_geometry(capacity)
    virtual = image_summary(
        virtual_path, VIRTUAL_AXIS, VIRTUAL_AXIS * CLIP_LEVELS
    )
    physical = image_summary(physical_path, columns, rows)
    legend = legend_summary(legend_path)
    failures = evaluate(
        virtual, physical, state, require_miss, require_invalidation
    )
    return {
        "schema": SCHEMA,
        "images": {
            "virtual_pages": virtual,
            "physical_occupancy": physical,
            "legend": legend,
        },
        "telemetry": state,
        "requirements": {
            "miss_unrendered": require_miss,
            "invalidation": require_invalidation,
        },
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--virtual", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--legend", type=Path, required=True)
    parser.add_argument(
        "--telemetry",
        type=Path,
        required=True,
        help="same-frame virtual-shadow-report.json (full quality telemetry also accepted)",
    )
    parser.add_argument("--require-miss", action="store_true")
    parser.add_argument("--require-invalidation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(
            args.virtual,
            args.physical,
            args.legend,
            args.telemetry,
            args.require_miss,
            args.require_invalidation,
        )
    except QualificationError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["failures"]:
        for failure in result["failures"]:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: VSM debug views and per-light costs are internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
