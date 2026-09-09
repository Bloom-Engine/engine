#!/usr/bin/env python3
"""Qualify a full-scene VSM camera transition against matched CSM controls."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path

try:
    from tools.quality.khronos_materials import QualificationError, png_rgb
except ModuleNotFoundError:
    from khronos_materials import QualificationError, png_rgb


SCHEMA = "bloom-vsm-motion-corpus-v1"
RESIDUAL_THRESHOLD = 0.03


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def luminance(pixel: tuple[int, int, int]) -> float:
    red, green, blue = pixel
    return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def largest_component(mask: bytearray, width: int, height: int) -> dict[str, float | int]:
    seen = bytearray(width * height)
    largest = {
        "pixels": 0,
        "pixel_ratio": 0.0,
        "width_ratio": 0.0,
        "height_ratio": 0.0,
        "bounding_fill": 0.0,
    }
    for start, selected in enumerate(mask):
        if not selected or seen[start]:
            continue
        seen[start] = 1
        pending = deque([start])
        pixels = 0
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0
        while pending:
            index = pending.popleft()
            x = index % width
            y = index // width
            pixels += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for adjacent_y in range(max(0, y - 1), min(height, y + 2)):
                row = adjacent_y * width
                for adjacent_x in range(max(0, x - 1), min(width, x + 2)):
                    adjacent = row + adjacent_x
                    if mask[adjacent] and not seen[adjacent]:
                        seen[adjacent] = 1
                        pending.append(adjacent)
        if pixels <= int(largest["pixels"]):
            continue
        box_width = max_x - min_x + 1
        box_height = max_y - min_y + 1
        largest = {
            "pixels": pixels,
            "pixel_ratio": pixels / (width * height),
            "width_ratio": box_width / width,
            "height_ratio": box_height / height,
            "bounding_fill": pixels / (box_width * box_height),
        }
    return largest


def residual_metrics(
    width: int,
    height: int,
    settled_vsm: list[tuple[int, int, int]],
    settled_csm: list[tuple[int, int, int]],
    motion_vsm: list[tuple[int, int, int]],
    motion_csm: list[tuple[int, int, int]],
) -> dict[str, object]:
    expected = width * height
    if any(
        len(pixels) != expected
        for pixels in (settled_vsm, settled_csm, motion_vsm, motion_csm)
    ):
        raise QualificationError("motion-corpus pixel count does not match dimensions")

    residuals = [
        (luminance(moving_vsm) - luminance(moving_csm))
        - (luminance(still_vsm) - luminance(still_csm))
        for still_vsm, still_csm, moving_vsm, moving_csm in zip(
            settled_vsm, settled_csm, motion_vsm, motion_csm
        )
    ]
    absolute = [abs(value) for value in residuals]
    selected = bytearray(value >= RESIDUAL_THRESHOLD for value in absolute)
    row_coverage = [
        sum(selected[y * width : (y + 1) * width]) / width for y in range(height)
    ]
    column_coverage = [
        sum(selected[y * width + x] for y in range(height)) / height
        for x in range(width)
    ]
    component = largest_component(selected, width, height)
    component["line_like"] = (
        int(component["pixels"]) >= 64
        and max(float(component["width_ratio"]), float(component["height_ratio"]))
        >= 0.45
        and float(component["bounding_fill"]) <= 0.25
    )
    return {
        "definition": "(motion_vsm-motion_csm)-(settled_vsm-settled_csm)",
        "threshold": RESIDUAL_THRESHOLD,
        "rmse": math.sqrt(sum(value * value for value in residuals) / expected),
        "mean_absolute": sum(absolute) / expected,
        "absolute_p95": percentile(absolute, 0.95),
        "absolute_p99": percentile(absolute, 0.99),
        "max_absolute": max(absolute, default=0.0),
        "ratio_over_0_02": sum(value >= 0.02 for value in absolute) / expected,
        "ratio_over_0_03": sum(value >= 0.03 for value in absolute) / expected,
        "ratio_over_0_05": sum(value >= 0.05 for value in absolute) / expected,
        "bright_ratio_over_0_05": sum(value >= 0.05 for value in residuals) / expected,
        "dark_ratio_over_0_05": sum(value <= -0.05 for value in residuals) / expected,
        "max_row_coverage_over_0_03": max(row_coverage, default=0.0),
        "max_column_coverage_over_0_03": max(column_coverage, default=0.0),
        "largest_component_over_0_03": component,
    }


def vsm_state(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        state = document["renderer_paths"]["virtual_shadows"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise QualificationError(f"invalid VSM telemetry {path}: {error}") from error
    if not isinstance(state, dict):
        raise QualificationError(f"invalid VSM telemetry state in {path}")
    return state


def count(state: dict[str, object], field: str, capture: str) -> int:
    value = state.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationError(
            f"{capture} telemetry field {field!r} must be an integer"
        )
    return value


def evaluate(
    metrics: dict[str, object],
    settled_vsm: dict[str, object],
    settled_csm: dict[str, object],
    motion_vsm: dict[str, object],
    motion_csm: dict[str, object],
) -> list[str]:
    failures = []
    for name, state in (("settled VSM", settled_vsm), ("motion VSM", motion_vsm)):
        if state.get("active") is not True:
            failures.append(f"{name} did not activate VSM")
        capacity = count(state, "physical_capacity", name)
        resident = count(state, "resident", name)
        dirty = count(state, "dirty", name)
        if capacity < 1 or resident > capacity or dirty > resident:
            failures.append(f"{name} residency is outside its physical budget")
    for name, state in (("settled CSM", settled_csm), ("motion CSM", motion_csm)):
        if state.get("active") is not False:
            failures.append(f"{name} unexpectedly activated VSM")
        if count(state, "physical_capacity", name) != 0:
            failures.append(f"{name} allocated a VSM physical pool")
        if count(state, "gpu_total_bytes", name) != 0:
            failures.append(f"{name} allocated VSM GPU bytes")

    if count(motion_vsm, "clipmap_level_rebases", "motion VSM") < 1:
        failures.append("motion VSM observed no clipmap rebase")
    if count(motion_vsm, "clipmap_pages_preserved", "motion VSM") < 1:
        failures.append("motion VSM preserved no cached page")
    if count(motion_vsm, "denied", "motion VSM") != 0:
        failures.append("motion VSM denied a demanded page")
    if count(motion_vsm, "evictions", "motion VSM") != 0:
        failures.append("motion VSM evicted a page during the fixed path")
    render_budget = count(motion_vsm, "render_budget", "motion VSM")
    if count(motion_vsm, "rendered", "motion VSM") > render_budget:
        failures.append("motion VSM exceeded its page render budget")
    if count(motion_vsm, "pending_render", "motion VSM") > render_budget:
        failures.append("motion VSM exceeded its pending-page budget")
    if count(motion_vsm, "gpu_total_bytes", "motion VSM") != count(
        settled_vsm, "gpu_total_bytes", "settled VSM"
    ):
        failures.append("camera motion changed the fixed VSM allocation")

    if float(metrics["rmse"]) > 0.01:
        failures.append("backend-isolated transition residual RMSE is too high")
    if float(metrics["absolute_p99"]) > 0.03:
        failures.append("backend-isolated transition residual p99 is too high")
    if float(metrics["ratio_over_0_03"]) > 0.01:
        failures.append("too much of the frame changed at page-artifact contrast")
    if float(metrics["bright_ratio_over_0_05"]) > 0.005:
        failures.append("transition contains a broad missing-shadow flash")
    if float(metrics["dark_ratio_over_0_05"]) > 0.005:
        failures.append("transition contains a broad stale/doubled shadow")
    if float(metrics["max_row_coverage_over_0_03"]) > 0.20:
        failures.append("transition contains a coherent horizontal seam or ring")
    if float(metrics["max_column_coverage_over_0_03"]) > 0.20:
        failures.append("transition contains a coherent vertical seam")
    component = metrics["largest_component_over_0_03"]
    if not isinstance(component, dict):
        raise QualificationError("invalid largest-component metrics")
    if float(component["pixel_ratio"]) > 0.01:
        failures.append("transition contains a large connected page artifact")
    if component["line_like"] is True:
        failures.append("transition contains a long seam/ring-like component")
    return failures


def compare(
    name: str,
    settled_vsm_path: Path,
    settled_csm_path: Path,
    motion_vsm_path: Path,
    motion_csm_path: Path,
    settled_vsm_telemetry: Path,
    settled_csm_telemetry: Path,
    motion_vsm_telemetry: Path,
    motion_csm_telemetry: Path,
) -> dict[str, object]:
    paths = {
        "settled_vsm": settled_vsm_path,
        "settled_csm": settled_csm_path,
        "motion_vsm": motion_vsm_path,
        "motion_csm": motion_csm_path,
    }
    decoded = {key: png_rgb(path) for key, path in paths.items()}
    dimensions = {(value[0], value[1]) for value in decoded.values()}
    if len(dimensions) != 1:
        raise QualificationError("motion-corpus captures have different dimensions")
    width, height = next(iter(dimensions))
    if width * 9 != height * 16:
        raise QualificationError("motion-corpus captures must use a 16:9 viewport")
    metrics = residual_metrics(
        width,
        height,
        decoded["settled_vsm"][2],
        decoded["settled_csm"][2],
        decoded["motion_vsm"][2],
        decoded["motion_csm"][2],
    )
    states = {
        "settled_vsm": vsm_state(settled_vsm_telemetry),
        "settled_csm": vsm_state(settled_csm_telemetry),
        "motion_vsm": vsm_state(motion_vsm_telemetry),
        "motion_csm": vsm_state(motion_csm_telemetry),
    }
    failures = evaluate(metrics, **states)
    return {
        "schema": SCHEMA,
        "name": name,
        "width": width,
        "height": height,
        "images": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "residual": metrics,
        "telemetry": states,
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--settled-vsm", type=Path, required=True)
    parser.add_argument("--settled-csm", type=Path, required=True)
    parser.add_argument("--motion-vsm", type=Path, required=True)
    parser.add_argument("--motion-csm", type=Path, required=True)
    parser.add_argument("--settled-vsm-telemetry", type=Path, required=True)
    parser.add_argument("--settled-csm-telemetry", type=Path, required=True)
    parser.add_argument("--motion-vsm-telemetry", type=Path, required=True)
    parser.add_argument("--motion-csm-telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(
            args.name,
            args.settled_vsm,
            args.settled_csm,
            args.motion_vsm,
            args.motion_csm,
            args.settled_vsm_telemetry,
            args.settled_csm_telemetry,
            args.motion_vsm_telemetry,
            args.motion_csm_telemetry,
        )
    except QualificationError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["residual"], sort_keys=True))
    for failure in result["failures"]:
        print(f"FAIL: {failure}")
    if result["failures"]:
        return 1
    print(f"PASS: {result['name']} VSM motion corpus is qualified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
