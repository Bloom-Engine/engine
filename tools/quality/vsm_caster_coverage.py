#!/usr/bin/env python3
"""Qualify alpha-tested and skinned casters in the fixed VSM motion scene."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.quality.khronos_materials import QualificationError, png_rgb
except ModuleNotFoundError:
    from khronos_materials import QualificationError, png_rgb


SCHEMA = "bloom-vsm-caster-coverage-v1"
ALPHA_ROI = (0.12, 0.68, 0.45, 0.83)
SKINNED_ROI = (0.27, 0.78, 0.45, 0.98)
CHANGE_THRESHOLD = 6 / 255


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


def difference_metrics(
    width: int,
    height: int,
    reference: list[tuple[int, int, int]],
    candidate: list[tuple[int, int, int]],
    roi: tuple[float, float, float, float],
) -> dict[str, float | int | list[float]]:
    if len(reference) != width * height or len(candidate) != width * height:
        raise QualificationError("caster-coverage pixel count does not match dimensions")
    x0, y0, x1, y1 = (
        round(roi[0] * width),
        round(roi[1] * height),
        round(roi[2] * width),
        round(roi[3] * height),
    )
    roi_width = x1 - x0
    roi_height = y1 - y0
    changed = [False] * (roi_width * roi_height)
    differences: list[float] = []
    changed_differences: list[float] = []
    occupied_rows = 0
    row_segments = 0
    changed_x: list[int] = []
    changed_y: list[int] = []

    for y in range(roi_height):
        in_segment = False
        row_changed = False
        source_row = (y0 + y) * width
        roi_row = y * roi_width
        for x in range(roi_width):
            source = source_row + x0 + x
            difference = abs(luminance(reference[source]) - luminance(candidate[source]))
            differences.append(difference)
            is_changed = difference >= CHANGE_THRESHOLD
            changed[roi_row + x] = is_changed
            if is_changed:
                changed_differences.append(difference)
                changed_x.append(x)
                changed_y.append(y)
                row_changed = True
                if not in_segment:
                    row_segments += 1
            in_segment = is_changed
        occupied_rows += row_changed

    changed_pixels = len(changed_differences)
    roi_pixels = roi_width * roi_height
    if changed_pixels:
        bbox_pixels = (
            (max(changed_x) - min(changed_x) + 1)
            * (max(changed_y) - min(changed_y) + 1)
        )
        bounding_fill = changed_pixels / bbox_pixels
    else:
        bounding_fill = 0.0
    return {
        "roi_normalized": list(roi),
        "roi_pixels": roi_pixels,
        "changed_pixels": changed_pixels,
        "changed_ratio": changed_pixels / roi_pixels,
        "difference_mean": sum(differences) / roi_pixels,
        "changed_difference_p95": percentile(changed_differences, 0.95),
        "occupied_rows": occupied_rows,
        "segments_per_occupied_row": row_segments / max(1, occupied_rows),
        "changed_bounding_fill": bounding_fill,
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
    alpha: dict[str, float | int | list[float]],
    skinned: dict[str, float | int | list[float]],
    full_state: dict[str, object],
    alpha_control_state: dict[str, object],
    later_state: dict[str, object],
) -> list[str]:
    failures = []
    for name, state in (("full", full_state), ("later", later_state)):
        if state.get("active") is not True:
            failures.append(f"{name} capture did not activate VSM")
        if count(state, "page_cutout_draws", name) < 1:
            failures.append(f"{name} capture submitted no alpha-tested VSM page draw")
        if count(state, "page_skinned_draws", name) < 1:
            failures.append(f"{name} capture submitted no skinned VSM page draw")
        if count(state, "dynamic_overlay_rendered_pages", name) < 1:
            failures.append(f"{name} capture rendered no current-frame overlay page")
    if count(alpha_control_state, "page_cutout_draws", "alpha control") != 0:
        failures.append("alpha control still submitted an alpha-tested VSM page draw")
    if count(alpha_control_state, "page_skinned_draws", "alpha control") < 1:
        failures.append("alpha control lost the independent skinned caster")

    if int(alpha["changed_pixels"]) < 500:
        failures.append("alpha caster does not change enough ground-shadow pixels")
    if float(alpha["changed_ratio"]) < 0.005:
        failures.append("alpha caster ground-shadow coverage is too small")
    if float(alpha["changed_ratio"]) > 0.75:
        failures.append("alpha caster behaves like an opaque ROI fill")
    if float(alpha["segments_per_occupied_row"]) < 1.5:
        failures.append("alpha caster does not preserve segmented cutout coverage")
    if float(alpha["changed_difference_p95"]) < 0.05:
        failures.append("alpha caster shadow contrast is too weak")

    if int(skinned["changed_pixels"]) < 500:
        failures.append("skinned pose change does not move enough ground-shadow pixels")
    if float(skinned["changed_ratio"]) < 0.005:
        failures.append("skinned ground-shadow motion coverage is too small")
    if float(skinned["changed_difference_p95"]) < 0.03:
        failures.append("skinned moving-shadow contrast is too weak")
    return failures


def compare(
    full_path: Path,
    alpha_control_path: Path,
    skinned_later_path: Path,
    telemetry_path: Path,
    alpha_control_telemetry_path: Path,
    skinned_later_telemetry_path: Path,
) -> dict[str, object]:
    full_width, full_height, full_pixels = png_rgb(full_path)
    alpha_width, alpha_height, alpha_pixels = png_rgb(alpha_control_path)
    later_width, later_height, later_pixels = png_rgb(skinned_later_path)
    if len({(full_width, full_height), (alpha_width, alpha_height), (later_width, later_height)}) != 1:
        raise QualificationError("VSM caster captures have different dimensions")
    if full_width * 9 != full_height * 16:
        raise QualificationError("VSM caster captures must use a 16:9 viewport")

    alpha = difference_metrics(
        full_width, full_height, alpha_pixels, full_pixels, ALPHA_ROI
    )
    skinned = difference_metrics(
        full_width, full_height, full_pixels, later_pixels, SKINNED_ROI
    )
    full_state = vsm_state(telemetry_path)
    alpha_control_state = vsm_state(alpha_control_telemetry_path)
    later_state = vsm_state(skinned_later_telemetry_path)
    failures = evaluate(alpha, skinned, full_state, alpha_control_state, later_state)
    return {
        "schema": SCHEMA,
        "width": full_width,
        "height": full_height,
        "change_threshold": CHANGE_THRESHOLD,
        "images": {
            "full": {"path": str(full_path), "sha256": sha256_file(full_path)},
            "alpha_control": {
                "path": str(alpha_control_path),
                "sha256": sha256_file(alpha_control_path),
            },
            "skinned_later": {
                "path": str(skinned_later_path),
                "sha256": sha256_file(skinned_later_path),
            },
        },
        "alpha_tested": alpha,
        "skinned": skinned,
        "telemetry": {
            "full": full_state,
            "alpha_control": alpha_control_state,
            "skinned_later": later_state,
        },
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--alpha-control", type=Path, required=True)
    parser.add_argument("--skinned-later", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--alpha-control-telemetry", type=Path, required=True)
    parser.add_argument("--skinned-later-telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(
            args.full,
            args.alpha_control,
            args.skinned_later,
            args.telemetry,
            args.alpha_control_telemetry,
            args.skinned_later_telemetry,
        )
    except QualificationError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "alpha_tested": result["alpha_tested"],
                "skinned": result["skinned"],
            },
            sort_keys=True,
        )
    )
    if result["failures"]:
        for failure in result["failures"]:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: alpha-tested and skinned VSM casters are qualified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
