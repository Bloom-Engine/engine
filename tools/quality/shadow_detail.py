#!/usr/bin/env python3
"""Measure directional-shadow contact detail in the fixed VSM oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.quality.khronos_materials import QualificationError, png_rgb
except ModuleNotFoundError:
    from khronos_materials import QualificationError, png_rgb


SCHEMA = "bloom-vsm-contact-detail-v1"
ROI = (19 / 128, 5 / 12, 109 / 128, 7 / 9)
NEUTRAL_CHROMA = 7
STRONG_EDGE = 0.10


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
        raise QualificationError("contact-detail ROI contains no selected pixels")
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[index]


def detail_metrics(
    width: int,
    height: int,
    vsm_pixels: list[tuple[int, int, int]],
    csm_pixels: list[tuple[int, int, int]],
) -> dict[str, object]:
    if len(vsm_pixels) != width * height or len(csm_pixels) != width * height:
        raise QualificationError("contact-detail pixel count does not match dimensions")
    x0 = round(ROI[0] * width)
    y0 = round(ROI[1] * height)
    x1 = round(ROI[2] * width)
    y1 = round(ROI[3] * height)
    roi_width = x1 - x0
    roi_height = y1 - y0
    selected = [False] * (roi_width * roi_height)
    vsm_luma = [0.0] * len(selected)
    csm_luma = [0.0] * len(selected)

    for y in range(roi_height):
        source_row = (y0 + y) * width
        roi_row = y * roi_width
        for x in range(roi_width):
            source = source_row + x0 + x
            target = roi_row + x
            vsm = vsm_pixels[source]
            csm = csm_pixels[source]
            selected[target] = (
                max(vsm) - min(vsm) <= NEUTRAL_CHROMA
                and max(csm) - min(csm) <= NEUTRAL_CHROMA
            )
            vsm_luma[target] = luminance(vsm)
            csm_luma[target] = luminance(csm)

    selected_luma = {"vsm": [], "csm": []}
    edge_values = {"vsm": [], "csm": []}
    strong_edges = {"vsm": 0, "csm": 0}
    for index, keep in enumerate(selected):
        if keep:
            selected_luma["vsm"].append(vsm_luma[index])
            selected_luma["csm"].append(csm_luma[index])

    for y in range(1, roi_height - 1):
        for x in range(1, roi_width - 1):
            index = y * roi_width + x
            neighbors = (
                index,
                index - 1,
                index + 1,
                index - roi_width,
                index + roi_width,
            )
            if not all(selected[neighbor] for neighbor in neighbors):
                continue
            for name, values in (("vsm", vsm_luma), ("csm", csm_luma)):
                horizontal = abs(values[index + 1] - values[index - 1]) * 0.5
                vertical = (
                    abs(values[index + roi_width] - values[index - roi_width]) * 0.5
                )
                edge = max(horizontal, vertical)
                edge_values[name].append(edge)
                strong_edges[name] += edge > STRONG_EDGE

    if len(edge_values["vsm"]) < 100:
        raise QualificationError("contact-detail ROI has too few neutral edge samples")

    per_mode: dict[str, dict[str, float | int]] = {}
    for name in ("vsm", "csm"):
        low = percentile(selected_luma[name], 0.05)
        high = percentile(selected_luma[name], 0.95)
        per_mode[name] = {
            "neutral_pixels": len(selected_luma[name]),
            "edge_samples": len(edge_values[name]),
            "luminance_p05": low,
            "luminance_p95": high,
            "shadow_contrast_p95_p05": high - low,
            "edge_p99": percentile(edge_values[name], 0.99),
            "strong_edge_pixels": strong_edges[name],
        }

    csm_strong = max(1, strong_edges["csm"])
    csm_edge_p99 = max(1e-9, float(per_mode["csm"]["edge_p99"]))
    comparisons = {
        "strong_edge_ratio": strong_edges["vsm"] / csm_strong,
        "edge_p99_ratio": float(per_mode["vsm"]["edge_p99"]) / csm_edge_p99,
        "shadow_contrast_gain": (
            float(per_mode["vsm"]["shadow_contrast_p95_p05"])
            - float(per_mode["csm"]["shadow_contrast_p95_p05"])
        ),
    }
    failures = []
    if comparisons["strong_edge_ratio"] < 5.0:
        failures.append("VSM does not retain five times as many strong shadow edges")
    if comparisons["edge_p99_ratio"] < 1.35:
        failures.append("VSM 99th-percentile shadow edge is not at least 35% stronger")
    if comparisons["shadow_contrast_gain"] < 0.004:
        failures.append("VSM shadow contrast gain is below 0.004")
    return {
        "roi_normalized": list(ROI),
        "neutral_chroma_max_code_value": NEUTRAL_CHROMA,
        "strong_edge_threshold": STRONG_EDGE,
        "vsm": per_mode["vsm"],
        "csm": per_mode["csm"],
        "comparisons": comparisons,
        "failures": failures,
        "passed": not failures,
    }


def compare(vsm_path: Path, csm_path: Path) -> dict[str, object]:
    vsm_width, vsm_height, vsm_pixels = png_rgb(vsm_path)
    csm_width, csm_height, csm_pixels = png_rgb(csm_path)
    if (vsm_width, vsm_height) != (csm_width, csm_height):
        raise QualificationError("VSM and CSM captures have different dimensions")
    if vsm_width * 9 != vsm_height * 16:
        raise QualificationError("contact-detail captures must use a 16:9 viewport")
    result = {
        "schema": SCHEMA,
        "images": {
            "vsm": {"path": str(vsm_path), "sha256": sha256_file(vsm_path)},
            "csm": {"path": str(csm_path), "sha256": sha256_file(csm_path)},
        },
        "width": vsm_width,
        "height": vsm_height,
    }
    result.update(detail_metrics(vsm_width, vsm_height, vsm_pixels, csm_pixels))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vsm", type=Path, required=True)
    parser.add_argument("--csm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.vsm, args.csm)
    except QualificationError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparisons"], sort_keys=True))
    if result["failures"]:
        for failure in result["failures"]:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: VSM retains qualified directional contact detail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
