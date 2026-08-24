#!/usr/bin/env python3
"""Capture and compare detailed-Bistro temporal owner controls."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, Sequence

try:
    from tools.quality.khronos_materials import QualificationError, png_rgb
except ModuleNotFoundError:
    from khronos_materials import QualificationError, png_rgb


SCHEMA = "bloom-bistro-temporal-matrix-v1"
CHANGE_THRESHOLD = 8
VARIANTS = {
    "full": {"taa": True, "ssgi": True, "ssr": True, "occlusion": True},
    "no-taa": {"taa": False, "ssgi": True, "ssr": True, "occlusion": True},
    "no-ssgi": {"taa": True, "ssgi": False, "ssr": True, "occlusion": True},
    "no-ssr": {"taa": True, "ssgi": True, "ssr": False, "occlusion": True},
    "no-occlusion": {"taa": True, "ssgi": True, "ssr": True, "occlusion": False},
}
PROBE_ENV = (
    "BLOOM_BISTRO_SSGI_SCENE",
    "BLOOM_BISTRO_PROBE_DUMP_DIR",
    "BLOOM_BISTRO_PROBE_DUMP_SIZE",
    "BLOOM_BISTRO_PROBE_DUMP_CAMERA",
    "BLOOM_BISTRO_PROBE_DUMP_SSGI",
    "BLOOM_BISTRO_PROBE_DUMP_SSR",
    "BLOOM_BISTRO_PROBE_DUMP_TAA",
    "BLOOM_BISTRO_PROBE_DUMP_OCCLUSION",
    "BLOOM_BISTRO_PROBE_DUMP_MOVE",
    "BLOOM_BISTRO_PROBE_DUMP_SEQUENCE_FRAMES",
    "BLOOM_BISTRO_PROBE_DUMP_SEQUENCE_DIAGNOSTICS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def histogram_percentile(histogram: list[int], fraction: float) -> int:
    total = sum(histogram)
    if total == 0:
        return 0
    target = round(fraction * (total - 1))
    accumulated = 0
    for value, count in enumerate(histogram):
        accumulated += count
        if accumulated > target:
            return value
    return len(histogram) - 1


def largest_component(mask: bytearray, width: int, height: int) -> dict[str, float | int]:
    seen = bytearray(width * height)
    largest_pixels = 0
    largest_width = 0
    largest_height = 0
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
            for adjacent in (index - 1, index + 1, index - width, index + width):
                if adjacent < 0 or adjacent >= width * height:
                    continue
                if adjacent == index - 1 and x == 0:
                    continue
                if adjacent == index + 1 and x + 1 == width:
                    continue
                if mask[adjacent] and not seen[adjacent]:
                    seen[adjacent] = 1
                    pending.append(adjacent)
        if pixels > largest_pixels:
            largest_pixels = pixels
            largest_width = max_x - min_x + 1
            largest_height = max_y - min_y + 1
    return {
        "pixels": largest_pixels,
        "pixel_ratio": largest_pixels / (width * height),
        "width_ratio": largest_width / width,
        "height_ratio": largest_height / height,
    }


def luma8(pixel: tuple[int, int, int]) -> int:
    red, green, blue = pixel
    return (54 * red + 183 * green + 19 * blue) >> 8


def sequence_metrics(
    width: int,
    height: int,
    frames: Iterable[Sequence[tuple[int, int, int]]],
) -> dict[str, object]:
    expected = width * height
    minimum: list[list[int]] | None = None
    maximum: list[list[int]] | None = None
    previous: Sequence[tuple[int, int, int]] | None = None
    frame_count = 0
    adjacent_pairs = 0
    adjacent_rgb_sum = 0
    adjacent_luma_sum = 0
    adjacent_histogram = [0] * 256
    adjacent_changed_pixels = 0
    largest_adjacent_component = {"pixels": 0, "pixel_ratio": 0.0}
    largest_adjacent_pair = None

    for frame_index, frame in enumerate(frames):
        if len(frame) != expected:
            raise QualificationError(
                f"frame {frame_index} has {len(frame)} pixels; expected {expected}"
            )
        frame_count += 1
        if minimum is None:
            minimum = [list(pixel) for pixel in frame]
            maximum = [list(pixel) for pixel in frame]
        else:
            assert maximum is not None
            for index, pixel in enumerate(frame):
                for channel in range(3):
                    minimum[index][channel] = min(minimum[index][channel], pixel[channel])
                    maximum[index][channel] = max(maximum[index][channel], pixel[channel])

        if previous is not None:
            mask = bytearray(expected)
            for index, (prior, current) in enumerate(zip(previous, frame)):
                red = abs(current[0] - prior[0])
                green = abs(current[1] - prior[1])
                blue = abs(current[2] - prior[2])
                maximum_change = max(red, green, blue)
                adjacent_rgb_sum += red + green + blue
                adjacent_luma_sum += abs(luma8(current) - luma8(prior))
                adjacent_histogram[maximum_change] += 1
                if maximum_change > CHANGE_THRESHOLD:
                    mask[index] = 1
                    adjacent_changed_pixels += 1
            component = largest_component(mask, width, height)
            if component["pixels"] > largest_adjacent_component["pixels"]:
                largest_adjacent_component = component
                largest_adjacent_pair = [frame_index - 1, frame_index]
            adjacent_pairs += 1
        previous = frame

    if frame_count < 2 or minimum is None or maximum is None:
        raise QualificationError("temporal analysis requires at least two frames")

    range_histogram = [0] * 256
    range_sum = 0
    for low, high in zip(minimum, maximum):
        temporal_range = max(high[channel] - low[channel] for channel in range(3))
        range_histogram[temporal_range] += 1
        range_sum += temporal_range

    adjacent_samples = adjacent_pairs * expected
    return {
        "frames": frame_count,
        "change_threshold_8bit": CHANGE_THRESHOLD,
        "temporal_max_channel_range_8bit": {
            "mean": range_sum / expected,
            "p95": histogram_percentile(range_histogram, 0.95),
            "p99": histogram_percentile(range_histogram, 0.99),
            "max": max(index for index, count in enumerate(range_histogram) if count),
            "ratio_over_threshold": sum(range_histogram[CHANGE_THRESHOLD + 1 :]) / expected,
        },
        "adjacent_frames": {
            "pairs": adjacent_pairs,
            "mean_absolute_rgb_8bit": adjacent_rgb_sum / (adjacent_samples * 3),
            "mean_absolute_luma_8bit": adjacent_luma_sum / adjacent_samples,
            "max_channel_change_p95": histogram_percentile(adjacent_histogram, 0.95),
            "max_channel_change_p99": histogram_percentile(adjacent_histogram, 0.99),
            "changed_pixel_ratio": adjacent_changed_pixels / adjacent_samples,
            "largest_component_over_threshold": largest_adjacent_component,
            "largest_component_pair": largest_adjacent_pair,
        },
    }


def analyze_variant(directory: Path, expected_frames: int) -> dict[str, object]:
    paths = sorted(directory.glob("sequence-*.png"))
    if len(paths) != expected_frames:
        raise QualificationError(
            f"{directory}: found {len(paths)} sequence frames; expected {expected_frames}"
        )
    first_width, first_height, first_pixels = png_rgb(paths[0])

    def decoded_frames() -> Iterable[Sequence[tuple[int, int, int]]]:
        yield first_pixels
        for path in paths[1:]:
            width, height, pixels = png_rgb(path)
            if (width, height) != (first_width, first_height):
                raise QualificationError(f"{path}: sequence dimensions changed")
            yield pixels

    metrics = sequence_metrics(first_width, first_height, decoded_frames())
    metrics.update(
        {
            "width": first_width,
            "height": first_height,
            "first_frame": str(paths[0]),
            "last_frame": str(paths[-1]),
            "first_sha256": sha256_file(paths[0]),
            "last_sha256": sha256_file(paths[-1]),
        }
    )
    return metrics


def capture_is_complete(directory: Path, expected_frames: int) -> bool:
    """Return true only for an exact, non-empty numbered frame sequence."""
    expected = [directory / f"sequence-{index:03}.png" for index in range(expected_frames)]
    actual = sorted(directory.glob("sequence-*.png"))
    return actual == expected and all(path.stat().st_size > 0 for path in expected)


def reduction(control: float, full: float) -> float | None:
    return None if full == 0.0 else 1.0 - control / full


def control_deltas(results: dict[str, dict[str, object]]) -> dict[str, object]:
    if "full" not in results:
        return {}
    full = results["full"]
    full_range = float(full["temporal_max_channel_range_8bit"]["mean"])
    full_adjacent = float(full["adjacent_frames"]["mean_absolute_rgb_8bit"])
    full_component = float(
        full["adjacent_frames"]["largest_component_over_threshold"]["pixel_ratio"]
    )
    deltas = {}
    for name, result in results.items():
        if name == "full":
            continue
        deltas[name] = {
            "temporal_range_mean_reduction": reduction(
                float(result["temporal_max_channel_range_8bit"]["mean"]), full_range
            ),
            "adjacent_rgb_mean_reduction": reduction(
                float(result["adjacent_frames"]["mean_absolute_rgb_8bit"]),
                full_adjacent,
            ),
            "largest_component_reduction": reduction(
                float(
                    result["adjacent_frames"]["largest_component_over_threshold"][
                        "pixel_ratio"
                    ]
                ),
                full_component,
            ),
        }
    return deltas


def enforce_component_limit(
    results: dict[str, dict[str, object]], maximum_pixels: int | None
) -> None:
    if maximum_pixels is None:
        return
    if "full" not in results:
        raise QualificationError("the coherent-component quality gate requires the full variant")
    measured = int(
        results["full"]["adjacent_frames"]["largest_component_over_threshold"]["pixels"]
    )
    if measured > maximum_pixels:
        raise QualificationError(
            f"full variant coherent component is {measured} pixels; limit is {maximum_pixels}"
        )


def capture_variant(
    repository: Path,
    scene: Path,
    output: Path,
    name: str,
    size: str,
    sequence_frames: int,
) -> None:
    settings = VARIANTS[name]
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    for key in PROBE_ENV:
        environment.pop(key, None)
    environment.update(
        {
            "BLOOM_BISTRO_SSGI_SCENE": str(scene),
            "BLOOM_BISTRO_PROBE_DUMP_DIR": str(output),
            "BLOOM_BISTRO_PROBE_DUMP_SIZE": size,
            "BLOOM_BISTRO_PROBE_DUMP_MOVE": "1",
            "BLOOM_BISTRO_PROBE_DUMP_SEQUENCE_FRAMES": str(sequence_frames),
        }
    )
    if settings["taa"]:
        environment["BLOOM_BISTRO_PROBE_DUMP_TAA"] = "1"
    if settings["ssr"]:
        environment["BLOOM_BISTRO_PROBE_DUMP_SSR"] = "1"
    if not settings["ssgi"]:
        environment["BLOOM_BISTRO_PROBE_DUMP_SSGI"] = "0"
    if not settings["occlusion"]:
        environment["BLOOM_BISTRO_PROBE_DUMP_OCCLUSION"] = "0"
    command = [
        "cargo",
        "test",
        "--release",
        "--test",
        "golden_render",
        "temporal_history::ssr_quality::ssgi_quality::dump_detailed_bistro_probe_state",
        "--",
        "--exact",
        "--nocapture",
    ]
    print(f"CAPTURE {name}: {' '.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=repository / "native" / "shared",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "capture.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise QualificationError(
            f"{name}: capture failed with exit {result.returncode}; see {output / 'capture.log'}"
        )


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in variants if item not in VARIANTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {', '.join(unknown)}")
    if not variants:
        raise argparse.ArgumentTypeError("at least one variant is required")
    return variants


def parse_size(value: str) -> str:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT") from error
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return f"{width}x{height}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=parse_size, default="512x288")
    parser.add_argument("--sequence-frames", type=int, default=32, choices=range(2, 65))
    parser.add_argument(
        "--variants", type=parse_variants, default=list(VARIANTS), help="comma-separated control names"
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="analyze captures already in --output"
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse complete variants and capture missing ones"
    )
    parser.add_argument(
        "--max-largest-component-pixels",
        type=int,
        help="fail if the full variant's largest >8-level adjacent component exceeds this size",
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    if not args.scene.is_file():
        parser.error(f"scene does not exist: {args.scene}")
    if args.analyze_only and args.resume:
        parser.error("--analyze-only and --resume are mutually exclusive")
    if not args.analyze_only:
        if args.output.exists() and not args.resume:
            parser.error(f"output already exists: {args.output}")
        args.output.mkdir(parents=True, exist_ok=args.resume)

    try:
        results = {}
        for name in args.variants:
            directory = args.output / name
            if not args.analyze_only:
                if args.resume and directory.exists():
                    if capture_is_complete(directory, args.sequence_frames):
                        print(f"REUSE {name}: {directory}", flush=True)
                    else:
                        print(f"RECAPTURE incomplete {name}: {directory}", flush=True)
                        shutil.rmtree(directory)
                        capture_variant(
                            repository,
                            args.scene.resolve(),
                            directory,
                            name,
                            args.size,
                            args.sequence_frames,
                        )
                else:
                    capture_variant(
                        repository,
                        args.scene.resolve(),
                        directory,
                        name,
                        args.size,
                        args.sequence_frames,
                    )
            print(f"ANALYZE {name}: {directory}", flush=True)
            results[name] = analyze_variant(directory, args.sequence_frames)
            (directory / "metrics.json").write_text(
                json.dumps(results[name], indent=2) + "\n", encoding="utf-8"
            )
        document = {
            "schema": SCHEMA,
            "scene": str(args.scene.resolve()),
            "size": args.size,
            "sequence_frames": args.sequence_frames,
            "variants": {name: VARIANTS[name] for name in args.variants},
            "results": results,
            "control_reductions_relative_to_full": control_deltas(results),
            "quality_limits": {
                "max_largest_component_pixels": args.max_largest_component_pixels,
            },
        }
        enforce_component_limit(results, args.max_largest_component_pixels)
        (args.output / "matrix.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
    except QualificationError as error:
        parser.error(str(error))
    print(json.dumps(document["control_reductions_relative_to_full"], indent=2))
    print(f"PASS: wrote deterministic temporal matrix to {args.output / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
