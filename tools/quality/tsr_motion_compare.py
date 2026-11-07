#!/usr/bin/env python3
"""Compare reconstructed camera motion with a matched native sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

try:
    from tools.quality.khronos_materials import QualificationError, png_rgb
except ModuleNotFoundError:
    from khronos_materials import QualificationError, png_rgb


SCHEMA = "bloom-tsr-motion-comparison-v1"
Pixel = tuple[int, int, int]
Frame = Sequence[Pixel]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numbered_sequence_paths(directory: Path, expected_frames: int) -> list[Path]:
    if expected_frames < 2:
        raise QualificationError("matched motion comparison requires at least two frames")
    paths = sorted(directory.glob("sequence-*.png"))
    expected = [directory / f"sequence-{index:03}.png" for index in range(expected_frames)]
    if paths != expected:
        raise QualificationError(
            f"{directory}: expected an exact sequence-000.."
            f"{expected_frames - 1:03}.png sequence"
        )
    return paths


def load_sequence(
    directory: Path, expected_frames: int
) -> tuple[int, int, list[Frame], list[Path]]:
    paths = numbered_sequence_paths(directory, expected_frames)
    width = 0
    height = 0
    frames: list[Frame] = []
    for path in paths:
        frame_width, frame_height, pixels = png_rgb(path)
        if frames and (frame_width, frame_height) != (width, height):
            raise QualificationError(f"{path}: sequence dimensions changed")
        width, height = frame_width, frame_height
        frames.append(pixels)
    return width, height, frames, paths


def reference_metrics(candidate: Sequence[Frame], native: Sequence[Frame]) -> dict[str, float]:
    if len(candidate) != len(native) or len(candidate) < 2:
        raise QualificationError("matched motion comparison requires equal multi-frame sequences")
    frame_squared_error = 0
    derivative_squared_error = 0
    frame_samples = 0
    derivative_samples = 0
    previous_candidate: Frame | None = None
    previous_native: Frame | None = None
    for frame_index, (candidate_frame, native_frame) in enumerate(zip(candidate, native)):
        if len(candidate_frame) != len(native_frame):
            raise QualificationError(f"frame {frame_index}: matched dimensions changed")
        for candidate_pixel, native_pixel in zip(candidate_frame, native_frame):
            for channel in range(3):
                difference = candidate_pixel[channel] - native_pixel[channel]
                frame_squared_error += difference * difference
                frame_samples += 1
        if previous_candidate is not None and previous_native is not None:
            for candidate_pixel, prior_candidate, native_pixel, prior_native in zip(
                candidate_frame, previous_candidate, native_frame, previous_native
            ):
                for channel in range(3):
                    candidate_delta = candidate_pixel[channel] - prior_candidate[channel]
                    native_delta = native_pixel[channel] - prior_native[channel]
                    difference = candidate_delta - native_delta
                    derivative_squared_error += difference * difference
                    derivative_samples += 1
        previous_candidate = candidate_frame
        previous_native = native_frame
    return {
        "native_frame_rmse": math.sqrt(frame_squared_error / frame_samples) / 255.0,
        "native_motion_derivative_rmse":
            math.sqrt(derivative_squared_error / derivative_samples) / 255.0,
    }


def enforce_no_regression(baseline: dict[str, float], candidate: dict[str, float]) -> None:
    for metric in ("native_frame_rmse", "native_motion_derivative_rmse"):
        if candidate[metric] > baseline[metric]:
            raise QualificationError(
                f"candidate {metric} {candidate[metric]:.9f} exceeds "
                f"baseline {baseline[metric]:.9f}"
            )


def relative_change(before: float, after: float) -> float:
    if before == 0.0:
        return 0.0
    return after / before - 1.0


def compare(
    baseline: Sequence[Frame], candidate: Sequence[Frame], native: Sequence[Frame]
) -> dict[str, object]:
    baseline_metrics = reference_metrics(baseline, native)
    candidate_metrics = reference_metrics(candidate, native)
    enforce_no_regression(baseline_metrics, candidate_metrics)
    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "relative_change": {
            metric: relative_change(baseline_metrics[metric], candidate_metrics[metric])
            for metric in baseline_metrics
        },
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        baseline_w, baseline_h, baseline, baseline_paths = load_sequence(
            args.baseline, args.expected_frames
        )
        candidate_w, candidate_h, candidate, candidate_paths = load_sequence(
            args.candidate, args.expected_frames
        )
        native_w, native_h, native, native_paths = load_sequence(
            args.native, args.expected_frames
        )
        if len({(baseline_w, baseline_h), (candidate_w, candidate_h), (native_w, native_h)}) != 1:
            raise QualificationError("baseline, candidate, and native dimensions differ")
        result = {
            "schema": SCHEMA,
            "width": baseline_w,
            "height": baseline_h,
            "frames": args.expected_frames,
            "comparison": compare(baseline, candidate, native),
            "artifacts": {
                "baseline": {
                    "directory": str(args.baseline.resolve()),
                    "first_sha256": sha256_file(baseline_paths[0]),
                    "last_sha256": sha256_file(baseline_paths[-1]),
                },
                "candidate": {
                    "directory": str(args.candidate.resolve()),
                    "first_sha256": sha256_file(candidate_paths[0]),
                    "last_sha256": sha256_file(candidate_paths[-1]),
                },
                "native": {
                    "directory": str(args.native.resolve()),
                    "first_sha256": sha256_file(native_paths[0]),
                    "last_sha256": sha256_file(native_paths[-1]),
                },
            },
        }
    except QualificationError as error:
        parser.error(str(error))
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
