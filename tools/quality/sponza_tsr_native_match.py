#!/usr/bin/env python3
"""Capture and qualify Sponza TSR camera motion against native rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Sequence

try:
    from tools.quality.khronos_materials import QualificationError, png_rgb
    from tools.quality.tsr_motion_compare import numbered_sequence_paths
except ModuleNotFoundError:
    from khronos_materials import QualificationError, png_rgb
    from tsr_motion_compare import numbered_sequence_paths


SCHEMA = "bloom-sponza-tsr-native-match-v1"
DIFF_METRICS = (
    "rmse_luminance",
    "ssim_luminance",
    "mean_oklab_delta",
    "mean_edge_delta",
    "percent_above_tolerance",
)
VARIANTS = (
    ("native", 1.0),
    ("fractional", 0.75),
    ("fractional-repeat", 0.75),
)
REPRODUCIBILITY_LIMITS = {
    "min_ssim": 0.999,
    "max_rmse": 0.002,
    "max_oklab_delta": 0.001,
    "max_edge_delta": 0.001,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise QualificationError(f"{path}: expected a PNG with an IHDR header")
    return struct.unpack(">II", header[16:24])


def verify_capture(
    directory: Path, expected_frames: int, width: int, height: int
) -> list[Path]:
    paths = numbered_sequence_paths(directory, expected_frames)
    for path in paths:
        if path.stat().st_size == 0:
            raise QualificationError(f"{path}: captured frame is empty")
        actual = png_dimensions(path)
        if actual != (width, height):
            raise QualificationError(
                f"{path}: captured {actual[0]}x{actual[1]}, expected {width}x{height}"
            )
    return paths


def capture_sequence(
    binary: Path,
    example_directory: Path,
    output: Path,
    render_scale: float,
    frames: int,
    warmup_frames: int,
    width: int,
    height: int,
    quality_preset: int,
    ssgi: int,
    fixed_timestep: str,
    timeout_seconds: float,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(binary),
        "--taa",
        "1",
        "--quality-preset",
        str(quality_preset),
        "--ssgi",
        str(ssgi),
        "--render-scale",
        str(render_scale),
        "--tsr-sequence",
        str(output),
        str(frames),
        str(warmup_frames),
        str(width),
        str(height),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "BLOOM_HEADLESS": "1",
            "BLOOM_HEADLESS_PIXEL_EXACT": "1",
            "BLOOM_QUALITY_FIXED_TIMESTEP": fixed_timestep,
        }
    )
    print(f"CAPTURE scale={render_scale}: {' '.join(command)}", flush=True)
    try:
        result = subprocess.run(
            command,
            cwd=example_directory,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise QualificationError(
            f"render scale {render_scale}: capture timed out after {timeout_seconds}s"
        ) from error
    (output / "capture.log").write_text(result.stdout, encoding="utf-8")
    done = f"BLOOM_TSR_SEQUENCE_DONE frames={frames} directory={output}"
    if result.returncode != 0 or done not in result.stdout or "BLOOM_TSR_SEQUENCE_ERROR" in result.stdout:
        raise QualificationError(
            f"render scale {render_scale}: capture failed; see {output / 'capture.log'}"
        )
    return verify_capture(output, frames, width, height)


def matched_reference_metrics(
    candidate_paths: Sequence[Path], native_paths: Sequence[Path]
) -> tuple[tuple[int, int], dict[str, float]]:
    if len(candidate_paths) != len(native_paths) or len(candidate_paths) < 2:
        raise QualificationError("native-match analysis requires equal multi-frame sequences")
    dimensions: tuple[int, int] | None = None
    frame_squared_error = 0
    derivative_squared_error = 0
    frame_samples = 0
    derivative_samples = 0
    prior_candidate = None
    prior_native = None
    for frame_index, (candidate_path, native_path) in enumerate(
        zip(candidate_paths, native_paths)
    ):
        candidate_w, candidate_h, candidate = png_rgb(candidate_path)
        native_w, native_h, native = png_rgb(native_path)
        if (candidate_w, candidate_h) != (native_w, native_h):
            raise QualificationError(f"frame {frame_index}: native-match dimensions differ")
        if dimensions is None:
            dimensions = (candidate_w, candidate_h)
        elif dimensions != (candidate_w, candidate_h):
            raise QualificationError(f"frame {frame_index}: sequence dimensions changed")
        for candidate_pixel, native_pixel in zip(candidate, native):
            for channel in range(3):
                difference = candidate_pixel[channel] - native_pixel[channel]
                frame_squared_error += difference * difference
                frame_samples += 1
        if prior_candidate is not None and prior_native is not None:
            for candidate_pixel, previous_candidate, native_pixel, previous_native in zip(
                candidate, prior_candidate, native, prior_native
            ):
                for channel in range(3):
                    candidate_delta = candidate_pixel[channel] - previous_candidate[channel]
                    native_delta = native_pixel[channel] - previous_native[channel]
                    difference = candidate_delta - native_delta
                    derivative_squared_error += difference * difference
                    derivative_samples += 1
        prior_candidate = candidate
        prior_native = native
    assert dimensions is not None
    return dimensions, {
        "native_frame_rmse": math.sqrt(frame_squared_error / frame_samples) / 255.0,
        "native_motion_derivative_rmse":
            math.sqrt(derivative_squared_error / derivative_samples) / 255.0,
    }


def sequence_activity(paths: Sequence[Path]) -> dict[str, float]:
    if len(paths) < 2:
        raise QualificationError("motion activity requires at least two frames")
    previous = None
    absolute_rgb = 0
    samples = 0
    for frame_index, path in enumerate(paths):
        _, _, pixels = png_rgb(path)
        if previous is not None:
            if len(pixels) != len(previous):
                raise QualificationError(f"frame {frame_index}: sequence dimensions changed")
            for current, prior in zip(pixels, previous):
                absolute_rgb += sum(abs(current[channel] - prior[channel]) for channel in range(3))
                samples += 3
        previous = pixels
    return {"adjacent_mean_absolute_rgb_8bit": absolute_rgb / samples}


def fractional_repeat_identity(
    fractional: Sequence[Path], repeated: Sequence[Path]
) -> dict[str, object]:
    if len(fractional) != len(repeated):
        raise QualificationError("fractional repeat frame count differs")
    hashes = []
    repeated_hashes = []
    divergent_frames = []
    for frame_index, (first, second) in enumerate(zip(fractional, repeated)):
        first_hash = sha256_file(first)
        second_hash = sha256_file(second)
        if first_hash != second_hash:
            divergent_frames.append(frame_index)
        hashes.append(first_hash)
        repeated_hashes.append(second_hash)
    return {
        "byte_identical": not divergent_frames,
        "divergent_frames": divergent_frames,
        "fractional_frame_sha256": hashes,
        "repeat_frame_sha256": repeated_hashes,
    }


def aggregate_frame_metrics(records: Sequence[dict[str, float]]) -> dict[str, object]:
    if not records:
        raise QualificationError("bloom-diff produced no frame metrics")
    aggregate = {}
    for metric in DIFF_METRICS:
        values = [float(record[metric]) for record in records]
        aggregate[metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return {"per_frame": list(records), "aggregate": aggregate}


def bloom_diff_sequence(
    diff_binary: Path,
    native_paths: Sequence[Path],
    candidate_paths: Sequence[Path],
    output: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for frame_index, (native, candidate) in enumerate(zip(native_paths, candidate_paths)):
        metrics_path = output / f"frame-{frame_index:03}.json"
        command = [
            str(diff_binary),
            "--reference",
            str(native),
            "--candidate",
            str(candidate),
            "--metrics-json",
            str(metrics_path),
            "--report-only",
            "--quiet",
        ]
        try:
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise QualificationError(f"bloom-diff timed out at frame {frame_index}") from error
        if result.returncode != 0 or not metrics_path.is_file():
            raise QualificationError(f"bloom-diff failed at frame {frame_index}: {result.stdout}")
        document = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = document.get("metrics", {})
        if any(metric not in metrics for metric in DIFF_METRICS):
            raise QualificationError(f"bloom-diff frame {frame_index}: incomplete metrics")
        records.append({metric: float(metrics[metric]) for metric in DIFF_METRICS})
    return aggregate_frame_metrics(records)


def enforce_reproducibility(frame_metrics: dict[str, object]) -> None:
    aggregate = frame_metrics["aggregate"]
    checks = (
        (
            "rmse_luminance",
            "max",
            REPRODUCIBILITY_LIMITS["max_rmse"],
            lambda measured, limit: measured <= limit,
        ),
        (
            "ssim_luminance",
            "min",
            REPRODUCIBILITY_LIMITS["min_ssim"],
            lambda measured, limit: measured >= limit,
        ),
        (
            "mean_oklab_delta",
            "max",
            REPRODUCIBILITY_LIMITS["max_oklab_delta"],
            lambda measured, limit: measured <= limit,
        ),
        (
            "mean_edge_delta",
            "max",
            REPRODUCIBILITY_LIMITS["max_edge_delta"],
            lambda measured, limit: measured <= limit,
        ),
    )
    for metric, statistic, limit, accepted in checks:
        measured = float(aggregate[metric][statistic])
        if not accepted(measured, limit):
            raise QualificationError(
                f"fractional repeat {metric}.{statistic} {measured:.9f} "
                f"violates governed reproducibility limit {limit:.9f}"
            )


def enforce_limits(metrics: dict[str, float], max_frame: float | None, max_motion: float | None) -> None:
    limits = (
        ("native_frame_rmse", max_frame),
        ("native_motion_derivative_rmse", max_motion),
    )
    for name, limit in limits:
        if limit is not None and metrics[name] > limit:
            raise QualificationError(f"{name} {metrics[name]:.9f} exceeds {limit:.9f}")


def enforce_visual_limits(
    frame_metrics: dict[str, object],
    max_luma_rmse: float | None,
    min_ssim: float | None,
    max_oklab_delta: float | None,
    max_edge_delta: float | None,
) -> None:
    aggregate = frame_metrics["aggregate"]
    limits = (
        ("rmse_luminance", "mean", max_luma_rmse, lambda measured, limit: measured <= limit),
        ("ssim_luminance", "mean", min_ssim, lambda measured, limit: measured >= limit),
        (
            "mean_oklab_delta",
            "mean",
            max_oklab_delta,
            lambda measured, limit: measured <= limit,
        ),
        (
            "mean_edge_delta",
            "mean",
            max_edge_delta,
            lambda measured, limit: measured <= limit,
        ),
    )
    for metric, statistic, limit, accepted in limits:
        if limit is None:
            continue
        measured = float(aggregate[metric][statistic])
        if not accepted(measured, limit):
            raise QualificationError(
                f"bloom-diff {metric}.{statistic} {measured:.9f} violates {limit:.9f}"
            )


def build_tools(repository: Path, build_sponza: bool, timeout_seconds: float) -> Path:
    if build_sponza:
        result = subprocess.run(
            [sys.executable, "tools/quality/build_example.py", "examples/sponza"],
            cwd=repository,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise QualificationError(f"failed to build Sponza:\n{result.stdout[-4000:]}")
    diff_binary = repository / "tools" / "bloom-diff" / "target" / "release" / "bloom-diff"
    if not diff_binary.is_file():
        result = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(repository / "tools" / "bloom-diff" / "Cargo.toml"),
            ],
            cwd=repository,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode != 0 or not diff_binary.is_file():
            raise QualificationError(f"failed to build bloom-diff:\n{result.stdout[-4000:]}")
    return diff_binary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--quality-preset", type=int, default=3, choices=range(5))
    parser.add_argument("--ssgi", type=int, default=1, choices=(0, 1))
    parser.add_argument("--fixed-timestep", default="0.016666666667")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--max-native-frame-rmse", type=float)
    parser.add_argument("--max-native-motion-derivative-rmse", type=float)
    parser.add_argument("--max-mean-luma-rmse", type=float)
    parser.add_argument("--min-mean-ssim", type=float)
    parser.add_argument("--max-mean-oklab-delta", type=float)
    parser.add_argument("--max-mean-edge-delta", type=float)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    example_directory = repository / "examples" / "sponza"
    binary = example_directory / "main"
    args.output = args.output.resolve()
    if args.frames < 2 or args.warmup_frames < 1 or args.width < 1 or args.height < 1:
        parser.error("frames must be >=2 and warmup/width/height must be positive")
    if args.analyze_only:
        if not args.output.is_dir():
            parser.error(f"analysis output does not exist: {args.output}")
    elif args.output.exists():
        parser.error(f"output already exists: {args.output}")

    try:
        diff_binary = build_tools(repository, not args.skip_build, args.timeout_seconds)
        if not args.analyze_only:
            if not binary.is_file():
                raise QualificationError(f"Sponza binary does not exist: {binary}")
            args.output.mkdir(parents=True)
            paths = {}
            for name, scale in VARIANTS:
                paths[name] = capture_sequence(
                    binary,
                    example_directory,
                    args.output / name,
                    scale,
                    args.frames,
                    args.warmup_frames,
                    args.width,
                    args.height,
                    args.quality_preset,
                    args.ssgi,
                    args.fixed_timestep,
                    args.timeout_seconds,
                )
        else:
            paths = {
                name: verify_capture(
                    args.output / name, args.frames, args.width, args.height
                )
                for name, _ in VARIANTS
            }

        repeat = fractional_repeat_identity(
            paths["fractional"], paths["fractional-repeat"]
        )
        repeat_metrics = bloom_diff_sequence(
            diff_binary,
            paths["fractional"],
            paths["fractional-repeat"],
            args.output / "fractional-repeat-diff",
            args.timeout_seconds,
        )
        enforce_reproducibility(repeat_metrics)
        dimensions, reference = matched_reference_metrics(paths["fractional"], paths["native"])
        native_activity = sequence_activity(paths["native"])
        fractional_activity = sequence_activity(paths["fractional"])
        if native_activity["adjacent_mean_absolute_rgb_8bit"] <= 0.0:
            raise QualificationError("native negative control did not move")
        enforce_limits(
            reference,
            args.max_native_frame_rmse,
            args.max_native_motion_derivative_rmse,
        )
        frame_metrics = bloom_diff_sequence(
            diff_binary,
            paths["native"],
            paths["fractional"],
            args.output / "bloom-diff",
            args.timeout_seconds,
        )
        enforce_visual_limits(
            frame_metrics,
            args.max_mean_luma_rmse,
            args.min_mean_ssim,
            args.max_mean_oklab_delta,
            args.max_mean_edge_delta,
        )
        document = {
            "schema": SCHEMA,
            "passed": True,
            "configuration": {
                "frames": args.frames,
                "warmup_frames": args.warmup_frames,
                "width": dimensions[0],
                "height": dimensions[1],
                "quality_preset": args.quality_preset,
                "ssgi_enabled": bool(args.ssgi),
                "native_render_scale": 1.0,
                "fractional_render_scale": 0.75,
                "fixed_timestep": args.fixed_timestep,
            },
            "fractional_repeat": {
                **repeat,
                "within_governed_reproducibility_bounds": True,
                "limits": REPRODUCIBILITY_LIMITS,
                "bloom_diff": repeat_metrics,
            },
            "native_match": reference,
            "sequence_activity": {
                "native": native_activity,
                "fractional": fractional_activity,
            },
            "bloom_diff": frame_metrics,
            "limits": {
                "max_native_frame_rmse": args.max_native_frame_rmse,
                "max_native_motion_derivative_rmse":
                    args.max_native_motion_derivative_rmse,
                "max_mean_luma_rmse": args.max_mean_luma_rmse,
                "min_mean_ssim": args.min_mean_ssim,
                "max_mean_oklab_delta": args.max_mean_oklab_delta,
                "max_mean_edge_delta": args.max_mean_edge_delta,
            },
        }
        (args.output / "result.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
    except (QualificationError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(document["native_match"], indent=2))
    print(f"PASS: wrote Sponza native-match evidence to {args.output / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
