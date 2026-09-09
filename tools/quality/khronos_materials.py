#!/usr/bin/env python3
"""Capture pinned Khronos alpha/transmission controls for human review."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "renderer-test"
KHRONOS_REVISION = "2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf"
RESULT_SCHEMA = "bloom-khronos-material-qualification-v1"
SOURCE_ROOT = (
    "https://raw.githubusercontent.com/KhronosGroup/"
    f"glTF-Sample-Assets/{KHRONOS_REVISION}/Models"
)


@dataclasses.dataclass(frozen=True)
class Case:
    id: str
    asset: str
    sha256: str
    license: str
    camera: tuple[float, float, float, float, float, float, float]
    resolution: tuple[int, int]
    semantic_gate: str

    @property
    def url(self) -> str:
        return f"{SOURCE_ROOT}/{self.asset}/glTF-Binary/{self.asset}.glb"

    @property
    def metadata_url(self) -> str:
        return f"{SOURCE_ROOT}/{self.asset}/README.md"


CASES = (
    Case(
        id="alpha-blend-mode",
        asset="AlphaBlendModeTest",
        sha256="37c3577d143071b42dd46e9d942b157837eb25c6340112171d7faecaa987b14e",
        license="CC-BY-4.0",
        camera=(0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 45.0),
        resolution=(960, 540),
        semantic_gate="green-checks",
    ),
    Case(
        id="transmission",
        asset="TransmissionTest",
        sha256="dd9732dae5517f8605ad4324d78b077b969c3e8357c056280d0a4e4b67797d15",
        license="CC0-1.0",
        camera=(
            -0.076755397,
            0.339420080,
            1.802846074,
            -0.115265710,
            0.199031961,
            0.813498748,
            34.515876027,
        ),
        resolution=(960, 720),
        semantic_gate="material-variation",
    ),
    Case(
        id="attenuation",
        asset="AttenuationTest",
        sha256="7ca161b7f8a9e4b2ac1f7f75816b5848bb31f3b4c226c4cb731b487c8809b756",
        license="CC-BY-4.0",
        camera=(0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 45.0),
        resolution=(960, 720),
        semantic_gate="attenuation-node-scale-ramp",
    ),
    Case(
        id="transmission-order",
        asset="TransmissionOrderTest",
        sha256="d904b6cd6c83792fd4a4d9ad4f0366bde76a63e347541c465f2ad4c5baf22a21",
        license="CC0-1.0",
        camera=(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 45.0),
        resolution=(960, 540),
        semantic_gate="material-variation",
    ),
)

DIAGNOSTIC_MARKERS = (
    "bloom gltf:",
    "validation error",
    "validation failed",
    "unsupported material",
    "unsupported extension",
    "invalid physical extension",
)


class QualificationError(RuntimeError):
    """A deterministic qualification prerequisite or invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_asset(case: Case, asset_dir: Path, timeout: float) -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / f"{case.asset}-{KHRONOS_REVISION[:7]}.glb"
    if path.exists():
        actual = sha256_file(path)
        if actual != case.sha256:
            raise QualificationError(
                f"{path}: hash mismatch (expected {case.sha256}, got {actual})"
            )
        return path

    request = urllib.request.Request(
        case.url,
        headers={"User-Agent": "Bloom-Khronos-Qualification/1"},
    )
    temporary = path.with_suffix(".download")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        temporary.write_bytes(data)
        actual = sha256_file(temporary)
        if actual != case.sha256:
            raise QualificationError(
                f"{case.asset}: downloaded hash mismatch "
                f"(expected {case.sha256}, got {actual})"
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def paeth(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    left_distance = abs(prediction - left)
    above_distance = abs(prediction - above)
    upper_left_distance = abs(prediction - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def png_rgb(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise QualificationError(f"{path}: not a PNG")
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while offset + 12 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
        elif chunk_type == b"IDAT":
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            break
    if (
        width is None
        or height is None
        or bit_depth != 8
        or color_type not in (2, 6)
        or interlace != 0
    ):
        raise QualificationError(f"{path}: expected non-interlaced RGB/RGBA8 PNG")

    channels = 3 if color_type == 2 else 4
    stride = width * channels
    raw = zlib.decompress(compressed)
    if len(raw) != height * (stride + 1):
        raise QualificationError(f"{path}: unexpected decompressed PNG size")
    previous = bytearray(stride)
    pixels: list[tuple[int, int, int]] = []
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        encoded = raw[cursor : cursor + stride]
        cursor += stride
        row = bytearray(stride)
        for index, value in enumerate(encoded):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                decoded = value
            elif filter_type == 1:
                decoded = value + left
            elif filter_type == 2:
                decoded = value + above
            elif filter_type == 3:
                decoded = value + ((left + above) // 2)
            elif filter_type == 4:
                decoded = value + paeth(left, above, upper_left)
            else:
                raise QualificationError(f"{path}: invalid PNG filter {filter_type}")
            row[index] = decoded & 0xFF
        pixels.extend(
            (row[index], row[index + 1], row[index + 2])
            for index in range(0, stride, channels)
        )
        previous = row
    return width, height, pixels


def attenuation_node_scale_samples(
    width: int,
    height: int,
    pixels: list[tuple[int, int, int]],
    pixel_density: int,
) -> list[dict[str, object]]:
    # Centers of the five front faces in the fixed face-on camera. Sampling a
    # small median patch avoids the chart grid lines and reflection outliers.
    centers = (
        (0.3385, 0.6875),
        (0.4115, 0.6875),
        (0.5000, 0.6875),
        (0.6145, 0.6875),
        (0.7760, 0.6875),
    )
    radius = max(1, 4 * pixel_density)
    samples: list[dict[str, object]] = []
    for normalized_x, normalized_y in centers:
        center_x = round(normalized_x * width)
        center_y = round(normalized_y * height)
        patch = [
            pixels[y * width + x]
            for y in range(max(0, center_y - radius), min(height, center_y + radius + 1))
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1))
        ]
        channels = [
            sorted(pixel[channel] for pixel in patch)
            for channel in range(3)
        ]
        middle = len(patch) // 2
        rgb = [channel[middle] for channel in channels]
        samples.append(
            {
                "rgb": rgb,
                "blue_minus_red": rgb[2] - rgb[0],
            }
        )
    return samples


def image_statistics(
    path: Path,
    requested_resolution: tuple[int, int],
    semantic_gate: str,
) -> dict[str, object]:
    width, height, pixels = png_rgb(path)
    requested_width, requested_height = requested_resolution
    if (
        width % requested_width != 0
        or height % requested_height != 0
        or width // requested_width != height // requested_height
    ):
        raise QualificationError(
            f"{path}: {width}x{height} is not an integer-density "
            f"{requested_width}x{requested_height} capture"
        )
    count = len(pixels)
    luma_sum = 0.0
    luma_min = 1.0
    luma_max = 0.0
    green_checks = 0
    blue_absorption = 0
    chromatic = 0
    for red, green, blue in pixels:
        value = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
        luma_sum += value
        luma_min = min(luma_min, value)
        luma_max = max(luma_max, value)
        green_checks += (
            green > red * 1.15 and green > blue * 1.05 and green > 80
        )
        blue_absorption += (
            blue > red * 1.10 and blue > green * 1.02 and blue > 80
        )
        chromatic += max(red, green, blue) - min(red, green, blue) > 25
    pixel_density = width // requested_width
    metrics: dict[str, object] = {
        "width": width,
        "height": height,
        "pixel_density": pixel_density,
        "mean_luminance": luma_sum / count,
        "luminance_range": luma_max - luma_min,
        "green_check_fraction": green_checks / count,
        "blue_absorption_fraction": blue_absorption / count,
        "chromatic_pixel_fraction": chromatic / count,
    }
    failures: list[str] = []
    if metrics["mean_luminance"] < 0.02:
        failures.append("capture is effectively black")
    if metrics["luminance_range"] < 0.10:
        failures.append("capture is effectively flat")
    if semantic_gate == "green-checks" and metrics["green_check_fraction"] < 0.0005:
        failures.append("expected alpha-mode green checks are absent")
    if semantic_gate == "attenuation-node-scale-ramp":
        samples = attenuation_node_scale_samples(
            width,
            height,
            pixels,
            pixel_density,
        )
        ramp = [int(sample["blue_minus_red"]) for sample in samples]
        metrics["node_scale_samples"] = samples
        metrics["node_scale_chroma_ramp"] = ramp
        if metrics["blue_absorption_fraction"] < 0.03:
            failures.append("expected volume absorption variation is absent")
        if ramp[-1] - ramp[0] < 15 or any(
            following < previous - 4
            for previous, following in zip(ramp, ramp[1:])
        ):
            failures.append(
                "node-scale row does not deepen absorption from 0.25 to 2.0"
            )
    if semantic_gate == "material-variation" and metrics["chromatic_pixel_fraction"] < 0.03:
        failures.append("expected material variation is absent")
    metrics["failures"] = failures
    return metrics


def diagnostic_lines(output: str) -> list[str]:
    return [
        line
        for line in output.splitlines()
        if any(marker in line.lower() for marker in DIAGNOSTIC_MARKERS)
    ]


def command_record(
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = -1
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True
    return {
        "argv": argv,
        "cwd": str(cwd),
        "returncode": returncode,
        "duration_ms": (time.perf_counter() - started) * 1000.0,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
    }


def render_case(
    case: Case,
    asset: Path,
    binary: Path,
    out_dir: Path,
    timeout: float,
) -> dict[str, object]:
    case_dir = out_dir / "cases" / case.id
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BLOOM_GLTF_REFRACTION"] = "1"
    captures: list[dict[str, object]] = []
    failures: list[str] = []
    camera = [format(value, ".12g") for value in case.camera]
    for suffix in ("a", "b"):
        image = case_dir / f"final-{suffix}.png"
        command = [
            str(binary),
            "--model",
            str(asset),
            "--camera",
            *camera,
            "--res",
            str(case.resolution[0]),
            str(case.resolution[1]),
            "--out",
            str(image),
        ]
        record = command_record(command, EXAMPLE_DIR, env, timeout)
        combined_log = f"{record['stdout']}\n{record['stderr']}"
        diagnostics = diagnostic_lines(combined_log)
        capture_failures: list[str] = []
        if record["returncode"] != 0:
            capture_failures.append(f"renderer exited {record['returncode']}")
        if diagnostics:
            capture_failures.append("material/import/validation diagnostics emitted")
        stats: dict[str, object] | None = None
        if image.is_file():
            try:
                stats = image_statistics(image, case.resolution, case.semantic_gate)
                capture_failures.extend(stats["failures"])
            except (OSError, QualificationError, zlib.error) as exc:
                capture_failures.append(str(exc))
        else:
            capture_failures.append("renderer did not produce a capture")
        image_hash = sha256_file(image) if image.is_file() else None
        captures.append(
            {
                "suffix": suffix,
                "image": str(image.relative_to(out_dir)),
                "sha256": image_hash,
                "statistics": stats,
                "diagnostics": diagnostics,
                "command": record,
                "failures": capture_failures,
            }
        )
        failures.extend(f"{suffix}: {failure}" for failure in capture_failures)
    deterministic = (
        captures[0]["sha256"] is not None
        and captures[0]["sha256"] == captures[1]["sha256"]
    )
    if not deterministic:
        failures.append("same-machine repeat captures are not byte-identical")
    return {
        "id": case.id,
        "asset": case.asset,
        "asset_path": str(asset),
        "asset_sha256": sha256_file(asset),
        "source": case.url,
        "metadata": case.metadata_url,
        "license": case.license,
        "camera": list(case.camera),
        "requested_resolution": list(case.resolution),
        "semantic_gate": case.semantic_gate,
        "captures": captures,
        "deterministic": deterministic,
        "automation_status": "pass" if not failures else "fail",
        "review_state": "candidate-human-review-required",
        "failures": failures,
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_summary(result: dict[str, object], out_dir: Path) -> None:
    lines = [
        "# Khronos material qualification",
        "",
        f"Automated status: **{str(result['automation_status']).upper()}**",
        "",
        "Review state: **candidate — human review required**. This command never "
        "installs or approves a visual baseline.",
        "",
        f"Pinned glTF-Sample-Assets revision: `{KHRONOS_REVISION}`",
        "",
        "| Case | Automated | Deterministic | Candidate |",
        "| --- | --- | --- | --- |",
    ]
    for case in result["cases"]:
        candidate = case["captures"][0]["image"]
        lines.append(
            f"| {case['id']} | {case['automation_status']} | "
            f"{case['deterministic']} | [{candidate}]({candidate}) |"
        )
    lines.extend(
        [
            "",
            "Automated gates cover pinned asset integrity, renderer exit status, "
            "supported-field diagnostics, valid/non-flat image output, coarse "
            "scene-specific color invariants, and exact same-machine repeatability. "
            "A reviewer must still judge semantic rendering and reference quality.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    asset_dir = (
        Path(args.asset_dir).resolve()
        if args.asset_dir
        else out_dir / "assets"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        case.id: fetch_asset(case, asset_dir, args.download_timeout)
        for case in CASES
    }
    binary = (
        Path(args.binary).resolve()
        if args.binary
        else EXAMPLE_DIR / "main"
    )
    build: dict[str, object] | None = None
    if not args.binary:
        build = command_record(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "quality" / "build_example.py"),
                str(EXAMPLE_DIR),
            ],
            REPO_ROOT,
            os.environ.copy(),
            args.timeout,
        )
        if build["returncode"] != 0:
            raise QualificationError("renderer-test build failed")
    if not binary.is_file():
        raise QualificationError(f"renderer-test binary is missing: {binary}")

    cases = [
        render_case(case, assets[case.id], binary, out_dir, args.timeout)
        for case in CASES
    ]
    failures = [
        f"{case['id']}: {failure}"
        for case in cases
        for failure in case["failures"]
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "automation_status": "pass" if not failures else "fail",
        "review_state": "candidate-human-review-required",
        "repository_commit": git_commit(),
        "khronos_revision": KHRONOS_REVISION,
        "binary": str(binary),
        "build": build,
        "cases": cases,
        "failures": failures,
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(result, out_dir)
    print(f"{result['automation_status'].upper()}: {out_dir / 'result.json'}")
    print(f"Review candidates: {out_dir / 'summary.md'}")
    return 0 if not failures else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Capture pinned Khronos alpha/transmission controls twice"
    )
    result.add_argument(
        "--out",
        default="tools/quality/out/khronos-materials",
        help="ignored output/evidence directory",
    )
    result.add_argument(
        "--asset-dir",
        help="asset cache (defaults to OUT/assets)",
    )
    result.add_argument(
        "--binary",
        help="existing renderer-test binary; skips the build",
    )
    result.add_argument("--timeout", type=float, default=120.0)
    result.add_argument("--download-timeout", type=float, default=60.0)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, QualificationError, urllib.error.URLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
