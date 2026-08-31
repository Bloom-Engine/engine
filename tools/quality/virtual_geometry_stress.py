#!/usr/bin/env python3
"""Prepare and run Bloom's cross-backend virtual-geometry stress gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as host_platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.quality.prepare_virtual_geometry_stress import build_stress_scene


LOGICAL_ID = "stress/10m"
QUALITY = "high"
DEFAULT_SCALING_INSTANCES = (1, 10, 100)


class StressQualificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_platform(system: str | None = None) -> str:
    system = (system or host_platform.system()).lower()
    if system == "darwin":
        return "macos"
    if system in {"linux", "windows"}:
        return system
    raise StressQualificationError(f"unsupported stress host platform: {system}")


def default_backend(platform_id: str) -> str:
    try:
        return {"macos": "metal", "linux": "vulkan", "windows": "dx12"}[
            platform_id
        ]
    except KeyError as error:
        raise StressQualificationError(
            f"no default stress backend for platform {platform_id!r}"
        ) from error


def select_index_entry(
    index: Mapping[str, Any], logical_id: str, platform_id: str, quality: str
) -> Mapping[str, Any]:
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise StressQualificationError("asset index has no entries array")
    selected = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("logical_id") == logical_id
        and entry.get("profile")
        == {"platform": platform_id, "quality": quality}
    ]
    if len(selected) != 1:
        raise StressQualificationError(
            f"asset index has {len(selected)} exact {logical_id} "
            f"{platform_id}/{quality} entries"
        )
    artifact = selected[0].get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        raise StressQualificationError("selected asset index entry has no artifact path")
    return selected[0]


def run_step(
    name: str, argv: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    print(f"==> {name}", flush=True)
    print("    " + " ".join(argv), flush=True)
    started = time.perf_counter()
    result = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        env=dict(environment) if environment is not None else None,
        check=False,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    if result.returncode != 0:
        raise StressQualificationError(
            f"{name} failed with exit code {result.returncode}"
        )
    return {"name": name, "duration_ms": duration_ms}


def pass_gpu_mean(summary: Mapping[str, Any], label: str) -> float:
    passes = summary.get("profile", {}).get("passes")
    if not isinstance(passes, list):
        raise StressQualificationError("stress summary has no profiler pass array")
    selected = [
        record
        for record in passes
        if isinstance(record, dict) and record.get("label") == label
    ]
    if len(selected) != 1 or not isinstance(selected[0].get("gpu_mean_ms"), (int, float)):
        raise StressQualificationError(f"stress summary has no exact {label} GPU mean")
    value = float(selected[0]["gpu_mean_ms"])
    if value <= 0.0:
        raise StressQualificationError(f"stress {label} GPU mean is not positive")
    return value


def evaluate_scaling(
    summaries: Sequence[Mapping[str, Any]], expected_instances: Sequence[int]
) -> dict[str, Any]:
    expected = sorted(set(expected_instances))
    by_instances: dict[int, Mapping[str, Any]] = {}
    for summary in summaries:
        placements = summary.get("placements")
        if not isinstance(placements, int):
            raise StressQualificationError("scaling summary has no integer placement count")
        if placements in by_instances:
            raise StressQualificationError(f"duplicate scaling point for {placements} instances")
        by_instances[placements] = summary
    if sorted(by_instances) != expected:
        raise StressQualificationError(
            f"scaling instances {sorted(by_instances)} do not match {expected}"
        )

    source_triangles = {summary.get("source_triangles") for summary in summaries}
    archive_clusters = {summary.get("archive_clusters") for summary in summaries}
    archive_pages = {summary.get("archive_pages") for summary in summaries}
    if len(source_triangles) != 1 or next(iter(source_triangles), 0) < 10_000_000:
        raise StressQualificationError("scaling sweep did not retain the same 10M source archive")
    if len(archive_clusters) != 1 or len(archive_pages) != 1:
        raise StressQualificationError("scaling sweep changed archive topology")

    points: list[dict[str, Any]] = []
    for instances in expected:
        summary = by_instances[instances]
        runtime = summary.get("runtime")
        profile = summary.get("profile")
        if not isinstance(runtime, dict) or not isinstance(profile, dict):
            raise StressQualificationError("scaling summary is missing runtime/profile telemetry")
        visible = runtime.get("last_visible_groups")
        frustum = runtime.get("last_frustum_culled_groups")
        selected = runtime.get("last_selected_count")
        measured_frames = summary.get("measured_frames")
        wall_ms = summary.get("measurement_wall_ms")
        if not all(isinstance(value, int) and value >= 0 for value in (visible, frustum, selected)):
            raise StressQualificationError("scaling summary has invalid cluster counters")
        if not isinstance(measured_frames, int) or measured_frames < 1:
            raise StressQualificationError("scaling summary has invalid measured frame count")
        if not isinstance(wall_ms, (int, float)) or wall_ms <= 0.0:
            raise StressQualificationError("scaling summary has invalid wall duration")
        candidate_groups = visible + frustum
        if candidate_groups < 1 or selected < 1:
            raise StressQualificationError("scaling point submitted no candidate/selected geometry")
        points.append(
            {
                "instances": instances,
                "candidate_groups": candidate_groups,
                "visible_groups": visible,
                "frustum_culled_groups": frustum,
                "selected_clusters": selected,
                "resident_pages": runtime.get("resident_pages"),
                "wall_frame_mean_ms": float(wall_ms) / measured_frames,
                "gpu_frame_mean_ms": profile.get("gpu_frame_mean_ms"),
                "selector_gpu_mean_ms": pass_gpu_mean(
                    summary, "virtual_geometry_hierarchy_selection"
                ),
                "draw_emission_gpu_mean_ms": pass_gpu_mean(
                    summary, "virtual_geometry_draw_emission"
                ),
            }
        )

    candidate_growth = points[-1]["candidate_groups"] / points[0]["candidate_groups"]
    selected_growth = points[-1]["selected_clusters"] / points[0]["selected_clusters"]
    selector_growth = points[-1]["selector_gpu_mean_ms"] / points[0]["selector_gpu_mean_ms"]
    instance_growth = points[-1]["instances"] / points[0]["instances"]
    if candidate_growth < instance_growth * 0.5 or selected_growth < instance_growth * 0.5:
        raise StressQualificationError(
            "fixed scaling sweep did not materially increase candidate/selected work"
        )
    if selector_growth > max(4.0, candidate_growth * 0.25):
        raise StressQualificationError(
            "hierarchy selection grew disproportionately to candidate groups"
        )
    return {
        "schema": "bloom-virtual-geometry-scaling-v1",
        "source_triangles": next(iter(source_triangles)),
        "archive_clusters": next(iter(archive_clusters)),
        "archive_pages": next(iter(archive_pages)),
        "instance_growth": instance_growth,
        "candidate_group_growth": candidate_growth,
        "selected_cluster_growth": selected_growth,
        "selector_gpu_growth": selector_growth,
        "points": points,
        "validation": "pass",
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_markdown_summary(path: Path, qualification: Mapping[str, Any]) -> None:
    runtime = qualification["runtime"]
    profile = runtime["profile"]
    streaming = runtime["runtime"]
    lines = [
        f"### Virtual geometry — {qualification['backend']}",
        "",
        "| Measurement | Result |",
        "|---|---:|",
        f"| Source triangles | {runtime['source_triangles']:,} |",
        f"| Artifact bytes | {qualification['artifact']['bytes']:,} |",
        f"| Resident pages | {runtime['resident_pages']:,} |",
        f"| I/O requests / failures | {streaming['streaming_io_requests']:,} / {streaming['streaming_io_failures']:,} |",
        f"| Wall mean | {runtime['measurement_wall_ms'] / runtime['measured_frames']:.4f} ms |",
        f"| GPU mean / p95 | {profile['gpu_frame_mean_ms']:.4f} / {profile['gpu_frame_p95_ms']:.4f} ms |",
        "",
        "| Instances | Candidate groups | Selected clusters | Selector GPU |",
        "|---:|---:|---:|---:|",
    ]
    for point in qualification["scaling"]["points"]:
        lines.append(
            f"| {point['instances']:,} | {point['candidate_groups']:,} | "
            f"{point['selected_clusters']:,} | {point['selector_gpu_mean_ms']:.4f} ms |"
        )
    lines.append("")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def qualify(arguments: argparse.Namespace) -> Path:
    work = arguments.work.resolve()
    output = arguments.out.resolve()
    if work == output or work in output.parents or output in work.parents:
        raise StressQualificationError("--work and --out must be separate trees")
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source_path = work / "source" / "stress-10m.gltf"
    store_path = work / "store"

    source = build_stress_scene(source_path)
    write_json(output / "source.json", source)
    steps: list[dict[str, Any]] = []
    cargo_cook = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        "tools/bloom-cook/Cargo.toml",
        "--",
    ]
    steps.append(
        run_step(
            "cook deterministic virtual geometry",
            [
                *cargo_cook,
                "geometry-store",
                arguments.logical_id,
                str(source_path),
                str(store_path),
                "--platform",
                arguments.platform,
                "--quality",
                arguments.quality,
                "--hierarchy-levels",
                "8",
                "--vertex-format",
                "quantized32",
            ],
        )
    )
    steps.append(run_step("build canonical asset index", [*cargo_cook, "asset-index", str(store_path)]))
    steps.append(
        run_step(
            "inspect canonical asset index",
            [*cargo_cook, "asset-index-inspect", str(store_path)],
        )
    )

    index_path = store_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = select_index_entry(
        index, arguments.logical_id, arguments.platform, arguments.quality
    )
    artifact_relative = entry["artifact"]["path"]
    artifact_path = (store_path / artifact_relative).resolve()
    try:
        artifact_path.relative_to(store_path.resolve())
    except ValueError as error:
        raise StressQualificationError("indexed stress artifact escapes store") from error
    if not artifact_path.is_file():
        raise StressQualificationError(f"indexed stress artifact is missing: {artifact_path}")

    environment = dict(os.environ)
    full_instance_count = max(arguments.scaling_instances)
    environment.update(
        {
            "BLOOM_VIRTUAL_STRESS_SCENE": str(source_path),
            "BLOOM_VIRTUAL_STRESS_ARCHIVE": str(artifact_path),
            "BLOOM_VIRTUAL_STRESS_STORE": str(store_path),
            "BLOOM_VIRTUAL_STRESS_LOGICAL_ID": arguments.logical_id,
            "BLOOM_VIRTUAL_STRESS_PLATFORM": arguments.platform,
            "BLOOM_VIRTUAL_STRESS_QUALITY": arguments.quality,
            "BLOOM_VIRTUAL_STRESS_BACKEND": arguments.backend,
            "BLOOM_VIRTUAL_STRESS_DIAGNOSTICS": str(output),
            "BLOOM_VIRTUAL_STRESS_WARMUP_FRAMES": str(arguments.warmup_frames),
            "BLOOM_VIRTUAL_STRESS_MEASURED_FRAMES": str(arguments.measured_frames),
            "BLOOM_VIRTUAL_STRESS_INSTANCE_LIMIT": str(full_instance_count),
        }
    )
    steps.append(
        run_step(
            f"run {arguments.backend} virtual-geometry stress",
            [
                "cargo",
                "test",
                "--release",
                "--manifest-path",
                "native/shared/Cargo.toml",
                "--test",
                "virtual_geometry_stress_runtime",
                "--features",
                "models3d",
                "--",
                "--nocapture",
            ],
            environment=environment,
        )
    )

    runtime_summary_path = output / "summary.json"
    runtime_summary = json.loads(runtime_summary_path.read_text(encoding="utf-8"))
    scaling_summaries = [runtime_summary]
    for instance_count in arguments.scaling_instances:
        if instance_count == full_instance_count:
            continue
        scaling_output = output / "scaling" / f"instances-{instance_count}"
        scaling_environment = dict(environment)
        scaling_environment["BLOOM_VIRTUAL_STRESS_DIAGNOSTICS"] = str(scaling_output)
        scaling_environment["BLOOM_VIRTUAL_STRESS_INSTANCE_LIMIT"] = str(instance_count)
        steps.append(
            run_step(
                f"run {arguments.backend} virtual-geometry scaling at {instance_count} instances",
                [
                    "cargo",
                    "test",
                    "--release",
                    "--manifest-path",
                    "native/shared/Cargo.toml",
                    "--test",
                    "virtual_geometry_stress_runtime",
                    "--features",
                    "models3d",
                    "--",
                    "--nocapture",
                ],
                environment=scaling_environment,
            )
        )
        scaling_summaries.append(
            json.loads((scaling_output / "summary.json").read_text(encoding="utf-8"))
        )
    scaling = evaluate_scaling(scaling_summaries, arguments.scaling_instances)
    qualification = {
        "schema": "bloom-cross-backend-virtual-geometry-stress-v2",
        "platform": arguments.platform,
        "backend": arguments.backend,
        "logical_id": arguments.logical_id,
        "quality": arguments.quality,
        "source": source,
        "index": {
            "schema": index.get("schema"),
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
            "entry": entry,
        },
        "artifact": {
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
        },
        "steps": steps,
        "runtime": runtime_summary,
        "scaling": scaling,
    }
    report = output / "qualification.json"
    write_json(report, qualification)
    write_markdown_summary(output / "qualification.md", qualification)
    print(f"virtual geometry qualification: {report}", flush=True)
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    platform_id = default_platform()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--platform", default=platform_id)
    parser.add_argument("--backend", choices=("metal", "vulkan", "dx12"))
    parser.add_argument("--quality", default=QUALITY)
    parser.add_argument("--logical-id", default=LOGICAL_ID)
    parser.add_argument("--warmup-frames", type=int, default=180)
    parser.add_argument("--measured-frames", type=int, default=120)
    parser.add_argument(
        "--scaling-instances",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCALING_INSTANCES),
    )
    arguments = parser.parse_args()
    if arguments.backend is None:
        arguments.backend = default_backend(arguments.platform)
    if arguments.warmup_frames < 1 or arguments.measured_frames < 1:
        parser.error("frame counts must be positive")
    arguments.scaling_instances = sorted(set(arguments.scaling_instances))
    if arguments.scaling_instances[0] < 1 or arguments.scaling_instances[-1] != 100:
        parser.error("scaling instances must be positive and include the full 100-instance point")
    return arguments


def main() -> int:
    try:
        qualify(parse_arguments())
    except (OSError, json.JSONDecodeError, StressQualificationError) as error:
        print(f"virtual geometry stress failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
