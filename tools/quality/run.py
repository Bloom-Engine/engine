#!/usr/bin/env python3
"""Deterministic Bloom visual/performance qualification runner.

The runner deliberately owns orchestration only. Scene executables own pixels
and per-pass telemetry; bloom-diff owns image metrics. That keeps qualification
code out of production frame paths and makes every artifact reproducible from
the versioned manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "scenes.toml"
RESULT_SCHEMA = "bloom-quality-result-v1"
CAPABILITY_SNAPSHOT_SCHEMA = "bloom-renderer-capability-snapshot-v1"
REVIEW_SCHEMA = "bloom-quality-baseline-review-v1"
INSTALL_SCHEMA = "bloom-quality-baseline-install-v1"
REPRO_SCHEMA = "bloom-quality-reproducibility-v1"
ALLOWED_SUITES = {"quick", "full"}
ALLOWED_STATUSES = {"pass", "fail", "skip", "error"}
KNOWN_INTERMEDIATE_NAMES = {
    "hdr-scene",
    "scene-depth",
    "shadow-cascade-0",
    "shadow-cascade-1",
    "shadow-cascade-2",
    "ssgi",
    "ssgi-rejection-reason",
    "ssgi-temporal-confidence",
    "ssr",
    "ssr-raw",
    "ssr-rejection-reason",
    "ssr-temporal-confidence",
    "taa-motion",
    "taa-rejection-reason",
    "taa-reprojected-uv",
    "taa-temporal-confidence",
    "taa-reconstruction-footprint",
    "taa-detail-lock",
    "pt-motion",
    "pt-rejection-reason",
    "pt-reprojected-uv",
    "pt-temporal-confidence",
}
STABLE_CASE_METADATA_KEYS = (
    "id",
    "description",
    "quality_tier",
    "resolution",
    "render_scale",
    "seed",
    "fixed_timestep",
    "warmup_frames",
    "measured_frames",
    "camera",
    "settings",
    "assets",
    "feature_decision",
    "reference_target",
)
STABLE_TELEMETRY_METADATA_KEYS = (
    "schema",
    "fixed_timestep",
    "warmup_frames",
    "measured_frames",
    "quality_preset",
    "render_scale",
    "present_mode",
    "present_mode_code",
    "uncapped",
    "gpu_timestamps_available",
    "warmup_excluded",
    "shader_compilation_excluded",
    "adapter",
    "renderer_paths",
)
RENDERER_CAPABILITY_TIERS = {"baseline", "modern", "high-end"}
RENDERER_CAPABILITY_PATH_KEYS = {
    "materials",
    "geometry",
    "shadows",
    "gi",
    "reflections",
    "anti_aliasing",
    "textures",
    "path_tracing",
}
RENDERER_CAPABILITY_FEATURE_KEYS = {
    "texture_binding_array",
    "non_uniform_indexing",
    "indirect_first_instance",
    "ray_query",
}
RENDERER_CAPABILITY_LIMIT_KEYS = {
    "max_binding_array_elements_per_shader_stage",
    "max_binding_array_sampler_elements_per_shader_stage",
    "max_texture_array_layers",
    "max_sampled_textures_per_shader_stage",
    "max_samplers_per_shader_stage",
    "max_bind_groups",
    "max_color_attachments",
}
DEVICE_NEGOTIATION_LIMIT_KEYS = {
    "max_bind_groups",
    "max_color_attachments",
    "max_sampled_textures_per_shader_stage",
    "max_samplers_per_shader_stage",
    "max_storage_buffers_per_shader_stage",
    "max_uniform_buffer_binding_size",
    "max_binding_array_elements_per_shader_stage",
    "max_binding_array_sampler_elements_per_shader_stage",
}
STEADY_STATE_BIND_GROUP_SITES = {
    "scene_compose",
    "ssr_temporal",
    "upscale",
    "taa",
    "taa_reactive",
    "depth_of_field",
    "motion_blur",
    "subsurface_scattering",
    "contrast_adaptive_sharpen",
    "auto_exposure",
    "final_composite",
    "custom_post_pass",
    "visibility_buffer",
}


class QualityError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    cwd: str
    returncode: int
    duration_ms: float
    stdout: str
    stderr: str
    timed_out: bool


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capability_snapshot_failures(adapter: Mapping[str, Any]) -> list[str]:
    """Validate the capability evidence embedded in every native quality run."""
    failures: list[str] = []
    capability = adapter.get("renderer_capabilities")
    if not isinstance(capability, dict):
        return ["adapter did not include renderer_capabilities snapshot"]

    for key in ("detected", "selected"):
        if capability.get(key) not in RENDERER_CAPABILITY_TIERS:
            failures.append(f"renderer_capabilities.{key} is not a known tier")
    for key in ("requested", "forced"):
        value = capability.get(key)
        if value is not None and value not in RENDERER_CAPABILITY_TIERS:
            failures.append(f"renderer_capabilities.{key} is not null or a known tier")
    if adapter.get("capability_tier") != capability.get("selected"):
        failures.append("adapter capability_tier does not match selected renderer tier")
    forced = capability.get("forced")
    if forced is not None and forced != capability.get("selected"):
        failures.append("forced renderer tier does not match selected renderer tier")
    if capability.get("diagnostic") is not None and not isinstance(
        capability.get("diagnostic"), str
    ):
        failures.append("renderer_capabilities.diagnostic is not null or text")

    available = capability.get("available")
    if not isinstance(available, dict):
        failures.append("renderer_capabilities.available is missing")
    else:
        features = available.get("features")
        if not isinstance(features, dict):
            failures.append("renderer_capabilities.available.features is missing")
        else:
            for key in RENDERER_CAPABILITY_FEATURE_KEYS:
                if not isinstance(features.get(key), bool):
                    failures.append(
                        f"renderer_capabilities.available.features.{key} is not boolean"
                    )
        limits = available.get("limits")
        if not isinstance(limits, dict):
            failures.append("renderer_capabilities.available.limits is missing")
        else:
            for key in RENDERER_CAPABILITY_LIMIT_KEYS:
                value = limits.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    failures.append(
                        f"renderer_capabilities.available.limits.{key} is invalid"
                    )

    paths = capability.get("paths")
    if not isinstance(paths, dict):
        failures.append("renderer_capabilities.paths is missing")
    else:
        for key in RENDERER_CAPABILITY_PATH_KEYS:
            value = paths.get(key)
            if not isinstance(value, str) or not value:
                failures.append(f"renderer_capabilities.paths.{key} is missing")

    negotiation = adapter.get("device_negotiation")
    if not isinstance(negotiation, dict):
        failures.append("adapter did not include device_negotiation snapshot")
        return failures
    for key in ("preferred_tier", "selected_tier"):
        if negotiation.get(key) not in RENDERER_CAPABILITY_TIERS:
            failures.append(f"device_negotiation.{key} is not a known tier")
    if negotiation.get("selected_tier") != capability.get("selected"):
        failures.append("device negotiation tier does not match selected renderer tier")
    if negotiation.get("profile") not in {"native-full", "folded-mobile"}:
        failures.append("device_negotiation.profile is invalid")
    if not isinstance(negotiation.get("selected_request"), str) or not negotiation.get(
        "selected_request"
    ):
        failures.append("device_negotiation.selected_request is missing")
    if negotiation.get("fallback_cause") is not None and not isinstance(
        negotiation.get("fallback_cause"), str
    ):
        failures.append("device_negotiation.fallback_cause is not null or text")
    if not isinstance(negotiation.get("required_features"), str):
        failures.append("device_negotiation.required_features is missing")
    limits = negotiation.get("required_limits")
    if not isinstance(limits, dict):
        failures.append("device_negotiation.required_limits is missing")
    else:
        for key in DEVICE_NEGOTIATION_LIMIT_KEYS:
            value = limits.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                failures.append(f"device_negotiation.required_limits.{key} is invalid")
    return failures


def repo_path(raw: str, *, must_exist: bool = False) -> Path:
    path = (REPO_ROOT / raw).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise QualityError(f"path escapes repository: {raw!r}") from exc
    if must_exist and not path.exists():
        raise QualityError(f"required path does not exist: {raw}")
    return path


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    # Preserve the leading status column from `git status --short`; stripping
    # it truncates the first dirty path (` M docs/...` became `ocs/...`).
    return result.stdout.rstrip() if result.returncode == 0 else "unknown"


def stable_environment(adapter: Mapping[str, Any] | None) -> dict[str, Any]:
    dirty_paths = [
        line[3:]
        for line in git_text("status", "--short", "--untracked-files=all").splitlines()
        if len(line) >= 4
    ]
    return {
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_dirty": bool(dirty_paths),
        "git_dirty_paths": sorted(dirty_paths),
        "build_profile": "release",
        "os": platform.system().lower(),
        "os_release": platform.release(),
        "architecture": platform.machine().lower(),
        "python": platform.python_version(),
        "adapter": dict(adapter or {"availability": "not-reported"}),
    }


def capability_snapshot_artifact(
    environment: Mapping[str, Any], machine_class: str | None
) -> dict[str, Any]:
    adapter = environment.get("adapter")
    return {
        "schema": CAPABILITY_SNAPSHOT_SCHEMA,
        "git_commit": environment.get("git_commit"),
        "machine_class": machine_class,
        "adapter": dict(adapter) if isinstance(adapter, Mapping) else {
            "availability": "not-reported"
        },
    }


def write_capability_snapshot_artifact(
    out_dir: Path, environment: Mapping[str, Any], machine_class: str | None
) -> str:
    name = "capabilities.json"
    (out_dir / name).write_text(
        json.dumps(
            capability_snapshot_artifact(environment, machine_class),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return name


def load_adapter(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        env_path = os.environ.get("BLOOM_QUALITY_ADAPTER_JSON", "")
        path = Path(env_path) if env_path else None
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityError(f"cannot read adapter metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityError(f"adapter metadata must be a JSON object: {path}")
    return value


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        manifest = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise QualityError(f"cannot load manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise QualityError("quality manifest schema_version must be 1")
    validate_manifest(manifest)
    return manifest, hashlib.sha256(raw).hexdigest()


def list_of_strings(value: Any, where: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise QualityError(f"{where} must be an array of strings")
    if not allow_empty and not value:
        raise QualityError(f"{where} must not be empty")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    workflows = manifest.get("workflow")
    if not isinstance(workflows, dict):
        raise QualityError("manifest requires [workflow]")
    cases = manifest.get("case")
    if not isinstance(cases, list) or not cases:
        raise QualityError("manifest requires at least one [[case]]")
    ids: set[str] = set()
    case_by_id: dict[str, Mapping[str, Any]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise QualityError(f"case {index} must be a table")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise QualityError(f"case {index} requires a non-empty id")
        if case_id in ids:
            raise QualityError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        case_by_id[case_id] = case
        resolution = case.get("resolution")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or not all(isinstance(v, int) and v > 0 for v in resolution)
        ):
            raise QualityError(f"{case_id}.resolution must be two positive integers")
        for scalar in ("warmup_frames", "measured_frames", "seed"):
            if not isinstance(case.get(scalar), int) or case[scalar] < 0:
                raise QualityError(f"{case_id}.{scalar} must be a non-negative integer")
        if not isinstance(case.get("fixed_timestep"), (int, float)) or case["fixed_timestep"] <= 0:
            raise QualityError(f"{case_id}.fixed_timestep must be positive")
        if (
            not isinstance(case.get("render_scale"), (int, float))
            or case["render_scale"] <= 0
        ):
            raise QualityError(f"{case_id}.render_scale must be positive")
        if not isinstance(case.get("quality_tier"), str) or not case["quality_tier"]:
            raise QualityError(f"{case_id}.quality_tier must be a non-empty string")
        required_intermediates = list_of_strings(
            case.get("required_intermediates"),
            f"{case_id}.required_intermediates",
        )
        unknown_intermediates = set(required_intermediates) - KNOWN_INTERMEDIATE_NAMES
        if unknown_intermediates:
            raise QualityError(
                f"{case_id}.required_intermediates contains unknown names: "
                f"{sorted(unknown_intermediates)}"
            )
        capture = case.get("capture")
        if not isinstance(capture, dict):
            raise QualityError(f"{case_id} requires [case.capture]")
        list_of_strings(capture.get("command"), f"{case_id}.capture.command")
        if "prepare" in capture:
            list_of_strings(capture["prepare"], f"{case_id}.capture.prepare")
        if "build" in capture:
            list_of_strings(capture["build"], f"{case_id}.capture.build")
        reference = case.get("reference")
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise QualityError(f"{case_id} requires case.reference.path")
        for key in ("kind", "generation"):
            if not isinstance(reference.get(key), str) or not reference[key]:
                raise QualityError(f"{case_id}.reference.{key} must be non-empty")
        thresholds = case.get("thresholds")
        if not isinstance(thresholds, dict):
            raise QualityError(f"{case_id} requires [case.thresholds]")
        if not any(
            key in thresholds
            for key in ("min_ssim", "max_rmse", "max_oklab_delta", "max_edge_delta")
        ):
            raise QualityError(f"{case_id} must configure at least one visual threshold")
        camera = case.get("camera")
        if not isinstance(camera, dict):
            raise QualityError(f"{case_id} requires a versioned camera table")
        for vector in ("position", "target", "up"):
            values = camera.get(vector)
            if (
                not isinstance(values, list)
                or len(values) != 3
                or not all(isinstance(v, (int, float)) for v in values)
            ):
                raise QualityError(f"{case_id}.camera.{vector} must have three numbers")
        if not isinstance(camera.get("fov_y_degrees"), (int, float)):
            raise QualityError(f"{case_id}.camera.fov_y_degrees must be numeric")
        if not isinstance(camera.get("animation"), str) or not camera["animation"]:
            raise QualityError(f"{case_id}.camera.animation must be non-empty")
        if not isinstance(case.get("settings"), dict):
            raise QualityError(f"{case_id} requires [case.settings]")
        budgets = case.get("budgets")
        if not isinstance(budgets, dict):
            raise QualityError(f"{case_id} requires [case.budgets]")
        for key in (
            "machine_class",
            "max_cpu_frame_p95_ms",
            "max_gpu_frame_p95_ms",
            "max_vram_peak_mb",
        ):
            if key not in budgets:
                raise QualityError(f"{case_id}.budgets.{key} is required")
        assets = case.get("assets")
        if not isinstance(assets, list) or not assets:
            raise QualityError(f"{case_id}.assets must be a non-empty array")
        for asset_index, asset in enumerate(assets):
            if not isinstance(asset, dict):
                raise QualityError(f"{case_id}.assets[{asset_index}] must be a table")
            for key in ("path", "source", "license"):
                if not isinstance(asset.get(key), str) or not asset[key]:
                    raise QualityError(
                        f"{case_id}.assets[{asset_index}].{key} must be non-empty"
                    )
            if not isinstance(asset.get("sha256"), str) and not isinstance(
                asset.get("revision"), str
            ):
                raise QualityError(
                    f"{case_id}.assets[{asset_index}] requires sha256 or revision"
                )
    machine_classes = manifest.get("machine_class")
    if not isinstance(machine_classes, list) or not machine_classes:
        raise QualityError("manifest requires at least one [[machine_class]]")
    machine_ids: set[str] = set()
    for item in machine_classes:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise QualityError("every machine class requires an id")
        if item["id"] in machine_ids:
            raise QualityError(f"duplicate machine class id: {item['id']}")
        machine_ids.add(item["id"])
        for key in ("description", "os", "backend", "gpu"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise QualityError(f"machine class {item['id']}.{key} is required")
        list_of_strings(item.get("hard_metrics"), f"{item['id']}.hard_metrics")
    for case in cases:
        machine_id = case["budgets"]["machine_class"]
        if machine_id not in machine_ids:
            raise QualityError(
                f"{case['id']} references unknown machine class {machine_id!r}"
            )
    for suite in ALLOWED_SUITES:
        selected = list_of_strings(workflows.get(suite), f"workflow.{suite}")
        missing = sorted(set(selected) - ids)
        if missing:
            raise QualityError(f"workflow.{suite} references missing cases: {missing}")
    if not set(workflows["quick"]).issubset(set(workflows["full"])):
        raise QualityError("workflow.quick must be a subset of workflow.full")
    reproducibility = manifest.get("reproducibility")
    if not isinstance(reproducibility, dict):
        raise QualityError("manifest requires [reproducibility] noise bounds")
    for key in (
        "min_ssim",
        "max_rmse",
        "max_oklab_delta",
        "max_edge_delta",
        "max_cpu_mean_relative_delta",
        "max_cpu_mean_absolute_delta_ms",
        "max_gpu_mean_relative_delta",
        "max_gpu_mean_absolute_delta_ms",
        "max_cpu_p95_relative_delta",
        "max_cpu_p95_absolute_delta_ms",
        "max_gpu_p95_relative_delta",
        "max_gpu_p95_absolute_delta_ms",
    ):
        value = reproducibility.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            raise QualityError(f"reproducibility.{key} must be a non-negative number")
    fault_controls = manifest.get("negative_control", [])
    if not isinstance(fault_controls, list):
        raise QualityError("negative_control must be an array of tables")
    expected_faults = {
        "brdf-energy",
        "shadow-placement",
        "gi-leakage",
        "motion-history",
        "texture-orientation",
    }
    configured_faults = {item.get("fault") for item in fault_controls if isinstance(item, dict)}
    missing_faults = expected_faults - configured_faults
    if missing_faults:
        raise QualityError(f"negative controls missing: {sorted(missing_faults)}")
    for item in fault_controls:
        if not isinstance(item, dict) or item.get("case") not in case_by_id:
            raise QualityError(f"negative control references an unknown case: {item!r}")


def verify_assets(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for asset in case.get("assets", []):
        if not isinstance(asset, dict):
            raise QualityError(f"{case['id']}: asset entry must be a table")
        raw_path = asset.get("path")
        expected = asset.get("sha256")
        revision = asset.get("revision")
        if not isinstance(raw_path, str) or (
            not isinstance(expected, str) and not isinstance(revision, str)
        ):
            raise QualityError(
                f"{case['id']}: asset requires path plus sha256 or revision"
            )
        path = repo_path(raw_path)
        if not path.exists():
            source = asset.get("source")
            records.append(
                {
                    "path": raw_path,
                    "expected_sha256": expected,
                    "expected_revision": revision,
                    "status": "missing",
                    "source": source,
                    "license": asset.get("license", "unspecified"),
                }
            )
            continue
        if isinstance(revision, str):
            if not path.is_dir():
                status = "not-a-directory"
                actual_revision = None
                dirty = None
            else:
                probe = subprocess.run(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                actual_revision = probe.stdout.strip() if probe.returncode == 0 else None
                dirty_probe = subprocess.run(
                    ["git", "-C", str(path), "status", "--porcelain"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                dirty = bool(dirty_probe.stdout.strip()) if dirty_probe.returncode == 0 else None
                status = (
                    "pass"
                    if actual_revision == revision
                    and (dirty is False or bool(asset.get("allow_dirty", False)))
                    else "revision-mismatch-or-dirty"
                )
            records.append(
                {
                    "path": raw_path,
                    "expected_revision": revision,
                    "actual_revision": actual_revision,
                    "dirty": dirty,
                    "status": status,
                    "source": asset.get("source"),
                    "license": asset.get("license", "unspecified"),
                }
            )
            continue
        actual = sha256_file(path)
        records.append(
            {
                "path": raw_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "pass" if actual == expected else "hash-mismatch",
                "source": asset.get("source"),
                "license": asset.get("license", "unspecified"),
            }
        )
    return records


def placeholders(case: Mapping[str, Any], case_dir: Path) -> dict[str, str]:
    width, height = case["resolution"]
    camera = case["camera"]
    pos = camera["position"]
    target = camera["target"]
    return {
        "repo": str(REPO_ROOT),
        "case_dir": str(case_dir),
        "candidate": str(case_dir / "final.png"),
        "telemetry": str(case_dir / "telemetry.json"),
        "intermediates": str(case_dir / "intermediates"),
        "width": str(width),
        "height": str(height),
        "warmup_frames": str(case["warmup_frames"]),
        "measured_frames": str(case["measured_frames"]),
        "seed": str(case["seed"]),
        "fixed_timestep": format(float(case["fixed_timestep"]), ".12g"),
        "render_scale": format(float(case.get("render_scale", 1.0)), ".6g"),
        "quality_tier": str(case.get("quality_tier", "high")),
        "camera_px": str(pos[0]),
        "camera_py": str(pos[1]),
        "camera_pz": str(pos[2]),
        "camera_tx": str(target[0]),
        "camera_ty": str(target[1]),
        "camera_tz": str(target[2]),
        "camera_fov": str(camera["fov_y_degrees"]),
    }


def expand_argv(argv: Sequence[str], values: Mapping[str, str], where: str) -> list[str]:
    try:
        return [part.format_map(values) for part in argv]
    except KeyError as exc:
        raise QualityError(f"{where} uses unknown placeholder {exc}") from exc


def run_command(
    argv: Sequence[str],
    cwd: Path,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(os.environ, **(dict(env or {}))),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr += f"\nquality runner: timed out after {timeout_seconds:.1f}s\n"
    return CommandResult(
        argv=list(argv),
        cwd=str(cwd),
        returncode=returncode,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


def command_record(result: CommandResult) -> dict[str, Any]:
    return dataclasses.asdict(result)


def read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityError(f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityError(f"{what} must be a JSON object: {path}")
    return value


def png_dimensions(path: Path) -> tuple[int, int]:
    try:
        header = path.read_bytes()[:24]
    except OSError as exc:
        raise QualityError(f"cannot read PNG {path}: {exc}") from exc
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise QualityError(f"artifact is not a valid PNG: {path}")
    return (
        int.from_bytes(header[16:20], "big"),
        int.from_bytes(header[20:24], "big"),
    )


def steady_state_renderer_failures(renderer_paths: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    uploads = renderer_paths.get("steady_state_uploads")
    lighting = uploads.get("lighting") if isinstance(uploads, dict) else None
    if not isinstance(lighting, dict):
        failures.append("renderer paths did not report steady-state lighting uploads")
    else:
        write_count = lighting.get("write_count")
        byte_count = lighting.get("byte_count")
        full_bytes = lighting.get("full_buffer_bytes")
        if (
            not isinstance(write_count, int)
            or isinstance(write_count, bool)
            or not 0 <= write_count <= 3
        ):
            failures.append("steady-state lighting write_count is not an integer in [0, 3]")
        if (
            not isinstance(full_bytes, int)
            or isinstance(full_bytes, bool)
            or full_bytes <= 0
        ):
            failures.append("steady-state lighting full_buffer_bytes is not positive")
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
            or (
                isinstance(full_bytes, int)
                and not isinstance(full_bytes, bool)
                and byte_count > full_bytes
            )
        ):
            failures.append("steady-state lighting byte_count is invalid")

    resources = renderer_paths.get("steady_state_resources")
    bind_groups = (
        resources.get("bind_group_creations") if isinstance(resources, dict) else None
    )
    if not isinstance(bind_groups, dict):
        failures.append("renderer paths did not report steady-state bind-group creations")
        return failures
    total = bind_groups.get("total")
    sites = bind_groups.get("sites")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        failures.append("steady-state bind-group total is not a non-negative integer")
    if not isinstance(sites, dict):
        failures.append("steady-state bind-group sites are unavailable")
        return failures
    if set(sites) != STEADY_STATE_BIND_GROUP_SITES:
        failures.append("steady-state bind-group site set is incomplete or unknown")
        return failures
    counts = list(sites.values())
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in counts
    ):
        failures.append("steady-state bind-group site counts must be non-negative integers")
    elif isinstance(total, int) and not isinstance(total, bool) and sum(counts) != total:
        failures.append("steady-state bind-group total does not match named sites")
    elif total != 0:
        failures.append("steady-state bind-group creation remained after warm-up")

    graph_compiles = resources.get("graph_compiles")
    if (
        not isinstance(graph_compiles, int)
        or isinstance(graph_compiles, bool)
        or graph_compiles < 0
    ):
        failures.append("steady-state graph compile count is not a non-negative integer")
    elif graph_compiles != 0:
        failures.append("render graph recompiled after warm-up")

    pipelines = resources.get("pipeline_creations")
    pipeline_first_use = (
        pipelines.get("first_use") if isinstance(pipelines, dict) else None
    )
    if (
        not isinstance(pipeline_first_use, int)
        or isinstance(pipeline_first_use, bool)
        or pipeline_first_use < 0
    ):
        failures.append("steady-state pipeline first-use count is invalid")
    elif pipeline_first_use != 0:
        failures.append("pipeline creation remained after warm-up")

    encoders = resources.get("command_encoder_creations")
    if not isinstance(encoders, dict):
        failures.append("renderer paths did not report command-encoder creations")
    else:
        encoder_total = encoders.get("total")
        encoder_sites = encoders.get("sites")
        if encoder_sites != {"frame_submission": encoder_total}:
            failures.append("command-encoder total does not match the frame-submission site")
        if encoder_total != 1:
            failures.append("steady frame must create exactly one submission encoder")

    physical = resources.get("transient_physical_creations")
    if not isinstance(physical, dict):
        failures.append("renderer paths did not report transient physical creations")
    else:
        for kind in ("textures", "buffers"):
            count = physical.get(kind)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                failures.append(
                    f"steady-state transient physical {kind} count is invalid"
                )
            elif count != 0:
                failures.append(
                    f"transient physical {kind} were created after warm-up"
                )
    return failures


def telemetry_contract_failures(
    case: Mapping[str, Any], telemetry: Mapping[str, Any] | None
) -> list[str]:
    if telemetry is None:
        return ["capture did not produce telemetry.json"]
    failures: list[str] = []
    if telemetry.get("schema") != "bloom-quality-telemetry-v1":
        failures.append(f"unsupported telemetry schema {telemetry.get('schema')!r}")
    for key in ("warmup_frames", "measured_frames"):
        if telemetry.get(key) != case.get(key):
            failures.append(
                f"telemetry {key} {telemetry.get(key)!r} != requested {case.get(key)!r}"
            )
    observed_step = telemetry.get("fixed_timestep")
    if not isinstance(observed_step, (int, float)) or abs(
        float(observed_step) - float(case["fixed_timestep"])
    ) > 1e-6:
        failures.append(
            f"telemetry fixed_timestep {observed_step!r} != requested "
            f"{case['fixed_timestep']!r}"
        )
    expected_preset = case.get("settings", {}).get("quality_preset")
    if expected_preset is not None and telemetry.get("quality_preset") != expected_preset:
        failures.append(
            f"telemetry quality_preset {telemetry.get('quality_preset')!r} "
            f"!= requested {expected_preset!r}"
        )
    observed_scale = telemetry.get("render_scale")
    if not isinstance(observed_scale, (int, float)) or abs(
        float(observed_scale) - float(case["render_scale"])
    ) > 1e-6:
        failures.append(
            f"telemetry render_scale {observed_scale!r} != requested "
            f"{case['render_scale']!r}"
        )
    if not telemetry.get("uncapped", False):
        failures.append("telemetry did not prove uncapped execution")
    if not telemetry.get("warmup_excluded", False):
        failures.append("telemetry did not prove warm-up exclusion")
    if not telemetry.get("shader_compilation_excluded", False):
        failures.append("telemetry did not prove shader-compilation exclusion")
    adapter = telemetry.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("availability") != "reported":
        failures.append("telemetry did not report the native adapter")
    else:
        failures.extend(capability_snapshot_failures(adapter))
    renderer_paths = telemetry.get("renderer_paths")
    if not isinstance(renderer_paths, dict):
        failures.append("telemetry did not report active renderer paths")
    else:
        failures.extend(steady_state_renderer_failures(renderer_paths))
    for key in ("cpu_frame_mean_ms", "cpu_frame_p95_ms", "measurement_wall_ms"):
        value = telemetry.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            failures.append(f"telemetry {key} is unavailable")
    if telemetry.get("gpu_timestamps_available", False):
        for key in ("gpu_frame_mean_ms", "gpu_frame_p95_ms"):
            value = telemetry.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                failures.append(f"telemetry {key} is unavailable despite GPU timestamps")
    passes = telemetry.get("passes")
    if not isinstance(passes, list) or not passes:
        failures.append("telemetry did not report per-pass timings")
    return failures


def selected_machine_class(
    manifest: Mapping[str, Any], machine_id: str | None
) -> dict[str, Any] | None:
    if machine_id is None:
        return None
    for item in manifest.get("machine_class", []):
        if isinstance(item, dict) and item.get("id") == machine_id:
            return dict(item)
    raise QualityError(f"unknown machine class: {machine_id}")


def effective_features(adapter: Mapping[str, Any] | None) -> set[str]:
    from_env = {
        feature.strip()
        for feature in os.environ.get("BLOOM_QUALITY_FEATURES", "").split(",")
        if feature.strip()
    }
    if adapter:
        values = adapter.get("features", [])
        if isinstance(values, list):
            from_env.update(str(value) for value in values)
    return from_env


def feature_decision(case: Mapping[str, Any], features: set[str]) -> dict[str, Any]:
    required = set(case.get("required_features", []))
    forbidden = set(case.get("forbidden_features", []))
    missing = sorted(required - features)
    present_forbidden = sorted(forbidden & features)
    if not missing and not present_forbidden:
        return {
            "status": "native",
            "required": sorted(required),
            "forbidden": sorted(forbidden),
            "missing": [],
            "present_forbidden": [],
            "fallback": None,
        }
    fallback = case.get("fallback")
    if isinstance(fallback, dict):
        return {
            "status": "fallback",
            "required": sorted(required),
            "forbidden": sorted(forbidden),
            "missing": missing,
            "present_forbidden": present_forbidden,
            "fallback": dict(fallback),
        }
    return {
        "status": "unsupported",
        "required": sorted(required),
        "forbidden": sorted(forbidden),
        "missing": missing,
        "present_forbidden": present_forbidden,
        "fallback": None,
    }


def pending_feature_decision(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "runtime-probe",
        "required": sorted(set(case.get("required_features", []))),
        "forbidden": sorted(set(case.get("forbidden_features", []))),
        "missing": [],
        "present_forbidden": [],
        "fallback": None,
    }


def performance_failures(
    case: Mapping[str, Any],
    telemetry: Mapping[str, Any] | None,
    machine: Mapping[str, Any] | None,
) -> list[str]:
    if machine is None or not machine.get("hard_gate", False):
        return []
    budgets = case.get("budgets", {})
    required_class = budgets.get("machine_class")
    if required_class and required_class != machine.get("id"):
        return []
    if telemetry is None:
        return ["hard-gated machine produced no telemetry.json"]
    failures: list[str] = []
    adapter = telemetry.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("availability") != "reported":
        failures.append("hard-gated machine did not report native adapter metadata")
    else:
        expected_backend = str(machine.get("backend", "")).lower()
        actual_backend = str(adapter.get("backend", "")).lower()
        if expected_backend and actual_backend != expected_backend:
            failures.append(
                f"adapter backend {actual_backend!r} != machine class {expected_backend!r}"
            )
        expected_gpu = str(machine.get("gpu", "")).lower()
        actual_gpu = str(adapter.get("name", "")).lower()
        if expected_gpu and expected_gpu not in actual_gpu:
            failures.append(
                f"adapter {adapter.get('name')!r} != machine class GPU {machine.get('gpu')!r}"
            )
    hard_metrics = set(machine.get("hard_metrics", ["cpu", "gpu", "vram"]))
    mappings = (
        ("cpu", "cpu_frame_p95_ms", "max_cpu_frame_p95_ms"),
        ("gpu", "gpu_frame_p95_ms", "max_gpu_frame_p95_ms"),
        ("vram", "vram_peak_mb", "max_vram_peak_mb"),
    )
    for metric, measured_key, budget_key in mappings:
        if metric not in hard_metrics:
            continue
        limit = budgets.get(budget_key)
        if limit is None:
            continue
        measured = telemetry.get(measured_key)
        if measured is None:
            failures.append(f"{measured_key} unavailable for hard budget {budget_key}")
        elif float(measured) > float(limit):
            failures.append(f"{measured_key} {float(measured):.4f} > {float(limit):.4f}")
    if not telemetry.get("uncapped", False):
        failures.append("performance capture did not prove an uncapped/headless present mode")
    if not telemetry.get("warmup_excluded", False):
        failures.append("telemetry did not prove warm-up exclusion")
    if not telemetry.get("shader_compilation_excluded", False):
        failures.append("telemetry did not prove shader-compilation exclusion")
    if not telemetry.get("gpu_timestamps_available", False):
        failures.append("hard-gated machine did not report GPU timestamps")
    return failures


def diff_command(
    diff_bin: Path,
    case: Mapping[str, Any],
    reference: Path,
    candidate: Path,
    case_dir: Path,
    report_only: bool,
    seeded_fault: str | None = None,
) -> list[str]:
    thresholds = case["thresholds"]
    argv = [
        str(diff_bin),
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--heatmap",
        str(case_dir / "heatmap.png"),
        "--composite",
        str(case_dir / "comparison.png"),
        "--metrics-json",
        str(case_dir / "metrics.json"),
        "--tolerance",
        str(thresholds.get("pixel_tolerance", 0.02)),
    ]
    for key, flag in (
        ("min_ssim", "--min-ssim"),
        ("max_rmse", "--max-rmse"),
        ("max_oklab_delta", "--max-oklab-delta"),
        ("max_edge_delta", "--max-edge-delta"),
    ):
        if key in thresholds:
            argv.extend([flag, str(thresholds[key])])
    mask = thresholds.get("mask")
    if mask:
        argv.extend(["--mask", str(repo_path(str(mask), must_exist=True))])
    if seeded_fault:
        argv.extend(
            [
                "--seed-fault",
                seeded_fault,
                "--fault-output",
                str(case_dir / f"fault-{seeded_fault}.png"),
            ]
        )
    if report_only:
        argv.append("--report-only")
    return argv


def build_diff_tool(timeout_seconds: float) -> tuple[Path, CommandResult | None]:
    binary_name = "bloom-diff.exe" if os.name == "nt" else "bloom-diff"
    binary = REPO_ROOT / "tools/bloom-diff/target/release" / binary_name
    manifest = REPO_ROOT / "tools/bloom-diff/Cargo.toml"
    result = run_command(
        ["cargo", "build", "--release", "--manifest-path", str(manifest)],
        REPO_ROOT,
        timeout_seconds,
    )
    if result.returncode != 0 or not binary.exists():
        raise QualityError(
            "failed to build bloom-diff:\n"
            + result.stdout[-4000:]
            + "\n"
            + result.stderr[-4000:]
        )
    return binary, result


def run_case(
    case: Mapping[str, Any],
    out_dir: Path,
    diff_bin: Path,
    features: set[str],
    features_known: bool,
    machine: Mapping[str, Any] | None,
    report_only: bool,
    timeout_seconds: float,
    built_commands: set[tuple[str, ...]],
) -> dict[str, Any]:
    case_id = str(case["id"])
    case_dir = out_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    values = placeholders(case, case_dir)
    assets = verify_assets(case)
    decision = (
        feature_decision(case, features)
        if features_known
        else pending_feature_decision(case)
    )
    record: dict[str, Any] = {
        "id": case_id,
        "description": case.get("description", ""),
        "quality_tier": case.get("quality_tier", "high"),
        "resolution": case["resolution"],
        "render_scale": case.get("render_scale", 1.0),
        "seed": case["seed"],
        "fixed_timestep": case["fixed_timestep"],
        "warmup_frames": case["warmup_frames"],
        "measured_frames": case["measured_frames"],
        "camera": case["camera"],
        "settings": case.get("settings", {}),
        "assets": assets,
        "feature_decision": decision,
        "commands": [],
        "artifacts": {},
        "reference_target": str(case["reference"]["path"]),
        "status": "error",
        "failures": [],
    }
    asset_failures = [
        f"asset {item['path']}: {item['status']}"
        for item in assets
        if item["status"] != "pass"
    ]
    if asset_failures:
        record["failures"].extend(asset_failures)
        record["status"] = "fail"
        return record
    if decision["status"] == "unsupported":
        record["status"] = "skip"
        record["failures"].append(
            f"unsupported features: missing={decision['missing']} "
            f"forbidden={decision['present_forbidden']}"
        )
        return record
    capture = case["capture"]
    cwd = repo_path(capture.get("working_dir", "."), must_exist=True)
    prepare_argv_raw = capture.get("prepare")
    if prepare_argv_raw:
        prepare_argv = expand_argv(
            prepare_argv_raw, values, f"{case_id}.capture.prepare"
        )
        prepare_result = run_command(prepare_argv, cwd, timeout_seconds)
        record["commands"].append(
            {"kind": "prepare", **command_record(prepare_result)}
        )
        if prepare_result.returncode != 0:
            record["status"] = "error"
            record["failures"].append(
                f"asset preparation failed with exit {prepare_result.returncode}"
            )
            return record
    build_argv_raw = capture.get("build")
    if build_argv_raw:
        build_argv = expand_argv(build_argv_raw, values, f"{case_id}.capture.build")
        build_key = tuple([str(cwd), *build_argv])
        if build_key not in built_commands:
            build_result = run_command(build_argv, cwd, timeout_seconds)
            record["commands"].append({"kind": "build", **command_record(build_result)})
            if build_result.returncode != 0:
                record["status"] = "error"
                record["failures"].append(f"build failed with exit {build_result.returncode}")
                return record
            built_commands.add(build_key)
    candidate = Path(values["candidate"])
    telemetry_path = Path(values["telemetry"])
    intermediates = Path(values["intermediates"])
    intermediates.mkdir(parents=True, exist_ok=True)
    capture_argv = expand_argv(capture["command"], values, f"{case_id}.capture.command")
    env = {
        "BLOOM_HEADLESS": "1",
        "BLOOM_HEADLESS_PIXEL_EXACT": "1",
        "BLOOM_NO_FULLSCREEN": "1",
        "BLOOM_QUALITY": "1",
        "BLOOM_QUALITY_CASE": case_id,
        "BLOOM_QUALITY_SEED": str(case["seed"]),
        "BLOOM_QUALITY_FIXED_TIMESTEP": str(case["fixed_timestep"]),
        "BLOOM_QUALITY_WARMUP_FRAMES": str(case["warmup_frames"]),
        "BLOOM_QUALITY_MEASURED_FRAMES": str(case["measured_frames"]),
        "BLOOM_QUALITY_TELEMETRY": str(telemetry_path),
        "BLOOM_QUALITY_INTERMEDIATES": str(intermediates),
        "RUST_BACKTRACE": "1",
    }
    capture_result = run_command(capture_argv, cwd, timeout_seconds, env)
    record["commands"].append({"kind": "capture", **command_record(capture_result)})
    if capture_result.returncode != 0:
        record["status"] = "error"
        record["failures"].append(f"capture failed with exit {capture_result.returncode}")
        return record
    if not candidate.exists():
        record["status"] = "error"
        record["failures"].append(f"capture did not produce {candidate}")
        return record
    candidate_dimensions = png_dimensions(candidate)
    if candidate_dimensions != tuple(case["resolution"]):
        record["failures"].append(
            f"candidate dimensions {candidate_dimensions} != requested "
            f"{tuple(case['resolution'])}"
        )
    telemetry = read_json(telemetry_path, "telemetry") if telemetry_path.exists() else None
    if telemetry is not None and isinstance(telemetry.get("adapter"), dict):
        runtime_features = effective_features(telemetry["adapter"])
        decision = feature_decision(case, runtime_features)
        record["feature_decision"] = decision
        if decision["status"] == "unsupported":
            record["status"] = "skip"
            record["failures"].append(
                f"runtime adapter unsupported: missing={decision['missing']} "
                f"forbidden={decision['present_forbidden']}"
            )
            return record
    if telemetry is not None and telemetry.get("vram_peak_mb") is None:
        case_key = "".join(
            character if character.isalnum() else "_"
            for character in case_id.upper()
        )
        external_vram = os.environ.get(
            f"BLOOM_QUALITY_VRAM_PEAK_MB_{case_key}",
            os.environ.get("BLOOM_QUALITY_VRAM_PEAK_MB"),
        )
        if external_vram:
            try:
                telemetry["vram_peak_mb"] = float(external_vram)
                telemetry["vram_measurement_source"] = "hardware-runner"
            except ValueError:
                record["failures"].append(
                    f"invalid external VRAM measurement {external_vram!r}"
                )
    record["telemetry"] = telemetry
    record["failures"].extend(telemetry_contract_failures(case, telemetry))
    record["artifacts"]["candidate"] = str(candidate.relative_to(out_dir))
    intermediate_files = sorted(
        str(path.relative_to(out_dir)) for path in intermediates.glob("*.png") if path.is_file()
    )
    record["artifacts"]["intermediates"] = intermediate_files
    required_intermediates = set(case.get("required_intermediates", ["final"]))
    produced_names = {"final"} | {Path(path).stem for path in intermediate_files}
    missing_intermediates = sorted(required_intermediates - produced_names)
    if missing_intermediates:
        record["failures"].append(f"missing named intermediates: {missing_intermediates}")
    # Baselines are independently governed and may legitimately be absent
    # during initial corpus bring-up. Do not let that suppress the runtime,
    # resource, or hard-performance contracts that the completed capture can
    # already prove. In particular, report-only bootstrap runs must expose
    # budget failures before a human installs the first visual reference.
    record["failures"].extend(performance_failures(case, telemetry, machine))
    reference = repo_path(case["reference"]["path"])
    if not reference.exists():
        record["status"] = "fail"
        record["failures"].append(f"approved baseline missing: {case['reference']['path']}")
        if report_only:
            record["report_only_failure"] = True
        return record
    diff_argv = diff_command(
        diff_bin, case, reference, candidate, case_dir, report_only=False
    )
    diff_result = run_command(diff_argv, REPO_ROOT, timeout_seconds)
    record["commands"].append({"kind": "diff", **command_record(diff_result)})
    metrics_path = case_dir / "metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path, "diff metrics")
        record["metrics"] = metrics.get("metrics", {})
        record["visual_passed"] = bool(metrics.get("passed", False))
        record["artifacts"].update(
            {
                "reference": os.path.relpath(reference, out_dir),
                "metrics": str(metrics_path.relative_to(out_dir)),
                "heatmap": str((case_dir / "heatmap.png").relative_to(out_dir)),
                "comparison": str((case_dir / "comparison.png").relative_to(out_dir)),
            }
        )
    else:
        record["visual_passed"] = False
        record["failures"].append("bloom-diff did not produce metrics.json")
    if diff_result.returncode != 0:
        record["failures"].append(f"visual thresholds failed (exit {diff_result.returncode})")
    record["status"] = "pass" if not record["failures"] else "fail"
    if report_only and record["status"] == "fail":
        record["report_only_failure"] = True
    return record


def case_summary_row(case: Mapping[str, Any]) -> list[str]:
    metrics = case.get("metrics", {})
    telemetry = case.get("telemetry") or {}
    return [
        str(case["id"]),
        str(case["status"]).upper(),
        f"{float(metrics.get('ssim_luminance', 0.0)):.5f}" if metrics else "-",
        f"{float(metrics.get('mean_oklab_delta', 0.0)):.5f}" if metrics else "-",
        f"{float(metrics.get('mean_edge_delta', 0.0)):.5f}" if metrics else "-",
        (
            f"{float(telemetry['cpu_frame_p95_ms']):.3f}"
            if telemetry.get("cpu_frame_p95_ms") is not None
            else "-"
        ),
        (
            f"{float(telemetry['gpu_frame_p95_ms']):.3f}"
            if telemetry.get("gpu_frame_p95_ms") is not None
            else "-"
        ),
        "; ".join(str(item) for item in case.get("failures", [])),
    ]


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    escaped_rows = [
        [cell.replace("|", "\\|").replace("\n", " ") for cell in row] for row in rows
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in escaped_rows)
    return "\n".join(lines)


def write_summaries(result: Mapping[str, Any], out_dir: Path) -> None:
    rows = [case_summary_row(case) for case in result["cases"]]
    headers = [
        "Case",
        "Status",
        "SSIM",
        "OKLab Δ",
        "Edge Δ",
        "CPU p95 ms",
        "GPU p95 ms",
        "Notes",
    ]
    md = (
        f"# Bloom quality qualification: {result['suite']}\n\n"
        f"Overall: **{str(result['status']).upper()}**  \n"
        f"Commit: `{result['environment']['git_commit']}`  \n"
        f"Manifest: `{result['manifest_sha256']}`  \n"
        f"Machine class: `{result.get('machine_class') or 'report-only'}`\n\n"
        + markdown_table(headers, rows)
        + "\n"
    )
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    html_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        + "</tr>"
        for row in rows
    )
    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Bloom quality qualification</title>
<style>
body{{font:14px system-ui;margin:2rem;color:#ddd;background:#181a1b}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #555;padding:.45rem;text-align:left}}
th{{background:#292c2f}}code{{color:#9cdcfe}}.pass{{color:#7ee787}}.fail{{color:#ff7b72}}
</style>
<h1>Bloom quality qualification: {html.escape(str(result["suite"]))}</h1>
<p class="{html.escape(str(result["status"]))}">Overall: {html.escape(str(result["status"]).upper())}</p>
<p>Commit <code>{html.escape(str(result["environment"]["git_commit"]))}</code></p>
<table><thead><tr>{"".join(f"<th>{html.escape(h)}</th>" for h in headers)}</tr></thead>
<tbody>{html_rows}</tbody></table>
"""
    (out_dir / "summary.html").write_text(html_doc, encoding="utf-8")


def stable_result_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case in result.get("cases", []):
        if not isinstance(case, dict):
            continue
        stable_case = {
            key: case.get(key)
            for key in STABLE_CASE_METADATA_KEYS
        }
        telemetry = case.get("telemetry")
        stable_case["telemetry"] = (
            {
                key: telemetry.get(key)
                for key in STABLE_TELEMETRY_METADATA_KEYS
            }
            if isinstance(telemetry, dict)
            else None
        )
        cases.append(stable_case)
    return {
        "schema": result.get("schema"),
        "manifest_path": result.get("manifest_path"),
        "manifest_sha256": result.get("manifest_sha256"),
        "suite": result.get("suite"),
        "machine_class": result.get("machine_class"),
        "report_only": result.get("report_only"),
        "environment": result.get("environment"),
        "features": result.get("features"),
        "cases": cases,
    }


def result_artifact(result_path: Path, raw: Any, what: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise QualityError(f"{what} artifact path is missing")
    source_dir = result_path.parent.resolve()
    path = (source_dir / raw).resolve()
    try:
        path.relative_to(source_dir)
    except ValueError as exc:
        raise QualityError(f"{what} artifact escapes its result directory") from exc
    if not path.is_file():
        raise QualityError(f"{what} artifact does not exist: {path}")
    return path


def artifact_map(
    result_path: Path, case: Mapping[str, Any]
) -> dict[str, Path]:
    artifacts = case.get("artifacts")
    if not isinstance(artifacts, dict):
        raise QualityError(f"{case.get('id')}: result has no artifacts")
    mapped = {
        "final": result_artifact(
            result_path, artifacts.get("candidate"), f"{case.get('id')} final"
        )
    }
    intermediates = artifacts.get("intermediates", [])
    if not isinstance(intermediates, list):
        raise QualityError(f"{case.get('id')}: intermediates must be an array")
    for raw in intermediates:
        path = result_artifact(
            result_path, raw, f"{case.get('id')} intermediate"
        )
        name = path.stem
        if name in mapped:
            raise QualityError(f"{case.get('id')}: duplicate artifact name {name!r}")
        mapped[name] = path
    return mapped


def reproducibility_diff_command(
    diff_bin: Path,
    reference: Path,
    candidate: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    name: str,
) -> list[str]:
    return [
        str(diff_bin),
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--heatmap",
        str(output_dir / f"{name}-heatmap.png"),
        "--composite",
        str(output_dir / f"{name}-comparison.png"),
        "--metrics-json",
        str(output_dir / f"{name}-metrics.json"),
        "--tolerance",
        str(config.get("pixel_tolerance", 0.002)),
        "--min-ssim",
        str(config["min_ssim"]),
        "--max-rmse",
        str(config["max_rmse"]),
        "--max-oklab-delta",
        str(config["max_oklab_delta"]),
        "--max-edge-delta",
        str(config["max_edge_delta"]),
    ]


def timing_delta(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    metric: str,
    absolute_limit: float,
    relative_limit: float,
) -> dict[str, Any]:
    first_value = first.get(metric)
    second_value = second.get(metric)
    if not isinstance(first_value, (int, float)) or not isinstance(
        second_value, (int, float)
    ):
        return {
            "metric": metric,
            "first": first_value,
            "second": second_value,
            "passed": False,
            "failure": "metric unavailable",
        }
    absolute_delta = abs(float(first_value) - float(second_value))
    denominator = max(abs(float(first_value)), abs(float(second_value)), 1e-9)
    relative_delta = absolute_delta / denominator
    passed = absolute_delta <= absolute_limit or relative_delta <= relative_limit
    return {
        "metric": metric,
        "first": float(first_value),
        "second": float(second_value),
        "absolute_delta_ms": absolute_delta,
        "relative_delta": relative_delta,
        "absolute_limit_ms": absolute_limit,
        "relative_limit": relative_limit,
        "passed": passed,
    }


def check_reproducibility(args: argparse.Namespace) -> int:
    manifest, manifest_hash = load_manifest(Path(args.manifest).resolve())
    config = manifest["reproducibility"]
    first_path = Path(args.first).resolve()
    second_path = Path(args.second).resolve()
    first = read_json(first_path, "first quality result")
    second = read_json(second_path, "second quality result")
    for label, result in (("first", first), ("second", second)):
        if result.get("schema") != RESULT_SCHEMA:
            raise QualityError(
                f"{label} result has unsupported schema {result.get('schema')!r}"
            )
        if result.get("manifest_sha256") != manifest_hash:
            raise QualityError(
                f"{label} result was not produced by the selected manifest"
            )
    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        try:
            out_dir.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise QualityError(
                "refusing to replace a reproducibility output outside the repository"
            ) from exc
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    failures: list[str] = []
    metadata_equal = stable_result_metadata(first) == stable_result_metadata(second)
    if not metadata_equal:
        failures.append("stable metadata differs")
        (out_dir / "metadata-first.json").write_text(
            json.dumps(stable_result_metadata(first), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "metadata-second.json").write_text(
            json.dumps(stable_result_metadata(second), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    first_cases = {
        str(case.get("id")): case
        for case in first.get("cases", [])
        if isinstance(case, dict)
    }
    second_cases = {
        str(case.get("id")): case
        for case in second.get("cases", [])
        if isinstance(case, dict)
    }
    if set(first_cases) != set(second_cases):
        failures.append("case sets differ")
    diff_bin, build_result = build_diff_tool(args.timeout)
    case_records: list[dict[str, Any]] = []
    timing_specs = (
        (
            "cpu_frame_mean_ms",
            "max_cpu_mean_absolute_delta_ms",
            "max_cpu_mean_relative_delta",
        ),
        (
            "gpu_frame_mean_ms",
            "max_gpu_mean_absolute_delta_ms",
            "max_gpu_mean_relative_delta",
        ),
        (
            "cpu_frame_p95_ms",
            "max_cpu_p95_absolute_delta_ms",
            "max_cpu_p95_relative_delta",
        ),
        (
            "gpu_frame_p95_ms",
            "max_gpu_p95_absolute_delta_ms",
            "max_gpu_p95_relative_delta",
        ),
    )
    for case_id in sorted(set(first_cases) & set(second_cases)):
        first_case = first_cases[case_id]
        second_case = second_cases[case_id]
        case_dir = out_dir / case_id
        case_dir.mkdir()
        first_artifacts = artifact_map(first_path, first_case)
        second_artifacts = artifact_map(second_path, second_case)
        artifact_records: list[dict[str, Any]] = []
        if set(first_artifacts) != set(second_artifacts):
            failures.append(f"{case_id}: artifact sets differ")
        for name in sorted(set(first_artifacts) & set(second_artifacts)):
            reference = first_artifacts[name]
            candidate = second_artifacts[name]
            first_sha = sha256_file(reference)
            second_sha = sha256_file(candidate)
            artifact_record: dict[str, Any] = {
                "name": name,
                "first_sha256": first_sha,
                "second_sha256": second_sha,
                "identical_bytes": first_sha == second_sha,
                "passed": True,
            }
            if first_sha != second_sha:
                safe_name = "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in name
                )
                argv = reproducibility_diff_command(
                    diff_bin, reference, candidate, config, case_dir, safe_name
                )
                diff_result = run_command(argv, REPO_ROOT, args.timeout)
                metrics_path = case_dir / f"{safe_name}-metrics.json"
                artifact_record["command"] = command_record(diff_result)
                artifact_record["metrics"] = (
                    read_json(metrics_path, "reproducibility metrics").get("metrics", {})
                    if metrics_path.exists()
                    else None
                )
                artifact_record["passed"] = diff_result.returncode == 0
            if not artifact_record["passed"]:
                failures.append(f"{case_id}: unstable pixels in {name}")
            artifact_records.append(artifact_record)
        first_telemetry = first_case.get("telemetry")
        second_telemetry = second_case.get("telemetry")
        timing_records: list[dict[str, Any]] = []
        if isinstance(first_telemetry, dict) and isinstance(second_telemetry, dict):
            for metric, absolute_key, relative_key in timing_specs:
                timing_records.append(
                    timing_delta(
                        first_telemetry,
                        second_telemetry,
                        metric,
                        float(config[absolute_key]),
                        float(config[relative_key]),
                    )
                )
        else:
            timing_records.append(
                {
                    "metric": "telemetry",
                    "passed": False,
                    "failure": "telemetry unavailable",
                }
            )
        for timing in timing_records:
            if not timing["passed"]:
                failures.append(
                    f"{case_id}: unstable or missing timing {timing['metric']}"
                )
        case_records.append(
            {
                "id": case_id,
                "artifacts": artifact_records,
                "timing": timing_records,
                "passed": all(item["passed"] for item in artifact_records)
                and all(item["passed"] for item in timing_records),
            }
        )
    report = {
        "schema": REPRO_SCHEMA,
        "manifest_sha256": manifest_hash,
        "first_result_sha256": sha256_file(first_path),
        "second_result_sha256": sha256_file(second_path),
        "metadata_identical": metadata_equal,
        "noise_bounds": dict(config),
        "build": command_record(build_result) if build_result else None,
        "cases": case_records,
        "failures": failures,
        "passed": not failures,
    }
    (out_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[reproducibility] {'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"  - {failure}")
    print(f"[reproducibility] result: {out_dir / 'result.json'}")
    return 0 if not failures else 1


def execute_suite(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest, manifest_hash = load_manifest(manifest_path)
    adapter = load_adapter(Path(args.adapter_json).resolve() if args.adapter_json else None)
    machine = selected_machine_class(manifest, args.machine_class)
    selected_ids = list(manifest["workflow"][args.suite])
    if args.case:
        requested = set(args.case)
        missing = requested - set(selected_ids)
        if missing:
            raise QualityError(
                f"--case entries are not part of suite {args.suite}: {sorted(missing)}"
            )
        selected_ids = [case_id for case_id in selected_ids if case_id in requested]
    case_by_id = {case["id"]: case for case in manifest["case"]}
    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        if args.keep:
            pass
        else:
            try:
                out_dir.relative_to(REPO_ROOT)
            except ValueError as exc:
                raise QualityError(
                    "refusing to replace an output directory outside the repository; use --keep"
                ) from exc
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_bin, build_result = build_diff_tool(args.timeout)
    started = time.perf_counter()
    features = effective_features(adapter)
    features_known = adapter is not None or bool(
        os.environ.get("BLOOM_QUALITY_FEATURES", "").strip()
    )
    built_commands: set[tuple[str, ...]] = set()
    cases: list[dict[str, Any]] = []
    for case_id in selected_ids:
        print(f"[quality] {case_id}", flush=True)
        cases.append(
            run_case(
                case_by_id[case_id],
                out_dir,
                diff_bin,
                features,
                features_known,
                machine,
                args.report_only,
                args.timeout,
                built_commands,
            )
        )
        print(f"[quality] {case_id}: {cases[-1]['status']}", flush=True)
    hard_fail = any(case["status"] in ("fail", "error") for case in cases)
    observed_adapter = adapter
    if observed_adapter is None:
        observed_adapter = next(
            (
                dict(case["telemetry"]["adapter"])
                for case in cases
                if isinstance(case.get("telemetry"), dict)
                and isinstance(case["telemetry"].get("adapter"), dict)
            ),
            None,
        )
    observed_features = effective_features(observed_adapter)
    environment = stable_environment(observed_adapter)
    machine_class = machine.get("id") if machine else None
    capability_snapshot_path = write_capability_snapshot_artifact(
        out_dir, environment, machine_class
    )
    result = {
        "schema": RESULT_SCHEMA,
        "manifest_path": os.path.relpath(manifest_path, REPO_ROOT),
        "manifest_sha256": manifest_hash,
        "suite": args.suite,
        "machine_class": machine_class,
        "report_only": bool(args.report_only),
        "environment": environment,
        "features": sorted(observed_features or features),
        "artifacts": {"capability_snapshot": capability_snapshot_path},
        "build": command_record(build_result) if build_result else None,
        "cases": cases,
        "status": "fail" if hard_fail else "pass",
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summaries(result, out_dir)
    print(f"[quality] result: {out_dir / 'result.json'}")
    print(f"[quality] summary: {out_dir / 'summary.html'}")
    return 0 if args.report_only or not hard_fail else 1


def check_manifest(args: argparse.Namespace) -> int:
    manifest, digest = load_manifest(Path(args.manifest).resolve())
    failures: list[str] = []
    for case in manifest["case"]:
        for record in verify_assets(case):
            if record["status"] != "pass":
                failures.append(f"{case['id']}: {record['path']}: {record['status']}")
        reference = repo_path(case["reference"]["path"])
        if not reference.exists():
            failures.append(f"{case['id']}: missing baseline {case['reference']['path']}")
    print(f"manifest schema: 1")
    print(f"manifest sha256: {digest}")
    print(f"cases: {len(manifest['case'])}")
    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASS: manifest, asset hashes, and approved baselines are complete")
    return 0


def run_negative_controls(args: argparse.Namespace) -> int:
    manifest, manifest_hash = load_manifest(Path(args.manifest).resolve())
    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    diff_bin, _ = build_diff_tool(args.timeout)
    case_by_id = {case["id"]: case for case in manifest["case"]}
    source_result_path = (
        Path(args.source_result).resolve() if args.source_result else None
    )
    source_cases: dict[str, Mapping[str, Any]] = {}
    source_result_sha256 = None
    if source_result_path is not None:
        source_result = read_json(source_result_path, "negative-control source result")
        if source_result.get("schema") != RESULT_SCHEMA:
            raise QualityError("negative-control source has an unsupported schema")
        if source_result.get("manifest_sha256") != manifest_hash:
            raise QualityError(
                "negative-control source was not produced by the selected manifest"
            )
        source_cases = {
            str(case.get("id")): case
            for case in source_result.get("cases", [])
            if isinstance(case, dict)
        }
        source_result_sha256 = sha256_file(source_result_path)
    records: list[dict[str, Any]] = []
    for control in manifest["negative_control"]:
        fault = control["fault"]
        case = case_by_id[control["case"]]
        if source_result_path is not None:
            source_case = source_cases.get(str(case["id"]))
            if source_case is None:
                raise QualityError(
                    f"negative-control source has no case {case['id']!r}"
                )
            artifacts = artifact_map(source_result_path, source_case)
            reference = artifacts["final"]
            reference_source = "unapproved-result-bundle"
        else:
            reference = repo_path(case["reference"]["path"], must_exist=True)
            reference_source = "approved-baseline"
        control_dir = out_dir / fault
        control_dir.mkdir(parents=True)
        argv = diff_command(
            diff_bin,
            case,
            reference,
            reference,
            control_dir,
            report_only=False,
            seeded_fault=fault,
        )
        result = run_command(argv, REPO_ROOT, args.timeout)
        detected = result.returncode == 1
        records.append(
            {
                "fault": fault,
                "case": case["id"],
                "detected": detected,
                "reference_source": reference_source,
                "reference_sha256": sha256_file(reference),
                "command": command_record(result),
                "metrics": (
                    read_json(control_dir / "metrics.json", "fault metrics")
                    if (control_dir / "metrics.json").exists()
                    else None
                ),
            }
        )
        print(f"[negative-control] {fault}: {'DETECTED' if detected else 'MISSED'}")
    passed = all(record["detected"] for record in records)
    result = {
        "schema": "bloom-quality-negative-controls-v1",
        "passed": passed,
        "manifest_sha256": manifest_hash,
        "source_result_sha256": source_result_sha256,
        "controls": records,
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if passed else 1


def write_baseline_review_summaries(
    review: Mapping[str, Any], review_dir: Path
) -> None:
    rows: list[list[str]] = []
    sections: list[str] = []
    for entry in review.get("entries", []):
        timing = entry.get("timing") or {}
        metrics = entry.get("metric_deltas") or {}
        rows.append(
            [
                str(entry["case"]),
                str(entry["baseline_state"]),
                f"{float(metrics['ssim_luminance']):.6f}"
                if metrics.get("ssim_luminance") is not None
                else "initial",
                f"{float(timing['cpu_frame_p95_ms']):.3f}"
                if timing.get("cpu_frame_p95_ms") is not None
                else "-",
                f"{float(timing['gpu_frame_p95_ms']):.3f}"
                if timing.get("gpu_frame_p95_ms") is not None
                else "-",
                f"[after]({entry['proposed_image']})",
            ]
        )
        images: list[tuple[str, str]] = [("Proposed", entry["proposed_image"])]
        if entry.get("before_image"):
            images.insert(0, ("Current", entry["before_image"]))
        for key, label in (("comparison", "Comparison"), ("heatmap", "Heatmap")):
            raw = entry.get("evidence", {}).get(key)
            if raw:
                images.append((label, raw))
        image_html = "".join(
            "<figure><figcaption>"
            + html.escape(label)
            + "</figcaption><a href=\""
            + html.escape(path)
            + "\"><img loading=\"lazy\" src=\""
            + html.escape(path)
            + "\"></a></figure>"
            for label, path in images
        )
        intermediate_html = "".join(
            "<figure><figcaption>"
            + html.escape(Path(path).stem)
            + "</figcaption><a href=\""
            + html.escape(path)
            + "\"><img loading=\"lazy\" src=\""
            + html.escape(path)
            + "\"></a></figure>"
            for path in entry.get("intermediates", [])
        )
        sections.append(
            "<section><h2>"
            + html.escape(str(entry["case"]))
            + "</h2><p>Baseline state: <code>"
            + html.escape(str(entry["baseline_state"]))
            + "</code> · CPU p95 "
            + html.escape(str(timing.get("cpu_frame_p95_ms", "-")))
            + " ms · GPU p95 "
            + html.escape(str(timing.get("gpu_frame_p95_ms", "-")))
            + " ms</p><div class=\"gallery\">"
            + image_html
            + "</div><details><summary>Named graph intermediates</summary>"
            + "<div class=\"gallery\">"
            + intermediate_html
            + "</div></details></section>"
        )
    markdown = (
        "# Bloom baseline review\n\n"
        f"Reason: {review['reason']}\n\n"
        f"Source commit: `{review['source_commit']}`  \n"
        f"Manifest: `{review['manifest_sha256']}`\n\n"
        + markdown_table(
            ["Case", "Baseline", "SSIM", "CPU p95 ms", "GPU p95 ms", "Image"],
            rows,
        )
        + "\n\n"
        + str(review["installation"])
        + "\n"
    )
    (review_dir / "review.md").write_text(markdown, encoding="utf-8")
    html_doc = """<!doctype html>
<meta charset="utf-8">
<title>Bloom baseline review</title>
<style>
body{font:14px system-ui;margin:2rem;color:#ddd;background:#181a1b}
section{border-top:1px solid #555;padding:1rem 0 2rem}.gallery{display:flex;flex-wrap:wrap;gap:1rem}
figure{margin:0;background:#242729;padding:.5rem}figcaption{margin-bottom:.4rem;font-weight:600}
img{display:block;max-width:520px;max-height:360px;object-fit:contain;background:#111}
code{color:#9cdcfe}summary{cursor:pointer;margin:1rem 0}
</style>
<h1>Bloom baseline review</h1>
<p><strong>Reason:</strong> """ + html.escape(str(review["reason"])) + """</p>
<p>Source commit <code>""" + html.escape(str(review["source_commit"])) + """</code><br>
Manifest <code>""" + html.escape(str(review["manifest_sha256"])) + """</code></p>
""" + "".join(sections) + """
<p>""" + html.escape(str(review["installation"])) + """</p>
"""
    (review_dir / "review.html").write_text(html_doc, encoding="utf-8")


def baseline_review(args: argparse.Namespace) -> int:
    result_path = Path(args.result).resolve()
    result = read_json(result_path, "quality result")
    if result.get("schema") != RESULT_SCHEMA:
        raise QualityError(f"unsupported result schema: {result.get('schema')!r}")
    review_dir = Path(args.out).resolve()
    if review_dir.exists():
        raise QualityError(f"review output already exists: {review_dir}")
    review_dir.mkdir(parents=True)
    requested = set(args.case or [case["id"] for case in result["cases"]])
    entries: list[dict[str, Any]] = []
    for case in result["cases"]:
        if case["id"] not in requested:
            continue
        artifacts = case.get("artifacts", {})
        candidate_raw = artifacts.get("candidate")
        if not candidate_raw:
            raise QualityError(f"{case['id']} has no candidate evidence")
        candidate = result_artifact(
            result_path, candidate_raw, f"{case['id']} candidate"
        )
        current_baseline = case.get("reference_target")
        if not isinstance(current_baseline, str):
            raise QualityError(f"{case['id']} has no baseline target")
        reference = repo_path(current_baseline)
        baseline_root = repo_path("tools/quality/baselines")
        try:
            reference.relative_to(baseline_root)
        except ValueError as exc:
            raise QualityError(
                f"{case['id']}: baseline target is outside tools/quality/baselines"
            ) from exc
        target = review_dir / case["id"]
        target.mkdir()
        shutil.copy2(candidate, target / "after.png")
        before_image = None
        if reference.exists():
            shutil.copy2(reference, target / "before.png")
            before_image = f"{case['id']}/before.png"
        review_intermediates: list[str] = []
        intermediate_dir = target / "intermediates"
        for raw in artifacts.get("intermediates", []):
            source = result_artifact(
                result_path, raw, f"{case['id']} intermediate"
            )
            intermediate_dir.mkdir(exist_ok=True)
            destination = intermediate_dir / source.name
            shutil.copy2(source, destination)
            review_intermediates.append(
                f"{case['id']}/intermediates/{destination.name}"
            )
        review_evidence: dict[str, str] = {}
        for key in ("heatmap", "comparison", "metrics"):
            raw = artifacts.get(key)
            if raw:
                source = result_artifact(
                    result_path, raw, f"{case['id']} {key}"
                )
                destination = target / source.name
                shutil.copy2(source, destination)
                review_evidence[key] = f"{case['id']}/{destination.name}"
        entries.append(
            {
                "case": case["id"],
                "reason": args.reason,
                "current_baseline": current_baseline,
                "proposed_image": f"{case['id']}/after.png",
                "before_image": before_image,
                "baseline_state": "present" if before_image else "absent",
                "intermediates": review_intermediates,
                "evidence": review_evidence,
                "metric_deltas": case.get("metrics", {}),
                "timing": case.get("telemetry"),
            }
        )
    missing = requested - {entry["case"] for entry in entries}
    if missing:
        raise QualityError(f"requested cases absent from result: {sorted(missing)}")
    review = {
        "schema": REVIEW_SCHEMA,
        "source_result_sha256": sha256_file(result_path),
        "source_commit": result["environment"]["git_commit"],
        "manifest_sha256": result["manifest_sha256"],
        "reason": args.reason,
        "entries": entries,
        "installation": (
            "This bundle never writes baselines. After human image review, run "
            "`tools/quality/run.py baseline-install --review REVIEW/review.json "
            "--approved-by NAME` in a separate, explicit change."
        ),
    }
    (review_dir / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_baseline_review_summaries(review, review_dir)
    print(f"baseline review bundle: {review_dir}")
    return 0


def baseline_install(args: argparse.Namespace) -> int:
    review_path = Path(args.review).resolve()
    review = read_json(review_path, "baseline review")
    if review.get("schema") != REVIEW_SCHEMA:
        raise QualityError(f"unsupported review schema: {review.get('schema')!r}")
    if not args.approved_by.strip():
        raise QualityError("--approved-by must name the human reviewer")
    review_dir = review_path.parent
    requested = set(args.case or [entry["case"] for entry in review.get("entries", [])])
    installed: list[dict[str, Any]] = []
    for entry in review.get("entries", []):
        if not isinstance(entry, dict) or entry.get("case") not in requested:
            continue
        case_id = str(entry["case"])
        proposed_raw = entry.get("proposed_image")
        target_raw = entry.get("current_baseline")
        if not isinstance(proposed_raw, str) or not isinstance(target_raw, str):
            raise QualityError(f"{case_id}: review entry is missing paths")
        proposed = (review_dir / proposed_raw).resolve()
        if not proposed.is_file():
            raise QualityError(f"{case_id}: proposed image missing: {proposed}")
        target = repo_path(target_raw)
        baseline_root = repo_path("tools/quality/baselines")
        try:
            target.relative_to(baseline_root)
        except ValueError as exc:
            raise QualityError(
                f"{case_id}: baseline target is outside tools/quality/baselines"
            ) from exc
        before_raw = entry.get("before_image")
        if isinstance(before_raw, str):
            before = (review_dir / before_raw).resolve()
            if not before.is_file() or not target.is_file():
                raise QualityError(f"{case_id}: existing baseline evidence is incomplete")
            if sha256_file(before) != sha256_file(target):
                raise QualityError(
                    f"{case_id}: baseline changed since review; create a new bundle"
                )
        elif target.exists():
            raise QualityError(
                f"{case_id}: review expected no baseline but target now exists"
            )
        before_sha = sha256_file(target) if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(proposed, target)
        installed.append(
            {
                "case": case_id,
                "target": os.path.relpath(target, REPO_ROOT),
                "before_sha256": before_sha,
                "after_sha256": sha256_file(target),
            }
        )
    missing = requested - {entry["case"] for entry in installed}
    if missing:
        raise QualityError(f"requested cases absent from review: {sorted(missing)}")
    receipt = {
        "schema": INSTALL_SCHEMA,
        "review_sha256": sha256_file(review_path),
        "approved_by": args.approved_by.strip(),
        "installed": installed,
    }
    receipt_path = Path(args.receipt).resolve() if args.receipt else review_dir / "installation.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"installed {len(installed)} approved baseline(s)")
    print(f"installation receipt: {receipt_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Bloom deterministic visual and GPU performance qualification"
    )
    root.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    sub = root.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a manifest suite")
    run.add_argument("suite", choices=sorted(ALLOWED_SUITES))
    run.add_argument("--out", default=str(SCRIPT_DIR / "out/latest"))
    run.add_argument("--case", action="append")
    run.add_argument("--machine-class")
    run.add_argument("--adapter-json")
    run.add_argument("--report-only", action="store_true")
    run.add_argument("--keep", action="store_true")
    run.add_argument("--timeout", type=float, default=900.0)
    run.set_defaults(func=execute_suite)

    check = sub.add_parser("check", help="validate manifest, assets, and baselines")
    check.set_defaults(func=check_manifest)

    faults = sub.add_parser("faults", help="prove all seeded negative controls fail")
    faults.add_argument("--out", default=str(SCRIPT_DIR / "out/faults"))
    faults.add_argument("--timeout", type=float, default=300.0)
    faults.add_argument(
        "--source-result",
        help=(
            "bootstrap detector validation from an unapproved result bundle; "
            "normal CI must omit this and use approved baselines"
        ),
    )
    faults.set_defaults(func=run_negative_controls)

    repro = sub.add_parser(
        "repro-check",
        help="compare two result bundles against documented determinism/noise bounds",
    )
    repro.add_argument("--first", required=True)
    repro.add_argument("--second", required=True)
    repro.add_argument("--out", default=str(SCRIPT_DIR / "out/reproducibility"))
    repro.add_argument("--timeout", type=float, default=300.0)
    repro.set_defaults(func=check_reproducibility)

    review = sub.add_parser(
        "baseline-review", help="write a review bundle; never overwrite a baseline"
    )
    review.add_argument("--result", required=True)
    review.add_argument("--out", required=True)
    review.add_argument("--reason", required=True)
    review.add_argument("--case", action="append")
    review.set_defaults(func=baseline_review)

    install = sub.add_parser(
        "baseline-install",
        help="install a human-approved review bundle with stale-baseline protection",
    )
    install.add_argument("--review", required=True)
    install.add_argument("--approved-by", required=True)
    install.add_argument("--case", action="append")
    install.add_argument("--receipt")
    install.set_defaults(func=baseline_install)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.func(args))
    except QualityError as exc:
        print(f"quality error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
