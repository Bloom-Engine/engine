#!/usr/bin/env python3
"""Capture cutout inputs on opaque leaf cards using a temporary shader build.

This is a diagnostic, not a quality or timing gate. Source and the native
library are restored before returning. Run without concurrent native builds.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import platform
import subprocess
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHADER = ROOT / "native/shared/src/renderer/shaders/core.rs"


def function_body(source: str, signature: str) -> tuple[int, int]:
    assert source.count(signature) == 1, signature
    start = source.index("{", source.index(signature)) + 1
    depth = 1
    for index in range(start, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return start, index
    raise ValueError(f"unterminated shader function: {signature}")


def probe_shader(source: str, channel: str) -> str:
    begin, end = function_body(source, "fn fs_depth_prepass(")
    decision = source[begin:end]
    assert decision.count("var survives = true;") == 1
    assert decision.count("if (!survives) { discard; }") == 1
    decision = "\n    var survives = true;\n" + decision.replace(
        "var survives = true;", ""
    ).replace("if (!survives) { discard; }", "")
    # Opaque cards expose both accepted and rejected fragments. This changes
    # occlusion: the values describe the nearest card, not the composited leaf.
    source = source[:begin] + "\n" + source[end:]
    begin, end = function_body(source, "fn shade_main_scene(")
    output = r"""
    let probe_lod = mask_texture_lod(
        base_uv, textureDimensions(base_color_tex), lighting.shadow_cascade_splits.w,
    );
    let probe_coverage = textureSampleLevel(
        base_color_tex, base_color_samp, base_uv, max(probe_lod, 1.0),
    ).a;
    let probe_authored_alpha = textureSampleLevel(
        base_color_tex, base_color_samp, base_uv, 0.0,
    ).a * in.color.a;
    let probe_threshold = mask_coverage_threshold(
        base_uv, textureDimensions(base_color_tex), probe_lod,
    );
    let bits = bitcast<u32>(PROBE_SCALAR);
    var result: SceneOut;
    // Keep alpha at one: the ordinary scene pipeline uses alpha blending.
    result.color = vec4<f32>(probe_lod, probe_coverage, probe_authored_alpha, 1.0);
    let flags = select(0u, 1u, survives) + select(0u, 2u, material.emissive.w > 0.5)
        + select(0u, 4u, alpha_cutoff > 0.0);
    result.material = vec2<f32>(probe_threshold, f32(flags) / 255.0);
    result.velocity = base_uv;
    // Rgba8Unorm stores all four bytes of the selected f32 exactly. HDR and
    // velocity are half precision and cannot establish tiny UV differences.
    result.albedo = vec4<f32>(vec4<u32>(
        bits & 255u, (bits >> 8u) & 255u, (bits >> 16u) & 255u, bits >> 24u,
    )) / 255.0;
    return result;
""".replace("PROBE_SCALAR", {"u": "base_uv.x", "v": "base_uv.y"}[channel])
    signature_start = source.index("fn shade_main_scene(")
    declaration = source[signature_start:begin]
    # Keep the original helper available: later renderer specialization
    # rewrites its marked lighting blocks even though this probe never calls it.
    original_helper = source[signature_start:].replace(
        "fn shade_main_scene(", "fn shade_main_scene_unprobed(", 1
    )
    return source[:signature_start] + declaration + decision + output + "}\n\n" + original_helper


def verify_capture(directory: Path) -> dict[str, int]:
    intermediates = directory / "intermediates"
    mrt = intermediates / "mrt"
    manifest = json.loads((mrt / "scene-mrt.json").read_text())
    expected = {"hdr-scene": 8, "material-properties": 2, "motion-vectors": 4, "albedo": 4}
    assert {item["name"] for item in manifest["attachments"]} == set(expected), manifest
    pixels = manifest["width"] * manifest["height"]
    for item in manifest["attachments"]:
        assert (mrt / (item["name"] + ".raw")).stat().st_size == pixels * expected[item["name"]]
    depth = (intermediates / "raw/scene-depth.raw").read_bytes()
    material = (mrt / "material-properties.raw").read_bytes()
    coordinates = (mrt / "albedo.raw").read_bytes()
    assert len(depth) == pixels * 4
    masked, accepted = 0, 0
    for (z,), flags, (uv,) in zip(struct.iter_unpack("<f", depth), material[1::2],
                                  struct.iter_unpack("<f", coordinates), strict=True):
        if z < 1.0 and flags & 6 == 6:
            assert math.isfinite(uv), uv
            masked += 1
            accepted += flags & 1
    assert 0 < accepted < masked, (accepted, masked)
    return {"masked_pixels": masked, "accepted_pixels": accepted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    host = {"Windows": "windows", "Darwin": "macos"}.get(platform.system())
    if host is None:
        parser.error("this native diagnostic supports Windows and macOS")
    original = SHADER.read_bytes()
    source = original.decode("utf-8").replace("\r\n", "\n")
    receipt = {
        "schema": "bloom-alpha-input-probe-v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "original_shader_sha256": hashlib.sha256(original).hexdigest(),
        "timing_qualification": False,
        "commands": [],
        "channels": {},
    }
    (out / "original-core.rs.txt").write_bytes(original)
    native = ["cargo", "build", "--release", "--manifest-path", f"native/{host}/Cargo.toml"]
    env = dict(os.environ)

    def run(command: list[str], cwd: Path, log: Path, run_env: dict[str, str]) -> None:
        print("+ " + " ".join(command), flush=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=cwd, env=run_env, stdout=output, stderr=subprocess.STDOUT)
        receipt["commands"].append({
            "argv": command, "cwd": str(cwd), "exit_code": result.returncode,
            "elapsed_seconds": time.monotonic() - started, "log": str(log.relative_to(out)),
        })
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}); see {log}")

    try:
        for channel in ("u", "v"):
            directory = out / channel
            directory.mkdir(exist_ok=True)
            candidate = probe_shader(source, channel)
            SHADER.write_bytes(candidate.encode("utf-8"))
            (directory / "shader.patch").write_text("".join(difflib.unified_diff(
                source.splitlines(keepends=True), candidate.splitlines(keepends=True),
                fromfile="core.rs", tofile=f"core-probe-{channel}.rs",
            )), encoding="utf-8")
            run(native, ROOT, directory / "native-build.log", env)
            build = out / "build"
            build.mkdir(exist_ok=True)
            executable = build / (f"quality-motion-{channel}" + (".exe" if host == "windows" else ""))
            run([sys.executable, "tools/quality/build_example.py", "examples/quality-motion",
                 "--output", str(executable)], ROOT, directory / "example-build.log", env)
            capture_env = dict(env, **{
                "BLOOM_HW_GI": "0", "BLOOM_FORCE_RENDER_TIER": "modern",
                "BLOOM_HEADLESS": "1", "BLOOM_HEADLESS_PIXEL_EXACT": "1",
                "BLOOM_NO_FULLSCREEN": "1", "BLOOM_QUALITY": "1", "BLOOM_QUALITY_RAW": "1",
                "BLOOM_QUALITY_CASE": "skinned-alpha-motion", "BLOOM_QUALITY_SEED": "0",
                "BLOOM_QUALITY_FIXED_TIMESTEP": "0.016666666667",
                "BLOOM_QUALITY_WARMUP_FRAMES": "120", "BLOOM_QUALITY_MEASURED_FRAMES": "240",
                "BLOOM_QUALITY_TELEMETRY": str(directory / "telemetry.json"),
                "BLOOM_QUALITY_INTERMEDIATES": str(directory / "intermediates"),
            })
            receipt["channels"][channel] = {
                "shader_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
                "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "capture_env": {key: value for key, value in capture_env.items() if key.startswith("BLOOM_")},
            }
            run([str(executable), "--quality-preset", "3", "--render-scale", "1",
                 "--quality-run", "120", "240", "0.016666666667", str(directory / "final.png"),
                 str(directory / "telemetry.json"), str(directory / "intermediates")],
                ROOT / "examples/quality-motion", directory / "capture.log", capture_env)
            telemetry = json.loads((directory / "telemetry.json").read_text())
            receipt["channels"][channel]["adapter"] = telemetry["adapter"]
            receipt["channels"][channel]["verification"] = verify_capture(directory)
    finally:
        SHADER.write_bytes(original)
        receipt["source_restored"] = SHADER.read_bytes() == original
        try:
            run(native, ROOT, out / "restored-native-build.log", env)
        finally:
            (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Alpha probes retained in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
