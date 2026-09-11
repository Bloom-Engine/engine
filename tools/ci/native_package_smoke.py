#!/usr/bin/env python3
"""Pack/install Bloom, compile a Perry game, and require its native frame."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.quality.khronos_materials import png_rgb  # noqa: E402


def npm_command() -> list[str]:
    executable = shutil.which("npm")
    if executable is None:
        raise RuntimeError("npm is required to install the packed engine")
    if os.name == "nt" and Path(executable).suffix.lower() in (".cmd", ".bat", ".ps1"):
        # Invoke npm's JavaScript entry directly; do not turn paths into shell
        # command text to work around Windows batch-file execution.
        cli = Path(executable).parent / "node_modules/npm/bin/npm-cli.js"
        node = shutil.which("node")
        if not cli.is_file() or node is None:
            raise RuntimeError(f"cannot locate Node/npm-cli.js beside {executable}")
        return [node, str(cli)]
    return [executable]


def check_frame(path: Path) -> dict:
    width, height, pixels = png_rgb(path)
    if (width, height) != (128, 128):
        raise RuntimeError(f"unexpected startup frame size: {width}x{height}")
    mismatches = 0
    for index, pixel in enumerate(pixels):
        x, y = index % width, index // width
        expected = (255, 255, 255) if 32 <= x < 96 and 32 <= y < 96 else (0, 0, 0)
        mismatches += pixel != expected
    if mismatches:
        raise RuntimeError(f"startup/physics frame differs at {mismatches} of {width * height} pixels")
    return {"width": width, "height": height, "exact_pixels": len(pixels),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "target/ci/native-package")
    parser.add_argument("--backend", choices=["dx12", "vulkan"], action="append")
    parser.add_argument("--mode", choices=["scene", "direct-2d"], action="append")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this installed-package smoke currently supports Windows")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "bloom-native-package-smoke-v3", "status": "running", "commands": [], "frames": [], "binaries": [],
              "scope": "Installed source package, native headless renderer and Jolt; window presentation and packaged DXC remain separate."}

    def save() -> None:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def run(name: str, command: list[str], cwd: Path, env: dict, timeout: int) -> str:
        print(name, flush=True)
        started = time.monotonic()
        stdout_path, stderr_path = out / f"{name}.log", out / f"{name}.stderr.log"
        record = {"name": name, "command": command, "cwd": str(cwd)}
        report["commands"].append(record)
        save()
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                result = subprocess.run(command, cwd=cwd, env=env, stdout=stdout,
                                        stderr=stderr, timeout=timeout, check=False)
            record["exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{name} exited {result.returncode}; see {stderr_path}")
            return stdout_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            record["error"] = str(exc)
            raise
        finally:
            record["duration_seconds"] = round(time.monotonic() - started, 3)
            save()

    temporary_root = Path(tempfile.gettempdir()).resolve()
    temporary = Path(tempfile.mkdtemp(prefix="bn-", dir=temporary_root)).resolve()
    save()
    try:
        env = os.environ.copy()
        env.setdefault("CARGO_BUILD_JOBS", "2")
        # Perry locates native output in the crate's own target directory.
        # Exercise dependency discovery without repository-specific overrides.
        for name in ("CARGO_TARGET_DIR", "BLOOM_JOLT_PREBUILT_DIR", "BLOOM_JOLT_FROM_SOURCE"):
            env.pop(name, None)
        compiler = env.get("BLOOM_PERRY") or shutil.which("perry")
        if not compiler:
            raise RuntimeError("Perry is required; prepare the pinned Windows toolchain first")
        report["compiler_version"] = run("perry-version", [compiler, "--version"], ROOT, env, 30).strip()
        report["compiler_sha256"] = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        npm = npm_command()
        packed = json.loads(run("pack", npm + ["pack", "--json", "--ignore-scripts", "--pack-destination", str(temporary)], ROOT, env, 120))[0]
        archive = temporary / packed["filename"]
        report["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        project = temporary / "game"
        project.mkdir()
        manifest = {"name": "bloom-native-smoke", "version": "1.0.0", "private": True, "main": "main.ts",
                    "perry": {"allow": {"nativeLibrary": ["@bloomengine/engine", "@bloomengine/engine/*"]}}}
        (project / "package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        fixture = ROOT / "tools/ci/fixtures/native-package.ts"
        shutil.copyfile(fixture, project / "main.ts")
        report["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        run("install", npm + ["install", "--ignore-scripts", "--no-audit", "--no-fund", str(archive)], project, env, 240)
        installed = project / "node_modules/@bloomengine/engine"
        report["installed_source_sha256"] = {
            name: hashlib.sha256((installed / name).read_bytes()).hexdigest()
            for name in ("native/shared/src/renderer/mod.rs", "native/shared/src/renderer/direct_frame.rs",
                         "native/shared/src/renderer/quality_capture.rs")
        }
        jolt = project / "node_modules/@bloomengine/jolt-prebuilt"
        report["jolt_version"] = json.loads((jolt / "package.json").read_text())["version"]
        report["jolt_archives"] = [{"name": name, "sha256": hashlib.sha256((jolt / "lib/win32-x64" / name).read_bytes()).hexdigest()}
                                   for name in ("Jolt.lib", "bloom_jolt.lib")]
        fixture_text = fixture.read_text(encoding="utf-8")
        mode_marker = "const BLOOM_SMOKE_DIRECT_2D = false;"
        if fixture_text.count(mode_marker) != 1:
            raise RuntimeError("native fixture must have exactly one render-mode marker")
        ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x8000)
        for mode in args.mode or ["scene", "direct-2d"]:
            entry = fixture_text.replace(mode_marker, "const BLOOM_SMOKE_DIRECT_2D = " + ("true;" if mode == "direct-2d" else "false;"))
            (project / "main.ts").write_text(entry, encoding="utf-8")
            binary = temporary / f"native-smoke-{mode}.exe"
            run("compile-" + mode, [compiler, "compile", "main.ts", "-o", str(binary)], project, env, 1800)
            with binary.open("rb") as stream:
                if stream.read(2) != b"MZ":
                    raise RuntimeError("Perry did not produce a native Windows executable")
            report["binaries"].append({"mode": mode, "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                                       "bytes": binary.stat().st_size, "entry_sha256": hashlib.sha256((project / "main.ts").read_bytes()).hexdigest()})
            report["used_cmake_fallback"] = (installed / "native/third_party/bloom_jolt/build").exists()
            if report["used_cmake_fallback"]:
                raise RuntimeError("installed prebuilt package was ignored; CMake fallback was used")
            for backend in args.backend or ["dx12"]:
                name = f"startup-{mode}-{backend}"
                run_dir = temporary / name
                run_dir.mkdir()
                runtime = env.copy()
                runtime.update(BLOOM_HEADLESS="1", BLOOM_HEADLESS_PIXEL_EXACT="1", BLOOM_WGPU_BACKEND=backend)
                run(name, [str(binary)], run_dir, runtime, 180)
                cleanup_path = run_dir / "native-cleanup.txt"
                if not cleanup_path.is_file() or cleanup_path.read_text(encoding="utf-8") != "1":
                    raise RuntimeError("native game did not complete exactly one cleanup callback")
                shutil.copyfile(cleanup_path, out / f"{name}.cleanup.txt")
                png = run_dir / "native-startup.png"
                if not png.is_file():
                    raise RuntimeError("native startup exited without its required frame capture")
                capture = out / f"{name}.png"
                shutil.copyfile(png, capture)
                report["frames"].append({"mode": mode, "backend": backend, "cleanup_count": 1, **check_frame(capture)})
                save()
        report["status"] = "pass"
        print("PASS: installed native package links, simulates Jolt, and renders its exact frame.")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report.update(status="fail", error=str(exc))
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        save()
        if temporary.parent != temporary_root or temporary.is_symlink() or temporary.is_junction():
            raise RuntimeError(f"refusing cleanup outside owned temporary directory: {temporary}")
        shutil.rmtree(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
