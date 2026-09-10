#!/usr/bin/env python3
"""Prepare the pinned Windows example compiler and its matching source libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


VERSION = "0.5.1220"
SOURCE_SHA = "06137858dc8c6f80975238377138f2f948d6ef88"
ARCHIVE_SHA256 = "f3f817c806ae296d7e7a2e58dcc9e335711e1715360df9df68ce12e43e011117"
REPOSITORY = "https://github.com/PerryTS/perry.git"
ARCHIVE_NAME = "perry-windows-x86_64.zip"
ARCHIVE_URL = f"https://github.com/PerryTS/perry/releases/download/v{VERSION}/{ARCHIVE_NAME}"
RUNTIME_FEATURES = "perry-runtime/full,perry-runtime/regex-engine,perry-stdlib/async-runtime,perry-stdlib/crypto"


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args, cwd=cwd, check=True, stdout=subprocess.PIPE,
        encoding="utf-8", errors="replace", timeout=600,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="toolchain directory outside the engine checkout")
    parser.add_argument("--archive", help="reuse a downloaded release ZIP; its hash is still verified")
    parser.add_argument("--github-env", action="store_true", help="export paths for subsequent Actions steps")
    args = parser.parse_args()
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        parser.error("this toolchain is for Windows x86_64")
    out = Path(args.out).resolve()
    engine = Path(__file__).resolve().parents[2]
    if out == engine or engine in out.parents:
        parser.error("--out must be outside the engine checkout (the compiler source has its own checks)")
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive).resolve() if args.archive else out / ARCHIVE_NAME
    if not archive.is_file():
        if args.archive:
            parser.error(f"archive does not exist: {archive}")
        print(f"Downloading Perry {VERSION}", flush=True)
        partial = out / (ARCHIVE_NAME + ".partial")
        request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "Bloom-CI"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        if sha256(partial) != ARCHIVE_SHA256:
            raise RuntimeError("Perry release archive SHA-256 does not match the pinned release")
        partial.replace(archive)
    if sha256(archive) != ARCHIVE_SHA256:
        raise RuntimeError("Perry release archive SHA-256 does not match the pinned release")

    binary_dir = out / "bin"
    binary_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            destination = (binary_dir / member.filename).resolve()
            if binary_dir != destination and binary_dir not in destination.parents:
                raise RuntimeError(f"archive entry escapes toolchain directory: {member.filename}")
        package.extractall(binary_dir)
    compiler = binary_dir / "perry.exe"
    actual_version = run(str(compiler), "--version")
    if actual_version != f"perry {VERSION}":
        raise RuntimeError(f"unexpected compiler version: {actual_version}")

    # The release's full stdlib references unbundled HTTP extension symbols.
    # Build the union of runtime features used by the canonical examples, with
    # the same unwind profile as Bloom. Per-app panic=abort archives can collide
    # with Bloom's Rust COMDAT sections on Windows. Keep release source and
    # Cargo.lock unmodified; no compiler patch or synthetic FFI stub.
    source = out / "source"
    if not source.exists():
        print(f"Checking out Perry source {SOURCE_SHA}", flush=True)
        run("git", "clone", "--depth", "1", "--branch", f"v{VERSION}",
            "--filter=blob:none", "--sparse", REPOSITORY, str(source))
    if run("git", "rev-parse", "HEAD", cwd=source) != SOURCE_SHA:
        raise RuntimeError(f"Perry source at {source} is not the pinned release; use a fresh --out directory")
    if run("git", "status", "--porcelain", "--untracked-files=no", cwd=source):
        raise RuntimeError(f"Perry source at {source} has tracked modifications")
    run("git", "sparse-checkout", "set", "crates", ".cargo",
        "docs/examples/_fixtures/native-libraries/my-bindings", cwd=source)

    build_dir = out / "runtime-build"
    build_env = os.environ.copy()
    build_env["CARGO_TARGET_DIR"] = str(build_dir)
    build_env["RUSTFLAGS"] = "-C panic=unwind"
    build_env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    build_command = [
        "cargo", "build", "--locked", "--release", "-p", "perry-runtime-static",
        "-p", "perry-stdlib-static", "--no-default-features", "--features", RUNTIME_FEATURES,
    ]
    print("Building matching Perry libraries for the native example profile", flush=True)
    with (out / "runtime-build.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            build_command, cwd=source, env=build_env, stdout=log,
            stderr=subprocess.STDOUT, timeout=1800, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Perry library build failed; see {out / 'runtime-build.log'}")
    if run("git", "status", "--porcelain", "--untracked-files=no", cwd=source):
        raise RuntimeError("Perry library build modified its pinned source")
    library_dir = build_dir / "release"
    libraries = {
        name: sha256(library_dir / name)
        for name in ("perry_runtime.lib", "perry_stdlib.lib")
    }
    environment = {
        "PERRY_WORKSPACE_ROOT": str(source),
        "PERRY_RUNTIME_DIR": str(library_dir),
        "PERRY_NO_AUTO_OPTIMIZE": "1",
    }

    receipt = {
        "schema": "bloom-windows-perry-toolchain-v1", "version": VERSION,
        "archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
        "source_repository": REPOSITORY, "source_sha": SOURCE_SHA,
        "source_lock_sha256": sha256(source / "Cargo.lock"),
        "compiler": str(compiler), "compiler_sha256": sha256(compiler),
        "environment": environment,
        "runtime_features": RUNTIME_FEATURES,
        "runtime_build_command": build_command,
        "runtime_rustflags": build_env["RUSTFLAGS"],
        "runtime_libraries_sha256": libraries,
        "rustc": run("rustc", "-Vv"),
    }
    (out / "toolchain.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if args.github_env:
        with Path(os.environ["GITHUB_PATH"]).open("a", encoding="utf-8") as stream:
            stream.write(str(binary_dir) + "\n")
        with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as stream:
            for key, value in environment.items():
                stream.write(f"{key}={value}\n")
    print(f"Ready: {compiler}\nEnvironment: {json.dumps(environment)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
