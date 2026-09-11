#!/usr/bin/env python3
"""Prepare the pinned official Windows wasm-pack binary for the starter gate."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

VERSION = '0.15.0'
ARCHIVE = f'wasm-pack-v{VERSION}-x86_64-pc-windows-msvc.tar.gz'
URL = f'https://github.com/wasm-bindgen/wasm-pack/releases/download/v{VERSION}/{ARCHIVE}'
SHA256 = '518dc51180c7bc864699c9279b3bc99025bc109123e0249a34ba34d130a509bf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--github-env', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = out / ARCHIVE
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        with urllib.request.urlopen(URL, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise RuntimeError('official wasm-pack archive SHA-256 differs from the pinned release')
        archive.write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes()), mode='r:gz') as tar:
        files = [member for member in tar.getmembers() if member.isfile() and Path(member.name).name == 'wasm-pack.exe']
        if len(files) != 1:
            raise RuntimeError('official wasm-pack archive must contain exactly one executable')
        data = tar.extractfile(files[0]).read()
    executable = out / 'wasm-pack.exe'
    executable.write_bytes(data)
    version = subprocess.check_output([str(executable), '--version'], text=True).strip()
    if version != f'wasm-pack {VERSION}':
        raise RuntimeError(f'unexpected wasm-pack version: {version}')
    receipt = dict(url=URL, archive_sha256=SHA256, executable_sha256=hashlib.sha256(data).hexdigest(), version=version)
    (out / 'toolchain.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    if args.github_env:
        with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as output:
            output.write(f'BLOOM_WASM_PACK={executable}\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
