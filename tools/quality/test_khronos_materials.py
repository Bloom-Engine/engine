#!/usr/bin/env python3
"""Unit tests for the opt-in Khronos material qualification command."""

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("khronos_materials.py")
SPEC = importlib.util.spec_from_file_location("bloom_khronos_materials", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
khronos = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = khronos
SPEC.loader.exec_module(khronos)


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def rgb_png(width: int, height: int, pixels: list[tuple[int, int, int]]) -> bytes:
    rows = bytearray()
    for row in range(height):
        rows.append(0)
        for pixel in pixels[row * width : (row + 1) * width]:
            rows.extend(pixel)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )


class KhronosMaterialQualificationTests(unittest.TestCase):
    def test_assets_are_revision_and_hash_pinned(self) -> None:
        self.assertEqual(len(khronos.KHRONOS_REVISION), 40)
        self.assertEqual(len(khronos.CASES), 4)
        self.assertEqual(len({case.id for case in khronos.CASES}), 4)
        for case in khronos.CASES:
            self.assertEqual(len(case.sha256), 64)
            self.assertIn(khronos.KHRONOS_REVISION, case.url)
            self.assertTrue(case.url.endswith(f"/{case.asset}.glb"))
            self.assertIn(case.license, {"CC-BY-4.0", "CC0-1.0"})
            self.assertTrue(case.metadata_url.endswith(f"/{case.asset}/README.md"))

    def test_diagnostic_filter_ignores_runtime_noise_but_rejects_import_gaps(
        self,
    ) -> None:
        expected_noise = "\n".join(
            [
                "[perry] warning: runtime-only stdlib stub",
                "bloom: surface acquire failed (count=1)",
                "bloom: ssgi trace backend = hw-ray-query",
            ]
        )
        self.assertEqual(khronos.diagnostic_lines(expected_noise), [])
        self.assertEqual(
            khronos.diagnostic_lines(
                "bloom glTF: material Glass uses unsupported extension"
            ),
            ["bloom glTF: material Glass uses unsupported extension"],
        )

    def test_png_decoder_and_semantic_statistics_are_dependency_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            pixels = [(0, 255, 0)] * 2 + [(0, 0, 255)] * 2
            path.write_bytes(rgb_png(2, 2, pixels))
            width, height, decoded = khronos.png_rgb(path)
            self.assertEqual((width, height), (2, 2))
            self.assertEqual(decoded, pixels)
            statistics = khronos.image_statistics(path, (2, 2), "green-checks")
            self.assertEqual(statistics["pixel_density"], 1)
            self.assertGreater(statistics["green_check_fraction"], 0.4)
            self.assertEqual(statistics["failures"], [])

    def test_non_integral_capture_density_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            path.write_bytes(rgb_png(3, 2, [(255, 0, 0)] * 6))
            with self.assertRaisesRegex(
                khronos.QualificationError,
                "integer-density",
            ):
                khronos.image_statistics(path, (2, 2), "material-variation")


if __name__ == "__main__":
    unittest.main()
