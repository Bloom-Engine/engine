#!/usr/bin/env python3
"""Unit tests for the VSM contact-detail image gate."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("shadow_detail.py")
SPEC = importlib.util.spec_from_file_location("bloom_shadow_detail", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
shadow_detail = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shadow_detail
SPEC.loader.exec_module(shadow_detail)


def fixture(sharp: bool) -> tuple[int, int, list[tuple[int, int, int]]]:
    width = 128
    height = 72
    pixels = [(180, 180, 180)] * (width * height)
    x0 = round(shadow_detail.ROI[0] * width)
    y0 = round(shadow_detail.ROI[1] * height)
    x1 = round(shadow_detail.ROI[2] * width)
    y1 = round(shadow_detail.ROI[3] * height)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if sharp:
                value = 90 if (x // 3) % 2 == 0 else 190
            else:
                value = 140 + ((x // 3) % 2) * 4
            pixels[y * width + x] = (value, value, value)
    return width, height, pixels


class ShadowDetailTests(unittest.TestCase):
    def test_sharp_neutral_shadow_lines_pass_against_blurred_control(self) -> None:
        width, height, vsm = fixture(True)
        _, _, csm = fixture(False)
        result = shadow_detail.detail_metrics(width, height, vsm, csm)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])
        self.assertGreater(result["comparisons"]["strong_edge_ratio"], 5.0)

    def test_reversed_candidate_fails_the_quality_gate(self) -> None:
        width, height, sharp = fixture(True)
        _, _, blurred = fixture(False)
        result = shadow_detail.detail_metrics(width, height, blurred, sharp)
        self.assertFalse(result["passed"])
        self.assertGreaterEqual(len(result["failures"]), 2)

    def test_colored_geometry_is_excluded_from_shadow_measurement(self) -> None:
        width, height, vsm = fixture(True)
        _, _, csm = fixture(False)
        for y in range(height):
            x = width // 2
            vsm[y * width + x] = (255, 0, 0)
            csm[y * width + x] = (255, 0, 0)
        result = shadow_detail.detail_metrics(width, height, vsm, csm)
        self.assertTrue(result["passed"])
        self.assertLess(
            result["vsm"]["neutral_pixels"],
            round((shadow_detail.ROI[2] - shadow_detail.ROI[0]) * width)
            * round((shadow_detail.ROI[3] - shadow_detail.ROI[1]) * height),
        )


if __name__ == "__main__":
    unittest.main()
