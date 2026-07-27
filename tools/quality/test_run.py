#!/usr/bin/env python3
"""Unit tests for qualification orchestration and baseline governance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("bloom_quality_run", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)


class BaselineGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.baseline = (
            self.root
            / "tools"
            / "quality"
            / "baselines"
            / "portable"
            / "sample.png"
        )
        self.run_dir = self.root / "run"
        self.candidate = self.run_dir / "cases" / "sample" / "final.png"
        self.intermediate = (
            self.run_dir
            / "cases"
            / "sample"
            / "intermediates"
            / "hdr-scene.png"
        )
        self.candidate.parent.mkdir(parents=True)
        self.intermediate.parent.mkdir(parents=True)
        self.candidate.write_bytes(b"candidate-png")
        self.intermediate.write_bytes(b"intermediate-png")
        self.result_path = self.run_dir / "result.json"
        self.result_path.write_text(
            json.dumps(
                {
                    "schema": quality.RESULT_SCHEMA,
                    "manifest_sha256": "manifest",
                    "environment": {"git_commit": "commit"},
                    "cases": [
                        {
                            "id": "sample",
                            "reference_target": (
                                "tools/quality/baselines/portable/sample.png"
                            ),
                            "artifacts": {
                                "candidate": "cases/sample/final.png",
                                "intermediates": [
                                    "cases/sample/intermediates/hdr-scene.png"
                                ],
                            },
                            "metrics": {"ssim_luminance": 1.0},
                            "telemetry": {"cpu_frame_p95_ms": 1.0},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.review_dir = self.root / "review"
        self.repo_patch = mock.patch.object(quality, "REPO_ROOT", self.root)
        self.repo_patch.start()

    def tearDown(self) -> None:
        self.repo_patch.stop()
        self.temp.cleanup()

    def make_review(self) -> Path:
        args = argparse.Namespace(
            result=str(self.result_path),
            out=str(self.review_dir),
            reason="unit-test review",
            case=None,
        )
        self.assertEqual(quality.baseline_review(args), 0)
        return self.review_dir / "review.json"

    def test_initial_review_never_writes_then_explicit_install_does(self) -> None:
        review_path = self.make_review()
        self.assertFalse(self.baseline.exists())
        self.assertTrue((self.review_dir / "review.md").is_file())
        self.assertTrue((self.review_dir / "review.html").is_file())
        review = quality.read_json(review_path, "review")
        entry = review["entries"][0]
        self.assertEqual(entry["baseline_state"], "absent")
        self.assertIsNone(entry["before_image"])
        self.assertEqual(
            entry["intermediates"],
            ["sample/intermediates/hdr-scene.png"],
        )

        receipt = self.root / "receipt.json"
        args = argparse.Namespace(
            review=str(review_path),
            approved_by="Human Reviewer",
            case=None,
            receipt=str(receipt),
        )
        self.assertEqual(quality.baseline_install(args), 0)
        self.assertEqual(self.baseline.read_bytes(), b"candidate-png")
        installed = quality.read_json(receipt, "receipt")
        self.assertEqual(installed["approved_by"], "Human Reviewer")
        self.assertIsNone(installed["installed"][0]["before_sha256"])

    def test_stale_existing_baseline_is_rejected_without_overwrite(self) -> None:
        self.baseline.parent.mkdir(parents=True)
        self.baseline.write_bytes(b"reviewed-baseline")
        review_path = self.make_review()
        self.baseline.write_bytes(b"changed-after-review")
        args = argparse.Namespace(
            review=str(review_path),
            approved_by="Human Reviewer",
            case=None,
            receipt=None,
        )
        with self.assertRaisesRegex(
            quality.QualityError, "baseline changed since review"
        ):
            quality.baseline_install(args)
        self.assertEqual(self.baseline.read_bytes(), b"changed-after-review")

    def test_baseline_target_outside_governed_tree_is_rejected(self) -> None:
        result = quality.read_json(self.result_path, "result")
        result["cases"][0]["reference_target"] = "outside.png"
        self.result_path.write_text(json.dumps(result), encoding="utf-8")
        args = argparse.Namespace(
            result=str(self.result_path),
            out=str(self.review_dir),
            reason="invalid target",
            case=None,
        )
        with self.assertRaisesRegex(
            quality.QualityError, "outside tools/quality/baselines"
        ):
            quality.baseline_review(args)
        self.assertFalse((self.root / "outside.png").exists())

    def test_candidate_artifact_escape_is_rejected(self) -> None:
        escaped = self.root / "escaped.png"
        escaped.write_bytes(b"not-result-evidence")
        result = quality.read_json(self.result_path, "result")
        result["cases"][0]["artifacts"]["candidate"] = "../escaped.png"
        self.result_path.write_text(json.dumps(result), encoding="utf-8")
        args = argparse.Namespace(
            result=str(self.result_path),
            out=str(self.review_dir),
            reason="invalid artifact",
            case=None,
        )
        with self.assertRaisesRegex(quality.QualityError, "artifact escapes"):
            quality.baseline_review(args)


class ReproducibilityTests(unittest.TestCase):
    def test_checked_in_manifest_satisfies_contract(self) -> None:
        manifest, digest = quality.load_manifest(MODULE_PATH.with_name("scenes.toml"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["case"]), 9)
        self.assertEqual(len(digest), 64)

    def test_stable_metadata_ignores_commands_and_duration(self) -> None:
        common = {
            "schema": quality.RESULT_SCHEMA,
            "manifest_path": "tools/quality/scenes.toml",
            "manifest_sha256": "manifest",
            "suite": "quick",
            "machine_class": None,
            "report_only": True,
            "environment": {"git_commit": "commit"},
            "features": ["ray-query"],
            "cases": [
                {
                    "id": "sample",
                    "description": "sample",
                    "commands": [{"duration_ms": 1.0}],
                    "telemetry": {
                        "schema": "telemetry",
                        "fixed_timestep": 1 / 60,
                        "cpu_frame_mean_ms": 1.0,
                    },
                }
            ],
            "duration_ms": 10.0,
        }
        changed = json.loads(json.dumps(common))
        changed["duration_ms"] = 999.0
        changed["cases"][0]["commands"][0]["duration_ms"] = 500.0
        changed["cases"][0]["telemetry"]["cpu_frame_mean_ms"] = 2.0
        self.assertEqual(
            quality.stable_result_metadata(common),
            quality.stable_result_metadata(changed),
        )

    def test_timing_delta_accepts_absolute_or_relative_bound(self) -> None:
        absolute = quality.timing_delta(
            {"metric": 1.0}, {"metric": 1.2}, "metric", 0.25, 0.01
        )
        relative = quality.timing_delta(
            {"metric": 10.0}, {"metric": 11.0}, "metric", 0.1, 0.10
        )
        rejected = quality.timing_delta(
            {"metric": 1.0}, {"metric": 2.0}, "metric", 0.25, 0.10
        )
        self.assertTrue(absolute["passed"])
        self.assertTrue(relative["passed"])
        self.assertFalse(rejected["passed"])

    def test_telemetry_contract_rejects_vsync_and_wrong_frame_count(self) -> None:
        case = {
            "fixed_timestep": 1 / 60,
            "warmup_frames": 120,
            "measured_frames": 300,
            "render_scale": 1.0,
            "settings": {"quality_preset": 3},
        }
        telemetry = {
            "schema": "bloom-quality-telemetry-v1",
            "fixed_timestep": 1 / 60,
            "warmup_frames": 120,
            "measured_frames": 299,
            "quality_preset": 3,
            "render_scale": 1.0,
            "uncapped": False,
            "warmup_excluded": True,
            "shader_compilation_excluded": True,
            "gpu_timestamps_available": True,
            "adapter": {"availability": "reported"},
            "renderer_paths": {},
            "cpu_frame_mean_ms": 1.0,
            "cpu_frame_p95_ms": 1.2,
            "gpu_frame_mean_ms": 2.0,
            "gpu_frame_p95_ms": 2.4,
            "measurement_wall_ms": 100.0,
            "passes": [{"label": "render_total"}],
        }
        failures = quality.telemetry_contract_failures(case, telemetry)
        self.assertTrue(any("measured_frames" in item for item in failures))
        self.assertTrue(any("uncapped" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
