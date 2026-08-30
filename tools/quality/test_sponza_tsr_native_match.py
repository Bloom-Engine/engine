import tempfile
import unittest
from pathlib import Path

from tools.quality.khronos_materials import QualificationError
from tools.quality.sponza_tsr_native_match import (
    aggregate_frame_metrics,
    enforce_limits,
    enforce_reproducibility,
    fractional_repeat_identity,
    verify_capture,
)


class SponzaTsrNativeMatchTests(unittest.TestCase):
    def test_fractional_repeat_requires_byte_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            result = fractional_repeat_identity([first], [second])
            self.assertTrue(result["byte_identical"])
            second.write_bytes(b"different")
            result = fractional_repeat_identity([first], [second])
            self.assertFalse(result["byte_identical"])
            self.assertEqual(result["divergent_frames"], [0])

    def test_capture_rejects_incomplete_numbering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sequence-001.png").touch()
            (root / "sequence-002.png").touch()
            with self.assertRaisesRegex(QualificationError, "exact sequence"):
                verify_capture(root, 2, 16, 9)

    def test_metrics_aggregate_mean_min_and_max(self) -> None:
        first = {
            "rmse_luminance": 0.1,
            "ssim_luminance": 0.8,
            "mean_oklab_delta": 0.2,
            "mean_edge_delta": 0.3,
            "percent_above_tolerance": 40.0,
        }
        second = {
            "rmse_luminance": 0.3,
            "ssim_luminance": 1.0,
            "mean_oklab_delta": 0.4,
            "mean_edge_delta": 0.5,
            "percent_above_tolerance": 60.0,
        }
        result = aggregate_frame_metrics([first, second])["aggregate"]
        self.assertAlmostEqual(result["rmse_luminance"]["mean"], 0.2)
        self.assertEqual(result["ssim_luminance"]["min"], 0.8)
        self.assertEqual(result["mean_edge_delta"]["max"], 0.5)

    def test_optional_reference_limits_fail_closed(self) -> None:
        metrics = {
            "native_frame_rmse": 0.02,
            "native_motion_derivative_rmse": 0.01,
        }
        enforce_limits(metrics, 0.02, 0.01)
        with self.assertRaisesRegex(QualificationError, "native_frame_rmse"):
            enforce_limits(metrics, 0.019, None)
        with self.assertRaisesRegex(QualificationError, "motion_derivative"):
            enforce_limits(metrics, None, 0.009)

    def test_repeat_uses_governed_reproducibility_bounds(self) -> None:
        aggregate = {
            "aggregate": {
                "rmse_luminance": {"max": 0.002},
                "ssim_luminance": {"min": 0.999},
                "mean_oklab_delta": {"max": 0.001},
                "mean_edge_delta": {"max": 0.001},
            }
        }
        enforce_reproducibility(aggregate)
        aggregate["aggregate"]["rmse_luminance"]["max"] = 0.0021
        with self.assertRaisesRegex(QualificationError, "governed reproducibility"):
            enforce_reproducibility(aggregate)


if __name__ == "__main__":
    unittest.main()
