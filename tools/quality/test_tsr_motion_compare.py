import tempfile
import unittest
from pathlib import Path

from tools.quality.khronos_materials import QualificationError
from tools.quality.tsr_motion_compare import (
    compare,
    enforce_no_regression,
    numbered_sequence_paths,
    reference_metrics,
)


class TsrMotionCompareTests(unittest.TestCase):
    def test_candidate_must_improve_frame_and_motion_reference(self) -> None:
        native = [[(0, 0, 0)], [(10, 20, 30)], [(20, 40, 60)]]
        baseline = [[(0, 0, 0)], [(7, 14, 21)], [(14, 28, 42)]]
        candidate = [[(0, 0, 0)], [(9, 18, 27)], [(18, 36, 54)]]
        result = compare(baseline, candidate, native)
        self.assertTrue(result["passed"])
        self.assertLess(
            result["candidate"]["native_frame_rmse"],
            result["baseline"]["native_frame_rmse"],
        )
        self.assertLess(
            result["candidate"]["native_motion_derivative_rmse"],
            result["baseline"]["native_motion_derivative_rmse"],
        )

    def test_history_lag_fails_even_if_frames_are_finite(self) -> None:
        baseline = {
            "native_frame_rmse": 0.02,
            "native_motion_derivative_rmse": 0.01,
        }
        candidate = {
            "native_frame_rmse": 0.019,
            "native_motion_derivative_rmse": 0.011,
        }
        with self.assertRaisesRegex(QualificationError, "motion_derivative"):
            enforce_no_regression(baseline, candidate)

    def test_mismatched_frame_counts_fail_closed(self) -> None:
        with self.assertRaisesRegex(QualificationError, "equal multi-frame"):
            reference_metrics([[(0, 0, 0)]], [[(0, 0, 0)], [(1, 1, 1)]])

    def test_numbered_sequence_rejects_a_shifted_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "sequence-001.png").touch()
            (directory / "sequence-002.png").touch()
            with self.assertRaisesRegex(QualificationError, "exact sequence"):
                numbered_sequence_paths(directory, 2)

    def test_zero_error_has_zero_relative_change(self) -> None:
        static = [[(1, 2, 3)], [(1, 2, 3)]]
        result = compare(static, static, static)
        self.assertEqual(
            result["relative_change"],
            {
                "native_frame_rmse": 0.0,
                "native_motion_derivative_rmse": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
