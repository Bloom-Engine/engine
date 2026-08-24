import unittest

from tools.quality.khronos_materials import QualificationError
from tools.quality.bistro_temporal_matrix import (
    CHANGE_THRESHOLD,
    control_deltas,
    parse_size,
    sequence_metrics,
)


class BistroTemporalMatrixTests(unittest.TestCase):
    def test_static_sequence_has_zero_temporal_residual(self) -> None:
        frame = [(30, 60, 90)] * 24
        metrics = sequence_metrics(6, 4, [frame, frame, frame])
        self.assertEqual(metrics["temporal_max_channel_range_8bit"]["max"], 0)
        self.assertEqual(metrics["adjacent_frames"]["mean_absolute_rgb_8bit"], 0)
        self.assertEqual(
            metrics["adjacent_frames"]["largest_component_over_threshold"]["pixels"],
            0,
        )

    def test_coherent_change_is_reported(self) -> None:
        first = [(0, 0, 0)] * 24
        second = first.copy()
        for y in range(1, 3):
            for x in range(1, 5):
                second[y * 6 + x] = (CHANGE_THRESHOLD + 1, 0, 0)
        metrics = sequence_metrics(6, 4, [first, second])
        component = metrics["adjacent_frames"]["largest_component_over_threshold"]
        self.assertEqual(component["pixels"], 8)
        self.assertAlmostEqual(component["pixel_ratio"], 8 / 24)
        self.assertEqual(metrics["temporal_max_channel_range_8bit"]["p99"], 9)

    def test_malformed_frame_fails_closed(self) -> None:
        with self.assertRaisesRegex(QualificationError, "expected 4"):
            sequence_metrics(2, 2, [[(0, 0, 0)] * 4, [(0, 0, 0)] * 3])

    def test_capture_size_is_canonicalized(self) -> None:
        self.assertEqual(parse_size("0512X0288"), "512x288")

    def test_control_delta_reports_owner_signal(self) -> None:
        def result(mean_range: float, mean_adjacent: float, component: float) -> dict:
            return {
                "temporal_max_channel_range_8bit": {"mean": mean_range},
                "adjacent_frames": {
                    "mean_absolute_rgb_8bit": mean_adjacent,
                    "largest_component_over_threshold": {"pixel_ratio": component},
                },
            }

        deltas = control_deltas(
            {
                "full": result(4.0, 2.0, 0.1),
                "no-taa": result(1.0, 0.5, 0.025),
            }
        )
        self.assertEqual(deltas["no-taa"]["temporal_range_mean_reduction"], 0.75)
        self.assertEqual(deltas["no-taa"]["adjacent_rgb_mean_reduction"], 0.75)
        self.assertEqual(deltas["no-taa"]["largest_component_reduction"], 0.75)


if __name__ == "__main__":
    unittest.main()
