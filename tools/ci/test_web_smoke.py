#!/usr/bin/env python3

import unittest

from tools.ci.web_smoke import enforce_tsr_limits, luma_stddev, sequence_activity


class WebSmokeMetricTests(unittest.TestCase):
    def test_motion_activity_rejects_static_negative_control(self) -> None:
        still = [[(10, 20, 30), (40, 50, 60)]] * 2
        moving = [
            [(10, 20, 30), (40, 50, 60)],
            [(11, 22, 33), (44, 55, 66)],
        ]
        self.assertEqual(sequence_activity(still), 0.0)
        self.assertGreater(sequence_activity(moving), 0.0)

    def test_luma_stddev_distinguishes_flat_and_structured_frames(self) -> None:
        self.assertEqual(luma_stddev([(20, 20, 20)] * 4), 0.0)
        self.assertGreater(luma_stddev([(0, 0, 0), (255, 255, 255)]), 100.0)

    def test_tsr_limits_use_the_governed_directions(self) -> None:
        passing = {
            "native_frame_rmse": 0.02,
            "native_motion_derivative_rmse": 0.02,
            "native_motion_activity_rgb_8bit": 0.1,
            "fractional_motion_activity_rgb_8bit": 0.1,
            "fractional_scene_luma_stddev_8bit": 20.0,
        }
        self.assertEqual(enforce_tsr_limits(passing), [])

        failing = {
            "native_frame_rmse": 1.0,
            "native_motion_derivative_rmse": 1.0,
            "native_motion_activity_rgb_8bit": 0.0,
            "fractional_motion_activity_rgb_8bit": 0.0,
            "fractional_scene_luma_stddev_8bit": 0.0,
        }
        self.assertEqual(len(enforce_tsr_limits(failing)), 5)


if __name__ == "__main__":
    unittest.main()
