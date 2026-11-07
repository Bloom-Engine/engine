import math
import unittest

from tools.quality.khronos_materials import QualificationError
from tools.quality.vsm_motion_corpus import evaluate, residual_metrics


def state(*, active: bool, motion: bool = False) -> dict[str, object]:
    return {
        "active": active,
        "physical_capacity": 256 if active else 0,
        "gpu_total_bytes": 19951824 if active else 0,
        "resident": 224 if active else 0,
        "dirty": 8 if active and motion else 0,
        "clipmap_level_rebases": 2 if active and motion else 0,
        "clipmap_pages_preserved": 180 if active and motion else 0,
        "denied": 0,
        "evictions": 0,
        "rendered": 8 if active and motion else 0,
        "pending_render": 8 if active and motion else 0,
        "render_budget": 8 if active else 0,
    }


class VsmMotionCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 160
        self.height = 90
        self.settled_csm = [(150, 150, 150)] * (self.width * self.height)
        self.settled_vsm = self.settled_csm.copy()
        for y in range(45, 70):
            for x in range(40, 120):
                self.settled_csm[y * self.width + x] = (115, 115, 115)
                self.settled_vsm[y * self.width + x] = (110, 110, 110)
        self.motion_csm = [
            tuple(min(255, channel + 2) for channel in pixel)
            for pixel in self.settled_csm
        ]
        self.motion_vsm = [
            tuple(min(255, channel + 2) for channel in pixel)
            for pixel in self.settled_vsm
        ]

    def failures(self, motion_vsm: list[tuple[int, int, int]]) -> list[str]:
        metrics = residual_metrics(
            self.width,
            self.height,
            self.settled_vsm,
            self.settled_csm,
            motion_vsm,
            self.motion_csm,
        )
        return evaluate(
            metrics,
            state(active=True),
            state(active=False),
            state(active=True, motion=True),
            state(active=False),
        )

    def modify(
        self, condition: object, delta: int
    ) -> list[tuple[int, int, int]]:
        pixels = self.motion_vsm.copy()
        for y in range(self.height):
            for x in range(self.width):
                if condition(x, y):
                    pixel = pixels[y * self.width + x]
                    pixels[y * self.width + x] = tuple(
                        max(0, min(255, channel + delta)) for channel in pixel
                    )
        return pixels

    def test_matched_control_removes_unrelated_temporal_change(self) -> None:
        self.assertEqual(self.failures(self.motion_vsm), [])

    def test_page_seam_fails_closed(self) -> None:
        failures = self.failures(self.modify(lambda x, _y: 78 <= x <= 81, 80))
        self.assertTrue(any("vertical seam" in item or "seam/ring" in item for item in failures))

    def test_clipmap_ring_fails_closed(self) -> None:
        def ring(x: int, y: int) -> bool:
            radius = math.sqrt(
                ((x - self.width / 2) / (self.width / 2)) ** 2
                + ((y - self.height / 2) / (self.height / 2)) ** 2
            )
            return 0.55 <= radius <= 0.60

        failures = self.failures(self.modify(ring, 80))
        self.assertTrue(any("seam/ring" in item or "connected" in item for item in failures))

    def test_missing_page_flash_fails_closed(self) -> None:
        failures = self.failures(
            self.modify(lambda x, y: 30 <= x < 110 and 25 <= y < 70, 90)
        )
        self.assertTrue(any("missing-shadow flash" in item for item in failures))

    def test_stale_doubled_shadow_fails_closed(self) -> None:
        failures = self.failures(
            self.modify(lambda x, y: 15 <= x < 75 and 35 <= y < 75, -80)
        )
        self.assertTrue(any("stale/doubled shadow" in item for item in failures))

    def test_missing_rebase_telemetry_fails_closed(self) -> None:
        metrics = residual_metrics(
            self.width,
            self.height,
            self.settled_vsm,
            self.settled_csm,
            self.motion_vsm,
            self.motion_csm,
        )
        failures = evaluate(
            metrics,
            state(active=True),
            state(active=False),
            state(active=True),
            state(active=False),
        )
        self.assertTrue(any("no clipmap rebase" in item for item in failures))

    def test_malformed_telemetry_fails_closed(self) -> None:
        metrics = residual_metrics(
            self.width,
            self.height,
            self.settled_vsm,
            self.settled_csm,
            self.motion_vsm,
            self.motion_csm,
        )
        malformed = state(active=True, motion=True)
        malformed["resident"] = "224"
        with self.assertRaisesRegex(QualificationError, "must be an integer"):
            evaluate(
                metrics,
                state(active=True),
                state(active=False),
                malformed,
                state(active=False),
            )


if __name__ == "__main__":
    unittest.main()
