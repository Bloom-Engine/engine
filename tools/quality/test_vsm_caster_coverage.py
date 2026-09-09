import unittest

from tools.quality.khronos_materials import QualificationError
from tools.quality.vsm_caster_coverage import (
    ALPHA_ROI,
    SKINNED_ROI,
    difference_metrics,
    evaluate,
)


def state(*, cutout: int, skinned: int, active: bool = True) -> dict[str, object]:
    return {
        "active": active,
        "page_cutout_draws": cutout,
        "page_skinned_draws": skinned,
        "dynamic_overlay_rendered_pages": 4,
    }


class VsmCasterCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 320
        self.height = 180
        self.neutral = [(180, 180, 180)] * (self.width * self.height)

    def patterned_candidate(
        self, roi: tuple[float, float, float, float], period: int
    ) -> list[tuple[int, int, int]]:
        pixels = self.neutral.copy()
        x0, y0, x1, y1 = (
            round(roi[0] * self.width),
            round(roi[1] * self.height),
            round(roi[2] * self.width),
            round(roi[3] * self.height),
        )
        for y in range(y0, y1):
            for x in range(x0, x1):
                if (x // period + y // period) % 2 == 0:
                    pixels[y * self.width + x] = (80, 80, 80)
        return pixels

    def test_patterned_alpha_and_moving_skinned_shadow_pass(self) -> None:
        alpha = difference_metrics(
            self.width,
            self.height,
            self.neutral,
            self.patterned_candidate(ALPHA_ROI, 2),
            ALPHA_ROI,
        )
        skinned = difference_metrics(
            self.width,
            self.height,
            self.neutral,
            self.patterned_candidate(SKINNED_ROI, 3),
            SKINNED_ROI,
        )
        failures = evaluate(
            alpha,
            skinned,
            state(cutout=4, skinned=4),
            state(cutout=0, skinned=4),
            state(cutout=4, skinned=4),
        )
        self.assertEqual(failures, [])

    def test_missing_caster_draws_fail_closed(self) -> None:
        changed = self.patterned_candidate(ALPHA_ROI, 2)
        alpha = difference_metrics(
            self.width, self.height, self.neutral, changed, ALPHA_ROI
        )
        skinned = difference_metrics(
            self.width,
            self.height,
            self.neutral,
            self.patterned_candidate(SKINNED_ROI, 2),
            SKINNED_ROI,
        )
        failures = evaluate(
            alpha,
            skinned,
            state(cutout=0, skinned=0),
            state(cutout=1, skinned=0),
            state(cutout=0, skinned=0),
        )
        self.assertTrue(any("alpha-tested VSM page draw" in item for item in failures))
        self.assertTrue(any("skinned VSM page draw" in item for item in failures))
        self.assertTrue(any("alpha control still" in item for item in failures))

    def test_empty_images_cannot_pass(self) -> None:
        alpha = difference_metrics(
            self.width, self.height, self.neutral, self.neutral, ALPHA_ROI
        )
        skinned = difference_metrics(
            self.width, self.height, self.neutral, self.neutral, SKINNED_ROI
        )
        failures = evaluate(
            alpha,
            skinned,
            state(cutout=4, skinned=4),
            state(cutout=0, skinned=4),
            state(cutout=4, skinned=4),
        )
        self.assertTrue(any("ground-shadow pixels" in item for item in failures))

    def test_opaque_alpha_fill_cannot_pass_as_cutout_coverage(self) -> None:
        opaque = self.neutral.copy()
        x0, y0, x1, y1 = (
            round(ALPHA_ROI[0] * self.width),
            round(ALPHA_ROI[1] * self.height),
            round(ALPHA_ROI[2] * self.width),
            round(ALPHA_ROI[3] * self.height),
        )
        for y in range(y0, y1):
            for x in range(x0, x1):
                opaque[y * self.width + x] = (80, 80, 80)
        alpha = difference_metrics(
            self.width, self.height, self.neutral, opaque, ALPHA_ROI
        )
        skinned = difference_metrics(
            self.width,
            self.height,
            self.neutral,
            self.patterned_candidate(SKINNED_ROI, 2),
            SKINNED_ROI,
        )
        failures = evaluate(
            alpha,
            skinned,
            state(cutout=4, skinned=4),
            state(cutout=0, skinned=4),
            state(cutout=4, skinned=4),
        )
        self.assertTrue(any("opaque ROI fill" in item for item in failures))

    def test_malformed_telemetry_fails_closed(self) -> None:
        metrics = difference_metrics(
            self.width,
            self.height,
            self.neutral,
            self.patterned_candidate(ALPHA_ROI, 2),
            ALPHA_ROI,
        )
        malformed = state(cutout=4, skinned=4)
        malformed["page_cutout_draws"] = "4"
        with self.assertRaisesRegex(QualificationError, "must be an integer"):
            evaluate(
                metrics,
                metrics,
                malformed,
                state(cutout=0, skinned=4),
                state(cutout=4, skinned=4),
            )
