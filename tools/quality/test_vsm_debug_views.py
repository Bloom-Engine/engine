import unittest

from tools.quality.vsm_debug_views import ORDER, evaluate


def summary(counts: dict[str, int], width: int, height: int) -> dict[str, object]:
    complete = {name: counts.get(name, 0) for name in (*ORDER, "unknown")}
    return {
        "width": width,
        "height": height,
        "scale": 1,
        "counts": complete,
    }


def state() -> dict[str, object]:
    return {
        "active": True,
        "physical_capacity": 4,
        "physical_bytes": 262_144,
        "gpu_overhead_bytes": 32_768,
        "resident": 3,
        "dirty": 2,
        "requested_pages": 3,
        "cache_hits": 2,
        "cache_misses": 1,
        "invalidated": 1,
        "rendered": 1,
        "clipmap_level_rebases": 1,
        "dynamic_overlay_draws": 2,
        "render_budget": 4,
        "levels": [
            {"level": 0, "resident": 2, "dirty": 2},
            {"level": 1, "resident": 1, "dirty": 0},
            {"level": 2, "resident": 0, "dirty": 0},
        ],
        "debug_views": {
            "available": True,
            "capture_only": True,
            "legend_order": list(ORDER),
            "colors": [
                "#080808",
                "#ffb423",
                "#ff37be",
                "#46d26e",
                "#4696ff",
                "#be64ff",
            ],
            "virtual_pages": "virtual-shadow-pages.png",
            "physical_occupancy": "virtual-shadow-physical.png",
            "legend": "virtual-shadow-legend.png",
            "report": "virtual-shadow-report.json",
        },
        "per_light_cost": [
            {
                "light": 0,
                "kind": "directional",
                "requested_pages": 3,
                "cache_hits": 2,
                "cache_misses": 1,
                "invalidated_pages": 1,
                "rendered_pages": 1,
                "resident_pages": 3,
                "dirty_pages": 2,
                "clipmap_level_rebases": 1,
                "dynamic_overlay_draws": 2,
                "physical_depth_bytes_owned": 196_608,
                "shared_pool_bytes": 262_144,
                "shared_metadata_staging_bytes": 32_768,
                "render_budget_pages": 4,
            }
        ],
    }


class VsmDebugViewTests(unittest.TestCase):
    def setUp(self) -> None:
        occupied = {
            "free": 1,
            "miss-unrendered": 1,
            "invalidated": 1,
            "clip-level-1": 1,
        }
        self.virtual = summary(
            {**occupied, "free": 32 * 96 - 3}, 32, 96
        )
        self.physical = summary(occupied, 4, 1)
        self.state = state()

    def test_consistent_views_and_costs_pass(self) -> None:
        self.assertEqual(
            evaluate(self.virtual, self.physical, self.state, True, True), []
        )

    def test_missing_required_event_colors_fail_closed(self) -> None:
        self.virtual["counts"]["miss-unrendered"] = 0
        self.virtual["counts"]["free"] += 1
        self.physical["counts"]["miss-unrendered"] = 0
        self.physical["counts"]["free"] += 1
        failures = evaluate(
            self.virtual, self.physical, self.state, True, False
        )
        self.assertTrue(any("never-rendered" in failure for failure in failures))
        self.assertTrue(any("dirty states" in failure for failure in failures))

    def test_unknown_colors_fail_closed(self) -> None:
        self.virtual["counts"]["unknown"] = 1
        failures = evaluate(
            self.virtual, self.physical, self.state, False, False
        )
        self.assertTrue(any("unknown color" in failure for failure in failures))

    def test_per_light_cost_disagreement_fails_closed(self) -> None:
        self.state["per_light_cost"][0]["shared_pool_bytes"] += 1
        failures = evaluate(
            self.virtual, self.physical, self.state, False, False
        )
        self.assertTrue(any("shared_pool_bytes" in failure for failure in failures))

    def test_clip_level_disagreement_fails_closed(self) -> None:
        self.state["levels"][1]["dirty"] = 1
        failures = evaluate(
            self.virtual, self.physical, self.state, False, False
        )
        self.assertTrue(any("clip-level-1" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
