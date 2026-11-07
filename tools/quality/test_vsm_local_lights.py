from __future__ import annotations

import unittest

from tools.quality.vsm_local_lights import evaluate


def fixture(submitted: int = 100) -> dict[str, object]:
    active = {0, 1, 2, 3, 4}
    rows: list[dict[str, object]] = [
        {
            "light": 0,
            "kind": "directional",
        }
    ]
    for light in range(submitted):
        is_active = light in active
        rows.append(
            {
                "light": light,
                "cache_light": light + 1,
                "kind": "point",
                "state": "active" if is_active else "suppressed",
                "requested_pages": 6 if is_active else 0,
                "resident_pages": 6 if is_active else 0,
                "dirty_pages": 0,
                "rendered_pages": 0,
                "render_budget_pages": 8,
            }
        )
    return {
        "enabled": True,
        "active": True,
        "physical_capacity": 256,
        "resident": 250,
        "render_budget": 8,
        "rendered": 0,
        "local_lights": {
            "submission_limit": 256,
            "admission_limit": 5,
            "faces_per_light": 6,
            "submitted": submitted,
            "visible": submitted,
            "admitted": 5,
            "active_shaded": 5,
            "visibility_rejected": 0,
            "budget_suppressed": submitted - 5,
            "requested_pages": 30,
            "resident_pages": 30,
            "dirty_pages": 0,
            "rendered_pages": 0,
            "shared_page_budget": True,
            "fallback": "suppress-direct-contribution",
        },
        "per_light_cost": rows,
    }


class LocalVsmValidationTests(unittest.TestCase):
    def test_one_hundred_light_fixture_passes(self) -> None:
        self.assertEqual(evaluate(fixture(), 100), [])

    def test_submission_below_requirement_fails(self) -> None:
        failures = evaluate(fixture(99), 100)
        self.assertTrue(any("require at least 100" in failure for failure in failures))

    def test_partial_cube_face_admission_fails(self) -> None:
        state = fixture()
        local = state["local_lights"]
        assert isinstance(local, dict)
        local["requested_pages"] = 29
        failures = evaluate(state, 100)
        self.assertTrue(any("projection faces" in failure for failure in failures))

    def test_unbounded_same_frame_rendering_fails(self) -> None:
        state = fixture()
        state["rendered"] = 9
        failures = evaluate(state, 100)
        self.assertTrue(any("hard page budget" in failure for failure in failures))

    def test_active_light_must_have_six_clean_pages(self) -> None:
        state = fixture()
        rows = state["per_light_cost"]
        assert isinstance(rows, list) and isinstance(rows[1], dict)
        rows[1]["dirty_pages"] = 1
        failures = evaluate(state, 100)
        self.assertTrue(any("fully resident and clean" in failure for failure in failures))

    def test_suppressed_light_cannot_request_pages(self) -> None:
        state = fixture()
        rows = state["per_light_cost"]
        assert isinstance(rows, list) and isinstance(rows[-1], dict)
        rows[-1]["requested_pages"] = 6
        failures = evaluate(state, 100)
        self.assertTrue(any("suppressed" in failure for failure in failures))

    def test_one_page_spot_projection_passes(self) -> None:
        state = fixture()
        local = state["local_lights"]
        rows = state["per_light_cost"]
        assert isinstance(local, dict) and isinstance(rows, list)
        local["spot_faces_per_light"] = 1
        local["requested_pages"] = 5
        local["resident_pages"] = 5
        for row in rows[1:]:
            assert isinstance(row, dict)
            row["kind"] = "spot"
            if row["state"] == "active":
                row["requested_pages"] = 1
                row["resident_pages"] = 1
        self.assertEqual(evaluate(state, 100, "spot"), [])


if __name__ == "__main__":
    unittest.main()
