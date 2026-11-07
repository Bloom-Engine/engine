import unittest

from tools.quality.virtual_geometry_stress import (
    StressQualificationError,
    default_backend,
    default_platform,
    evaluate_scaling,
    select_index_entry,
)


class VirtualGeometryStressQualificationTests(unittest.TestCase):
    @staticmethod
    def scaling_summary(
        instances: int, candidates: int, selected: int, selector_ms: float
    ) -> dict[str, object]:
        return {
            "placements": instances,
            "available_placements": 100,
            "source_triangles": 10_000_000,
            "archive_clusters": 245_500,
            "archive_pages": 8_496,
            "measured_frames": 120,
            "measurement_wall_ms": 720.0,
            "runtime": {
                "last_visible_groups": candidates,
                "last_frustum_culled_groups": 0,
                "last_selected_count": selected,
                "resident_pages": 100,
            },
            "profile": {
                "gpu_frame_mean_ms": 4.0,
                "passes": [
                    {
                        "label": "virtual_geometry_hierarchy_selection",
                        "gpu_mean_ms": selector_ms,
                    },
                    {
                        "label": "virtual_geometry_draw_emission",
                        "gpu_mean_ms": 0.1,
                    },
                ],
            },
        }

    def test_host_platforms_choose_explicit_primary_backends(self) -> None:
        self.assertEqual(default_platform("Darwin"), "macos")
        self.assertEqual(default_platform("Linux"), "linux")
        self.assertEqual(default_platform("Windows"), "windows")
        self.assertEqual(default_backend("macos"), "metal")
        self.assertEqual(default_backend("linux"), "vulkan")
        self.assertEqual(default_backend("windows"), "dx12")
        with self.assertRaisesRegex(StressQualificationError, "unsupported"):
            default_platform("Plan9")

    def test_selects_exact_profile_and_rejects_missing_or_duplicate_entries(self) -> None:
        expected = {
            "logical_id": "stress/10m",
            "profile": {"platform": "linux", "quality": "high"},
            "artifact": {"path": "chunks/sha256/a.bgeo"},
        }
        other = {
            "logical_id": "stress/10m",
            "profile": {"platform": "macos", "quality": "high"},
            "artifact": {"path": "chunks/sha256/b.bgeo"},
        }
        index = {"entries": [other, expected]}
        self.assertIs(
            select_index_entry(index, "stress/10m", "linux", "high"), expected
        )
        with self.assertRaisesRegex(StressQualificationError, "0 exact"):
            select_index_entry(index, "stress/10m", "windows", "high")
        with self.assertRaisesRegex(StressQualificationError, "2 exact"):
            select_index_entry(
                {"entries": [expected, dict(expected)]},
                "stress/10m",
                "linux",
                "high",
            )

    def test_scaling_requires_fixed_archive_and_candidate_bounded_growth(self) -> None:
        summaries = [
            self.scaling_summary(1, 4, 70, 0.8),
            self.scaling_summary(10, 40, 700, 1.0),
            self.scaling_summary(100, 440, 10_000, 1.6),
        ]
        result = evaluate_scaling(summaries, [1, 10, 100])
        self.assertEqual(result["validation"], "pass")
        self.assertEqual(result["candidate_group_growth"], 110.0)
        self.assertEqual(result["selector_gpu_growth"], 2.0)

        changed_archive = [dict(summary) for summary in summaries]
        changed_archive[1]["source_triangles"] = 1_000_000
        with self.assertRaisesRegex(StressQualificationError, "same 10M"):
            evaluate_scaling(changed_archive, [1, 10, 100])

        disproportionate = [dict(summary) for summary in summaries]
        disproportionate[-1] = self.scaling_summary(100, 440, 10_000, 30.0)
        with self.assertRaisesRegex(StressQualificationError, "disproportionately"):
            evaluate_scaling(disproportionate, [1, 10, 100])


if __name__ == "__main__":
    unittest.main()
