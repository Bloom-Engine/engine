import unittest

from tools.quality.virtual_geometry_stress import (
    StressQualificationError,
    default_backend,
    default_platform,
    select_index_entry,
)


class VirtualGeometryStressQualificationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
