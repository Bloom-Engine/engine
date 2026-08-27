import json
import tempfile
import unittest
from pathlib import Path

from tools.quality.prepare_virtual_geometry_stress import build_stress_scene


class VirtualGeometryStressSourceTests(unittest.TestCase):
    def test_source_is_deterministic_and_counts_logical_triangles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = build_stress_scene(root / "first" / "stress.gltf", 4, 3, 2, 2.0)
            second = build_stress_scene(root / "second" / "stress.gltf", 4, 3, 2, 2.0)
            self.assertEqual(first["source_triangles"], 48)
            self.assertEqual(first["triangles_per_mesh"], 12)
            self.assertEqual(first["vertices_per_mesh"], 12)
            self.assertEqual(first["gltf_sha256"], second["gltf_sha256"])
            self.assertEqual(first["binary_sha256"], second["binary_sha256"])
            document = json.loads((root / "first" / "stress.gltf").read_text())
            self.assertEqual(len(document["meshes"]), 4)
            self.assertEqual(len(document["nodes"]), 4)
            self.assertTrue(
                all(mesh["primitives"][0]["indices"] == 1 for mesh in document["meshes"])
            )

    def test_rejects_invalid_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "mesh_count"):
                build_stress_scene(root / "stress.gltf", mesh_count=0)
            with self.assertRaisesRegex(ValueError, "grid dimensions"):
                build_stress_scene(root / "stress.gltf", quads_x=0)
            with self.assertRaisesRegex(ValueError, "extension"):
                build_stress_scene(root / "stress.glb")


if __name__ == "__main__":
    unittest.main()
