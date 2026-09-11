"""Negative controls for installed native startup acceptance."""

from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from tools.ci.native_package_smoke import check_frame


def fixture_png(path: Path, square: tuple[int, int, int] | None) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = bytearray()
    for y in range(128):
        rows.append(0)
        for x in range(128):
            rows.extend(square if square and 32 <= x < 96 and 32 <= y < 96 else (0, 0, 0))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class NativeFrameControls(unittest.TestCase):
    def test_blank_frame_cannot_pass_startup(self):
        with tempfile.TemporaryDirectory(prefix="bloom-frame-") as directory:
            path = Path(directory) / "blank.png"
            fixture_png(path, None)
            with self.assertRaisesRegex(RuntimeError, "4096 of 16384"):
                check_frame(path)

    def test_physics_failure_color_cannot_pass_startup(self):
        with tempfile.TemporaryDirectory(prefix="bloom-frame-") as directory:
            path = Path(directory) / "physics-failed.png"
            fixture_png(path, (255, 0, 0))
            with self.assertRaisesRegex(RuntimeError, "4096 of 16384"):
                check_frame(path)


if __name__ == "__main__":
    unittest.main()
