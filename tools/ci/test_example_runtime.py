import unittest

from tools.ci.example_runtime import EXAMPLES, bounded_native_source, check_pixels, validate_native_state


class ExampleAcceptanceTests(unittest.TestCase):
    def test_native_completion_requires_readback_and_one_cleanup(self):
        self.assertEqual(validate_native_state('9,1,true')['cleanups'], 1)
        for value in ['', '8,1,true', '120,1,true', '9,0,true', '9,2,true', '9,1,false', '9,1,true,extra']:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                validate_native_state(value)

    def test_adapter_preserves_bodies_and_rejects_ambiguous_loops(self):
        source = 'runGame((dt) => { update(dt); draw(); }, () => { dispose(); });'
        bounded = bounded_native_source(source)
        self.assertTrue(bounded.endswith(source.replace('runGame(', '__bloomExampleSmoke(')))
        for bad in ['', source + source, '__bloomSmokeFrames = 0;\n' + source]:
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                bounded_native_source(bad)

    def test_a_nonblank_hud_cannot_pass_for_game_content(self):
        for name, (width, height) in EXAMPLES.items():
            pixels = [(255, 255, 255)] * (width * 40) + [(0, 0, 0)] * (width * (height - 40))
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                check_pixels(name, width, height, pixels)

    def test_isometric_terrain_alone_does_not_prove_the_player_rendered(self):
        width, height = EXAMPLES['isometric-rpg']
        pixels = [(80, 160, 60)] * (width * height)
        with self.assertRaisesRegex(RuntimeError, 'player'):
            check_pixels('isometric-rpg', width, height, pixels)
        for y in range(200, 221):
            for x in range(470, 484):
                pixels[y * width + x] = (50, 100, 255)
        self.assertEqual(check_pixels('isometric-rpg', width, height, pixels)['content_pixels']['player'], 294)


if __name__ == '__main__':
    unittest.main()
