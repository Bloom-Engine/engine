import copy
import unittest

from tools.ci.starter_browser_smoke import validate_state, check_pixels


def valid_state():
    return dict(frames=8, cleanups=1, registrations=1, stopped=True, errors=[],
                reads=[dict(path='assets/welcome.txt', value='Hello, Bloom!\n')],
                fetches=[dict(path='/assets/welcome.txt', status=200)],
                texts=[['Hello, Bloom!\n', 24, 24, 24, 255, 255, 255, 255] for _ in range(8)],
                rects=[[368, 193, 64, 64, 255, 255, 255, 255] for _ in range(8)])


class StarterBrowserAcceptanceTests(unittest.TestCase):
    def test_requires_real_asset_draw_progress_and_one_cleanup(self):
        state = valid_state()
        validate_state(state)
        for change in (dict(frames=7), dict(cleanups=2), dict(registrations=True), dict(stopped=False),
                       dict(errors=['startup failed']), dict(reads=[]), dict(fetches=[]), dict(texts=[]),
                       dict(rects=[]), dict(reads=[dict(path='assets/welcome.txt', value='')])):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_state({**state, **change})
        for field, position, value in [('texts', 0, 'wrong asset'), ('rects', 0, float('nan')), ('rects', 2, 0)]:
            wrong = copy.deepcopy(state)
            wrong[field][0][position] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError): validate_state(wrong)

    def test_pixels_reject_empty_missing_text_and_unexpected_background(self):
        pixels = [(0, 0, 0)] * (800 * 450)
        for y in range(193, 257):
            for x in range(368, 432): pixels[y * 800 + x] = (255, 255, 255)
        with self.assertRaises(RuntimeError): check_pixels(800, 450, pixels, [368])
        for x in range(24, 224): pixels[32 * 800 + x] = (200, 200, 200)
        check_pixels(800, 450, pixels, [368])
        pixels[225 * 800 + 400] = (0, 0, 0)
        with self.assertRaises(RuntimeError): check_pixels(800, 450, pixels, [368])
        pixels[225 * 800 + 400] = (255, 255, 255)
        pixels[0] = (255, 255, 255)
        with self.assertRaises(RuntimeError): check_pixels(800, 450, pixels, [368])


if __name__ == '__main__': unittest.main()
