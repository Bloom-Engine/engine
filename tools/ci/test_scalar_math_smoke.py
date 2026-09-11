import copy
import json
import unittest

from tools.ci.scalar_math_smoke import PREFIX, validate_observations


class ScalarAcceptanceTests(unittest.TestCase):
    def valid(self):
        # Independently evaluated sample values, including both branches and
        # extrapolation. These controls test rejection by the result verifier.
        return {'samples': [
            [-.25, -4.3125, .0625, -.5625, .125, -.015625, -.953125, -.0625, 0, -4.3125],
            [0, -2.5, 0, 0, 0, 0, 0, 0, 0, -2.5],
            [.1, -1.775, .01, .19, .02, .001, .271, .004, .1, -1.775],
            [.25, -.6875, .0625, .4375, .125, .015625, .578125, .0625, .25, -.6875],
            [.5, 1.125, .25, .75, .5, .125, .875, .5, .5, 1.125],
            [.75, 2.9375, .5625, .9375, .875, .421875, .984375, .9375, .75, 2.9375],
            [1, 4.75, 1, 1, 1, 1, 1, 1, 1, 4.75],
            [1.25, 6.5625, 1.5625, .9375, .875, 1.953125, 1.015625, 1.0625, 1, 6.5625],
        ], 'nanPreserved': True, 'infinityPreserved': True, 'negativeZeroPreserved': True}

    def test_rejects_integer_truncation_and_bad_numeric_types(self):
        expected = self.valid()
        validate_observations(PREFIX + json.dumps(expected))
        for column in range(1, 8):
            for bad in [0, True, '0', None, float('nan'), float('inf')]:
                actual = copy.deepcopy(expected)
                actual['samples'][3][column] = bad
                with self.subTest(column=column, bad=bad), self.assertRaises(RuntimeError):
                    validate_observations(PREFIX + json.dumps(actual))

    def test_rejects_missing_duplicate_and_incomplete_reports(self):
        line = PREFIX + json.dumps(self.valid())
        for output in ['', line + '\n' + line, PREFIX + '{}']:
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                validate_observations(output)
        for key in ['nanPreserved', 'infinityPreserved', 'negativeZeroPreserved']:
            actual = self.valid()
            actual[key] = False
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_observations(PREFIX + json.dumps(actual))
        actual = self.valid()
        actual['samples'].pop()
        with self.assertRaises(RuntimeError):
            validate_observations(PREFIX + json.dumps(actual))


if __name__ == '__main__':
    unittest.main()
