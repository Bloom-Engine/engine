import copy
import json
import unittest

from tools.ci.palette_smoke import COLORS, INPUTS, PREFIX, ROOT_CONSTANTS, validate_observations


def reference():
    return dict(colors=[[name] + list(bytes.fromhex(rgba)) * 4 + [True] * 3 for name, rgba in COLORS.items()],
                inputs=INPUTS, rootInputs=INPUTS, rootConstants=ROOT_CONSTANTS,
                aliasMutation=True, restored=True, sharedMaps=True)


class PublicConstantAcceptanceTests(unittest.TestCase):
    def test_reference_and_failed_constant_alias_controls(self):
        valid = reference()
        validate_observations(PREFIX + json.dumps(valid))
        for key, value in [('aliasMutation', False), ('restored', False), ('sharedMaps', False),
                           ('inputs', []), ('rootInputs', []), ('rootConstants', []), ('colors', [])]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_observations(PREFIX + json.dumps({**valid, key: value}))
        for field, index in [('inputs', 0), ('rootInputs', 0), ('rootConstants', 1)]:
            for value in [None, '1', True, float('nan'), -100]:
                bad = copy.deepcopy(valid)
                bad[field][index] = value
                with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                    validate_observations(PREFIX + json.dumps(bad))
        for index, value in [(0, 'duplicate'), (1, None), (5, 0), (9, True), (13, -1), (17, False), (18, False), (19, False)]:
            bad = copy.deepcopy(valid)
            bad['colors'][0][index] = value
            with self.subTest(index=index), self.assertRaises(RuntimeError): validate_observations(PREFIX + json.dumps(bad))
        output = PREFIX + json.dumps(valid)
        for bad in ['', output + '\n' + output]:
            with self.assertRaises(RuntimeError): validate_observations(bad)


if __name__ == '__main__': unittest.main()
