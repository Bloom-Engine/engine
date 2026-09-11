import copy
import unittest

from tools.ci.compile_web_examples import validate_imports
from tools.ci.example_runtime import EXAMPLES
from tools.ci.web_example_smoke import validate_prepared, validate_state


def state_for(name):
    return dict(frames=8, cleanups=1, registrations=1, stopped=True, errors=[], badArguments=[],
                windows=[list(EXAMPLES[name])], frameDraws=[1] * 8,
                calls=dict(bloom_close_window=2, bloom_init_audio=1, bloom_close_audio=1,
                           bloom_disable_cursor=1, bloom_enable_cursor=1))


class WebExampleAcceptanceTests(unittest.TestCase):
    def test_requires_complete_frames_valid_arguments_viewport_and_cleanup(self):
        for name in EXAMPLES:
            valid = state_for(name)
            validate_state(name, valid)
            for changes in [dict(frames=7), dict(frames=True), dict(cleanups=2), dict(registrations=0),
                            dict(stopped=False), dict(windows=[[128, 128]]), dict(errors=['render failed']),
                            dict(badArguments=[dict(name='bloom_draw_rect', args=['undefined'])]),
                            dict(frameDraws=[1]*7+[0]), dict(frameDraws=[True]*8), dict(calls={})]:
                with self.subTest(name=name, changes=changes), self.assertRaises(RuntimeError):
                    validate_state(name, {**valid, **changes})
        for name, key in [('space-blaster', 'bloom_close_audio'), ('voxel-sandbox', 'bloom_enable_cursor')]:
            bad = state_for(name)
            bad['calls'][key] = 0
            with self.subTest(name=name), self.assertRaises(RuntimeError): validate_state(name, bad)

    def test_artifact_requires_exact_inventory_source_and_pinned_compiler(self):
        valid = dict(status='pass', source_commit='candidate', compiler_version='perry 0.5.1220',
                     examples=[dict(name=name, status='pass', source_unchanged=True) for name in EXAMPLES])
        validate_prepared(valid, 'candidate')
        for changes in [dict(status='fail'), dict(source_commit='older'), dict(compiler_version='perry 0.5.999'),
                        dict(examples=valid['examples'][:-1]), dict(examples=valid['examples'] + [valid['examples'][0]])]:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                validate_prepared({**valid, **changes}, 'candidate')
        for field, value in [('status', 'fail'), ('source_unchanged', False), ('name', 'unknown')]:
            bad = copy.deepcopy(valid)
            bad['examples'][0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError): validate_prepared(bad, 'candidate')

    def test_compiler_success_cannot_hide_unresolved_or_missing_engine_imports(self):
        imports = [dict(module='ffi', name=name) for name in ['bloom_init_window', 'bloom_run_game_with_cleanup',
                   'bloom_close_window', 'bloom_clear_background', 'bloom_draw_rect']]
        self.assertEqual(len(validate_imports('', imports)), 5)
        with self.assertRaises(RuntimeError): validate_imports('Could not resolve import bloom/core', imports)
        for index in range(len(imports)):
            with self.subTest(index=index), self.assertRaises(RuntimeError): validate_imports('', imports[:index] + imports[index+1:])


if __name__ == '__main__': unittest.main()
