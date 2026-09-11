"""Shared startup content checks for the six portable game examples.

These detect missing game content, including the retained HUD-only failures.
They do not replace renderer quality goldens or prove complete gameplay.
"""

import hashlib
import re

from tools.quality.khronos_materials import png_rgb

EXAMPLES = {
    'test3d': (800, 600),
    'dungeon-crawl': (800, 600),
    'isometric-rpg': (960, 640),
    'kart-racer': (960, 540),
    'space-blaster': (800, 600),
    'voxel-sandbox': (960, 540),
}

NATIVE_WRAPPER = '''import { runGame as __bloomSmokeRun, closeWindow as __bloomSmokeClose,
  captureFrameToPng as __bloomSmokeCapture, isFrameCaptureReady as __bloomSmokeReady,
  writeFile as __bloomSmokeWrite } from "bloom/core";
let __bloomSmokeFrames = 0;
let __bloomSmokeCleanups = 0;
let __bloomSmokeRequested = false;
function __bloomExampleSmoke(update: (dt: number) => void, cleanup: () => void): void {
  __bloomSmokeRun((dt) => {
    update(dt);
    __bloomSmokeFrames = __bloomSmokeFrames + 1;
    if (__bloomSmokeFrames === 8) __bloomSmokeRequested = __bloomSmokeCapture("frame.png");
    if (__bloomSmokeFrames > 8 && __bloomSmokeReady()) __bloomSmokeClose();
    if (__bloomSmokeFrames >= 120) __bloomSmokeClose();
  }, () => {
    cleanup();
    __bloomSmokeCleanups = __bloomSmokeCleanups + 1;
    __bloomSmokeWrite("state.txt", __bloomSmokeFrames + "," + __bloomSmokeCleanups + "," + __bloomSmokeRequested);
  });
}
'''


def bounded_native_source(source):
    # Limit adaptation to one known public loop call; preserve both bodies.
    if '__bloomSmoke' in source or len(re.findall(r'\brunGame\(', source)) != 1:
        raise RuntimeError('example must contain exactly one uninstrumented runGame call')
    return NATIVE_WRAPPER + re.sub(r'\brunGame\(', '__bloomExampleSmoke(', source)


def validate_native_state(value):
    fields = value.split(',')
    if len(fields) != 3 or not fields[0].isdigit() or not 9 <= int(fields[0]) < 120 or fields[1:] != ['1', 'true']:
        raise RuntimeError(f'example did not capture and complete one cleanup before its frame limit: {value}')
    return dict(frames=int(fields[0]), cleanups=1, capture_requested=True)


def check_pixels(name, width, height, pixels):
    if name not in EXAMPLES or (width, height) != EXAMPLES[name] or len(pixels) != width * height:
        raise RuntimeError(f'{name}: capture does not have the complete expected viewport')
    counts = dict(red=0, green=0, blue=0, dungeon_floor=0, player=0)
    for index, (r, g, b) in enumerate(pixels):
        x, y = index % width, index // width
        if 60 <= y < height - 40:
            counts['red'] += r > 100 and r > g * 1.25 and r > b * 1.25
            counts['green'] += g > 60 and g > r * 1.2 and g > b * 1.2
            counts['blue'] += r < 140 and b > 100 and b > r * 1.3 and b > g * 1.1
            counts['dungeon_floor'] += (r, g, b) == (40, 40, 50)
        if name == 'dungeon-crawl' and 250 <= x < 550 and 150 <= y < 450:
            counts['player'] += (r, g, b) == (50, 150, 255)
        elif name == 'isometric-rpg' and 60 <= y < height - 40:
            counts['player'] += (r, g, b) == (50, 100, 255)
        elif name == 'space-blaster' and 250 <= x < 550 and 450 <= y < 570:
            counts['player'] += (r, g, b) == (50, 200, 255)
    requirements = {
        'test3d': {'red': 1000},
        'dungeon-crawl': {'player': 500, 'dungeon_floor': 5000},
        'isometric-rpg': {'player': 100, 'green': 10000},
        'kart-racer': {'blue': 10, 'green': 10000},
        'space-blaster': {'player': 300},
        'voxel-sandbox': {'green': 10000},
    }[name]
    for feature, minimum in requirements.items():
        if counts[feature] < minimum:
            raise RuntimeError(f'{name}: missing required game content ({feature}: {counts[feature]}, need {minimum})')
    return dict(width=width, height=height, content_pixels=counts)


def check_frame(name, path):
    return {**check_pixels(name, *png_rgb(path)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
