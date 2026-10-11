import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'eclipse-aligner.py'
EXAMPLES = ROOT / 'examples'

# BGR color of the synthetic sun
SUN_COLOR = (60, 170, 240)


@pytest.fixture(scope='session')
def aligner():
    '''The eclipse-aligner.py script imported as a module.'''
    spec = importlib.util.spec_from_file_location('eclipse_aligner', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_frame(width, height, sun_xy, sun_r, moon_offset=None, moon_r=None):
    '''A synthetic eclipse frame: a bright sun disk, optionally covered by a black moon disk, on a black sky.'''
    img = np.zeros((height, width, 3), np.uint8)
    cv2.circle(img, sun_xy, sun_r, SUN_COLOR, -1)
    if moon_offset is not None:
        moon_xy = (sun_xy[0] + moon_offset[0], sun_xy[1] + moon_offset[1])
        cv2.circle(img, moon_xy, moon_r, (0, 0, 0), -1)
    return img


def write_jpg(path, img):
    assert cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def read_jpg(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, path
    return img


def run_cli(*args, cwd, timeout=120, input=None):
    '''Run eclipse-aligner.py in a subprocess, a hang fails the test with subprocess.TimeoutExpired.'''
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *[str(a) for a in args]],
        cwd=cwd,
        input=input,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=timeout,
        env=env,
    )
    # show the output of failing tests
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    return result
