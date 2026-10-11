'''End to end tests that run eclipse-aligner.py on synthetic and example images in a temporary directory.'''
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

from conftest import EXAMPLES, SCRIPT, draw_frame, read_jpg, run_cli, write_jpg

OPTIONS = [
    '--input', '--sun', '--sun_radius', '--sun_threshold', '--fix_angle', '--moon_radius_mod', '--moon_threshold_mod',
    '--max_phase_rotation_group_threshold', '--min_phase_rotation_group_threshold', '--filter', '--clipped_cutoff',
    '--empty_cutoff', '--workers', '--output_suffix', '--clipped_suffix', '--circled_suffix', '--sun_binary_suffix',
    '--moon_binary_suffix', '--circles'
]


def make_input(tmp_path, frames, name='jpg'):
    '''Write {filename: image} to tmp_path/name and return the directory.'''
    jpg_dir = tmp_path / name
    jpg_dir.mkdir()
    for filename, img in frames.items():
        write_jpg(jpg_dir / filename, img)
    return jpg_dir


def read_circles(tmp_path):
    return json.loads((tmp_path / 'circles_data.json').read_text())


def bright_pixels(img, threshold=60):
    '''(x, y) arrays of the pixels brighter than threshold.'''
    ys, xs = np.nonzero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > threshold)
    return xs, ys


def assert_sun_centered(img, sun_r, tolerance=3):
    '''All bright pixels are inside the sun circle at the center of the image.'''
    h, w = img.shape[:2]
    xs, ys = bright_pixels(img)
    assert len(xs) > 0
    assert np.hypot(xs - w // 2, ys - h // 2).max() <= sun_r + tolerance


def moon_offset(distance, degree):
    '''Offset of the moon from the sun, degree is measured from the down direction towards the right (like the script).'''
    return round(distance * math.sin(math.radians(degree))), round(distance * math.cos(math.radians(degree)))


def test_help_lists_all_options(tmp_path):
    result = run_cli('-h', cwd=tmp_path)
    assert result.returncode == 0
    for option in OPTIONS:
        assert option in result.stdout


def test_missing_input_is_an_error(tmp_path):
    result = run_cli(cwd=tmp_path)
    assert result.returncode == 2
    assert '--input' in result.stderr


def test_centers_the_sun(tmp_path):
    width, height, sun_r, moon_r = 960, 640, 120, 118
    rng = np.random.default_rng(1)
    truth, frames = {}, {}
    for i, offset in enumerate([(230, 40), (150, 100), (-90, 140), (-200, -60)]):
        sun_xy = (int(rng.integers(140, width - 140)), int(rng.integers(140, height - 140)))
        truth[f'f{i}.jpg'] = (sun_xy, offset)
        frames[f'f{i}.jpg'] = draw_frame(width, height, sun_xy, sun_r, offset, moon_r)
    jpg_dir = make_input(tmp_path, frames)

    result = run_cli('--input', jpg_dir, '--sun_radius', sun_r, '--workers', 2, cwd=tmp_path)
    assert result.returncode == 0

    circles = read_circles(tmp_path)
    assert sorted(circles) == sorted(frames)
    for filename, (sun_xy, offset) in truth.items():
        sun_x, sun_y, r = circles[filename]['sun']
        assert abs(sun_x - sun_xy[0]) <= 2 and abs(sun_y - sun_xy[1]) <= 2 and r == sun_r
        output = read_jpg(tmp_path / 'jpg-output' / filename)
        assert output.shape == (height, width, 3)
        # the output looks like the same eclipse phase drawn at the center of the image
        expected = draw_frame(width, height, (width // 2, height // 2), sun_r, offset, moon_r)
        assert np.abs(output.astype(int) - expected).mean() < 2
    for suffix in ('-circled', '-sun-binary'):
        assert sorted(p.name for p in (tmp_path / f'jpg{suffix}').iterdir()) == sorted(frames)


@pytest.mark.parametrize('fix_angle', [False, True], ids=['center', 'fix_angle'])
def test_small_sun_in_large_frame(tmp_path, fix_angle):
    # (sun_r + 2 * moon_r) * 2 = 344 is smaller than the 1600x1000 frame, np.pad got negative values (#3)
    width, height, sun_r, moon_r, model_moon_r = 1600, 1000, 60, 59, 57
    rng = np.random.default_rng(3)
    truth, frames = {}, {}
    for i, offset in enumerate([(70, 40), (35, 20), (2, 1), (-35, -20), (-70, -40)]):
        sun_xy = (int(rng.integers(80, width - 80)), int(rng.integers(80, height - 80)))
        truth[f'f{i}.jpg'] = (sun_xy, offset)
        frames[f'f{i}.jpg'] = draw_frame(width, height, sun_xy, sun_r, offset, moon_r)
    jpg_dir = make_input(tmp_path, frames)

    args = ['--input', jpg_dir, '--sun_radius', sun_r, '--workers', 2]
    if fix_angle:
        args += [
            '--fix_angle', 0, '--moon_radius_mod', model_moon_r - sun_r, '--max_phase_rotation_group_threshold', 10,
            '--min_phase_rotation_group_threshold', 20
        ]
    result = run_cli(*args, cwd=tmp_path)
    assert result.returncode == 0

    circles = read_circles(tmp_path)
    for filename, (sun_xy, offset) in truth.items():
        sun_x, sun_y, _ = circles[filename]['sun']
        assert abs(sun_x - sun_xy[0]) <= 2 and abs(sun_y - sun_xy[1]) <= 2
        if fix_angle:
            moon_x, moon_y, r, _, moon_signal_center = circles[filename]['moon']
            assert r == model_moon_r
            # the smaller model moon can sit up to 2 px closer to the sun than the drawn moon
            assert abs(moon_x - moon_signal_center - offset[0]) <= 3
            assert abs(moon_y - moon_signal_center - offset[1]) <= 3
        assert_sun_centered(read_jpg(tmp_path / 'jpg-output' / filename), sun_r)


def test_fix_angle_aligns_the_moon_path(tmp_path):
    width, height, sun_r, moon_r, model_moon_r, fix_angle = 800, 600, 150, 148, 146, 90
    # the moon path rotates by 10 degrees per frame (field rotation), f3 is at annularity and keeps the f2 angle
    phases = [(190, 30), (130, 40), (70, 50), (10, 50), (70, 240), (130, 250)]
    rng = np.random.default_rng(5)
    frames, offsets = {}, {}
    for i, (distance, degree) in enumerate(phases):
        sun_xy = (int(rng.integers(360, 440)), int(rng.integers(260, 340)))
        offsets[f'f{i}.jpg'] = moon_offset(distance, degree)
        frames[f'f{i}.jpg'] = draw_frame(width, height, sun_xy, sun_r, offsets[f'f{i}.jpg'], moon_r)
    jpg_dir = make_input(tmp_path, frames)

    result = run_cli(
        '--input', jpg_dir, '--sun_radius', sun_r, '--moon_radius_mod', model_moon_r - sun_r, '--fix_angle', fix_angle,
        '--workers', 2, cwd=tmp_path
    )
    assert result.returncode == 0
    assert 'Using MOON RADIUS: 146' in result.stdout
    assert sorted(p.name for p in (tmp_path / 'jpg-moon-binary').iterdir()) == sorted(frames)

    circles = read_circles(tmp_path)
    for filename, (distance, _) in zip(frames, phases):
        moon_x, moon_y, _, min_dist, moon_signal_center = circles[filename]['moon']
        offset = offsets[filename]
        assert abs(moon_x - moon_signal_center - offset[0]) <= 3
        assert abs(moon_y - moon_signal_center - offset[1]) <= 3
        assert abs(min_dist - distance) <= 3
        if distance < 30:
            continue
        output = read_jpg(tmp_path / 'jpg-output' / filename)
        assert_sun_centered(output, sun_r)
        # the bright crescent is opposite the moon, both are on the line at fix_angle through the center
        xs, ys = bright_pixels(output)
        crescent_degree = math.degrees(math.atan2(xs.mean() - width // 2, ys.mean() - height // 2))
        error = (crescent_degree - fix_angle) % 180
        assert min(error, 180 - error) < 4, (filename, crescent_degree)

    # reuse the detected coordinates with --circles, without running the detection again
    first_output = (tmp_path / 'jpg-output' / 'f2.jpg').read_bytes()
    shutil.rmtree(tmp_path / 'jpg-output')
    result = run_cli(
        '--input', jpg_dir, '--sun_radius', sun_r, '--moon_radius_mod', model_moon_r - sun_r, '--fix_angle', fix_angle,
        '--circles', tmp_path / 'circles_data.json', cwd=tmp_path
    )
    assert result.returncode == 0
    assert 'Using existing coordinates data' in result.stdout
    assert 'Starting' not in result.stdout
    assert (tmp_path / 'jpg-output' / 'f2.jpg').read_bytes() == first_output


def test_detects_a_small_sun_next_to_noise(tmp_path):
    # the --sun radius was detected as 0 and a radius below 50 px made cv2.circle assert (#4)
    width, height, sun_r = 900, 600, 30
    sun = draw_frame(width, height, (450, 300), sun_r)
    sun[10, 100] = sun[40, 700] = 255
    write_jpg(tmp_path / 'sun.jpg', sun)
    frames = {
        'f0.jpg': draw_frame(width, height, (200, 150), sun_r, (25, 10), 29),
        'f1.jpg': draw_frame(width, height, (650, 420), sun_r, (-20, 15), 29),
    }
    jpg_dir = make_input(tmp_path, frames)

    result = run_cli('--input', jpg_dir, '--sun', tmp_path / 'sun.jpg', '--workers', 1, cwd=tmp_path)
    assert result.returncode == 0
    assert f'Detected SUN RADIUS: {sun_r}' in result.stdout
    circles = read_circles(tmp_path)
    assert circles['f0.jpg']['sun'][:2] == [200, 150]
    assert circles['f1.jpg']['sun'][:2] == [650, 420]
    for filename in frames:
        assert_sun_centered(read_jpg(tmp_path / 'jpg-output' / filename), sun_r)


def test_undetectable_sun_gives_guidance(tmp_path):
    write_jpg(tmp_path / 'dark.jpg', np.zeros((300, 400, 3), np.uint8))
    jpg_dir = make_input(tmp_path, {'f0.jpg': draw_frame(400, 300, (200, 150), 50)})
    result = run_cli('--input', jpg_dir, '--sun', tmp_path / 'dark.jpg', cwd=tmp_path)
    assert result.returncode == 1
    assert 'Unable to detect the sun' in result.stdout
    assert '--sun_threshold' in result.stdout and '--sun_radius' in result.stdout


def test_too_small_sun_radius_is_an_error(tmp_path):
    jpg_dir = make_input(tmp_path, {'f0.jpg': draw_frame(400, 300, (200, 150), 50)})
    result = run_cli('--input', jpg_dir, '--sun_radius', 2, cwd=tmp_path)
    assert result.returncode == 1
    assert '--sun_radius must be at least 5 pixels' in result.stdout


def test_filter_moves_clipped_and_empty_images_without_hanging(tmp_path):
    # workers sent no result for filtered images and the main process waited forever (#6)
    width, height, sun_r = 600, 400, 60
    frames = {
        'f0.jpg': draw_frame(width, height, (300, 200), sun_r),
        'f1.jpg': draw_frame(width, height, (30, 200), sun_r),  # clipped at the left edge
        'f2.jpg': np.zeros((height, width, 3), np.uint8),  # no sun
        'f3.jpg': draw_frame(width, height, (350, 180), sun_r, (40, 0), 58),
    }
    jpg_dir = make_input(tmp_path, frames)

    result = run_cli('--filter', '--input', jpg_dir, '--sun_radius', sun_r, '--workers', 1, cwd=tmp_path, timeout=60)
    assert result.returncode == 0
    assert 'Moved f1.jpg with CLIPPED SUN' in result.stdout + result.stderr
    assert 'Moved f2.jpg with NO SUN' in result.stdout + result.stderr
    assert sorted(p.name for p in (tmp_path / 'jpg-clipped').iterdir()) == ['f1.jpg', 'f2.jpg']
    assert sorted(p.name for p in jpg_dir.iterdir()) == ['f0.jpg', 'f3.jpg']
    assert sorted(p.name for p in (tmp_path / 'jpg-output').iterdir()) == ['f0.jpg', 'f3.jpg']
    assert sorted(read_circles(tmp_path)) == ['f0.jpg', 'f3.jpg']


def test_worker_errors_are_reported(tmp_path):
    # errors in a worker are reported with exit status 1 instead of hanging (#6)
    jpg_dir = make_input(tmp_path, {'f0.jpg': draw_frame(400, 300, (200, 150), 50)})
    (jpg_dir / 'z_broken.jpg').write_bytes(b'not a jpg')
    result = run_cli('--input', jpg_dir, '--sun_radius', 50, '--workers', 2, cwd=tmp_path, timeout=60)
    assert result.returncode == 1
    assert 'Failed to process' in result.stdout and 'z_broken.jpg' in result.stdout
    assert 'Unable to read the image' in result.stdout


@pytest.mark.skipif(sys.platform == 'win32', reason='sends SIGINT to the process group like Ctrl+C in a terminal')
def test_ctrl_c_exits_without_hanging(tmp_path):
    # the input paths not yet read by the stopped workers filled the pipe, and the main process waited forever on exit
    # for them to be sent
    jpg_dir = tmp_path / ('j' * 200)
    jpg_dir.mkdir()
    write_jpg(jpg_dir / 'f0.jpg', draw_frame(1600, 1200, (800, 600), 150))
    # long paths, more than the 64 KiB that a pipe holds
    for i in range(600):
        os.link(jpg_dir / 'f0.jpg', jpg_dir / f'f{i:04d}{"x" * 40}.jpg')
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), '--input', str(jpg_dir), '--sun_radius', '150', '--workers', '2'],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        for line in process.stdout:
            if 'Starting 2 workers' in line:
                break
        time.sleep(2)
        os.killpg(process.pid, signal.SIGINT)
        output, _ = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f'Still running 60 seconds after Ctrl+C:\n{output}')
    print(output)
    assert process.returncode == 130
    assert 'Interrupted, stopping the workers' in output


@pytest.mark.parametrize(
    'example, args, expected',
    [
        ('example0', ['--sun', 'example0/ring.jpg'], {'ring.jpg': [240, 180, 127]}),
        (
            'example1', ['--sun_radius', 110], {
                'ex1.jpg': [238, 153, 110],
                'ex2.jpg': [251, 147, 110],
                'ex3.jpg': [276, 132, 110],
                'ex4.jpg': [284, 140, 110],
                'ex5.jpg': [302, 144, 110],
                'ex6.jpg': [189, 131, 110]
            }
        ),
        (
            'example2', ['--sun_radius', 141, '--sun_threshold', 75], {
                'DSC05686.jpg': [713, 1099, 141],
                'DSC05688.jpg': [1382, 1166, 141],
                'DSC05690.jpg': [1569, 1360, 141],
                'DSC05702.jpg': [477, 1539, 141],
                'DSC05703.jpg': [544, 1549, 141],
                'DSC05704.jpg': [653, 919, 141]
            }
        ),
    ],
)
def test_examples(tmp_path, example, args, expected):
    # the expected coordinates are from the README of #5
    shutil.copytree(EXAMPLES / example, tmp_path / example)
    result = run_cli('--input', example, *args, '--workers', 2, cwd=tmp_path)
    assert result.returncode == 0
    assert {name: data['sun'] for name, data in read_circles(tmp_path).items()} == expected
    assert sorted(p.name for p in (tmp_path / f'{example}-output').iterdir()) == sorted(expected)
