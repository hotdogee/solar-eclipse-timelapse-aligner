'''Unit tests for the detection helpers, using the script imported as a module.'''
import multiprocessing
import queue
import sys

import cv2
import numpy as np
import pytest

from conftest import EXAMPLES, draw_frame, read_jpg


def first_component_radius(img, sun_threshold):
    '''The sun radius detection used before #4 was fixed: the first connected component instead of the largest.'''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, sun_threshold, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(img_binary)
    return int(max(stats[1][2:4]) // 2)


@pytest.mark.parametrize('width, height, sun_r', [(600, 400, 80), (900, 600, 30), (3000, 2000, 142)])
def test_detect_sun_radius_ignores_noise_above_the_sun(aligner, width, height, sun_r):
    img = draw_frame(width, height, (width // 2, height // 2), sun_r)
    # hot pixels found before the sun in scan order
    img[5, 10] = img[20, width - 30] = 255
    assert first_component_radius(img, 25) == 0  # the #4 bug
    assert aligner.detect_sun_radius(img, 25) == sun_r


def test_detect_sun_radius_ignores_border_artifacts(aligner):
    img = draw_frame(416, 312, (240, 150), 110)
    # a faint frame around the image, like the example1 crops
    cv2.rectangle(img, (0, 0), (415, 311), (60, 60, 60), 1)
    assert first_component_radius(img, 25) == 208
    assert aligner.detect_sun_radius(img, 25) == 110


def test_detect_sun_radius_slightly_eclipsed_sun(aligner):
    img = draw_frame(800, 600, (400, 300), 150, moon_offset=(200, 0), moon_r=148)
    assert aligner.detect_sun_radius(img, 25) == 150


def test_detect_sun_radius_without_sun(aligner):
    assert aligner.detect_sun_radius(np.zeros((100, 120, 3), np.uint8), 25) == 0


@pytest.mark.parametrize(
    'path, sun_threshold, expected',
    [
        ('example0/ring.jpg', 25, 127),
        ('example1/ex1.jpg', 25, 110),
        ('example1/ex5.jpg', 25, 110),
        # 1872x2648 photo with noise above the sun, detected as 0 before #4 was fixed
        ('example2/DSC05686.jpg', 25, 144),
        ('example2/DSC05686.jpg', 75, 141),
    ],
)
def test_detect_sun_radius_examples(aligner, path, sun_threshold, expected):
    img = read_jpg(EXAMPLES / path)
    assert aligner.detect_sun_radius(img, sun_threshold) == expected


@pytest.mark.parametrize('sun_r', [5, 10, 49, 50, 99, 100, 142, 1210])
def test_make_sun_mask_ring_width_scales_with_radius(aligner, sun_r):
    # radius - 50 was negative for radius < 50 and made cv2.circle assert (#4)
    mask = aligner.make_sun_mask(sun_r)
    assert mask.shape == (sun_r * 2 + 1, sun_r * 2 + 1)
    ring_width = int(mask[sun_r, sun_r:].sum())
    assert ring_width == min(50, sun_r // 2)


def test_move_to_center_crops_and_pads(aligner):
    img = np.arange(1, 13, dtype=np.uint8).reshape(3, 4)
    # crop: the 3x3 neighbourhood of (1, 1)
    assert aligner.move_to_center(img, 1, 1, 3, 3).tolist() == [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
    # pad: (3, 2) is the bottom right pixel
    assert aligner.move_to_center(img, 3, 2, 3, 3).tolist() == [[7, 8, 0], [11, 12, 0], [0, 0, 0]]


class FakeProcess:

    def __init__(self, exitcode=None):
        self.exitcode = exitcode


def test_collect_results_yields_all_results(aligner):
    q = queue.Queue()
    q.put(('ok', 'a.jpg', {'sun': (1, 2, 3)}))
    q.put(('skipped', 'b.jpg', 'NO SUN'))
    results = list(aligner.collect_results(q, [FakeProcess()], 2, poll_interval=0.01))
    assert [r[0] for r in results] == ['ok', 'skipped']


def test_collect_results_raises_worker_errors(aligner):
    q = queue.Queue()
    q.put(('error', 'bad.jpg', 'Traceback (most recent call last):\nValueError: boom'))
    with pytest.raises(aligner.WorkerError, match='bad.jpg') as e:
        list(aligner.collect_results(q, [FakeProcess()], 1, poll_interval=0.01))
    assert 'ValueError: boom' in str(e.value)


def test_collect_results_raises_when_workers_exit_early(aligner):
    q = queue.Queue()
    q.put(('ok', 'a.jpg', {'sun': (1, 2, 3)}))
    with pytest.raises(aligner.WorkerError, match=r'1 of 2 images were not processed .*exit codes: 0'):
        list(aligner.collect_results(q, [FakeProcess(exitcode=0)], 2, poll_interval=0.01))


def test_collect_results_raises_when_a_real_worker_crashes(aligner):
    # a worker killed while processing an image (e.g. out of memory) used to hang the main process (#6)
    ctx = multiprocessing.get_context('spawn')
    output_queue = ctx.Queue()
    crashed = ctx.Process(target=sys.exit, args=(3,))
    crashed.start()
    crashed.join(timeout=60)
    with pytest.raises(aligner.WorkerError, match='exit codes: 3'):
        list(aligner.collect_results(output_queue, [crashed], 1, poll_interval=0.1))
