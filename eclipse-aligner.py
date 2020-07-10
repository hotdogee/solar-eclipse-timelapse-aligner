import argparse
import errno
import json
import os
import shutil
import psutil
import sys
import time
from multiprocessing import Manager, Process, Queue, Value, freeze_support
from pathlib import Path

import cv2
import click
import numpy as np
from scipy import signal, spatial
from tqdm import tqdm


def find_sun_and_moon_worker(input_queue, output_queue, params):
    try:
        for jpg_path in iter(input_queue.get, 'STOP'):
            # put the sun in the center of an image with enough border for moon detection by conv2d
            # we'll need a square image with sides of
            # (sun_radius + moon_radius * 2) * 2 = (1210 + 1191 * 2) * 2 = 7184
            # read jpg
            img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
            # (3456, 4608, 3)
            # remove images with white values on border (CLIPPED SUN)
            if params['filter']:
                if img[:, (0, -1), 1].max() > params['clipped_cutoff'] or img[(0, -1), :,
                                                                              1].max() > params['clipped_cutoff']:
                    # print(f'Moving {jpg_path.name} with CLIPPED SUN ({clipped_count})')
                    shutil.move(jpg_path, jpg_clipped_path / jpg_path.name)
                    continue
                # remove images with all black values (NO SUN)
                if img.max() < params['empty_cutoff']:
                    # print(f'Moving {jpg_path.name} with NO SUN ({empty_count})')
                    shutil.move(jpg_path, jpg_clipped_path / jpg_path.name)
                    continue
            ####################
            # find center of sun
            ####################
            # grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # otsu
            # _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, img_binary = cv2.threshold(img_gray, params['sun_threshold'], 255, cv2.THRESH_BINARY)
            # save sun_binary
            cv2.imwrite(str(params['jpg_sun_binary_path'] / jpg_path.name), img_binary)
            # fft convolve
            sun_signal = signal.fftconvolve(img_binary.astype('float') / 255, params['sun_mask'], mode='same')
            # find coordinates of the largest value in sun_signal
            y, x = np.unravel_index(np.argmax(sun_signal, axis=None), sun_signal.shape)
            sun_x, sun_y = int(x), int(y)
            # draw sun circle on circled
            circled = img.copy()
            a = cv2.circle(circled, (sun_x, sun_y), params['sun_mask_r'], (0, 0, 255), 4)
            a = cv2.rectangle(circled, (sun_x - 5, sun_y - 5), (sun_x + 5, sun_y + 5), (0, 128, 255), -1)
            #####################
            # find center of moon
            #####################
            if params['fix_angle'] != -1:
                img_h, img_w = img.shape[:2]
                # pad image to canvas_size
                pad_x, pad_y = (params['canvas_size'] - img_w) // 2, (params['canvas_size'] - img_h) // 2
                padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))
                # calculate the distance between the center of the sun and the center of the image
                center_x, center_y = img_w // 2, img_h // 2
                canvas = np.zeros((params['canvas_size'], params['canvas_size'], img.shape[2]), dtype=np.uint8)
                dx, dy = center_x - sun_x, center_y - sun_y
                # draw sun at the center of the canvas
                left, right = (0, dx) if dx > 0 else (-dx, 0)
                up, down = (0, dy) if dy > 0 else (-dy, 0)
                pad_h, pad_w = padded.shape[:2]
                canvas[down:(pad_h - up), right:(pad_w - left), :] = padded[up:(pad_h - down), left:(pad_w - right), :]
                # locate moon
                # grayscale
                canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                # otsu
                th, _ = cv2.threshold(canvas_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, canvas_binary = cv2.threshold(canvas_gray, th + params['moon_threshold_mod'], 255, cv2.THRESH_BINARY)
                # save moon_binary
                _, moon_binary = cv2.threshold(img_gray, th + params['moon_threshold_mod'], 255, cv2.THRESH_BINARY)
                cv2.imwrite(str(params['jpg_moon_binary_path'] / jpg_path.name), moon_binary)
                # fft convolve
                moon_signal = signal.fftconvolve(canvas_binary.astype('float') / 255, params['moon_mask'], mode='valid')
                # find coordinates of the smallest values in moon_signal that are also inside the moon_signal_mask
                rows, cols = np.where((moon_signal < 1) * (params['moon_signal_mask'] > 0))
                # find the coordinate closest to the center of the sun (closest to the center of the moon_signal image)
                moon_signal_center = moon_signal.shape[0] // 2
                moon_x, moon_y, min_dist = 0, 0, 99999
                for i in range(rows.shape[0]):
                    dist = spatial.distance.euclidean([cols[i], rows[i]], [moon_signal_center, moon_signal_center])
                    if dist < min_dist:
                        moon_x, moon_y, min_dist = int(cols[i]), int(rows[i]), float(dist)
                # calculate moon coordinates in the original image
                # moon_x is relative to moon_signal (4802, 4802)
                original_moon_x = moon_x - (params['moon_signal_mask_size'] - img_w) // 2 - dx
                original_moon_y = moon_y - (params['moon_signal_mask_size'] - img_h) // 2 - dy
                # DSCN0437.jpg: 3010, 4375, 1288, -77 -  4624, 2914, 1864, 154
                # tqdm.write(
                #     f'{jpg_path.name}: {original_moon_x}, {moon_x}, {pad_x}, {dx} - {original_moon_y}, {moon_y}, {pad_y}, {dy}'
                # )
                # draw moon circle on circled
                a = cv2.circle(circled, (original_moon_x, original_moon_y), params['moon_mask_r'], (0, 255, 0), 4)
                a = cv2.rectangle(
                    circled, (original_moon_x - 5, original_moon_y - 5), (original_moon_x + 5, original_moon_y + 5),
                    (0, 128, 255), -1
                )
                # put in queue
                output_queue.put((jpg_path, sun_x, sun_y, moon_x, moon_y, min_dist, moon_signal_center))
            else:
                # put in queue
                output_queue.put((jpg_path, sun_x, sun_y))
            # save circled
            cv2.imwrite(str(params['jpg_circled_path'] / jpg_path.name), circled)
    except Exception as e:
        if params['fix_angle'] != -1:
            output_queue.put((jpg_path, -1, e, -1, -1, -1, -1))
        else:
            output_queue.put((jpg_path, -1, e))


angle = 0
if __name__ == '__main__':
    freeze_support()

    def output_image(jpg_path, circles_data):
        global angle
        # circles_data = {
        #     'moon': (moon_x, moon_y, moon_mask_r, min_dist, moon_signal_center),
        #     'sun': (sun_x, sun_y, sun_mask_r)
        # }
        sun_x, sun_y, sun_mask_r = circles_data['sun']
        # read jpg
        img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
        # (3456, 4608, 3)
        img_h, img_w = img.shape[:2]
        # pad image to canvas_size
        pad_x, pad_y = (canvas_size - img_w) // 2, (canvas_size - img_h) // 2
        padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))
        # calculate the distance between the center of the sun and the center of the image
        center_x, center_y = img_w // 2, img_h // 2
        canvas = np.zeros((canvas_size, canvas_size, img.shape[2]), dtype=np.uint8)
        canvas_h, canvas_w = canvas.shape[:2]
        dx, dy = center_x - sun_x, center_y - sun_y
        # draw sun at the center of the canvas
        left, right = (0, dx) if dx > 0 else (-dx, 0)
        up, down = (0, dy) if dy > 0 else (-dy, 0)
        pad_h, pad_w = padded.shape[:2]
        canvas[down:(pad_h - up), right:(pad_w - left), :] = padded[up:(pad_h - down), left:(pad_w - right), :]
        if args.fix_angle != -1:
            # update angle if min_dist > 30 and min_dist < (sun_mask_r + moon_mask_r - 100)
            moon_x, moon_y, moon_mask_r, min_dist, moon_signal_center = circles_data['moon']
            if min_dist > args.max_phase_rotation_group_threshold and min_dist < (
                sun_mask_r + moon_mask_r - args.min_phase_rotation_group_threshold
            ):
                dx, dy = moon_x - moon_signal_center, moon_y - moon_signal_center
                moon_degree = np.arctan2(dx, dy) * 180 / np.pi
                if moon_degree > 0:
                    angle = args.fix_angle - moon_degree
                else:
                    angle = args.fix_angle - 180 - moon_degree
            # tqdm.write(f'{jpg_path.name} angle: {angle}')
            # rotate canvas
            center = (canvas_w // 2, canvas_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            canvas = cv2.warpAffine(canvas, M, (canvas_w, canvas_h))
        # crop to original size
        crop_x, crop_y = (canvas_w - crop_w) // 2, (canvas_h - crop_h) // 2
        cropped = canvas[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        # save cropped
        cv2.imwrite(str(jpg_output_path / jpg_path.name), cropped)

    def verify_input_path(p):
        # get absolute path to dataset directory
        path = Path(os.path.abspath(os.path.expanduser(p)))
        # doesn't exist
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        # is dir
        if path.is_dir():
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
        return path

    def verify_indir_path(p):
        # get absolute path
        path = Path(os.path.abspath(os.path.expanduser(p)))
        # doesn't exist
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        # existing file
        if not path.is_dir():
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), path)
        else:  # got existing directory
            assert len([x for x in path.iterdir()]) != 0, 'Directory is empty'
        return path

    parser = argparse.ArgumentParser(
        description='This specialized program is used to align and stabilize solar eclipse time-lapse photos',
        prog='eclipse-aligner'
    )
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v1.0')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input directory, will only read JPGs, develop you DNGs into JPGs first.'
    )
    parser.add_argument(
        '--sun',
        type=str,
        default='',
        help=
        'Path to an image with an un-eclipsed sun, used to auto detect the radius of the sun in pixels, if you do not have an image with an un-eclipsed sun, you can try using an image with the least partially eclipsed sun.'
    )
    parser.add_argument(
        '--sun_radius',
        type=int,
        default=-1,
        help='Radius of the sun in pixels, if not given, auto detect from full sun image. (default: %(default)s)'
    )
    parser.add_argument(
        '--sun_threshold',
        type=int,
        default=25,
        help=
        'Brightness threshold used to convert the image to black and white, choose a value that gives clean edges for best results. (default: %(default)s)'
    )
    parser.add_argument(
        '--fix_angle',
        type=int,
        default=-1,
        help=
        'Enables moon angle stabilization if given a non-negative value in degrees, do not enable if processing partial solar eclipse images. (default: %(default)s)'
    )
    parser.add_argument(
        '--moon_radius_mod',
        type=int,
        default=-24,
        help=
        '(MOON_RADIUS - SUN_RADIUS) in pixels, use a negative value if the moon is smaller than the sun (as is the case of annular solar eclipse). (default: %(default)s)'
    )
    parser.add_argument(
        '--moon_threshold_mod',
        type=int,
        default=-1,
        help=
        'Use a negative value to use a smaller threshold, or a positive value to use a larger threshold, choose a value that gives clean moon edges. TECHNICAL INFO: this value is added the dynamic Otsu threshold. (default: %(default)s)'
    )
    parser.add_argument(
        '--max_phase_rotation_group_threshold',
        type=int,
        default=30,
        metavar='THRESHOLD',
        help=
        'To solve the problem that the angle of moon is ambiguous at totality or annularity, we group the images where the distance of the sun and moon is smaller than the given value in pixels, and assume that the require rotation if the same for the whole group, and the required rotation of the image with a distance just greater than the given value is used for the whole group. (default: %(default)s)'
    )
    parser.add_argument(
        '--min_phase_rotation_group_threshold',
        type=int,
        default=100,
        metavar='THRESHOLD',
        help=
        'Moon detection for images near the start or end of the eclipse may be hard to get right, we group the images where the shadowed width is less than the given value in pixels, and assume that the require rotation if the same for the images in the same group. (default: %(default)s)'
    )
    parser.add_argument('--filter', action='store_true', help='Enable the clipped sun filter. (default: %(default)s)')
    parser.add_argument(
        '--clipped_cutoff',
        type=int,
        default=40,
        help=
        'The clipped sun filter will filter out images if any edge pixel\'s brightness exceeds this value. (default: %(default)s)'
    )
    parser.add_argument(
        '--empty_cutoff',
        type=int,
        default=80,
        help=
        'The clipped sun filter will filter out images if the brightest pixel in the image is lower than this value, in other words, there is nothing in the image. (default: %(default)s)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=-1,
        help=
        'Number of processing workers, each workers requires 3.5GB of RAM for 16M pixel images, the larger the image the more ram is required. (default: %(default)s)'
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='-output',
        help=
        'The stabilized results will be save to a directory with the same name as the input directory appended with the given suffix. (default: %(default)s)'
    )
    parser.add_argument(
        '--clipped_suffix',
        type=str,
        default='-clipped',
        help=
        'The clipped sun filer will move images to a directory with the same name as the input directory appended with the given suffix. (default: %(default)s)'
    )
    parser.add_argument(
        '--circled_suffix',
        type=str,
        default='-circled',
        help=
        'Images with the sun circled in red for validation will be saved to a directory with the same name as the input directory appended with the given suffix. (default: %(default)s)'
    )
    parser.add_argument(
        '--sun_binary_suffix',
        type=str,
        default='-sun-binary',
        help=
        'Black and white images using the sun threshold will be saved to a directory with the same name as the input directory appended with the given suffix. Used for finetuning the sun threshold value. (default: %(default)s)'
    )
    parser.add_argument(
        '--moon_binary_suffix',
        type=str,
        default='-moon-binary',
        help=
        'Black and white images using the moon threshold mod will be saved to a directory with the same name as the input directory appended with the given suffix. Used for finetuning the moon threshold mod value. (default: %(default)s)'
    )
    parser.add_argument(
        '--circles',
        type=str,
        default='',
        help=
        'If not given, by default, the program will save the detected sun and moon location data in JSON format to a file named circles_data.json located in the same directory as the input directory, and will overwrite the file if exists. If the detected location are good, you can choose to move the file to another location for reuse if you decide to re-develop the DNGs with better settings. If given the path to the existing good circles_data.json, the program will read the sun and moon location data from the JSON file and perform the centering and angle stabilization using the data in the JSON file. (default: circles_data.json)'
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify input directory
    JPG_GLOB = '*.[jJ][pP][gG]'
    try:
        jpg_dir_path = verify_indir_path(args.input)
    except FileNotFoundError:
        click.secho('ERROR: --input does not exist, please provide directory containing JPG files.', fg='red')
        sys.exit(1)
    except NotADirectoryError:
        click.secho('ERROR: --input is a file, please provide directory containing JPG files.', fg='red')
        sys.exit(1)

    input_paths = sorted(jpg_dir_path.glob(JPG_GLOB))
    num_images = len(input_paths)
    if num_images == 0:
        click.secho(
            'ERROR: --input does not have any JPG files, please provide directory containing JPG files.', fg='red'
        )
        sys.exit(1)
    click.secho(f'INFO: {num_images} images found in {jpg_dir_path}')

    # get image size from first image
    img = cv2.imread(str(input_paths[0]), cv2.IMREAD_COLOR)
    if img is None:
        click.secho(f'ERROR: Unable to read first image in --input: {str(input_paths[0])}', fg='red')
        sys.exit(1)
    img_h, img_w = img.shape[:2]
    img_size = img_h * img_w
    crop_w, crop_h = img_w, img_h
    click.secho(f'INFO: Detected image size (height, width): {img.shape[:2]}')

    # set sun radius
    if args.sun_radius != -1:
        # use given sun radius
        sun_mask_r = args.sun_radius
        click.secho(f'INFO: Using given SUN RADIUS: {sun_mask_r}')
    elif args.sun != '':
        # detect sun radius
        try:
            sun_path = verify_input_path(args.sun)
        except FileNotFoundError:
            click.secho('ERROR: --sun does not exist, please provide a JPG file.', fg='red')
            sys.exit(1)
        except IsADirectoryError:
            click.secho('ERROR: --sun is a directory, please provide a JPG file.', fg='red')
            sys.exit(1)
        sun = cv2.imread(str(sun_path), cv2.IMREAD_COLOR)
        sun_gray = cv2.cvtColor(sun, cv2.COLOR_BGR2GRAY)
        _, sun_binary = cv2.threshold(sun_gray, args.sun_threshold, 255, cv2.THRESH_BINARY)
        _, _, stats, _ = cv2.connectedComponentsWithStats(sun_binary)
        sun_mask_r = int(max(stats[1][2:4]) // 2)
        click.secho(f'INFO: Detected SUN RADIUS: {sun_mask_r}')
    else:
        # print error message
        click.secho(f'ERROR: Please provide either --sun or --sun_radius', fg='red')
        sys.exit(1)

    # moon_mask_r = 1187
    moon_mask_r = sun_mask_r + args.moon_radius_mod
    if args.fix_angle != -1:
        click.secho(f'INFO: Using MOON RADIUS: {moon_mask_r} (sun_radius + moon_radius_mod)')
    # 7184
    canvas_size = (sun_mask_r + moon_mask_r * 2) * 2
    # 4802
    moon_signal_mask_size = canvas_size - moon_mask_r * 2
    moon_signal_mask_r = moon_signal_mask_size // 2

    # create masks
    sun_mask = np.zeros((sun_mask_r * 2 + 1, sun_mask_r * 2 + 1), np.float32)
    _ = cv2.circle(sun_mask, (sun_mask_r, sun_mask_r), sun_mask_r, 1.0, -1)
    _ = cv2.circle(sun_mask, (sun_mask_r, sun_mask_r), sun_mask_r - 50, 0.0, -1)
    moon_mask = np.zeros((moon_mask_r * 2 + 1, moon_mask_r * 2 + 1), np.float32)
    _ = cv2.circle(moon_mask, (moon_mask_r, moon_mask_r), moon_mask_r, 1.0, -1)
    moon_signal_mask = np.zeros((moon_signal_mask_size, moon_signal_mask_size), np.float32)
    _ = cv2.circle(moon_signal_mask, (moon_signal_mask_r, moon_signal_mask_r), moon_signal_mask_r, 1.0, -1)

    # setup dirs
    if args.filter:
        jpg_clipped_path = jpg_dir_path.parent / (jpg_dir_path.name + args.clipped_suffix)
        jpg_clipped_path.mkdir(parents=True, exist_ok=True)
    jpg_sun_binary_path = jpg_dir_path.parent / (jpg_dir_path.name + args.sun_binary_suffix)
    jpg_sun_binary_path.mkdir(parents=True, exist_ok=True)
    jpg_moon_binary_path = jpg_dir_path.parent / (jpg_dir_path.name + args.moon_binary_suffix)
    if args.fix_angle != -1:
        jpg_moon_binary_path.mkdir(parents=True, exist_ok=True)
    jpg_circled_path = jpg_dir_path.parent / (jpg_dir_path.name + args.circled_suffix)
    jpg_circled_path.mkdir(parents=True, exist_ok=True)
    jpg_output_path = jpg_dir_path.parent / (jpg_dir_path.name + args.output_suffix)
    jpg_output_path.mkdir(parents=True, exist_ok=True)

    # read circles if given
    if args.circles != '':
        # get absolute path
        circles_data_json_path = Path(os.path.abspath(os.path.expanduser(args.circles)))
        if circles_data_json_path.exists():
            # error out if given a directory
            if circles_data_json_path.is_dir():
                click.secho('ERROR: --circles is a directory, please provide a JSON file.', fg='red')
                sys.exit(0)
                # raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
            with circles_data_json_path.open('r') as f:
                circles_data = json.load(f)
            # check if JSON file has data for all JPGs in INPUT directory
            for jpg_path in input_paths:
                if jpg_path.name not in circles_data or 'sun' not in circles_data[jpg_path.name]:
                    click.secho(
                        f'ERROR: The sun coordinates data for the JPG image "{jpg_path.name}" was not found in --circles "{str(circles_data_json_path)}". Remove the --circles argument to process the image, or provide the correct --circles JSON file that contains the sun coordinates data for the JPG image "{jpg_path.name}"',
                        fg='red'
                    )
                    sys.exit(0)
                if args.fix_angle != -1 and 'moon' not in circles_data[jpg_path.name]:
                    click.secho(
                        f'ERROR: The moon coordinates data for the JPG image "{jpg_path.name}" was not found in --circles "{str(circles_data_json_path)}". Remove the --circles argument to process the image, or provide the correct --circles JSON file that contains the moon coordinates data for the JPG image "{jpg_path.name}"',
                        fg='red'
                    )
                    sys.exit(0)
            # looks good
            click.secho(f'INFO: Using existing coordinates data: "{args.circles}".')
        else:
            # doesn't exist, prompt user
            ok = click.confirm(
                click.style(
                    f'ATTENTION: --circles file does not exist. Continue to run the detection algorithm and save the results to: \'{str(circles_data_json_path)}\'?',
                    fg='yellow'
                ),
                default=True,
                abort=False
            )
            if ok:
                # initialize circles_data
                circles_data = {}
            else:
                # user does not want to continue, exit
                # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
                sys.exit(0)
    else:
        circles_data_json_path = jpg_dir_path.parent / 'circles_data.json'
        circles_data = {}

    if not circles_data:
        # get number of workers
        workers = args.workers
        if args.workers < 1:
            # detect cpu and memory
            # 32 cpu = 98 GB - 26GB
            estimated_memory_per_worker = 3000000000 / (3456 * 4608) * img_size
            ram_workers = psutil.virtual_memory().available // estimated_memory_per_worker
            cpu_workers = psutil.cpu_count(logical=False)
            workers = int(max(1, min(ram_workers, cpu_workers)))
        # don't need more workers than input images
        if workers > num_images:
            workers = num_images
        # log workers
        if workers == 1:
            click.secho('INFO: Starting 1 worker.')
        else:
            click.secho(f'INFO: Starting {workers} workers.')

        # setup params
        params = {
            'filter': args.filter,
            'clipped_cutoff': args.clipped_cutoff,
            'empty_cutoff': args.empty_cutoff,
            'sun_threshold': args.sun_threshold,
            'sun_mask': sun_mask,
            'sun_mask_r': sun_mask_r,
            'fix_angle': args.fix_angle,
            'canvas_size': canvas_size,
            'moon_threshold_mod': args.moon_threshold_mod,
            'moon_mask': moon_mask,
            'moon_signal_mask': moon_signal_mask,
            'moon_signal_mask_size': moon_signal_mask_size,
            'moon_mask_r': moon_mask_r,
            'jpg_circled_path': jpg_circled_path,
            'jpg_sun_binary_path': jpg_sun_binary_path,
            'jpg_moon_binary_path': jpg_moon_binary_path,
        }

        # create queues
        input_queue = Queue()
        output_queue = Queue()

        # submit tasks
        for jpg_path in input_paths:
            input_queue.put(jpg_path)

        # input process
        input_processes = []
        for i in range(workers):
            ip = Process(target=find_sun_and_moon_worker, args=(input_queue, output_queue, params))
            ip.start()
            input_processes.append(ip)
            input_queue.put('STOP')

        # save circles data
        for _ in tqdm(input_paths, desc='Processing', ascii=True, unit='img', dynamic_ncols=True, smoothing=0.1):
            if args.fix_angle != -1:
                jpg_path, sun_x, sun_y, moon_x, moon_y, min_dist, moon_signal_center = output_queue.get()
            else:
                jpg_path, sun_x, sun_y = output_queue.get()
            # check for errors
            if sun_x == -1:
                click.secho(f'\nERROR: {jpg_path}: {sun_y}.', fg='red')
                for p in input_processes:
                    p.kill()
                sys.exit(0)
            if args.fix_angle != -1:
                circles_data[jpg_path.name] = {
                    'moon': (moon_x, moon_y, moon_mask_r, min_dist, moon_signal_center),
                    'sun': (sun_x, sun_y, sun_mask_r)
                }
            else:
                circles_data[jpg_path.name] = {'sun': (sun_x, sun_y, sun_mask_r)}

        circles_data_json_path.write_text(json.dumps(circles_data))

    # find the first valid angle
    if args.fix_angle != -1:
        for jpg_path in input_paths:
            moon_x, moon_y, moon_mask_r, min_dist, moon_signal_center = circles_data[jpg_path.name]['moon']
            if min_dist < (sun_mask_r + moon_mask_r - args.min_phase_rotation_group_threshold):
                dx, dy = moon_x - moon_signal_center, moon_y - moon_signal_center
                moon_degree = np.arctan2(dx, dy) * 180 / np.pi
                if moon_degree > 0:
                    angle = args.fix_angle - moon_degree
                else:
                    angle = args.fix_angle - 180 - moon_degree
                break

    # compute output image and update circles_data
    for jpg_path in tqdm(input_paths, desc='    Saving', ascii=True, unit='img', dynamic_ncols=True, smoothing=0.1):
        output_image(jpg_path, circles_data[jpg_path.name])

    # exit
    sys.exit(0)
