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
from scipy import signal, spatial, stats
from tqdm import tqdm

import math
import doctest
import exifread
from datetime import datetime

#### constants
# colours
RED = (0, 0, 255)
ORANGE = (0, 128, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)


##############################################################
######### STAGE 1: input some calibration photos
######### figure out threshold and radius for sun
##############################################################

def get_sun_radius(sun_path, sun_threshold):
    '''(str) -> int
    Given filename path, return radius of sun using Han Lin's original method.
    >>> get_sun_radius('example0/ring.jpg', 25)
    127
    >>> get_sun_radius('example1/ex1.jpg', 25)
    110
    >>> get_sun_radius('example1/ex3.jpg', 55)
    110
    >>> get_sun_radius('example2/DSC05686.jpg', 60)
    141
    >>> get_sun_radius('example2/DSC05686.jpg', 25)
    0
    '''
    sun = cv2.imread(sun_path, cv2.IMREAD_COLOR)
    sun_gray = cv2.cvtColor(sun, cv2.COLOR_BGR2GRAY)
    _, sun_binary = cv2.threshold(sun_gray, sun_threshold, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(sun_binary)
    sun_mask_r = int(max(stats[1][2:4]) // 2)
    
    return sun_mask_r


def get_centre_of_sun(fname, sun_threshold, sun_radius):
    '''str, dict, CV2 image -> int, int
    Find the centre of the sun in the image img.
    Return: x and y of the centre
    >>> get_centre_of_sun('example0/ring.jpg', 25, 127)
    (240, 180)
    >>> get_centre_of_sun('example1/ex1.jpg', 25, 110)
    (238, 153)
    >>> get_centre_of_sun('example1/ex3.jpg', 55, 110)
    (276, 132)
    >>> get_centre_of_sun('example2/DSC05686.jpg', 60, 141)
    (713, 1097)
    >>> get_centre_of_sun('example2/DSC05686.jpg', 25, 0)    
    Error: invalid radius for sun. Sun radius must be greater than 50 pixels.
    '''
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    
    # STAGE 1: FIND CENTRE OF SUN
    # grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # otsu
    # _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_binary = cv2.threshold(img_gray, sun_threshold, 255, cv2.THRESH_BINARY)
    # save sun_binary
    cv2.imwrite(fname.replace('/', '-sun-binary/'), img_binary)
    #print(np.sum(img_binary))

    sun_mask_r = sun_radius # 141 for my photos, 110 for OG's # sun radius
    sun_mask = np.zeros((sun_mask_r * 2 + 1, sun_mask_r * 2 + 1), np.float32) # making zeros on a canvas. sum: 0
    sun_mask = cv2.circle(sun_mask, (sun_mask_r, sun_mask_r), sun_mask_r, 1.0, -1) # this runs: img, centre, radius, colour, thickness
    #print(sun_mask)
    try:
        sun_mask = cv2.circle(sun_mask, (sun_mask_r, sun_mask_r), sun_mask_r - 50, 0.0, -1) # THIS is the source of the assertion error

        # fft convolve
        sun_signal = signal.fftconvolve(img_binary.astype('float') / 255, sun_mask, mode='same')
        # find coordinates of the largest value in sun_signal
        y, x = np.unravel_index(np.argmax(sun_signal, axis=None), sun_signal.shape)
        sun_x, sun_y = int(x), int(y) # these are the centre of the sun's circle
        
        return sun_x, sun_y
    except:
        if sun_radius < 50:
            print('Error: invalid radius for sun. Sun radius must be greater than 50 pixels.')


def output_encircled(img, path, sun_radius, sun_x, sun_y, original_moon_x = -1, original_moon_y = -1, moon_radius = -1, moon_colour = GREEN, dirname = 'circled'):
    '''
    (CV2, str, int, int, int) -> None
    >>> img = cv2.imread('example0/ring.jpg', cv2.IMREAD_COLOR)
    >>> output_encircled(img, 'example0/ring.jpg', 126, 0, 0, dirname='stage1')
    Drawing to example0-stage1/ring.jpg;  radius: 126; centre: (0,0)
    >>> output_encircled(img, 'example0/ring.jpg', 100, 50, 200, dirname='stage1')
    Drawing to example0-stage1/ring.jpg;  radius: 100; centre: (50,200)
    '''
    
    # STAGE 3: output circles of sun and moon
    # draw sun circle on circled
    circled = img.copy()
    a = cv2.circle(circled, (sun_x, sun_y), sun_radius, RED, 4)
    a = cv2.rectangle(circled, (sun_x - 5, sun_y - 5), (sun_x + 5, sun_y + 5), ORANGE, -1)

    if original_moon_x != -1:
        a = cv2.circle(circled, (original_moon_x, original_moon_y), moon_radius, moon_colour, 4)
        a = cv2.rectangle(
            circled, (original_moon_x - 5, original_moon_y - 5), (original_moon_x + 5, original_moon_y + 5),
            (0, 128, 255), -1
        )

    write_to_fname = path.replace('/', '-' + dirname + '/')   #params['path'] + params['input_dir'].replace('/', '-' + dirname + ' /') + fname

    # ensure the directory exists where we want to write
    if not os.path.exists(get_directory(write_to_fname)):
        os.makedirs(get_directory(write_to_fname))
    
    print('Drawing to ', write_to_fname, ';  radius: ', sun_radius, '; centre: (', sun_x, ',', sun_y, ')', sep='')
    #print(write_to_fname)
    cv2.imwrite(write_to_fname, circled)


def make_json_filename(sun_path, stage):
    '''(str) -> str
    Create a json filename for get_sun_central_radius.
    >>> make_json_filename('example2/DSC05686.jpg', 'stage1')
    'example2-stage1/DSC05686.json'
    >>> make_json_filename('example2/DSC05688.JPG', 'stage1')
    'example2-stage1/DSC05688.json'
    '''
    return sun_path.replace('/','-' + stage + '/').replace('.jpg', '.json').replace('.JPG', '.json')


def get_sun_central_radius(sun_path, min_threshold = 5, max_threshold = 100, threshold_increment = 5):
    '''(str) -> int
    Given filename path, try a range of thresholds and see what the median radius is.
    >>> get_sun_central_radius('example0/ring.jpg')
    126
    >>> get_sun_central_radius('example1/ex1.jpg')
    109
    >>> get_sun_central_radius('example1/ex3.jpg')
    110
    >>> get_sun_central_radius('example2/DSC05686.jpg')
    141
    >>> get_sun_central_radius('example2/DSC05702.jpg')
    143
    '''
    # to avoid recomputing unneccissarily
    json_filename = make_json_filename(sun_path, 'stage1') 
    data = {}
    if os.path.isfile(json_filename):
        with open(json_filename, 'r') as f:
            data = json.load(f)    
            
    # did we get saved data?
    if data and (str(min_threshold) in data):
        #print('data found')
        radiuses = list(data.values())
    else:
        #print('data not found, output to', json_filename)
        data = {}
        radiuses = []
        for i in range(min_threshold, max_threshold, threshold_increment):
            tentative_radius = get_sun_radius(sun_path, i)
            data[i] = tentative_radius
            radiuses.append(tentative_radius)
        # save results to avoid recomputing
        with open(json_filename, 'w+') as f:
            json.dump(data, f) # note: this converts the keys to strings
        
    r = np.array(radiuses)
    return round(np.median( r[ r > 2 ]))


def get_thresholds_for_radius(sun_path, target_radius, tolerance, min_threshold = 5, max_threshold = 100, threshold_increment = 5):
    '''(str, int, int) -> list
    Given a sun image, return a list of sun thresholds that are within tolerance of the target radius.
    This is used for figuring out a suitable sun threshold in stage 1.
    A value of 2 for tolerance seems to be a generally good value.
    
    >>> get_thresholds_for_radius('example0/ring.jpg', 126, 1)
    [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example0/ring.jpg', 126, 2)
    [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example1/ex1.jpg', 109, 1)
    [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    >>> get_thresholds_for_radius('example1/ex3.jpg', 110, 1)
    [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example1/ex3.jpg', 110, 2)
    [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example2/DSC05686.jpg', 141, 1)
    [40, 45, 50, 55, 60, 65, 70, 75, 80]
    >>> get_thresholds_for_radius('example2/DSC05702.jpg', 143, 1)
    [65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example2/DSC05702.jpg', 143, 5)
    [65, 70, 75, 80, 85, 90, 95]
    '''
    # to avoid recomputing unneccissarily
    json_filename = make_json_filename(sun_path, 'stage1') 
    data = {}
    if os.path.isfile(json_filename):
        with open(json_filename, 'r') as f:
            data = json.load(f)    
   
    thresholds = []         
    if data and (str(min_threshold) in data):
        saved_thresholds = np.array(list(data.keys())).astype(int)
        radiuses = np.array(list(data.values())).astype(int)
        # there's probably a nicer way to do this
        for i, v in enumerate(radiuses):
            if abs(v - target_radius) <= tolerance:
                thresholds.append( saved_thresholds[i] )        
    else:     
        for i in range(min_threshold, max_threshold, threshold_increment):
            tentative_radius = get_sun_radius(sun_path, i)
            if abs(tentative_radius - target_radius) <= tolerance:
                thresholds.append(i)
    return thresholds


def get_common_threshold(sun_list, blended_radius, tolerance = 2):
    '''(list of str, int, int) -> int
    Given the list of images, and their blended radius, pick a threshold that is common to them all.
    >>> get_common_threshold(['example0/ring.jpg'], 126, 2)    
    15
    >>> get_common_threshold(['example0/ring.jpg'], 126, 1)    
    15
    >>> get_common_threshold(['example1/ex1.jpg', 'example1/ex3.jpg'], 110, 2)    
    35
    >>> get_common_threshold(['example2/DSC05686.jpg', 'example2/DSC05702.jpg'], 142, 2)
    65
    >>> get_common_threshold(['example2/DSC05686.jpg', 'example2/DSC05702.jpg'], 142, 1)
    65
    >>> get_common_threshold(['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'], 141, 2)
    70
    >>> get_common_threshold(['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'], 141, 1)
    Error: no common thresholds among the sun list. Rerun with a higher tolerance.
    '''
    common_thresholds = set([])
    for i, image_path in enumerate(sun_list):
        threshold_list = get_thresholds_for_radius(image_path, blended_radius, tolerance)
        if i == 0:
            common_thresholds = set(threshold_list)
        else:
            common_thresholds = common_thresholds.intersection( set(threshold_list) )

    if len(common_thresholds) == 0:
        print('Error: no common thresholds among the sun list. Rerun with a higher tolerance.')
    else:
        return min(common_thresholds)
        

def get_directory(path):
    '''
    str -> str
    >>> get_directory('path/to/a/file.type')
    'path/to/a/'
    '''
    return '/'.join(path.split('/')[:-1]) + '/'
        

def whether_calculate_stage1(data, sun_list):
    '''
    (dict, list) -> bool
    Return false if every element in sun_list is in the data dict.
    >>> whether_calculate_stage1({}, ['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'])
    True
    >>> whether_calculate_stage1({"example2/DSC05686.jpg": {"sun_radius": 141, "sun_x": 713, "sun_y": 1098, "sun_threshold": 70}, "example2/DSC05688.jpg": {"sun_radius": 141, "sun_x": 1382, "sun_y": 1165, "sun_threshold": 70}, "example2/DSC05702.jpg": {"sun_radius": 141, "sun_x": 477, "sun_y": 1538, "sun_threshold": 70}}, ['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'])
    False
    '''
    for sun_path in sun_list:
        if sun_path not in data:    
            return True
    return False
        

def get_s1_json_name(sun_list, tolerance):
    '''Generate a name for the stage 1 JSON file. 
    Unlike other stages, the unit tests need different JSON files
    so this is why it is not simply stage1.json like the other stages.
    
    >>> get_s1_json_name(['t/a', 't/b', 't/c'], 2)
    't-stage1/stage1n3t2.json'
    >>> get_s1_json_name(['example1/a.jpg'], 5)
    'example1-stage1/stage1n1t5.json'
    '''
    stage = 'stage1'
    return get_directory(sun_list[0]).replace('/', '-' + stage + '/') + stage + 'n' + str(len(sun_list)) + 't' + str(tolerance) + '.json'
        


def find_moon_in_partial_eclipse(path):
    pass
    '''
    >>> find_moon_in_partial_eclipse('example1/ex1.jpg')
    
    >>> find_moon_in_partial_eclipse('example2/DSC05686.jpg')
    
    >>> find_moon_in_partial_eclipse('example2/DSC05688.jpg')
    '''
    json_s2 = make_json_filename(get_directory(path), 'stage2') + 'stage2.json'
    data = open_and_load(json_s2)
    sun_x = data[path]['sun_x']
    sun_y = data[path]['sun_y']
    sun_radius = data[path]['sun_radius']        
    print(sun_x, sun_y, sun_radius)
        
    moon_x, moon_y = get_centre_of_moon(path, sun_x, sun_y, sun_radius, sun_radius)
    print(moon_x, moon_y)
    
    # get_moon_radius assumes the sun's position and moon's position are close (totality)
    # instead we will feed it the moon's position but say it's the sun
    altered = {}
    altered[path] = {'sun_x': moon_x, 'sun_y': moon_y, 'sun_radius': sun_radius, 'sun_threshold': data[path]['sun_threshold']}
    moon_radius = get_moon_radius(altered, [path], 15)
    print(moon_radius)
        
    new_moon_x, new_moon_y = get_centre_of_moon(path, sun_x, sun_y, sun_radius, moon_radius)
    print(new_moon_x, new_moon_y)
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    output_encircled(img, path, sun_radius, sun_x, sun_y, new_moon_x, new_moon_y, moon_radius, moon_colour = GREEN, dirname = 'test')
    
        
def stage_1(sun_list, tolerance = 2, force_recalculation = False):
    '''(list of str, int)
    Given a list of images for calibration, use these images to determine
    a sun radius and sun threshold that work for all of these images.
    Output pictures to a stage1 directory showing red circles for the sun 
    radius for you to manually confirm these work.
    If you get an error, you probably need to increase the tolerance.
    If desired, you can manually edit the json files.

    >>> stage_1(['example0/ring.jpg'], 2, True)    
    JSON file found at example0-stage1/stage1n1t2.json
    No JSON file for stage 1 found. Computing and saving one.
    Drawing to example0-stage1/ring.jpg;  radius: 126; centre: (239,180)
    (15, 126)
    >>> stage_1(['example0/ring.jpg'], 2)    
    JSON file found at example0-stage1/stage1n1t2.json
    Drawing to example0-stage1/ring.jpg;  radius: 126; centre: (239,180)
    (15, 126)
    >>> stage_1(['example0/ring.jpg'], 1)    
    JSON file found at example0-stage1/stage1n1t1.json
    Drawing to example0-stage1/ring.jpg;  radius: 126; centre: (239,180)
    (15, 126)
    >>> stage_1(['example1/ex1.jpg', 'example1/ex3.jpg'], 2)    
    JSON file found at example1-stage1/stage1n2t2.json
    Drawing to example1-stage1/ex1.jpg;  radius: 110; centre: (238,153)
    Drawing to example1-stage1/ex3.jpg;  radius: 110; centre: (276,132)
    (35, 110)
    >>> stage_1(['example2/DSC05686.jpg', 'example2/DSC05702.jpg'], 2)
    JSON file found at example2-stage1/stage1n2t2.json
    Drawing to example2-stage1/DSC05686.jpg;  radius: 142; centre: (713,1099)
    Drawing to example2-stage1/DSC05702.jpg;  radius: 142; centre: (477,1539)
    (65, 142)
    >>> stage_1(['example2/DSC05686.jpg', 'example2/DSC05702.jpg'], 1)
    JSON file found at example2-stage1/stage1n2t1.json
    Drawing to example2-stage1/DSC05686.jpg;  radius: 142; centre: (713,1099)
    Drawing to example2-stage1/DSC05702.jpg;  radius: 142; centre: (477,1539)
    (65, 142)
    >>> stage_1(['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'], 2)
    JSON file found at example2-stage1/stage1n3t2.json
    Drawing to example2-stage1/DSC05686.jpg;  radius: 141; centre: (713,1098)
    Drawing to example2-stage1/DSC05688.jpg;  radius: 141; centre: (1382,1165)
    Drawing to example2-stage1/DSC05702.jpg;  radius: 141; centre: (477,1538)
    (70, 141)
    >>> stage_1(['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'], 1)
    JSON file found at example2-stage1/stage1n3t1.json
    Drawing to example2-stage1/DSC05686.jpg;  radius: 141; centre: (713,1098)
    Drawing to example2-stage1/DSC05688.jpg;  radius: 141; centre: (1382,1165)
    Drawing to example2-stage1/DSC05702.jpg;  radius: 141; centre: (477,1538)
    (70, 141)
    '''
    # have we already calculated everything?
    stage = 'stage1'
    calculating = True
    json_filename =  get_s1_json_name(sun_list, tolerance)
    
    data = open_and_load(json_filename)

    calculating = whether_calculate_stage1(data, sun_list) or force_recalculation
    if calculating:
        print('No JSON file for stage 1 found. Computing and saving one.')
        # first calculate a radius for each sun picture, and average them
        radiuses = []
        for image_path in sun_list:
            radius = get_sun_central_radius(image_path)
            radiuses.append(radius)
        sun_radius = int(round(np.average([radiuses])))

        # now pick sun threshold that is common to them all
        sun_threshold = int(get_common_threshold(sun_list, sun_radius, tolerance = 2))

        for sun_path in sun_list:
            sun_x, sun_y = get_centre_of_sun(sun_path, sun_threshold, sun_radius)
            data[sun_path] = {'sun_radius': sun_radius, 'sun_x': sun_x, 'sun_y': sun_y, 'sun_threshold': sun_threshold}

        # save data
        with open(json_filename, 'w') as g:
            json.dump(data, g)

    # visualize the data to make it easy to check
    # we want to repaint these pictures in case you want to manually adjust the json file
    for sun_path in sun_list:
        img = cv2.imread(sun_path, cv2.IMREAD_COLOR)
        output_encircled(img, sun_path, data[sun_path]['sun_radius'], data[sun_path]['sun_x'], data[sun_path]['sun_y'], dirname=stage)

    return data[sun_path]['sun_threshold'], data[sun_path]['sun_radius']

##############################################################
######### STAGE 2: 
##############################################################

def is_image_file(fname):
    '''
    Is this a jpg?
    >>> is_image_file('image.JPG')
    True
    >>> is_image_file('image.jpg')
    True
    >>> is_image_file('image.json')
    False
    '''
    return fname.endswith('.JPG') or fname.endswith('.jpg')


def open_and_load(json_filename):
    '''
    Attempt to open json_filename. If the path doesn't exist, create
    the directory. If the file exists, load and return it.
    If the file doesn't exist, return empty dictionary.
    >>> open_and_load('example0-stage1/shouldnotexist.json')
    {}
    >>> open_and_load('example0-stage2/stage2.json')
    JSON file found at example0-stage2/stage2.json
    {'example0/ring.jpg': {'sun_radius': 126, 'sun_x': 239, 'sun_y': 180, 'sun_threshold': 15}}
    '''
    if os.path.exists(get_directory(json_filename)):        
        if os.path.isfile(json_filename):
            print('JSON file found at', json_filename)
            with open(json_filename, 'r') as f:
                data = json.load(f)
            return data
    else:
        os.makedirs(get_directory(json_filename))
    return {}        


def stage_2(directory, sun_threshold, sun_radius, force_recalculation = False):
    '''
    Now that we have figured out a radius and threshold from stage1,
    Calculate (and draw) sun location for every file, and save this to
    a JSON file in thte stage2 directory.
    You can manually edit this JSON file and rerun this code to visualize
    your changes to the JSON file.

    >>> stage_2('example0/', 15, 126, True)
    JSON file found at example0-stage2/stage2.json
    JSON file for stage 2 not found, calculating and saving to  example0-stage2/stage2.json
    Drawing to example0-stage2/ring.jpg;  radius: 126; centre: (239,180)
    >>> stage_2('example0/', 15, 126)
    JSON file found at example0-stage2/stage2.json
    Drawing to example0-stage2/ring.jpg;  radius: 126; centre: (239,180)
    >>> stage_2('example1/', 35, 110)
    JSON file found at example1-stage2/stage2.json
    Drawing to example1-stage2/ex1.jpg;  radius: 110; centre: (238,153)
    Drawing to example1-stage2/ex2.jpg;  radius: 110; centre: (252,143)
    Drawing to example1-stage2/ex3.jpg;  radius: 110; centre: (276,132)
    Drawing to example1-stage2/ex4.jpg;  radius: 110; centre: (284,140)
    Drawing to example1-stage2/ex5.jpg;  radius: 110; centre: (302,144)
    Drawing to example1-stage2/ex6.jpg;  radius: 110; centre: (187,130)
    >>> stage_2('example2/', 70, 141)
    JSON file found at example2-stage2/stage2.json
    Drawing to example2-stage2/DSC05686.jpg;  radius: 141; centre: (713,1098)
    Drawing to example2-stage2/DSC05688.jpg;  radius: 141; centre: (1382,1165)
    Drawing to example2-stage2/DSC05690.jpg;  radius: 141; centre: (1568,1356)
    Drawing to example2-stage2/DSC05702.jpg;  radius: 141; centre: (477,1538)
    Drawing to example2-stage2/DSC05703.jpg;  radius: 141; centre: (544,1549)
    Drawing to example2-stage2/DSC05704.jpg;  radius: 141; centre: (653,919)
    Drawing to example2-stage2/DSC05709.jpg;  radius: 141; centre: (1174,1239)
    '''
    
    # where we save results of this function
    stage = 'stage2'
    json_filename = make_json_filename(directory, stage) + stage + '.json'
    data = open_and_load(json_filename)
    
    files = os.listdir(directory)
    paths = []
    for fname in sorted(files):
        if is_image_file(fname):
            paths.append( directory + fname )

    if data:
        to_calculate = whether_calculate_stage1(data, paths)
    if not data or to_calculate or force_recalculation:
        print('JSON file for stage 2 not found, calculating and saving to ', json_filename)
        
        # make sure this directory exists as get_centre_of_sun will try to write to it
        bw_directory = directory.replace('/', '-sun-binary/')
        if not os.path.exists(bw_directory):
            os.makedirs(bw_directory)

        # find the sun in every file and save it to data
        for image_path in paths:
            sun_x, sun_y = get_centre_of_sun(image_path, sun_threshold, sun_radius)
            data[image_path] = {'sun_radius': sun_radius, 'sun_x': sun_x, 'sun_y': sun_y, 'sun_threshold': sun_threshold}

        # saving for later
        with open(json_filename, 'w') as g:
            json.dump(data, g)

    # visualize the data to make it easy to check
    # we want to repaint these pictures in case you want to manually adjust the json file
    for sun_path in data:
        img = cv2.imread(sun_path, cv2.IMREAD_COLOR)
        output_encircled(img, sun_path, data[sun_path]['sun_radius'], data[sun_path]['sun_x'], data[sun_path]['sun_y'], dirname=stage)




##############################################################
######### STAGE 3: Correct and refine stage 2. Get the moon radius!
##############################################################


def get_amount_of_sun(fname):
    '''str, dict, CV2 -> float
    How much of the black and white version of img is white?
    >>> get_amount_of_sun('example0-sun-binary/ring.jpg')
    4.202748842592593
    >>> get_amount_of_sun('example1-sun-binary/ring.jpg')
    Error example1-sun-binary/ring.jpg not found
    >>> get_amount_of_sun('example1-sun-binary/ex1.jpg')
    26.712632519723865
    >>> get_amount_of_sun('example1-sun-binary/ex2.jpg')
    9.622496012759171
    >>> get_amount_of_sun('example1-sun-binary/ex3.jpg')
    3.961301139147477
    '''
    if os.path.exists(fname): 
        img_bw = cv2.imread(fname, cv2.COLOR_BGR2GRAY)
        # for figuring out percentage of coverage
        all_ones = np.ones(img_bw.shape)
        
        return np.sum(img_bw) / np.sum(all_ones)
    else:
        print('Error', fname, 'not found')


def hough_circles(jpg_path, threshold=30):
    '''
    Use the Hough Circles approach to detect circles
    x, y, radius
    
    Threshold: a smaller value yields more circles, but also more false positives.
    
    >>> hough_circles('example0/ring.jpg')
    array([[[240, 178, 125],
            [222, 170, 144],
            [232, 196, 142],
            [240, 158, 146],
            [262, 178, 145],
            [252, 196, 107],
            [260, 158,  97],
            [212, 208,  88]]], dtype=uint16)
    >>> hough_circles('example1/ex3.jpg', 40)
    array([[[276, 132, 108],
            [260, 120,  91],
            [290, 114,  88],
            [288, 150,  90],
            [262, 146,  90]]], dtype=uint16)
    >>> hough_circles('example4/DSC05735.jpg', 40)
    array([[[1028, 1934,  210],
            [1010, 1946,  208],
            [1012, 1918,  204],
            ...,
            [1200, 1772,   56],
            [1158, 1760,   53],
            [ 752, 1994,   71]]], dtype=uint16)
    '''
    # STAGE 0: read in the input image
    img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)

    # https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
    # Convert to grayscale. 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
      
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 
      
    # Apply Hough transform on the blurred image.
    min_dist_bw_circles = 1
    detected_circles = cv2.HoughCircles(gray_blurred,  
                       cv2.HOUGH_GRADIENT, min_dist_bw_circles, 20, param1 = 50, 
                   param2 = threshold, minRadius = 50, maxRadius = 1000) 

    if np.sum(detected_circles):
        #print(detected_circles)
        detected_circles = np.uint16(np.around(detected_circles)) 
        
    return detected_circles
    '''
    # Draw circles that are detected. 
    if detected_circles is not None: 
        # Convert the circle parameters a, b and r to integers. 

      
        for pt in detected_circles[0, :]: 
            #print('DC', fname, pt)
            a, b, r = pt[0], pt[1], pt[2] 
      
            # Draw the circumference of the circle. 
            cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
      
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
            #cv2.imwrite(params['path'] + params['input_dir'].replace('/', '-test/'), img) 
            #cv2.waitKey(0) 
    '''


def find_sun_with_hough(path, sun_radius):
    '''
    >>> find_sun_with_hough('example0/ring.jpg', 126)
    (240, 180)
    >>> find_sun_with_hough('example1/ex3.jpg', 110)
    (274, 132)
    >>> find_sun_with_hough('example1/ex4.jpg', 110)
    (284, 142)
    >>> find_sun_with_hough('example4/DSC05735.jpg', 141)
    (1026, 1938)
    >>> find_sun_with_hough('example4/DSC05736.jpg', 141)
    (1090, 1938)
    '''

    bw = path.replace('/', '-sun-binary/')
    circs = hough_circles(bw)
    working_radius = np.max(circs)

    # go through the circles from Hough and find the one closest
    # in radius to what we know to be the sun's radius
    for c in circs[0]:
        x = c[0]
        y = c[1]
        r = c[2]
        if abs(r - sun_radius) < abs(working_radius - sun_radius):
            working_radius = r
            working_sun_x = x
            working_sun_y = y
        
    return working_sun_x, working_sun_y

        
def get_moon_radius(data, totalities, threshold = 30):
    '''
    >>> data = open_and_load('example0-stage3/stage3.json')
    JSON file found at example0-stage3/stage3.json
    >>> get_moon_radius(data, ['example0/ring.jpg'])
    123
    >>> data = open_and_load('example1-stage3/stage3.json')
    JSON file found at example1-stage3/stage3.json
    >>> get_moon_radius(data, ['example1/ex3.jpg', 'example1/ex4.jpg'])
    109
    >>> data = open_and_load('example4-stage3/stage3.json')
    JSON file found at example4-stage3/stage3.json
    >>> get_moon_radius(data, ['example4/DSC05735.jpg', 'example4/DSC05736.jpg', 'example4/DSC05737.jpg'])
    142
    '''
    radiuses = []
    xs = []
    ys = []
    
    # TODO: xs and ys not being used any more
    # possibility that there are multiple results for a given file
    
    for path in totalities:
        #print(path)
        sun_radius = data[path]['sun_radius']

        tolerance = sun_radius/8
        
        #print('SR', sun_radius)
        sun_threshold = data[path]['sun_threshold']
        sun_x = data[path]['sun_x']
        sun_y = data[path]['sun_y']

        bw = path.replace('/', '-sun-binary/')
        circs = hough_circles(bw, threshold)
        for c in circs[0]:
            x = c[0]
            y = c[1]
            r = c[2]
            
            d = get_moon_distance(sun_x, sun_y, x, y)
            if d < tolerance:                
                if abs(r - sun_radius) < tolerance:
                    #print(c, d)
                    radiuses.append(r)
                    xs.append(x)
                    ys.append(y)
    return round(np.average(radiuses))


def draw_example(sun_x, sun_y, sun_radius, moon_x, moon_y, moon_radius, canvas_size):
    #print('skipped doctest')
    '''Draw an example eclipse in black and white. Used for checking goodness of fit.
    >>> draw_example(200, 200, 100, 150, 100, 100, (400, 400))
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    >>> draw_example(200, 200, 100, 200, 100, 100, (400, 400))
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    >>> draw_example(200, 200, 100, 200, 200, 100, (400, 400))
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    >>> draw_example(200, 200, 100, 200, 200, 98, (400, 400))
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    >>> draw_example(200, 200, 100, 200, 200, 50, (400, 400))
    array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    '''
    directory = 'calibration/'
    path = directory + 'calib-' + '-'.join(map(str, [sun_x, sun_y, sun_radius, moon_x, moon_y, moon_radius])) + '.jpg'
    #print(path) 
    
    # create canvas
    img =  np.zeros((canvas_size[0], canvas_size[1], 3), np.uint8) 
    
    # draw the sun
    sun_image = cv2.circle(img, (sun_x, sun_y), sun_radius, WHITE, -1) 
    
    # draw the moon
    eclipse = cv2.circle(img, (moon_x, moon_y), moon_radius, BLACK, -1) 
    
    # convert to greyscale
    eclipse = cv2.cvtColor(eclipse, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(path, sun_image)

    return eclipse
    

def goodness_of_fit(bw_path, sun_x, sun_y, sun_radius, moon_x, moon_y, moon_radius):
    '''
    >>> goodness_of_fit('example0-sun-binary/ring.jpg', 239, 180, 126, 239, 180, 124)
    0.09976273148148149
    >>> goodness_of_fit('example0-sun-binary/ring.jpg', 239, 180, 126, 239, 180, 125)
    0.10785300925925925
    >>> goodness_of_fit('example0-sun-binary/ring.jpg', 239, 180, 126, 239, 180, 123)
    0.10199652777777778
    >>> goodness_of_fit('example0-sun-binary/ring.jpg', 239, 180, 126, 239, 180, 100)
    0.2153298611111111
    >>> goodness_of_fit('example1-sun-binary/ex2.jpg', 252, 143, 110, 270, 147, 108)
    0.09003987240829346
    >>> goodness_of_fit('example1-sun-binary/ex2.jpg', 252, 143, 110, 270, 147, 110)
    0.09852472089314195
    >>> goodness_of_fit('example1-sun-binary/ex2.jpg', 252, 143, 110, 260, 150, 110)
    0.12650717703349282
    '''

    image = cv2.imread(bw_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(image.shape)

    model = draw_example(sun_x, sun_y, sun_radius, moon_x, moon_y, moon_radius, image.shape)
    #print(model.shape)
    
    # https://www.tutorialspoint.com/how-to-compare-two-images-in-opencv-python
    h, w = image.shape
    
    #diff = cv2.subtract(image, model)
    diff = np.subtract(image, model)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    
    plain_r = np.sum(diff)
    perc = plain_r / (float(h*w))
    
    return mse

    

def stage_3(directory, totalities):
    '''
    Get the moon radius!
    >>> stage_3('example0/', ['example0/ring.jpg'])
    JSON file found at example0-stage2/stage2.json
    Drawing to example0-stage3/ring.jpg;  radius: 126; centre: (239,180)
    127
    >>> stage_3('example1/', ['example1/ex3.jpg', 'example1/ex4.jpg'])
    JSON file found at example1-stage2/stage2.json
    Drawing to example1-stage3/ex3.jpg;  radius: 110; centre: (276,132)
    Drawing to example1-stage3/ex4.jpg;  radius: 110; centre: (284,140)
    109
    >>> stage_3('example4/', ['example4/DSC05735.jpg', 'example4/DSC05736.jpg', 'example4/DSC05737.jpg'])
    JSON file found at example4-stage2/stage2.json
    Drawing to example4-stage3/DSC05735.jpg;  radius: 141; centre: (1004,1841)
    Drawing to example4-stage3/DSC05736.jpg;  radius: 141; centre: (1071,1840)
    Drawing to example4-stage3/DSC05737.jpg;  radius: 141; centre: (1208,2057)
    142
    '''
    # for this stage
    stage = 'stage3'
    json_filename = make_json_filename(directory, stage) + stage + '.json'
    # from last stage
    json_s2 = make_json_filename(directory, 'stage2') + 'stage2.json'
    data = open_and_load(json_s2)

    # recalculate sun location
    for sun_path in totalities:
        sun_radius = data[sun_path]['sun_radius']
        sun_x, sun_y = find_sun_with_hough(sun_path, sun_radius)
        
        # visualize the data to make it easy to check
        # we want to repaint these pictures in case you want to manually adjust the json file
        img = cv2.imread(sun_path, cv2.IMREAD_COLOR)
        output_encircled(img, sun_path, sun_radius, data[sun_path]['sun_x'], data[sun_path]['sun_y'], sun_x, sun_y, sun_radius, moon_colour=MAGENTA, dirname=stage)
    
        # save these updated coordinates
        data[sun_path]['sun_x'] = int(sun_x)
        data[sun_path]['sun_y'] = int(sun_y)
        
    # find the moon
    moon_radius = get_moon_radius(data, totalities)
        
    with open(json_filename, 'w') as g:
        json.dump(data, g)
            
    # TODO update the json with the new x and y
    return moon_radius


    
##############################################################
######### STAGE 4: Locate the moon!
##############################################################

def get_centre_of_moon(fname, sun_x, sun_y, sun_radius, moon_radius, moon_threshold = -1):
    '''(str, int, int) -> int, int
    Find the centre of the moon.
    Needs: name of original file, radius of sun (in pixels), radius of moon (in pixels)
    This does not appear to work properly around the totality.
    '''
    
    '''
    >>> get_centre_of_moon('example0/ring.jpg', 240, 180, 127, 130)
    >>> get_centre_of_moon('example1/ex1.jpg', 238, 153, 110, 108)
    >>> get_centre_of_moon('example2/DSC05686.jpg', 713, 1097, 141, 137)
    '''    
    img = cv2.imread(fname, cv2.IMREAD_COLOR)

    # STAGE 2: FIND CENTRE OF MOON

    moon_mask_r = moon_radius 
    moon_mask = np.zeros((moon_mask_r * 2 + 1, moon_mask_r * 2 + 1), np.float32)
    moon_mask = cv2.circle(moon_mask, (moon_mask_r, moon_mask_r), moon_mask_r, 1.0, -1)

    # STAGE 2a: set up canvas. I have no idea how to simplify it, because I have no idea what is happening here.
    sun_mask_r = sun_radius # 141 for my photos, 110 for OG's # sun radius
    
    canvas_size_original = (sun_mask_r + moon_mask_r * 2) * 2
    img_h, img_w = img.shape[:2]
    canvas_size_alternate = max( img_h, img_w )
    canvas_size = max(canvas_size_alternate, canvas_size_original)
    
    # pad image to canvas_size
    pad_x, pad_y = (canvas_size - img_w) // 2, (canvas_size - img_h) // 2
    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))
    # calculate the distance between the center of the sun and the center of the image
    center_x, center_y = img_w // 2, img_h // 2
    canvas = np.zeros((canvas_size, canvas_size, img.shape[2]), dtype=np.uint8)
    dx, dy = center_x - sun_x, center_y - sun_y
    # draw sun at the center of the canvas
    left, right = (0, dx) if dx > 0 else (-dx, 0)
    up, down = (0, dy) if dy > 0 else (-dy, 0)
    pad_h, pad_w = padded.shape[:2]
    canvas[down:(pad_h - up), right:(pad_w - left), :] = padded[up:(pad_h - down), left:(pad_w - right), :]

    # STAGE 2b: look for moon
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grayscale
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # otsu
    th, canvas_gray = cv2.threshold(canvas_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canvas_gray, canvas_binary = cv2.threshold(canvas_gray, th + moon_threshold, 255, cv2.THRESH_BINARY)
    # save moon_binary
    canvas_gray, moon_binary = cv2.threshold(img_gray, th + moon_threshold, 255, cv2.THRESH_BINARY)
    bname = fname.replace('/', '-moon-binary/')
    bw_directory = get_directory(bname)
    if not os.path.exists(bw_directory):
        os.makedirs(bw_directory)
    cv2.imwrite(bname, moon_binary)

    # STAGE 2c: ???
    # fft convolve
    moon_signal = signal.fftconvolve(canvas_binary.astype('float') / 255, moon_mask, mode='valid')
    # find coordinates of the smallest values in moon_signal that are also inside the moon_signal_mask
    moon_signal_mask_size = canvas_size - moon_mask_r * 2
    moon_signal_mask_r = moon_signal_mask_size // 2
    moon_signal_mask = np.zeros((moon_signal_mask_size, moon_signal_mask_size), np.float32)
    moon_signal_mask = cv2.circle(moon_signal_mask, (moon_signal_mask_r, moon_signal_mask_r), moon_signal_mask_r, 1.0, -1)
    rows, cols = np.where((moon_signal < 1) * (moon_signal_mask > 0))
    # find the coordinate closest to the center of the sun (closest to the center of the moon_signal image)
    moon_signal_center = moon_signal.shape[0] // 2
    moon_x, moon_y, min_dist = 0, 0, 99999
    for i in range(rows.shape[0]):
        dist = spatial.distance.euclidean([cols[i], rows[i]], [moon_signal_center, moon_signal_center])
        if dist < min_dist:
            moon_x, moon_y, min_dist = int(cols[i]), int(rows[i]), float(dist)
    # calculate moon coordinates in the original image
    # moon_x is relative to moon_signal (4802, 4802)
    original_moon_x = moon_x - (moon_signal_mask_size - img_w) // 2 - dx
    original_moon_y = moon_y - (moon_signal_mask_size - img_h) // 2 - dy
    
    return original_moon_x, original_moon_y


def get_moon_distance(sun_x, sun_y, moon_x, moon_y):
    '''
    (int, int, int, int) -> float
    Calculate the distance in pixels between the sun's and moon's centres.

    >>> get_moon_distance(100, 100, 0, 100)
    100.0
    >>> get_moon_distance(100, 100, 100, 0)
    100.0
    >>> get_moon_distance(100, 100, 200, 100)
    100.0
    >>> get_moon_distance(100, 100, 100, 200)
    100.0
    >>> get_moon_distance(100, 100, 150, 150)
    70.71067811865476
    >>> get_moon_distance(123, 345, 67, 89)
    262.0534296665472
    '''
    return math.dist([sun_x, sun_y], [moon_x, moon_y])


def get_moon_angle(sun_x, sun_y, moon_x, moon_y):
    '''
    (int, int, int, int) -> float
    Calculate the angle in degrees between the sun's and moon's centres.
    Return the angle in degrees between 0 and 365.
    
    https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-vectors
    https://www.reddit.com/r/learnpython/comments/17qixkb/difficulty_calculating_angle_between_two_points/
    >>> get_moon_angle(100, 100, 0, 100)
    0.0
    >>> get_moon_angle(100, 100, 100, 0)
    90.0
    >>> get_moon_angle(100, 100, 200, 100)
    180.0
    >>> get_moon_angle(100, 100, 100, 200)
    270.0
    >>> get_moon_angle(100, 100, 150, 150)
    225.0
    >>> get_moon_angle(123, 345, 67, 89)
    77.6609127217
    '''
    radians =  math.atan2(sun_y-moon_y, sun_x-moon_x)
    return round(math.degrees(radians) % 360, 10)
    

def stage_4(directory, moon_radius, force_recalculation = False):
    '''    
    Find the moon! This does not appear to work during totality.
    
    Example 1: ex3-ex6 are broken
    Example 3: starts to drift
    
    >>> redo = False
    >>> stage_4('example1/', 108, redo)
    JSON file found at example1-stage4/stage4.json
    JSON file found at example1-stage2/stage2.json
    Drawing to example1-stage4/ex1.jpg;  radius: 110; centre: (238,153)
    Drawing to example1-stage4/ex2.jpg;  radius: 110; centre: (252,143)
    Drawing to example1-stage4/ex3.jpg;  radius: 110; centre: (276,132)
    Drawing to example1-stage4/ex4.jpg;  radius: 110; centre: (284,140)
    Drawing to example1-stage4/ex5.jpg;  radius: 110; centre: (302,144)
    Drawing to example1-stage4/ex6.jpg;  radius: 110; centre: (187,130)
    >>> stage_4('example2/', 137, redo)
    JSON file found at example2-stage4/stage4.json
    JSON file found at example2-stage2/stage2.json
    Drawing to example2-stage4/DSC05686.jpg;  radius: 141; centre: (713,1098)
    Drawing to example2-stage4/DSC05688.jpg;  radius: 141; centre: (1382,1165)
    Drawing to example2-stage4/DSC05690.jpg;  radius: 141; centre: (1568,1356)
    Drawing to example2-stage4/DSC05702.jpg;  radius: 141; centre: (477,1538)
    Drawing to example2-stage4/DSC05703.jpg;  radius: 141; centre: (544,1549)
    Drawing to example2-stage4/DSC05704.jpg;  radius: 141; centre: (653,919)
    Drawing to example2-stage4/DSC05709.jpg;  radius: 141; centre: (1174,1239)
    '''
    stage = 'stage4'
    json_filename = make_json_filename(directory, stage) + stage + '.json'
    data = open_and_load(json_filename)
    
    json_s2 = make_json_filename(directory, 'stage2') + 'stage2.json'
    s2_data = open_and_load(json_s2)
    
    if not data or force_recalculation:
        for image_path in sorted(s2_data):
            #print(image_path, get_amount_of_sun(image_path))
            sun_x = s2_data[image_path]['sun_x']
            sun_y = s2_data[image_path]['sun_y']
            original_moon_x, original_moon_y = get_centre_of_moon(image_path, sun_x, sun_y, s2_data[image_path]['sun_radius'], moon_radius)
            
            moon_angle = get_moon_angle(sun_x, sun_y, original_moon_x, original_moon_y)
            moon_distance = get_moon_distance(sun_x, sun_y, original_moon_x, original_moon_y)
            
            # save these to moon_x and moon_y
            data[image_path] = s2_data[image_path]
            data[image_path]['moon_x'] = original_moon_x
            data[image_path]['moon_y'] = original_moon_y
            data[image_path]['moon_radius'] = moon_radius          
            data[image_path]['moon_angle'] = moon_angle
            data[image_path]['moon_dist'] = moon_distance

        with open(json_filename, 'w') as g:
            json.dump(data, g)

    # visualize the data to make it easy to check
    # we want to repaint these pictures in case you want to manually adjust the json file
    for sun_path in data:
        img = cv2.imread(sun_path, cv2.IMREAD_COLOR)
        output_encircled(img, sun_path, data[sun_path]['sun_radius'], data[sun_path]['sun_x'], data[sun_path]['sun_y'], data[sun_path]['moon_x'], data[sun_path]['moon_y'], moon_radius, dirname=stage)
    # TODO for example 3 the circle stops aligning nicely before totality    


##############################################################
######### STAGE 5: output centred pictures
##############################################################

def output_image(jpg_path, sun_x, sun_y, sun_mask_r):
    '''
    Given the x and y of the centre of the sun in jpg_path, and the
    radius of the sun (sun_mask_r), centre the image.
    
    >>> output_image('example0/ring.jpg', 239, 180, 126)
    Moved the sun in example0/ring.jpg by (1,0)
    >>> output_image('example1/ex1.jpg', 238, 153, 110)
    Moved the sun in example1/ex1.jpg by (-30,3)
    >>> output_image("example2/DSC05686.jpg", 713, 1098, 141)
    Moved the sun in example2/DSC05686.jpg by (223,226)
    '''

    if os.path.exists(jpg_path):
        # read jpg
        img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
        img_h, img_w = img.shape[:2]
        crop_w, crop_h = img_w, img_h
        
        # for output
        jpg_output_path = jpg_path.replace('/', '-stage5/')
        if not os.path.exists(get_directory(jpg_output_path)):        
            os.makedirs(get_directory(jpg_output_path))
    
        # TODO moon size
        moon_mask_r = sun_mask_r
        canvas_size_original = (sun_mask_r + moon_mask_r * 2) * 2
        canvas_size_alternate = max( img_h, img_w )
        canvas_size = max(canvas_size_alternate, canvas_size_original)

        # pad image to canvas_size
        pad_x, pad_y = (canvas_size - img_w) // 2, (canvas_size - img_h) // 2
        padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))
        
        # calculate the distance between the center of the sun and the center of the image
        center_x, center_y = img_w // 2, img_h // 2
        canvas = np.zeros((canvas_size, canvas_size, img.shape[2]), dtype=np.uint8)
        canvas_h, canvas_w = canvas.shape[:2]
        dx, dy = center_x - sun_x, center_y - sun_y
        print('Moved the sun in ', jpg_path, ' by (', dx, ',', dy, ')', sep='')
        
        # draw sun at the center of the canvas
        left, right = (0, dx) if dx > 0 else (-dx, 0)
        up, down = (0, dy) if dy > 0 else (-dy, 0)
        pad_h, pad_w = padded.shape[:2]
        canvas[down:(pad_h - up), right:(pad_w - left), :] = padded[up:(pad_h - down), left:(pad_w - right), :]

        # crop to original size
        crop_x, crop_y = (canvas_w - crop_w) // 2, (canvas_h - crop_h) // 2
        cropped = canvas[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        # save cropped
        cv2.imwrite(jpg_output_path, cropped)

    else:
        print('Error, could not find', jpg_path)


def stage_5(directory, stage_of_data):
    '''
    Given the files in directory "directory", centre them using the 
    information from the stage given by stage_of_data.

    >>> stage_5('example0/', 'stage2')
    JSON file found at example0-stage2/stage2.json
    Moved the sun in example0/ring.jpg by (1,0)
    >>> stage_5('example1/', 'stage2')
    JSON file found at example1-stage2/stage2.json
    Moved the sun in example1/ex1.jpg by (-30,3)
    Moved the sun in example1/ex2.jpg by (-43,7)
    Moved the sun in example1/ex3.jpg by (-68,25)
    Moved the sun in example1/ex4.jpg by (-77,16)
    Moved the sun in example1/ex5.jpg by (-92,13)
    Moved the sun in example1/ex6.jpg by (22,27)
    >>> stage_5('example2/', 'stage2')
    JSON file found at example2-stage2/stage2.json
    Moved the sun in example2/DSC05686.jpg by (223,226)
    Moved the sun in example2/DSC05688.jpg by (-446,159)
    Moved the sun in example2/DSC05690.jpg by (-632,-32)
    Moved the sun in example2/DSC05702.jpg by (459,-214)
    Moved the sun in example2/DSC05703.jpg by (392,-225)
    Moved the sun in example2/DSC05704.jpg by (283,405)
    Moved the sun in example2/DSC05709.jpg by (-238,85)
    >>> stage_5('example2/', 'doesnotexist')
    Error: no data available from example2-doesnotexist/doesnotexist.json
    '''

    json_s2 = make_json_filename(directory, stage_of_data) + stage_of_data + '.json'
    s2_data = open_and_load(json_s2)
    
    if s2_data:        
        for image_path in sorted(s2_data):
            output_image(image_path, s2_data[image_path]['sun_x'], s2_data[image_path]['sun_y'], s2_data[image_path]['sun_radius'])
    else:
        print('Error: no data available from', json_s2)
        

##############################################################
######### STAGE 6: video
##############################################################

def get_timestamp(path):
    '''
    >>> get_timestamp('example0/ring.jpg')
    Error: no timestamp in EXIF info for example0/ring.jpg
    >>> get_timestamp('example2/DSC05686.jpg')
    datetime.datetime(2024, 4, 8, 14, 59, 34)
    >>> get_timestamp('example2/DSC05709.jpg')
    datetime.datetime(2024, 4, 8, 15, 15, 43)
    '''
    # https://stackoverflow.com/questions/4764932/in-python-how-do-i-read-the-exif-data-for-an-image
    f = open(path, 'rb')

    # Return Exif tags
    tags = exifread.process_file(f)
    if 'EXIF DateTimeOriginal' in tags:
        time_info = tags['EXIF DateTimeOriginal']
        # convert it to ISO format so it can be a datetime object
        time_info = str(time_info).split(' ')
        date_str = time_info[0].replace(':','-')
        time_str = time_info[1]
        timestamp = datetime.fromisoformat(date_str + 'T' + time_str)
        return timestamp
    else:
        print('Error: no timestamp in EXIF info for', path)


def make_video(images, fps, force_creation = False):
    '''
    Given a list of images, make a video from their stage5 versions.
    Informed by  https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/
    >>> make_video(['example1/ex1.jpg', 'example1/ex2.jpg', 'example1/ex3.jpg'], 30)
    Video already exists at example1.mp4 -> not rewriting it
    '''    
    frame = cv2.imread(images[0].replace('/', '-stage5/')) 
    height, width, layers = frame.shape
    
    video_name = get_directory(images[0]).replace('/','') + '.mp4'

    if not os.path.exists(video_name) or force_creation:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_name,  fourcc, fps, (width, height))  
            
        # Appending the images to the video one by one 
        for image in images:
            centred_image = image.replace('/', '-stage5/')
            video.write(cv2.imread(centred_image))  

        # Deallocating memories taken for window creation   
        video.release()  # releasing the video generated 
    else:
        print('Video already exists at', video_name, '-> not rewriting it')
        

def get_durations_per_image(paths):
    '''(dict of str) -> list
    Given a dictionary with filenames for its keys, find out
    how many seconds elapsed between them.
    
    >>> get_durations_per_image(['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05690.jpg', 'example2/DSC05702.jpg', 'example2/DSC05703.jpg', 'example2/DSC05704.jpg', 'example2/DSC05709.jpg']) 
    [84, 150, 296, 22, 179, 238, 162]
    >>> get_durations_per_image(['example1/ex1.jpg', 'example1/ex2.jpg', 'example1/ex3.jpg', 'example1/ex4.jpg', 'example1/ex5.jpg', 'example1/ex6.jpg'])
    Error: no timestamp in EXIF info for example1/ex1.jpg
    [1, 1, 1, 1, 1, 1]
    '''
    times = []  
    diffs = []
    has_exif = True
    
    for i, image_path in enumerate(sorted(paths)):
        timestamp = get_timestamp(image_path)
        
        if timestamp == None or not has_exif:
            has_exif = False
            break
        else:
            times.append(timestamp)
            if i > 0:
                diff = timestamp - times[-2]
                diffs.append(diff.seconds)

    if has_exif:
        # diffs will be short by one. Use average diff to pad the last one.
        # use the diffs to weight the durations of the different images
        # in forming a video
        diffs.append( round(np.average(diffs)) )
        return diffs
    else:
        return [1]*len(paths)


def weight_frames(names, diffs):
    '''Repeat each element in names by its corresponding value in diffs.
    >>> weight_frames(['a', 'b'], [1, 2])
    ['a', 'b', 'b']
    >>> weight_frames(['a', 'b'], [3, 2])
    ['a', 'a', 'a', 'b', 'b']
    '''
    images = []
    for i, duration in enumerate(diffs):
        #print(names[i], duration)
        images += [names[i]]*duration
    return images
        
        
def stage_6(directory, stage_of_data, force_video_creation = False):
    '''Make a video!
    Assumes images are titled in a way that when sorted has them in chronological order.
    >>> stage_6('example1/', 'stage2')
    JSON file found at example1-stage2/stage2.json
    Error: no timestamp in EXIF info for example1/ex1.jpg
    Video already exists at example1.mp4 -> not rewriting it
    >>> stage_6('example1/', 'stage2', True)
    JSON file found at example1-stage2/stage2.json
    Error: no timestamp in EXIF info for example1/ex1.jpg
    >>> stage_6('example2/', 'stage2')
    JSON file found at example2-stage2/stage2.json
    Video already exists at example2.mp4 -> not rewriting it
    '''
    json_s2 = make_json_filename(directory, stage_of_data) + stage_of_data + '.json'
    s2_data = open_and_load(json_s2)
    
    if s2_data:      
        diffs = get_durations_per_image(s2_data)
        names = sorted(list(s2_data.keys()))

        images = weight_frames(names, diffs)
        
        fps = 180
        if len(images) == len(names):
            # no exif info
            fps = fps / 3
        make_video(images, fps, force_video_creation)
    else:
        print('Error: no data available from', json_s2)
        


if __name__ == '__main__':
    
    testing = True
    
    if testing:
        doctest.testmod()
    else:
        directory = 'example4/'
        moon_radius = 137
        
        sun_threshold, sun_radius = stage_1(['example4/DSC05686.jpg', 'example4/DSC05688.jpg', 'example4/DSC05702.jpg'], 2)
        stage_2(directory, sun_threshold, sun_radius)
        moon_radius = stage_3(directory, ['example4/DSC05735.jpg', 'example4/DSC05736.jpg', 'example4/DSC05737.jpg'])
        stage_4(directory, moon_radius)
        stage_5(directory, 'stage3')
        stage_6(directory, 'stage3', False)
    '''
    '''
    
    # Todo add exifread to requirements

    '''
    TODO
    1. split moon detection and totality correction of stage 3 into two stages
    2. expand totality correction to fix the rest of the totality shots not in the manual list
    
    - expand the totality detection; separate stage 3 into two parts
    - clipped pictures
    - moon detection
    - angle fixing
    - fixing totality circles
    - smooth video
    - command-line interface & parser
    - redirect messages currently going to standard out

    '''

