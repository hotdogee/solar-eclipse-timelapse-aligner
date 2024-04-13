# solar-eclipse-timelapse-aligner

This specialized program is used to align and stabilize solar eclipse time-lapse photos.
It was originally developed by Han Lin (@hotdogee) in to process images of the 2020-06-21 annular solar eclipse taken from a Nikon P950 at a 2 second interval.

# About

The eclipse images that the original creator of this project took had a black background with no details and a smooth surface sun with also no high contrast features for general image stabilization methods to match across the images. This program detects the location of the sun in the input images and saves a new image in the output directory with the sun moved to the center of the image.

Note: If your solar images were taken with a narrow band Ha filter, your images will likely have enough details for general stabilization methods to work.


# Before and after videos

## Before stabilization (links to YouTube)
[![Before stabilization](https://img.youtube.com/vi/71vtskpdcXA/0.jpg)](https://youtu.be/71vtskpdcXA)

## After stabilization (links to YouTube)
[![After stabilization](https://img.youtube.com/vi/M4H9w8yh4Fg/0.jpg)](https://youtu.be/M4H9w8yh4Fg)

## After stabilization
![Results](docs/tile-square.jpg)


# Features

* Location stabilization: Outputs new images with the sun moved to the center of the image.
* Rotation stabilization: Outputs new images with the sun moved to the center of the image and rotated so that the moon enters from the same angle.
* Multi-core CPU support: Will estimate the number of processes to use using the detected available memory and number of real CPU cores.
* Detect the center coordinates of the sun and the moon, the results are saved in a JSON file.
* Detect the radius of the sun given a full sun image.
* Command line interface to manually fine-tune the detection parameters if the defaults are not perfect.


# Input Image requirements

Sun detection and sun centering works best if your images meet the following requirements

1. Needs to be images of solar eclipse.
2. Images need to have clean dark backgrounds.
3. The sun needs to have clean edges when the program converts the image into black and white with a given threshold.
4. The size (in pixels) of the sun in the images needs to be constant, this should be true unless you changed the focal length during the shooting session.
5. The radius (in pixels) of the sun needs to be greater than 50. If your photos have the sun smaller than this, you'll need to batch resize them.
6. The sun should be fully visible in all images. In other words, it should not be partially cut off at the edges. This program has a clipped sun filter, when enabled, will filter out images with edge pixels over a certain brightness level.
7. (Optional) For the auto sun radius detect feature, a photo of the sun where the sun is clear of clouds and mostly or completely un-eclipsed is used, this is usually the first or last clear photo in the shooting session, a full sun with no clouds works best. 

Moon detection and moon angle stabilization currently only works if your images meet the following additional requirements:

1. The photos need to be taken at a location where the moon passes through the center of the sun, in other words, the eclipse needs to be a total solar eclipse or an annular solar eclipse. Partial solar eclipse images will not work with angle stabilization (centering should be fine).
2. When the images are sorted by filename, they should be in the correct chronological order.


# Installation

## Run the Python source code (Windows, Mac, Linux)

This assumes you are familiar with Python. You can clone the project, pip install the requirements, and directly run the source code.

```bash
# Install Python 3.7 on your OS, and optionally create and activate a new virtual environment
# Clone this repo
git clone https://github.com/patitsas/solar-eclipse-timelapse-aligner.git
# Change directory
cd solar-eclipse-timelapse-aligner
# Install required packages. If this command fails, open requirements.txt and install each one individually.
pip install -r requirements.txt
# Run the program, you should see a list of arguments and an error message for missing arguments: --input
python eclipse-aligner.py
# Print the full help message
python eclipse-aligner.py -h
# Optional: run these examples of input to confirm there are no errors (see below for information on workflow)
python eclipse-aligner.py --input=example0/ --sun=example0/ring.jpg --circles=example0.json
python eclipse-aligner.py --input=example1/ --sun_radius=110 --circles=example1.json
python eclipse-aligner.py --input=example2/ --sun_radius=141 --sun_threshold=75 --circles=example2.json
```
The optional commands at the end are to check the program is functioning. It will ask you if it can save the detection location information to example0.json, example1.json, etc; enter yes. If everything is functioning smoothly, delete those files.

If things are not running smoothly, check the contents of the json files. The filenames may appear in a different order and this is normal (Python dictionary). You want to check the numbers are the same for each filename.
```
>> cat example0.json
{"ring.jpg": {"sun": [240, 180, 127]}}
>> cat example1.json
{"ex1.jpg": {"sun": [238, 153, 110]}, "ex2.jpg": {"sun": [251, 147, 110]}, "ex3.jpg": {"sun": [276, 132, 110]}, "ex4.jpg": {"sun": [284, 140, 110]}, "ex5.jpg": {"sun": [302, 144, 110]}, "ex6.jpg": {"sun": [189, 131, 110]}}
>> cat example2.json
{"DSC05688.jpg": {"sun": [1382, 1166, 141]}, "DSC05690.jpg": {"sun": [1569, 1360, 141]}, "DSC05686.jpg": {"sun": [713, 1099, 141]}, "DSC05702.jpg": {"sun": [477, 1539, 141]}, "DSC05703.jpg": {"sun": [544, 1549, 141]}, "DSC05704.jpg": {"sun": [653, 919, 141]}, "DSC05712.jpg": {"sun": [1322, 1357, 141]}, "DSC05709.jpg": {"sun": [1174, 1239, 141]}, "DSC05717.jpg": {"sun": [927, 1451, 141]}, "DSC05722.jpg": {"sun": [1234, 1654, 141]}, "DSC05719.jpg": {"sun": [1032, 1516, 141]}, "DSC05724.jpg": {"sun": [1248, 1729, 141]}, "DSC05725.jpg": {"sun": [1103, 1740, 141]}, "DSC05728.jpg": {"sun": [1137, 1781, 141]}}
```

# High level workflow

0. (Optional) Test the program with the provided examples. There are three sets of examples provided: example0, example1, and example2.
1. (Optional) Test the program with the example images to learn what results you should expect.
2. (Recommended) Pick a couple images (3 to 10) representative of the different stages of the eclipse, and process them with the program to quickly find a set of parameters that work well with all of the representative images.
3. Use the same parameters found in step 2 to process the full image set, if the results are not perfect, use the images that had incorrect detection to fine tune the parameters.


# Using the program

This program currently only has a command line interface.

## Sun detection and Location stabilization

1. `--input` Copy your eclipse images in JPG format to a new input directory. Develop your RAW images into JPGs first if you shot RAW.

2. `--sun` Start by letting the program auto detect the sun radius, choose one of the photos in the input directory where the sun is clear of clouds and mostly or completely un-eclipsed, this is usually the first or last clear photo in the shooting session, a full sun with no clouds works best.

   `--sun_radius` You can also directly provide the sun radius in pixels to fine-tune the sun coordinate detection if the results are not perfect.

3. For example, if I put the 8 example images inside `D:\eclipse\jpg` and use `D:\eclipse\jpg\DSCN3815.jpg` as the `--sun` image, I would run the following command:
    ```
    eclipse-aligner.exe --input=D:\eclipse\jpg --sun=D:\eclipse\jpg\DSCN3815.jpg
    ```

    and receive something like the following output:

    ```
    INFO: 8 images found in D:\eclipse\jpg
    INFO: Detected image size (height, width): (3456, 4608)
    INFO: Detected SUN RADIUS: 1210
    INFO: Starting 8 workers.
    Processing: 100%|############################| 8/8 [00:02<00:00,  3.58img/s]
        Saving: 100%|############################| 8/8 [00:04<00:00,  1.99img/s]
    ```

    Three new directories with suffixes `-circled`, `-output`, and `-sun-binary` and one new JSON file named `circles_data.json` will be created alongside the input directory:

    ![Output Directories](docs/output-directories.jpg)
    ![Sun Results](docs/sun-results.jpg)

4. If you get an error that the sun radius is computed as 0 pixels, you have two options: adjust the threshold for the sun, and/or calculate the radius yourself. It is probably easiest to do the latter. To calculate the radius, open up your example file in an image editor. Find the pixel coordinates of two ends of the diameter, and use them to calculate the diameter, and then radius.

5. Check the resulting centered images inside the `-output` directory. The directory will created along side the `--input` directory, for example, if your input directory is `./jpg`, the `-output` directory will be at `./jpg-output`.

6. If the results are not perfect, fine-tuning a few parameters may fix the problem. Start the fine-tuning process by viewing the images inside the `-circled` directory. The red circle represents the detected sun location, if the circles are off, follow the steps for `Fine-tuning the sun detection` below.


## Enable the clipped sun filter

* `--filter` will enable the clipped sun filer, when, enabled clipped and empty images will be moved to a directory with the same name as the input directory appended with `-clipped` by default. For example, if your input directory is `./jpg`, the images will be moved to `./jpg-clipped`.
* Example command:
  ```
  eclipse-aligner.exe --filter --input=D:\eclipse\jpg --sun=D:\eclipse\jpg\DSCN3815.jpg
  ```

## Fine-tuning the sun detection

1. Check the images inside the `-circled` directory, if the red circle looks larger or smaller than the actual sun, decrease of increase the `--sun_radius` and re-process the images. If you used the auto sun detection argument `--sun` you can find the detected sun radius in the output `INFO: Detected SUN RADIUS: 1210` and use start fine-tuning from that value.

2. If the size of the red circle looks good, but the location is off, check the images inside the `-sun-binary` directory, and see if there are any images with fuzzy sun edges, increase or decrease the `--sun_threshold` (default: 25) until the edges of the sun are nice and clean like the example.

* Example command:
  ```
  eclipse-aligner.exe --input=D:\eclipse\jpg --sun_radius=1210 --sun_threshold=25
  ```

## Moon detection and Rotation stabilization

NOTE: Only works for total solar eclipse or annular solar eclipse, this part of the program assumes that the center of the moon pass through the center of the sun at the maximum point in your images.
Use this feature if after viewing the results sun centering, you determine that the rotation of the sun also needs to be fixed.

1. `--fix_angle` This parameter will rotate the outputted images by this angle. Set a value between `0` and `359` in degrees, check the image below of examples when this argument is set to `0`, `90`, `180`, and `270`.
   ![Angle Example](docs/doc-angle.jpg)

3. `--moon_radius_mod` (MOON_RADIUS - SUN_RADIUS) in pixels, use a negative value if the moon is smaller than the sun (as is the case of annular solar eclipse). There is currently no auto-detect feature for the moon radius, use Photoshop or similar software to figure out how much larger or smaller in pixels is the moon when compared to the sun. For example, if the width of the moon is 2372, the moon radius would be 1186, and if the sun radius is 1210, set `--moon_radius_mod` to -24 (because 1186 - 1210 = -24).

4. Check the resulting centered images inside the `-output` directory. The directory will created along side the `--input` directory, for example, if your input directory is `./jpg`, the `-output` directory will be at `./jpg-output`.

5. If the results are not perfect, fine-tuning a few parameters may fix the problem. Start the fine-tuning process by viewing the images inside the `-circled` directory. The red circle represents the detected sun location, the green circle represents the detected moon location, if the moon circles are off, follow the steps for `Fine-tuning the moon detection` below.

* Example command:
  ```
  eclipse-aligner.exe --input=D:\eclipse\jpg --sun=D:\eclipse\jpg\DSCN3815.jpg --fix_angle=17 --moon_radius_mod=-24
  ```
  Output:
  ```
  INFO: 8 images found in E:\test3\jpg
  INFO: Detected image size (height, width): (3456, 4608)
  INFO: Detected SUN RADIUS: 1210
  INFO: Using MOON RADIUS: 1186 (sun_radius + moon_radius_mod)
  INFO: Starting 8 workers.
  Processing: 100%  %|############################| 8/8  [01:30<00:00, 11.27s/img]
      Saving: 100%  %|############################|  8/8 [00:04<00:00,  1.74img/s]
  ```
  Result:
  ![Moon Results](docs/moon-results.jpg)

## Fine-tuning the moon detection

Moon detection errors for images taken near totality or annularity and near the start or end of the eclipse can be safely ignored. To solve the problem that the angle of moon is ambiguous at totality or annularity, we group the images where the distance of the sun and moon is smaller than the `--max_phase_rotation_group_threshold` (default: 30 pixels), and assume that the require rotation if the same for the whole group, and the required rotation of the image with a distance just greater than the given value is used for the whole group. Moon detection for images near the start or end of the eclipse may be hard to get right, we group the images where the shadowed width is less than the `--min_phase_rotation_group_threshold` (default: 100 pixels), and assume that the require rotation if the same for the images in the same group.

1. Try smaller or larger values of `--moon_radius_mod` until the size of the green circle of images inside the `-circled` directory best matches the moons shadow in the image.

2. Check the images inside the `-moon-binary` directory, and see if there are any images with fuzzy moon edges, increase or decrease the `--moon_threshold_mod` (default: 25) until the edges of the moon are nice and clean like the example.


# Complete list of arguments
```
usage: eclipse-aligner [-h] [-v] --input INPUT [--sun SUN]
                       [--sun_radius SUN_RADIUS]
                       [--sun_threshold SUN_THRESHOLD] [--fix_angle FIX_ANGLE]
                       [--moon_radius_mod MOON_RADIUS_MOD]
                       [--moon_threshold_mod MOON_THRESHOLD_MOD]
                       [--max_phase_rotation_group_threshold THRESHOLD]
                       [--min_phase_rotation_group_threshold THRESHOLD]
                       [--filter] [--clipped_cutoff CLIPPED_CUTOFF]
                       [--empty_cutoff EMPTY_CUTOFF] [--workers WORKERS]
                       [--output_suffix OUTPUT_SUFFIX]
                       [--clipped_suffix CLIPPED_SUFFIX]
                       [--circled_suffix CIRCLED_SUFFIX]
                       [--sun_binary_suffix SUN_BINARY_SUFFIX]
                       [--moon_binary_suffix MOON_BINARY_SUFFIX]
                       [--circles CIRCLES]

This specialized program is used to align and stabilize solar eclipse time-
lapse photos

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --input INPUT         Path to the input directory, will only read JPGs,
                        develop you DNGs into JPGs first.
  --sun SUN             Path to an image with an un-eclipsed sun, used to auto
                        detect the radius of the sun in pixels, if you do not
                        have an image with an un-eclipsed sun, you can try
                        using an image with the least partially eclipsed sun.
  --sun_radius SUN_RADIUS
                        Radius of the sun in pixels, if not given, auto detect
                        from full sun image. (default: -1)
  --sun_threshold SUN_THRESHOLD
                        Brightness threshold used to convert the image to
                        black and white, choose a value that gives clean edges
                        for best results. (default: 25)
  --fix_angle FIX_ANGLE
                        Rotates the outputted images. Enables moon angle
                        stabilization if given a non-negative value in degrees,
                        do not enable if processing partial solar eclipse images.
                        (default: -1)
  --moon_radius_mod MOON_RADIUS_MOD
                        (MOON_RADIUS - SUN_RADIUS) in pixels, use a negative
                        value if the moon is smaller than the sun (as is the
                        case of annular solar eclipse). (default: -24)
  --moon_threshold_mod MOON_THRESHOLD_MOD
                        Use a negative value to use a smaller threshold, or a
                        positive value to use a larger threshold, choose a
                        value that gives clean moon edges. TECHNICAL INFO:
                        this value is added the dynamic Otsu threshold.
                        (default: -1)
  --max_phase_rotation_group_threshold THRESHOLD
                        To solve the problem that the angle of moon is
                        ambiguous at totality or annularity, we group the
                        images where the distance of the sun and moon is
                        smaller than the given value in pixels, and assume
                        that the require rotation if the same for the whole
                        group, and the required rotation of the image with a
                        distance just greater than the given value is used for
                        the whole group. (default: 30)
  --min_phase_rotation_group_threshold THRESHOLD
                        Moon detection for images near the start or end of the
                        eclipse may be hard to get right, we group the images
                        where the shadowed width is less than the given value
                        in pixels, and assume that the require rotation if the
                        same for the images in the same group. (default: 100)
  --filter              Enable the clipped sun filter. (default: False)
  --clipped_cutoff CLIPPED_CUTOFF
                        The clipped sun filter will filter out images if any
                        edge pixel's brightness exceeds this value. (default:
                        40)
  --empty_cutoff EMPTY_CUTOFF
                        The clipped sun filter will filter out images if the
                        brightest pixel in the image is lower than this value,
                        in other words, there is nothing in the image.
                        (default: 80)
  --workers WORKERS     Number of processing workers, each workers requires
                        3.5GB of RAM for 16M pixel images, the larger the
                        image the more ram is required. (default: -1)
  --output_suffix OUTPUT_SUFFIX
                        The stabilized results will be save to a directory
                        with the same name as the input directory appended
                        with the given suffix. (default: -output)
  --clipped_suffix CLIPPED_SUFFIX
                        The clipped sun filer will move images to a directory
                        with the same name as the input directory appended
                        with the given suffix. (default: -clipped)
  --circled_suffix CIRCLED_SUFFIX
                        Images with the sun circled in red for validation will
                        be saved to a directory with the same name as the
                        input directory appended with the given suffix.
                        (default: -circled)
  --sun_binary_suffix SUN_BINARY_SUFFIX
                        Black and white images using the sun threshold will be
                        saved to a directory with the same name as the input
                        directory appended with the given suffix. Used for
                        finetuning the sun threshold value. (default: -sun-
                        binary)
  --moon_binary_suffix MOON_BINARY_SUFFIX
                        Black and white images using the moon threshold mod
                        will be saved to a directory with the same name as the
                        input directory appended with the given suffix. Used
                        for finetuning the moon threshold mod value. (default:
                        -moon-binary)
  --circles CIRCLES     If not given, by default, the program will save the
                        detected sun and moon location data in JSON format to
                        a file named circles_data.json located in the same
                        directory as the input directory, and will overwrite
                        the file if exists. If the detected location are good,
                        you can choose to move the file to another location
                        for reuse if you decide to re-develop the DNGs with
                        better settings. If given the path to the existing
                        good circles_data.json, the program will read the sun
                        and moon location data from the JSON file and perform
                        the centering and angle stabilization using the data
                        in the JSON file. (default: circles_data.json)
```


# Contribute

* Pull request for bug fixes and new features are welcome!
  

# Donations
The original creator of this code has a paypal here: [![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)]([YOUR_EMAIL_CODE](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=hotdogee@gmail.com&lc=US&item_name=Solar+eclipse+timelapse+aligner+Development+Support&no_note=0&cn=&currency_code=USD&bn=PP-DonationsBF:btn_donateCC_LG.gif:NonHosted))
