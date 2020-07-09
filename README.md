# solar-eclipse-timelapse-aligner

This specialized program is used to align and stabilize solar eclipse time-lapse photos.
It was originally developed to process images of the 2020/6/21 annular solar eclipse taken from a Nikon P950 at a 2 second interval.

# The problem with general image stabilization methods and what this program does

The eclipse images I took had a black background with no details and a smooth surface sun with also no high contrast features for general image stabilization methods to match across the images. This program detects the location of the sun in the input images and saves a new image in the output directory with the sun moved to the center of the image.

Note: If your solar images were taken with a narrow band Ha filter, your images will likely have enough details for general stabilization methods to work.

# Image requirements

Sun detection and sun centering works best if your images meet the following requirements

1. Needs to be images of solar eclipse.
2. Images need to have clean dark backgrounds.
3. The sun needs to have clean edges when the program converts the image into black and white with a given threshold.
4. The size (in pixels) of the sun in the images needs to be constant, this should be true unless you changed the focal length during the shooting session.
5. The sun should be fully visible in all images, in other words, it should not be partially cut off at the edges. This program has a clipped sun filter, when enabled, will filter out images with edge pixels over a certain brightness level.
6. (Optional) For the auto sun radius detect feature, a photo of the sun where the sun is clear of clouds and mostly or completely un-eclipsed is used, this is usually the first or last clear photo in the shooting session, a full sun with no clouds works best. 

Moon detection and moon angle stabilization currently only works if your images meet the following additional requirements:

1. The photos need to be taken at a location where the moon passes through the center of the sun, in other words, the eclipse needs to be a total solar eclipse or an annular solar eclipse. Partial solar eclipse images will not work with angle stabilization (centering should be fine).
2. When the images are sorted by filename, they should be in the correct chronological order.

# Installation

## Windows executable
Download the compiled EXE from the releases page on github and run the program from the command line or powershell.

## Python source
If you are comfortable working with the Python environment, you can clone the project, pip install the requirements, and directly run the source code.

# Workflow

1. (Optional) Test the program with the example images to learn what results you should expect.
2. (Recommended) Pick a couple images (3 to 10) representative of the different stages of the eclipse, and process them with the program to quickly find a set of parameters that work well with all of the representative images.
3. Use the same parameters found in step 2 to process the full image set, if the results are not perfect, use the images that had incorrect detection to fine tune the parameters.
# Example images

Download the example input images here:

# Using the program

This program currently only has a command line interface.

## Sun detection and sun centering

1. `--input` Copy your eclipse images in JPG format to a new input directory. Develop your RAW images into JPGs first if you shot RAW.

2. `--sun` Start by letting the program auto detect the sun radius, choose one of the photos in the input directory where the sun is clear of clouds and mostly or completely un-eclipsed, this is usually the first or last clear photo in the shooting session, a full sun with no clouds works best.

   `--sun_radius` You can also directly provide the sun radius in pixels to fine-tune the sun coordinate detection if the results are not perfect.

3. Check the resulting centered images inside the `-output` directory. The directory will created along side the `--input` directory, for example, if your input directory is `./jpg`, the `-output` directory will be at `./jpg-output`.

4. If the results are not perfect, fine-tuning a few parameters may fix the problem. Start the fine-tuning process by viewing the images inside the `-circled` directory. The red circle represents the detected sun location, if the circles are off, follow the steps for `Fine-tuning the sun detection` below.

## Enable the clipped sun filter

* `--filter` will enable the clipped sun filer, when, enabled clipped and empty images will be moved to a directory with the same name as the input directory appended with `-clipped` by default. For example, if your input directory is `./jpg`, the images will be moved to `./jpg-clipped`.

## Fine-tuning the sun detection

1. Check the images inside the `-circled` directory, if the red circle looks larger or smaller than the actual sun, decrease of increase the `--sun_radius` and re-process the images. If you used the auto sun detection argument `--sun` you can find the detected sun radius in the output `INFO: Detected SUN RADIUS: 1210` and use start fine-tuning from that value.

2. If the size of the red circle looks good, but the location is off, check the images inside the `-sun-binary` directory, and see if there are any images with fuzzy sun edges, increase or decrease the `--sun_threshold` (default: 25) until the edges of the sun are nice and clean like the example.

## Moon detection and moon angle stabilization

NOTE: Only works for total solar eclipse or annular solar eclipse, this part of the program assumes that the center of the moon pass through the center of the sun at the maximum point in your images.
Use this feature if after viewing the results sun centering, you determine that the rotation of the sun also needs to be fixed.

1. `--fix_angle` Set a value between `0` and `359` in degrees, check the image below of examples when this argument is set to `0`, `90`, `180`, and `270`.

2. `--moon_radius_mod` (MOON_RADIUS - SUN_RADIUS) in pixels, use a negative value if the moon is smaller than the sun (as is the case of annular solar eclipse). There is currently no auto-detect feature for the moon radius, use Photoshop or similar software to figure out how much larger or smaller in pixels is the moon when compared to the sun. For example, if the width of the moon is 2372, the moon radius would be 1186, and if the sun radius is 1210, set `--moon_radius_mod` to -24 (because 1186 - 1210 = -24).

3. Check the resulting centered images inside the `-output` directory. The directory will created along side the `--input` directory, for example, if your input directory is `./jpg`, the `-output` directory will be at `./jpg-output`.

4. If the results are not perfect, fine-tuning a few parameters may fix the problem. Start the fine-tuning process by viewing the images inside the `-circled` directory. The red circle represents the detected sun location, the green circle represents the detected moon location, if the moon circles are off, follow the steps for `Fine-tuning the moon detection` below.

## Fine-tuning the moon detection

Moon detection errors for images taken near totality or annularity and near the start or end of the eclipse can be safely ignored. To solve the problem that the angle of moon is ambiguous at totality or annularity, we group the images where the distance of the sun and moon is smaller than the `--max_phase_rotation_group_threshold` (default: 30 pixels), and assume that the require rotation if the same for the whole group, and the required rotation of the image with a distance just greater than the given value is used for the whole group. Moon detection for images near the start or end of the eclipse may be hard to get right, we group the images where the shadowed width is less than the `--min_phase_rotation_group_threshold` (default: 100 pixels), and assume that the require rotation if the same for the images in the same group.

1. Try smaller or larger values of `--moon_radius_mod` until the size of the green circle of images inside the `-circled` directory best matches the moons shadow in the image.

2. Check the images inside the `-moon-binary` directory, and see if there are any images with fuzzy moon edges, increase or decrease the `--moon_threshold_mod` (default: 25) until the edges of the moon are nice and clean like the example.

# Complete list of arguments
```
usage: solar-eclipse-timelapse-aligner [-h] [-v] --input INPUT [--sun SUN]
                                       [--sun_radius SUN_RADIUS]
                                       [--sun_threshold SUN_THRESHOLD]
                                       [--fix_angle FIX_ANGLE]
                                       [--moon_radius_mod MOON_RADIUS_MOD]
                                       [--moon_threshold_mod MOON_THRESHOLD_MOD]
                                       [--max_phase_rotation_group_threshold THRESHOLD]
                                       [--min_phase_rotation_group_threshold THRESHOLD]
                                       [--filter]
                                       [--clipped_cutoff CLIPPED_CUTOFF]
                                       [--empty_cutoff EMPTY_CUTOFF]
                                       [--workers WORKERS]
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
                        Enables moon angle stabilization if given a non-
                        negative value in degrees, do not enable if processing
                        partial solar eclipse images. (default: -1)
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
