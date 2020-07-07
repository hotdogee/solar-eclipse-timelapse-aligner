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
4. The size (in pixels) of the sun in the images needs to be constant, this should be true unless you changed the focus length during the shooting session.
5. The sun should be fully visible in all images, in other words, it should not be partially cut off at the edges. This program has a clipped sun filter, when enabled, will filter out images with edge pixels over a certain brightness level.

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

1. Copy the eclipse images in JPG format to a new input directory. Develop your RAW images into JPGs first if you shot RAW.


# Parameters to tune if the defaults are not perfect
