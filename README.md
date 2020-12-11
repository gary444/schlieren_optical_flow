
# Various apps for testing flow estimation methods for Schlieren images

## Apps:

### minima_and_maxima_sift_feature_matching

Requires a directory containing Schlieren images, and another directory containing blurred Schlieren images as input. Blur can be done using a third-party software like [ImageMagick](https://imagemagick.org/script/mogrify.php).

Key points are found at the minima and maxima of the blurred images. SIFT features are created at these keypoints, and the best feature matches are used to estimate flow between pairs of adjacent images.

### image_pair_optical_flow_estimation_and_vis
	
Estimates optical flow for a pair of given images. Allows selection of OF type.

### batch_optical_flow_estimation

Estimates optical flow for an ordered set of given images in a directory. 

### visualise_mean_optical_flow

Visualises an optical flow field with an arrow in each grid cell.  

### mean_flow_on_path

Calculates mean flow magnitude on a line across an image.

### of_lukas_kanade

Detect keypoints and track with Lukas-Kanade optical flow.



## Dependencies

OpenCV is required. Note that nonfree features (e.g. SIFT) are also used, meaning that the opencv_contrib github repo is required, and the library should be built with the ENABLE_NONFREE switch on.

CMake is required. Cmake Command line instructions below.

## Building using cmake command line

 * Open terminal in root directory
 * enter **mkdir build**
 * enter **cd build**
 * enter **ccmake ..**
 * press C to configure
 * press G to generate makefiles
 * enter **make**

After the above instructions, apps will be built in the **build** folder.

You may need to point CMake to your OpenCV library, by editing the OpenCV_DIR option in the Cmake GUI.







Single comparison example:
./ofComparison image0.png image1.png -t 0 -o out_image.png


Batch processing of image sequence example:

./ofBatchComparison -dir  pngimages/ -t 0 -o output/out.png

 - this outputs a mean image using the path given by the -o flag, as well as a matrix containing the mean optical flow values in the x and y dimensions
 - optical flow values are pure means, not normalised
 - image may be normalised depending on flag provided


Visualisation of mean optical flows with arrows

Example:
./visMean -f ../output/retest_8_8_20_result.mat -o ../output/retest_vismean.png 

