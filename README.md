


Single comparison example:
./ofComparison ../../../PIVlab/pngimages/Highflow\ 60L-min.MOV6076.png ../../../PIVlab/pngimages/Highflow\ 60L-min.MOV6077.png -t 0 -o ../output/retest_8_8_20_a.png


Batch processing of image sequence example:

./ofBatchComparison -dir  ../../../PIVlab/pngimages/ -t 0 -o ../output/retest_8_8_20.png

 - this outputs a mean image using the path given by the -o flag, as well as a matrix containing the mean optical flow values in the x and y dimensions
 - optical flow values are pure means, not normalised
 - image may be normalised depending on flag provided




Visualisation of mean optical flows with arrows

Example:
./visMean -f ../output/retest_8_8_20_result.mat -o ../output/retest_vismean.png 

