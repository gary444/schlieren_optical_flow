#include <stdio.h>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

// #include <opencv2/cudaoptflow.hpp>

#include <opencv2/optflow.hpp>



#include "helpers/OpenCVHelper.hpp"
#include "helpers/optical_flow_helpers.hpp"
#include "helpers/util.hpp"



void generate_grid_points(std::vector<cv::Point2f>& p0, const uint32_t width, const uint32_t height, const uint32_t px_spacing ) {
    for (int y = px_spacing/2; y < height; y+=px_spacing){
        for (int x = px_spacing/2; x < width; x+=px_spacing){
            p0.push_back( cv::Point2f(x,y) );
        }
    }
}

using namespace cv;

int main(int argc, char* argv[] )
{

    using namespace std;


    //capture properties
    const float real_width_mm = 1481.f;
    const float time_between_imgs_ms = 20.0f;

    // optical flow settings
    uint32_t OF_TYPE = 0;
    bool NORMALISE = false;

    bool CREATE_LEGEND = false;

    std::string outpath = "";
    std::string img1path = "";
    std::string img2path = "";

    if (cmd_option_exists(argv, argv+argc, "-h")
        || argc < 3){
        std::cout << "Optical flow app: Lukas Kanade sparse optical flow\n\n"
                  << "args 1 and 2 should be images to compare\n\n"
                  << "flags:\n" 
                  << "-o: output file path\n" 
                  // << "-norm: normalise magnitude image \n"
                  // << "-legend: create legend image to help interpret optical flow results \n"
                  // << "-flo: write flow as float binary with dimensions \n"
                  << std::endl; 

        return 0;
    } 

    img1path = argv[1];
    img2path = argv[2];

    if (cmd_option_exists(argv, argv+argc, "-o")){
        outpath = get_cmd_option(argv, argv+argc, "-o");
    } 

    const bool USE_SUGGESTED_CORNER_POINTS = false;



    // load images
    std::cout << "Comparing:\n\t" << img1path << "\n\t" << img2path << "\n\n";
    Mat image1, image2;
    image1 = imread( img1path, 1 );
    image2 = imread( img2path, 1 );

    if ( !image1.data && !image2.data ){
        printf("No image data \n");
        return -1;
    }
    printMatInfo(image1, "image1");
    printMatInfo(image2, "image2");


    // optical flow -----------------------------------------------------------------
    using namespace cv::optflow;

    // convert to single channel
    Mat image1_bw, image2_bw;
    cvtColor(image1, image1_bw, cv::COLOR_RGB2GRAY);
    cvtColor(image2, image2_bw, cv::COLOR_RGB2GRAY);


    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(image1.size(), image1.type());

    vector<Point2f> p0, p1;


    std::cout << "Creating features to track...\n\n";
    
    if (USE_SUGGESTED_CORNER_POINTS){
        // image, corners, maxCorners, qualityLevel, minDistance between corners, ROI mask,  blockSize for ovariance matrix, useHarrisDetector, free param of harris detector
        goodFeaturesToTrack(image1_bw, p0, 100, 0.01, 7, Mat(), 7, false, 0.04);
    } else {
        const uint32_t spacing = 20;
        generate_grid_points(p0, image1.cols, image1.rows, spacing);
    }


    std::cout << "Calculating optical flow...\n\n";
    std::vector<uint8_t> status;
    std::vector<float> err;

    // termination criteria
    // count = num iterations, eps = change in parameters at which iteration stops
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

    //prevImg, nextImg, prevPts, nextPts, status (flow found, 1 or 0), 
    // err (amount of error for each point), size of search window at each pyramid level, maxLevel, 
    // termination criteria
    calcOpticalFlowPyrLK(image1_bw, image2_bw, p0, p1, status, err, Size(20,20), 2, criteria);


    std::cout << "Drawing points...\n\n";

    std::vector<cv::Scalar> colors = create_random_colours(100);

    // std::vector<Point2f> good_new;
    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            line(mask,p1[i], p0[i],  colors[i % 100], 2);
            circle(image2, p1[i], 5, colors[i % 100], -1);
        }
    }

    std::cout << "finished drawing points...\n\n";



    Mat out_img;
    add(image2, mask, out_img);




    if ("" != outpath){
        std::cout << "Writing image to " << outpath << std::endl;
        imwrite( outpath, out_img );
    } 





    imshow("Frame", out_img);
    waitKey(0);



    return 0;
}