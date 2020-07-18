#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

// #include <opencv2/cudaoptflow.hpp>

#include <opencv2/optflow.hpp>


#include "OpenCVHelper.hpp"



using namespace cv;

int main(int argc, char** argv )
{

    uint32_t OF_TYPE = 0;
    std::string outpath = "";

    Mat image1, image2;
    image1 = imread( "../images/lena_sq1.png", 1 );
    image2 = imread( "../images/lena_sq2.png", 1 );


    if (argc > 1) image1 = imread(argv[1], 1 );
    if (argc > 2) image2 = imread(argv[2], 1 );
    if (argc > 3) OF_TYPE = atoi(argv[3]);
    if (argc > 4) outpath = argv[4];



    if ( !image1.data && !image2.data ){
        printf("No image data \n");
        return -1;
    }


    Mat of_result = image1;

    // std::cout << "image1: " << image1.rows << " x " << image1.cols << std::endl;
    // std::cout << "image2: " << image2.rows << " x " << image2.cols << std::endl;
    // std::cout << "of_result: " << of_result.rows << " x " << of_result.cols << std::endl;

    printMatInfo(image1, "image1");
    printMatInfo(image2, "image2");
    printMatInfo(of_result, "of_result");



    // optical flow -----------------------------------------------------------------
    using namespace cv::optflow;

    if (OF_TYPE == 0){

        std::cout << "DenseRLOFOpticalFlow...\n\n";

        Ptr<DenseRLOFOpticalFlow> of = DenseRLOFOpticalFlow::create();
        of->calc(image1, image2, of_result);
    }
    else if (OF_TYPE == 1) {

        std::cout << "DualTVL1OpticalFlow...\n\n";

        Mat image1_bw, image2_bw;

        cvtColor(image1, image1_bw, cv::COLOR_RGB2GRAY);
        cvtColor(image2, image2_bw, cv::COLOR_RGB2GRAY);

        Ptr<DualTVL1OpticalFlow> of = DualTVL1OpticalFlow::create();
        of->calc(image1_bw, image2_bw, of_result);

    }





    // visualise 

    const float max_pixel_movement_shown = 10.f;

    // create legend
    const uint32_t legend_size = 599;
    const float mag_per_pixel = (2.f*max_pixel_movement_shown) / float(legend_size)  ;

    std::vector<float> range (legend_size);
    for (int i = 0; i < legend_size; ++i){
        range[i] = -max_pixel_movement_shown + i*mag_per_pixel;
    }

    Mat legend_x_flow;
    Mat legend_y_flow;

    cv::repeat(range, legend_size, 1, legend_x_flow);
    cv::transpose(legend_x_flow, legend_y_flow);


    double min, max;
    cv::minMaxLoc(legend_x_flow, &min, &max);
    std::cout << "Min magnitude = " << min << std::endl;
    std::cout << "Max magnitude = " << max << std::endl;
    std::cout << "magnitude divided by " << max_pixel_movement_shown << std::endl;


    printMatInfo(legend_x_flow);
    printMatInfo(legend_y_flow);



    std::cout << "Calculate per-pixel magnitude and angle" << std::endl;

    Mat flow_parts[2];
    split(of_result, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(legend_x_flow, legend_y_flow, magnitude, angle, true);

    // cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    // normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    magn_norm = magnitude / max_pixel_movement_shown;
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    std::cout << "Create HSV image" << std::endl;


    //build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);


    const uint32_t cntr = (legend_size / 2) + 1;
    cv::circle(bgr, Point(cntr, cntr), legend_size/2, Scalar(0,0,0));
    

    if ("" != outpath) imwrite( outpath, bgr );


    imshow("frame2", bgr);








    waitKey(0);
    // if (keyboard == 'q' || keyboard == 27)


    return 0;
}