#include <stdio.h>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <opencv2/optflow.hpp>


#include "OpenCVHelper.hpp"
#include "optical_flow_helpers.hpp"
#include "util.hpp"


using namespace cv;



int main(int argc, char* argv[] )
{


    //capture properties
    // const float real_width_mm = 1481.f;
    // const float time_between_imgs_ms = 20.0f;

    // optical flow settings
    // uint32_t OF_TYPE = 0;

    // bool NORMALISE = false;
    // bool CREATE_LEGEND = false;
    std::string outpath = "";

    if (cmd_option_exists(argv, argv+argc, "-h")
        || !cmd_option_exists(argv, argv+argc, "-f")
        ){
        std::cout << "Visualise optical flow with arrows\n\n"
                  << "flags:\n" 
                  << "-f: input optical flow image\n" 
                  << "-o: output file path\n" 
                  << std::endl; 

        return 0;
    } 

    std::string img_path = get_cmd_option(argv, argv+argc, "-f");

    if (cmd_option_exists(argv, argv+argc, "-o")){
        outpath = get_cmd_option(argv, argv+argc, "-o");
    } 
    else {
        std::cout << "\nWARNING: no output path specified (use -o)\n\n";
    }

	std::cout << "Reading matrix from " << img_path << std::endl;
	cv::FileStorage file(img_path, cv::FileStorage::READ);

	Mat of_image;
	file["mean_of"] >> of_image;
	
    // load image
    // Mat of_image = imread(img_path);
    // const uint32_t img_width = of_image.cols;
    // const uint32_t img_height = of_image.rows;


    // test mean image
    Mat flow_parts[3];
    split(of_image, flow_parts);
    Mat magnitude, angle, magn_norm;

    printMatInfo(flow_parts[0]);

    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

    Mat out_img = flow_parts[0];
    mats_to_hsv(flow_parts[0], flow_parts[1], 10, out_img);
    
	imwrite( "../output/test_load.png", out_img );



return 0;

    // std::cout << "Calculate per-pixel magnitude and angle" << std::endl;

    // // Mat flow_parts[2];
    // // split(of_image, flow_parts);
    // // Mat magnitude, angle, magn_norm;

    // // cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);


    // // downsample images to get cell means
    // const uint32_t cell_size = 20;
    // const uint32_t reduced_height = img_height / 20;
    // const uint32_t reduced_width = img_width / 20;

    // // Mat ds_magnitude;
    // // Mat ds_angle;



    // // resize(magnitude, ds_magnitude, Size(reduced_width, reduced_height),0,0,INTER_NEAREST);
    // // resize(angle, ds_angle, Size(reduced_width, reduced_height),0,0,INTER_NEAREST);
    
    // printMatInfo(of_image, "of result");

    // Mat ds_of (Size (reduced_width, reduced_height), CV_8UC3);
    // resize(of_image, ds_of, Size(reduced_width, reduced_height),0,0,INTER_NEAREST);

    // std::cout << "Resized Image\n";

    // Mat ds_parts[3];
    // split(ds_of, ds_parts);
    // // Mat ds_magnitude, ds_angle, magn_norm;

    // Mat out_img;
    // mats_to_hsv(ds_parts[0], ds_parts[1], 10, out_img);




    // // const float pixel_movement_shown_limit = 10.f;

    // // // calculate and report min and max magnitude before normalisation
    // // double min_px_movement, max_px_movement;
    // // cv::minMaxLoc(magnitude, &min_px_movement, &max_px_movement);
    // // std::cout << "Min magnitude = " << min_px_movement << std::endl;
    // // std::cout << "Max magnitude = " << max_px_movement << std::endl;


    // // // calculate max displacement and speed
    // // const float mm_per_pixel = real_width_mm / img_width;
    // // const float max_displacement = max_px_movement * mm_per_pixel;
    // // const float max_speed = max_displacement / time_between_imgs_ms;

    // // std::cout << "Max displacement = " << max_displacement << std::endl;
    // // std::cout << "Max speed        = " << max_speed << std::endl;




    // if ("" != outpath){
    //     std::cout << "Writing image to " << outpath << std::endl;
    //     imwrite( outpath, out_img );
    // } 


    // // #if !__APPLE__
    // //     cv::imshow("optical flow", out_img);
    // //     waitKey(0);
    // // #endif


    // return 0;
}