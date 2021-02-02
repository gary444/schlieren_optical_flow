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

#include <opencv2/cudaoptflow.hpp>



#include "helpers/OpenCVHelper.hpp"
#include "helpers/optical_flow_helpers.hpp"
#include "helpers/util.hpp"



using namespace cv;

int main(int argc, char* argv[] )
{


    //capture properties
    const float real_width_mm = 1481.f;
    const float time_between_imgs_ms = 20.0f;

    // optical flow settings
    // uint32_t OF_TYPE = 0;
    bool NORMALISE = false;

    bool CREATE_LEGEND = false;

    std::string outpath = "";
    std::string img1path = "../images/lena_sq1.png";
    std::string img2path = "../images/lena_sq2.png";

    if (cmd_option_exists(argv, argv+argc, "-h")
        || argc < 3){
        std::cout << "Optical flow app\n\n"
                  << "args 1 and 2 should be images to compare\n\n"
                  << "flags:\n" 
                  // << "-t: optical flow type:\n" 
                  // << "\t0: DenseRLOFOpticalFlow\n" 
                  // << "\t1: DualTVL1OpticalFlow\n" 
                  << "-o: output file path\n" 
                  << "-norm: normalise magnitude image \n"
                  << "-legend: create legend image to help interpret optical flow results \n"
                  << "-flo: write flow as float binary with name \n"
                  << "-show: show result in popup window \n"
                  << std::endl; 

        return 0;
    } 

    if (argc > 1) img1path = argv[1];
    if (argc > 2) img2path = argv[2];

    if (cmd_option_exists(argv, argv+argc, "-o")){
        outpath = get_cmd_option(argv, argv+argc, "-o");
    } 
    if (cmd_option_exists(argv, argv+argc, "-norm")){
        std::cout << "normalisation active\n";
        NORMALISE = true;
    } 
    // if (cmd_option_exists(argv, argv+argc, "-t")){
    //     OF_TYPE = atoi(get_cmd_option(argv, argv+argc, "-t"));
    // } 

    CREATE_LEGEND = cmd_option_exists(argv, argv+argc, "-legend");

    bool SHOW_RESULT = cmd_option_exists(argv, argv+argc, "-show");

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
    Mat of_result (image1.rows, image1.cols, CV_32FC1);


    std::cout << "CUDA accelerated Brox Optical Flow...\n\n";

    Mat image1_bw, image2_bw;

    cvtColor(image1, image1_bw, cv::COLOR_RGB2GRAY);
    cvtColor(image2, image2_bw, cv::COLOR_RGB2GRAY);

    image1_bw.convertTo(image1_bw, CV_32FC1, 1.0/255.0);
    image2_bw.convertTo(image2_bw, CV_32FC1, 1.0/255.0);

    // create parameters:
	// double alpha=0.197, double gamma=50.0, double scale_factor=0.8, int inner_iterations=5, int outer_iterations=150, int solver_iterations=10
    Ptr<cv::cuda::BroxOpticalFlow> of = cv::cuda::BroxOpticalFlow::create();
    
    // convert to GPU mats
    
    cuda::GpuMat gpu_img_1 = cuda::GpuMat(image1_bw);
    cuda::GpuMat gpu_img_2 = cuda::GpuMat(image2_bw);
	cuda::GpuMat gpu_result = cuda::GpuMat(of_result); 

    of->calc(gpu_img_1, gpu_img_2, of_result);

    printMatInfo(of_result, "optical flow result");


    // visualise 

    std::cout << "Calculate per-pixel magnitude and angle" << std::endl;

    Mat flow_parts[2];
    split(of_result, flow_parts);
    Mat magnitude, angle, magn_norm;

    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);


    const float pixel_movement_shown_limit = 10.f;

    // rcalculate and eport min and max magnitude before normalisation
    double min_px_movement, max_px_movement;
    cv::minMaxLoc(magnitude, &min_px_movement, &max_px_movement);
    std::cout << "Min magnitude = " << min_px_movement << std::endl;
    std::cout << "Max magnitude = " << max_px_movement << std::endl;


    // calculate max displacement and speed
    const float mm_per_pixel = real_width_mm / image1.cols;
    const float max_displacement = max_px_movement * mm_per_pixel;
    const float max_speed = max_displacement / time_between_imgs_ms;

    std::cout << "Max displacement = " << max_displacement << std::endl;
    std::cout << "Max speed        = " << max_speed << std::endl;


    Mat out_img;
    if (NORMALISE){
        mats_to_hsv(flow_parts[0], flow_parts[1], max_px_movement, out_img);
    }
    else {
        mats_to_hsv(flow_parts[0], flow_parts[1], pixel_movement_shown_limit, out_img);
    }


    if (CREATE_LEGEND){
        const float leg_max_mag = NORMALISE ? max_px_movement : pixel_movement_shown_limit;
        const float max_speed_legend = NORMALISE ? max_speed : (pixel_movement_shown_limit*mm_per_pixel/time_between_imgs_ms);
        const std::string legend_path = "../images/legend_" + std::to_string(leg_max_mag) + ".png";
        create_legend(legend_path, leg_max_mag, max_speed_legend);
    }
    




    if ("" != outpath){
        std::cout << "Writing image to " << outpath << std::endl;
        imwrite( outpath, out_img );
    } 

    if (cmd_option_exists(argv, argv+argc, "-flo")){
        const std::string ofilepath = get_cmd_option(argv, argv+argc, "-flo");
        // write to file

        std::ofstream outfile(ofilepath, std::ios::binary);

        if (!outfile.is_open()){
            std::cout << "Error creating file " << ofilepath << std::endl;
            return false;
        }
        uint32_t width  = of_result.cols;
        uint32_t height = of_result.rows;

        outfile.write(reinterpret_cast<char*> (&width), sizeof(uint32_t));
        outfile.write(reinterpret_cast<char*> (&height), sizeof(uint32_t));
        outfile.write(reinterpret_cast<char*> (of_result.ptr<float>(0,0)), sizeof(float) * 2 * width * height);

        outfile.close();
        if (!outfile.good()){
            std::cout << "Error creating file " << ofilepath << std::endl;
            return false;
        }

        std::cout << "Successfully saved Optical Flow\n";
        return true;

    }


#if !__APPLE__
    if (SHOW_RESULT){
        cv::imshow("optical flow", out_img);
        waitKey(0);
    }
#endif


    return 0;
}