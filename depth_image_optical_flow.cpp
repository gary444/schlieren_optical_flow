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



using namespace cv;

const float min_depth = 0.5f;
const float max_depth = 3.0f;

float normalize_depth(float depth) {
  return (depth - min_depth)/(max_depth - min_depth);
}

Mat load_depth_image_to_mat (const std::string& path, const uint32_t depth_image_width, const uint32_t depth_image_height) {

    std::vector<float> depth_image_data (depth_image_height * depth_image_width);

    std::ifstream infile (path, std::ios::binary);
    infile.read(reinterpret_cast<char*> (depth_image_data.data()), sizeof(float) * depth_image_width * depth_image_height);
    infile.close();

    std::transform(depth_image_data.begin(), depth_image_data.end(), depth_image_data.begin(), [&](const float depth) {
        return normalize_depth(depth);
    });

    Mat img (depth_image_height, depth_image_width, CV_32FC1, depth_image_data.data());
    Mat img_8bit;
    img.convertTo(img_8bit, CV_8UC1, 255.0);


    cv::imshow("depth_image", img_8bit);
    waitKey(0);

    return img_8bit;
}


int main(int argc, char* argv[] )
{


    //capture properties
    const float real_width_mm = 1481.f;
    const float time_between_imgs_ms = 20.0f;

    // optical flow settings
    uint32_t OF_TYPE = 0;
    bool NORMALISE = false;

    bool CREATE_LEGEND = false;

    std::string outpath = "";
    std::string img1path = "../images/lena_sq1.png";
    std::string img2path = "../images/lena_sq2.png";

    if (cmd_option_exists(argv, argv+argc, "-h")
        || argc < 5){
        std::cout << "Optical flow app for depth images\n\n"
                  << "./depth_image_optical_flow [img1path] [img2path] [imgwidth] [imgheight] [options....]\n\n"
                  << "flags:\n" 
                  << "-t: optical flow type:\n" 
                  << "\t0: DenseRLOFOpticalFlow\n" 
                  << "\t1: DualTVL1OpticalFlow\n" 
                  << "-o: output file path\n" 
                  << "-norm: normalise magnitude image \n"
                  << "-legend: create legend image to help interpret optical flow results \n"
                  << "-flo: write flow as float binary with dimensions \n"
                  << std::endl; 

        return 0;
    } 

    img1path = argv[1];
    img2path = argv[2];
    const uint32_t width  = atoi(argv[3]);
    const uint32_t height = atoi(argv[4]);

    if (cmd_option_exists(argv, argv+argc, "-o")){
        outpath = get_cmd_option(argv, argv+argc, "-o");
    } 
    if (cmd_option_exists(argv, argv+argc, "-norm")){
        std::cout << "normalisation active\n";
        NORMALISE = true;
    } 
    if (cmd_option_exists(argv, argv+argc, "-t")){
        OF_TYPE = atoi(get_cmd_option(argv, argv+argc, "-t"));
    } 

    CREATE_LEGEND = cmd_option_exists(argv, argv+argc, "-legend");

    // load images
    std::cout << "Comparing:\n\t" << img1path << "\n\t" << img2path << "\n\n";
    // Mat image1in, image2in;
    Mat image1, image2;
    
    image1 = load_depth_image_to_mat(img1path, width, height);
    image2 = load_depth_image_to_mat(img2path, width, height);

    // image1in.convertTo(image1, CV_8UC1, 255.0);
    // image1in.convertTo(image1, CV_8UC1, 255.0);

    if ( !image1.data && !image2.data ){
        printf("No image data \n");
        return -1;
    }
    printMatInfo(image1, "image1");
    printMatInfo(image2, "image2");


    // optical flow -----------------------------------------------------------------
    using namespace cv::optflow;
    Mat of_result (image1.rows, image1.cols, CV_32FC2);

    if (OF_TYPE == 0){

        std::cout << "DenseRLOFOpticalFlow...\n\n";

        Ptr<DenseRLOFOpticalFlow> of = DenseRLOFOpticalFlow::create();
        of->calc(image1, image2, of_result);
    }
    else if (OF_TYPE == 1) {

        std::cout << "DualTVL1OpticalFlow...\n\n";

        Ptr<DualTVL1OpticalFlow> of = DualTVL1OpticalFlow::create();
        of->calc(image1, image2, of_result);

    }
    
    std::cout << "Done." << std::endl;

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
    cv::imshow("optical flow", out_img);
    waitKey(0);
#endif


    return 0;
}