
#include <fstream>
#include <stdio.h>


#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <opencv2/optflow.hpp>

#include <opencv2/video/tracking.hpp>



#include "helpers/OpenCVHelper.hpp"
#include "helpers/optical_flow_helpers.hpp"
#include "helpers/util.hpp"

#include "descriptors/Descriptors.hpp"

using namespace cv;
using namespace std;





int main(int argc, char* argv[] )
{


    //capture properties
    const float real_width_mm = 1481.f;
    const float time_between_imgs_ms = 20.0f;

    uint32_t default_num_imgs_to_process = 0;

    std::string outpath = "../output/maxima2.png";

    std::vector<std::vector<float> > all_path_vals;

    if (cmd_option_exists(argv, argv+argc, "-h")
        || !cmd_option_exists(argv, argv+argc, "-dirb")
        || !cmd_option_exists(argv, argv+argc, "-diro")
        || argc < 3){
        std::cout << "App for tracking flow at min max points with SIFT features\n\n"
                  << "flags:\n" 
                  << "-diro: directory containing original sequence of images\n" 
                  << "-dirb: directory containing blurred image sequence\n" 
                  << "-o: output file path\n"
                  << "-n: number of images from directory to use \n"
                  << std::endl; 

        return 0;
    } 

    if (cmd_option_exists(argv, argv+argc, "-o")){
        outpath = get_cmd_option(argv, argv+argc, "-o");
    } 
    else {
        std::cout << "\nWARNING: no output path specified (use -o)\n\n";
    }

    // load original images
    const std::string og_in_dir = get_cmd_option(argv, argv+argc, "-diro");
    std::cout << "Searching for images in directory: " << og_in_dir << std::endl;
    const std::vector<std::string> og_img_paths = get_file_paths_from_directory(og_in_dir, ".png");

    std::cout << "Original images, in order: \n";
    for (auto path : og_img_paths) std::cout << path << std::endl;
    std::cout << "Total: " << og_img_paths.size() << " images " << std::endl;




    // load blurred images
    const std::string blur_in_dir = get_cmd_option(argv, argv+argc, "-dirb");
    std::cout << "Searching for images in directory: " << blur_in_dir << std::endl;
    const std::vector<std::string> blur_img_paths = get_file_paths_from_directory(blur_in_dir, ".png");

    std::cout << "Blurred images, in order: \n";
    for (auto path : blur_img_paths) std::cout << path << std::endl;
    std::cout << "Total: " << blur_img_paths.size() << " images " << std::endl;


    // check numbers of images
    if (blur_img_paths.size() != og_img_paths.size()){
        std::cout << "Must be the same number of original and blurred images!" << std::endl;
        return 0;
    } 
    default_num_imgs_to_process = blur_img_paths.size();
    if (cmd_option_exists(argv, argv+argc, "-n")){
        const uint32_t requested_num_imgs = atoi(get_cmd_option(argv, argv+argc, "-n"));
        default_num_imgs_to_process = std::min(default_num_imgs_to_process, requested_num_imgs);
    }
    const uint32_t num_imgs_to_process = default_num_imgs_to_process;



    std::vector<Mat> descriptor_array (num_imgs_to_process);

    for (uint32_t i = 0; i < num_imgs_to_process; ++i){

      // create grey images for min/max detection
      Mat blr_img = imread(blur_img_paths[i]);
      Mat og_img  = imread(og_img_paths[i]);

      Mat blr_img_grey;
      cvtColor(blr_img, blr_img_grey, cv::COLOR_RGB2GRAY);

      std::vector<cv::Point> points = find_min_max_keypoints_in_image (blr_img_grey);

      calculate_descriptors_at_points(points, og_img, descriptor_array[i], outpath + "_" + std::to_string(i) + ".png");

    }


	return 0;

}