
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
#include <opencv2/features2d.hpp>



#include "helpers/OpenCVHelper.hpp"
#include "helpers/optical_flow_helpers.hpp"
#include "helpers/util.hpp"

#include "feature_matching/Descriptors.hpp"
#include "feature_matching/Matching.hpp"

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
    if (0 == blur_img_paths.size()){
        std::cout << "No images found" << std::endl;
        return 0;
    }


    default_num_imgs_to_process = blur_img_paths.size();
    if (cmd_option_exists(argv, argv+argc, "-n")){
        const uint32_t requested_num_imgs = atoi(get_cmd_option(argv, argv+argc, "-n"));
        default_num_imgs_to_process = std::min(default_num_imgs_to_process, requested_num_imgs);
    }
    const uint32_t num_imgs_to_process = default_num_imgs_to_process;



    // get size of first image
    Mat first_img = imread(blur_img_paths[0]);
    const uint32_t image_w = first_img.rows;
    const uint32_t image_h = first_img.cols;
    const float mm_per_pixel = real_width_mm / image_w;


    std::vector<std::vector<cv::KeyPoint> > keypoint_arrays (num_imgs_to_process /*std::vector<cv::KeyPoint>()*/);
    std::vector<Mat> descriptor_arrays (num_imgs_to_process);

    // TODO could use below function for batch processing
  // void cv::Feature2D::compute   (   InputArrayOfArrays    images, std::vector< std::vector< KeyPoint > > &    keypoints, OutputArrayOfArrays   descriptors )   


    for (uint32_t i = 0; i < num_imgs_to_process; ++i){

      std::cout << "Finding descriptors in image: " << i << std::endl;

      // create grey images for min/max detection
      Mat blr_img = imread(blur_img_paths[i]);
      Mat og_img  = imread(og_img_paths[i]);

      Mat blr_img_grey;
      cvtColor(blr_img, blr_img_grey, cv::COLOR_RGB2GRAY);

      std::vector<cv::Point> points = find_min_max_keypoints_in_image (blr_img_grey);
      keypoint_arrays[i] = convert_points_to_keypoints(points);

      calculate_descriptors_at_points(keypoint_arrays[i], og_img, descriptor_arrays[i], outpath + "_" + std::to_string(i) + ".png");

    }

    const float max_displacement_mm = 300.0;
    const float max_displacement_px = max_displacement_mm / mm_per_pixel;

    const float min_highlight_speed_ms = 0.5f;
    const float max_highlight_speed_ms = 1.0f;

    const float min_highlight_disp_m = min_highlight_speed_ms * (time_between_imgs_ms / 1000.f);
    const float max_highlight_disp_m = max_highlight_speed_ms * (time_between_imgs_ms / 1000.f);

    const float min_highlight_distance_px = min_highlight_disp_m * 1000.f / mm_per_pixel;
    const float max_highlight_distance_px = max_highlight_disp_m * 1000.f / mm_per_pixel;



    // compare images (adjacent frames)
    for (uint32_t i = 1; i < num_imgs_to_process; ++i){

      std::cout << "Finding matches between images " << i-1 << " and " << i << std::endl;

      std::vector<DMatch> matches = match_descriptors(descriptor_arrays[i-1], descriptor_arrays[i]);

      Mat og_imgA  = imread(og_img_paths[i-1]);
      Mat og_imgB  = imread(og_img_paths[i]);


      matches = filter_matches_by_displacement_in_pixels(matches, keypoint_arrays[i-1], keypoint_arrays[i], max_displacement_px);

      // Mat out_img_matches;
      // drawMatches( og_imgA, keypoint_arrays[i-1], og_imgB, keypoint_arrays[i], matches, out_img_matches, 
      //              Scalar(0,255,0), // match colour
      //              Scalar(128,0,0) // no-match colour
      //              );
      // imwrite("../../../images/output/sift_descriptors/matches_" + std::to_string(i-1) + "_" + std::to_string(i) + ".png", out_img_matches);


      // plot matches on the same images as arrows
      Mat out_arrow_img = og_imgB.clone();
      draw_matches_as_arrows(matches, keypoint_arrays[i-1], keypoint_arrays[i], out_arrow_img, min_highlight_distance_px, max_highlight_distance_px);
      imwrite("../../../images/output/sift_descriptors/movement_1f_" + std::to_string(i-1) + "_" + std::to_string(i) + ".png", out_arrow_img);


    }

    // compare images  (2 frame step)
    for (uint32_t i = 2; i < num_imgs_to_process; ++i){

      std::cout << "Finding matches between images " << i-2 << " and " << i << std::endl;

      std::vector<DMatch> matches = match_descriptors(descriptor_arrays[i-2], descriptor_arrays[i]);

      Mat og_imgA  = imread(og_img_paths[i-2]);
      Mat og_imgB  = imread(og_img_paths[i]);

      matches = filter_matches_by_displacement_in_pixels(matches, keypoint_arrays[i-2], keypoint_arrays[i], max_displacement_px);

      // plot matches on the same images as arrows
      Mat out_arrow_img = og_imgB.clone();
      draw_matches_as_arrows(matches, keypoint_arrays[i-2], keypoint_arrays[i], out_arrow_img, min_highlight_distance_px*2, max_highlight_distance_px*2);
      imwrite("../../../images/output/sift_descriptors/movement_2f_" + std::to_string(i-2) + "_" + std::to_string(i) + ".png", out_arrow_img);


    }


    std::cout << "Highlighted matches with speed between " << min_highlight_speed_ms << "  and " << max_highlight_speed_ms << " m/s" << std::endl; 
    std::cout << "Corresponds to displacement of between " << min_highlight_disp_m << "  and " << max_highlight_disp_m
       << " m, when time between frames is " << time_between_imgs_ms << " ms" << std::endl; 



	return 0;

}