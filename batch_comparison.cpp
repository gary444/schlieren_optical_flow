
#include <dirent.h>
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


using namespace cv;

// app calculates the optical flow between each tepmorally adjacent pair of images in an image sequence
// outputs the mean optical flow across the whole sequence

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

    std::vector<std::vector<float> > all_path_vals;

    if (cmd_option_exists(argv, argv+argc, "-h")
        || !cmd_option_exists(argv, argv+argc, "-dir")
        || argc < 3){
        std::cout << "Optical flow batch processing app\n\n"
                  << "flags:\n" 
                  << "-dir: directory containing a sequence of images to be processed to find mean optical flow:\n" 
                  << "-t: optical flow type:\n" 
                  << "\t0: DenseRLOFOpticalFlow\n" 
                  << "\t1: DualTVL1OpticalFlow\n" 
                  << "-o: output file path\n" 
                  << "-norm: normalise magnitude image \n"
                  << "-legend: create legend image to help interpret optical flow results \n"
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

    if (cmd_option_exists(argv, argv+argc, "-norm")){
        std::cout << "normalisation active\n";
        NORMALISE = true;
    } 

    if (cmd_option_exists(argv, argv+argc, "-t")){
        OF_TYPE = atoi(get_cmd_option(argv, argv+argc, "-t"));
    } 

    CREATE_LEGEND = cmd_option_exists(argv, argv+argc, "-legend");

    // load images
    const std::string inDirectory = get_cmd_option(argv, argv+argc, "-dir");
    std::cout << "Loading images from directory:" << inDirectory << std::endl;

    //check directory for image files
    std::vector<std::string> img_paths;
    if (auto dir = opendir( inDirectory.c_str() )) {
        while (auto f = readdir(dir)) {
            if (!f->d_name || f->d_name[0] == '.')
                continue; // Skip everything that starts with a dot
            std::string fname = inDirectory + "/" + f->d_name;
            
            // std::cout << fname << std::endl;
            // file type checking if necessary
            // if (fname.substr(fname.length()-4) == suffix){
            img_paths.push_back(fname);
        }
        closedir(dir);
        std::sort( img_paths.begin(), img_paths.end() );
    }

    if (0 == img_paths.size()){
        std::cout << "Error: no files found in this folder\n";
        return 0;
    }

    std::cout << "Images for comparison, in order: \n";
    for (auto path : img_paths) std::cout << path << std::endl;
    std::cout << "Total: " << img_paths.size() << " images " << std::endl;

    // load a test image to get size
    Mat test = imread(img_paths[0]);
    const uint32_t img_width = test.cols;
    const uint32_t img_height = test.rows;

    // optical flow -----------------------------------------------------------------

    // for each pair of images, get the 2 components and add to running sums
    Mat accum_of(cv::Size(img_width, img_height), CV_32FC2, Scalar(0,0));


    Mat of_result = accum_of;
    Mat last_mat = test; // already loaded first image to get dimensions


    Mat of_accum_parts[2];
    of_accum_parts[0] = Mat(cv::Size(img_width, img_height), CV_32FC1, Scalar(0,0));
    of_accum_parts[1] = Mat(cv::Size(img_width, img_height), CV_32FC1, Scalar(0,0));


    using namespace cv::optflow;

    int IMAGES_TO_USE = img_paths.size();
    if (cmd_option_exists(argv, argv+argc, "-n")){
        IMAGES_TO_USE = atoi(get_cmd_option(argv, argv+argc, "-n"));
        IMAGES_TO_USE = std::min(int(img_paths.size()), std::max(IMAGES_TO_USE, 0));
    } 

    for (int i = 1; i < IMAGES_TO_USE; ++i){


        std::cout << "Comparison " << i << std::endl;

        Mat mat_to_compare = imread(img_paths[i]);


        // Mat mat_to_compare = imread(img_paths[1]);


        if (OF_TYPE == 0){

            std::cout << "DenseRLOFOpticalFlow...\n\n";

            Ptr<DenseRLOFOpticalFlow> of = DenseRLOFOpticalFlow::create();
            of->setForwardBackward(8.f);

            of->calc(last_mat, mat_to_compare, of_result);
        }
        else if (OF_TYPE == 1) {

            std::cout << "DualTVL1OpticalFlow...\n\n";

            // convert to B+W
            Mat image1_bw, image2_bw;
            cvtColor(mat_to_compare, image2_bw, cv::COLOR_RGB2GRAY);
            cvtColor(last_mat, image1_bw, cv::COLOR_RGB2GRAY);

            Ptr<DualTVL1OpticalFlow> of = DualTVL1OpticalFlow::create();
            of->calc(image1_bw, image2_bw, of_result);

        }
        else if (OF_TYPE == 2) {


            Mat image1_bw, image2_bw;
            cvtColor(mat_to_compare, image2_bw, cv::COLOR_RGB2GRAY);
            cvtColor(last_mat, image1_bw, cv::COLOR_RGB2GRAY);

            Ptr<FarnebackOpticalFlow> of = cv::FarnebackOpticalFlow::create();
            of->calc(image1_bw, image2_bw, of_result);


        }

// printMatInfo(accum_of, "accum");
printMatInfo(of_result, "of result");

        Mat flow_parts[2];
        split(of_result, flow_parts);

        of_accum_parts[0] += flow_parts[0];
        of_accum_parts[1] += flow_parts[1];


        std::cout << "result " << of_result.at<float>(0,0,0) << std::endl;

        // accumulate of result
        // accum_of = accum_of + of_result;

        std::cout << "accum " << of_accum_parts[0].at<float>(0,0) << std::endl;


        // get magnitude along path
        std::vector<float> path_vals =  get_flow_magnitude_on_path(flow_parts[0], flow_parts[1], real_width_mm, time_between_imgs_ms);
        all_path_vals.push_back(path_vals);

        last_mat = mat_to_compare;
    }

printMatInfo(accum_of, "accum of");


    

    std::cout << "Calculating mean... " << std::endl;
    
    of_accum_parts[0] = of_accum_parts[0] / float(IMAGES_TO_USE-1); // divide by number of comparisons, not images
    of_accum_parts[1] = of_accum_parts[1] / float(IMAGES_TO_USE-1); // divide by number of comparisons, not images





    std::cout << "Calculate per-pixel magnitude and angle" << std::endl;

    Mat magnitude, angle, magn_norm;
    cartToPolar(of_accum_parts[0], of_accum_parts[1], magnitude, angle, true);


    const float pixel_movement_shown_limit = 10.f;

    // calculate and report min and max magnitude before normalisation
    double min_px_movement, max_px_movement;
    cv::minMaxLoc(magnitude, &min_px_movement, &max_px_movement);
    std::cout << "Min magnitude = " << min_px_movement << std::endl;
    std::cout << "Max magnitude = " << max_px_movement << std::endl;


    // calculate max displacement and speed
    const float mm_per_pixel = real_width_mm / img_width;
    const float max_displacement = max_px_movement * mm_per_pixel;
    const float max_speed = max_displacement / time_between_imgs_ms;

    std::cout << "Max displacement = " << max_displacement << std::endl;
    std::cout << "Max speed        = " << max_speed << std::endl;


    Mat out_img;
    if (NORMALISE){
        mats_to_hsv(of_accum_parts[0], of_accum_parts[1], max_px_movement, out_img);
    }
    else {
        mats_to_hsv(of_accum_parts[0], of_accum_parts[1], pixel_movement_shown_limit, out_img);
    }


    if (CREATE_LEGEND){
        const float leg_max_mag = NORMALISE ? max_px_movement : pixel_movement_shown_limit;
        const float max_speed_legend = NORMALISE ? max_speed : (pixel_movement_shown_limit*mm_per_pixel/time_between_imgs_ms);
        const std::string legend_path = "../images/mean_legend_" + std::to_string(leg_max_mag) + ".png";
        create_legend(legend_path, leg_max_mag, max_speed_legend);
    }

    // printMatInfo(mean_of, "mean of result");

    if ("" != outpath){

        std::cout << "Writing image to " << outpath << std::endl;
        imwrite( outpath, out_img );

        // store optical flow directly as an image
        std::string result_path = outpath.substr(0, outpath.length() - 4) + "_result.mat";
        // Mat third_channel ( Size(mean_of.cols, mean_of.rows), mean_of.type() );

        Mat mean_of;
        std::vector<cv::Mat> mean_of_channels;
        mean_of_channels.push_back(of_accum_parts[0]);
        mean_of_channels.push_back(of_accum_parts[1]);
        cv::merge(mean_of_channels, mean_of);

        // imwrite( result_path, mean_of );

        std::cout << "Writing result to " << result_path << std::endl;
        cv::FileStorage file(result_path, cv::FileStorage::WRITE);
        file << "mean_of" << mean_of;


        // imwrite(result_path, mean_of);

    } 


    // save path values
    std::ofstream path_val_file("../output/path_vals.txt");

    for (auto list : all_path_vals)
    {
        for (auto v : list)
        {
            path_val_file << v << ", ";
        }
        path_val_file << std::endl;
    }
    path_val_file.close();


    #if !__APPLE__
        cv::imshow("optical flow", out_img);
        waitKey(0);
    #endif


    return 0;
}