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
    const float real_width_mm = 1481.f;
    const float time_between_imgs_ms = 20.0f;

    // optical flow settings
    // uint32_t OF_TYPE = 0;

    bool NORMALISE = true;
    // bool CREATE_LEGEND = false;
    std::string outpath = "";

    if (cmd_option_exists(argv, argv+argc, "-h")
        || !cmd_option_exists(argv, argv+argc, "-f")
        ){
        std::cout << "Visualise optical flow with arrows\n\n"
                  << "flags:\n" 
                  << "-f: input mean optical flow matrix (.mat)\n" 
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

    // load image
	std::cout << "Reading matrix from " << img_path << std::endl;
	cv::FileStorage file(img_path, cv::FileStorage::READ);

	Mat of_image;
	file["mean_of"] >> of_image;

    printMatInfo(of_image, "of from file");

	
    const uint32_t img_width = of_image.cols;
    const uint32_t img_height = of_image.rows;

	// downsampling

	const uint32_t ds_square_size = 20;

    const uint32_t reduced_width = img_width / ds_square_size;
    const uint32_t reduced_height = img_height / ds_square_size;





    // test mean image

#if 0

	// downsample magnitude and angle

    Mat flow_parts[2];
    split(of_image, flow_parts);

    Mat magnitude, angle, magn_norm;

    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

    Mat ds_magnitude, ds_angle;

    resize(magnitude, ds_magnitude, Size(reduced_width, reduced_height),0,0,INTER_NEAREST);
    resize(angle, ds_angle, Size(reduced_width, reduced_height),0,0,INTER_NEAREST);


    // upsample again to get blocky effect
    resize(ds_magnitude, magnitude, Size(img_width, img_height),0,0,INTER_NEAREST);
    resize(ds_angle, angle, Size(img_width, img_height),0,0,INTER_NEAREST);

#else

    // downsample x and y components

    Mat ds_of_image;

    resize(of_image, ds_of_image, Size(reduced_width, reduced_height),0,0,INTER_NEAREST);
    // resize(angle, ds_angle, Size(reduced_width, reduced_height),0,0,INTER_NEAREST);


    // upsample again to get blocky effect
    resize(ds_of_image, of_image, Size(img_width, img_height),0,0,INTER_NEAREST);
    // resize(ds_angle, angle, Size(img_width, img_height),0,0,INTER_NEAREST);


    Mat flow_parts[2];
    split(of_image, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);



#endif

    // TODO draw arrows on to image
    // TODO sample optical flow along the line from mouth

    const float scale_factor = 3.f;
    struct Line
    {
    	Point start;
    	Point end;
    };
    std::vector<Line> lines;
    lines.reserve(reduced_height*reduced_width);

    for (int y = 0; y < reduced_height; ++y)
    {
    	for (int x = 0; x < reduced_width; ++x)
    	{
    		// start is centre of block
    		Line new_line;
    		new_line.start = Point( (x+0.5)*ds_square_size, (y+0.5)*ds_square_size );

    		// end is centre plus 'magnitude' pixels in direction 'angle'
    		float magnitude_p = magnitude.at<float>(new_line.start);
    		float angle_p = angle.at<float>(new_line.start);
    		int x_offset = int( magnitude_p * cos(angle_p*M_PI/180.f) * scale_factor);
    		int y_offset = int( magnitude_p * sin(angle_p*M_PI/180.f) * scale_factor);



    		new_line.end = Point(new_line.start.x + x_offset, new_line.start.y + y_offset);

    		lines.push_back(new_line);
    	}
    }


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
        mats_to_hsv(flow_parts[0], flow_parts[1], max_px_movement, out_img);
    }
    else {
        mats_to_hsv(flow_parts[0], flow_parts[1], pixel_movement_shown_limit, out_img);
    }


    for (const auto& line : lines){
    	cv::arrowedLine(out_img, line.start, line.end, Scalar(255,0,0));
    }


    std::cout << "Writing image to " << outpath << std::endl;
	imwrite( outpath, out_img );

    
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