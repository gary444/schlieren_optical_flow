
#include <sstream>


#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


void mats_to_hsv(cv::Mat x, cv::Mat y, const float normalisation_val, cv::Mat& out) {

	using namespace cv;

    std::cout << "Create HSV image" << std::endl;

    Mat magnitude, angle, magn_norm;

    cartToPolar(x, y, magnitude, angle, true);

    magn_norm = magnitude / normalisation_val;

    // convert angle to hue
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    //build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);

    out = bgr;
}

void create_legend(const std::string& outpath, const float normalisation_val, const float speed){

	using namespace cv;


    const uint32_t legend_size = 599;
    const float mag_per_pixel = (2.f*normalisation_val) / float(legend_size)  ;

    std::vector<float> range (legend_size);
    for (int i = 0; i < legend_size; ++i){
        range[i] = -normalisation_val + i*mag_per_pixel;
    }

    Mat legend_x_flow;
    Mat legend_y_flow;

    cv::repeat(range, legend_size, 1, legend_x_flow);
    cv::transpose(legend_x_flow, legend_y_flow);

    Mat legend;
    mats_to_hsv(legend_x_flow, legend_y_flow, normalisation_val, legend);


    const uint32_t cntr = (legend_size / 2) + 1;
    cv::circle(legend, Point(cntr, cntr), legend_size/2, Scalar(0,0,0));

    std::ostringstream stringStream;
    stringStream << std::setprecision(4) << normalisation_val << " ms";
    std::string str = stringStream.str();

    cv::putText(legend, //target image
            str,
            cv::Point(10, legend_size / 2), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            0.5,
            CV_RGB(0,0,0), //font color
            1);


    std::cout << "Writing legend to " << outpath << std::endl;
    imwrite( outpath, legend );
}


std::vector<float>  get_flow_magnitude_on_path(cv::Mat x_flow, 
											cv::Mat y_flow, 
											const float real_width_mm,
											const float time_between_imgs_ms
											 ) {

	using namespace cv;

    // define the path line 
    const Point line_start (415, 450);
    const Point line_end (1030, 237);
	const uint32_t NUM_SAMPLES = 100;


    Mat magnitude, angle, magn_norm;
    cartToPolar(x_flow, y_flow, magnitude, angle, true);


 //    // verify line is correct by adding it to image
 //    double min_px_movement, max_px_movement;
 //    cv::minMaxLoc(magnitude, &min_px_movement, &max_px_movement);
	// Mat out_img;
	// mats_to_hsv(flow_parts[0], flow_parts[1], max_px_movement, out_img);
 //    cv::arrowedLine(out_img, line_start, line_end, Scalar(255,255,255));

 //    std::cout << "Writing line image to " << outpath << std::endl;
	// imwrite( outpath, out_img );


	// sampling
    // step along the line, sampling magnitude values from the image

	std::vector<float> sampled_values;

	const float x_step = float(line_end.x - line_start.x) / NUM_SAMPLES;
	const float y_step = float(line_end.y - line_start.y) / NUM_SAMPLES;

	const int img_width = x_flow.cols;

    // calculate max displacement and speed
    const float mm_per_pixel = real_width_mm / img_width;
    // const float max_displacement = max_px_movement * mm_per_pixel;
    // const float max_speed = max_displacement / time_between_imgs_ms;

	for (int i = 0; i < NUM_SAMPLES; ++i)
	{
		Point sample_point (line_start.x + (i*x_step), line_start.y + (i*y_step) );
		float sample_mag = magnitude.at<float>(sample_point);
		float sample_speed = sample_mag * mm_per_pixel / time_between_imgs_ms;

		sampled_values.push_back(sample_speed);

		// std::cout << sampled_values.back() << std::endl;
	}

	// float line_length_in_mm = cv::norm(line_end-line_start) * mm_per_pixel;
	// std::cout << "Line length = " << line_length_in_mm << std::endl;

	return sampled_values;
}

