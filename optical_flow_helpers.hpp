
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

