#ifndef OPENCV_HELPER_HPP
#define OPENCV_HELPER_HPP

#include <opencv2/core/mat.hpp>

#include <string>
#include <sstream>

std::string getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

void printMatInfo(const cv::Mat& mat, const std::string name = "mat") {

    std::cout << name << " " << mat.rows << " x " << mat.cols 
    		  << " (type " << getImageType(mat.type()) << ")"
    		  << std::endl;

}

#endif