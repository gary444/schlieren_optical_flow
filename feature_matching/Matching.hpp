#ifndef MATCHING_CPP
#define MATCHING_CPP

#include <opencv2/features2d.hpp>


std::vector<cv::DMatch> match_descriptors(const cv::Mat& desc1, const cv::Mat& desc2);

#endif