#ifndef DESCRIPTORS_HPP
#define DESCRIPTORS_HPP

#include <vector>

#include <opencv2/opencv.hpp>

std::vector<cv::Point> GetLocalMaxima(const cv::Mat& Src,int MatchingSize, int Threshold, int GaussKernel  );

std::vector<cv::Point> find_min_max_keypoints_in_image (const cv::Mat& img);

std::vector<cv::KeyPoint> convert_points_to_keypoints(const std::vector<cv::Point>& points);

void calculate_descriptors_at_points(std::vector<cv::KeyPoint>& points, const cv::Mat& img, cv::Mat& descriptors, const std::string& outpath = "");


#endif