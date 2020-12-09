#ifndef MATCHING_CPP
#define MATCHING_CPP

#include <opencv2/features2d.hpp>


std::vector<cv::DMatch> match_descriptors(const cv::Mat& desc1, const cv::Mat& desc2);

std::vector<cv::DMatch> filter_matches_by_displacement_in_pixels(const std::vector<cv::DMatch>& matches, 
															   const std::vector<cv::KeyPoint>& kpsA, 
															   const std::vector<cv::KeyPoint>& kpsB,
															   const float max_displacement);

void draw_matches_as_arrows(const std::vector<cv::DMatch>& matches, 
							const std::vector<cv::KeyPoint>& kpsA, 
							const std::vector<cv::KeyPoint>& kpsB, 
							cv::Mat& bg_img,
							const float min_dist_for_highlighting,
							const float max_dist_for_highlighting);



#endif