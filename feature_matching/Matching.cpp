#include "Matching.hpp"



std::vector<cv::DMatch> match_descriptors(const cv::Mat& desc1, const cv::Mat& desc2){

	using namespace cv;

 	const int normType = NORM_L2; // best for SIFT. ORB and other binary descriptors should be compared with NORM_HAMMING2
	const bool crossCheck = false; // check in both directions for more reliable matches
	auto matcher = BFMatcher::create(normType, crossCheck); 


	std::vector<DMatch> matches;
    matcher->match( desc1, desc2, matches); // alternative: knn matches for multiple matches

    return matches;
}