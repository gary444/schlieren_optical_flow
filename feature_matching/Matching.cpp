#include "Matching.hpp"

#include <opencv2/imgproc.hpp>


std::vector<cv::DMatch> match_descriptors(const cv::Mat& desc1, const cv::Mat& desc2){

	using namespace cv;

 	const int normType = NORM_L2; // best for SIFT. ORB and other binary descriptors should be compared with NORM_HAMMING2
	const bool crossCheck = true; // check in both directions for more reliable matches
	auto matcher = BFMatcher::create(normType, crossCheck); 


	std::vector<DMatch> matches;
    matcher->match( desc1, desc2, matches); // alternative: knn matches for multiple matches

    return matches;
}


std::vector<cv::DMatch> filter_matches_by_displacement_in_pixels(const std::vector<cv::DMatch>& matches, 	
																   const std::vector<cv::KeyPoint>& kpsA, 
																   const std::vector<cv::KeyPoint>& kpsB,
																   const float max_displacement) {
	using namespace cv;

	std::vector<cv::DMatch> rtn_matches;

	for (const auto& match : matches) {
		const auto& kpA = kpsA[match.queryIdx];
		const auto& kpB = kpsB[match.trainIdx];
		
		const float distance = norm(kpA.pt - kpB.pt);

		if (distance < max_displacement){
			rtn_matches.push_back(match);
		}
	}

	return rtn_matches;
}




void draw_matches_as_arrows(const std::vector<cv::DMatch>& matches, 	
							const std::vector<cv::KeyPoint>& kpsA, 
							const std::vector<cv::KeyPoint>& kpsB, 
							cv::Mat& bg_img,
							const float min_dist_for_highlighting,
							const float max_dist_for_highlighting) {
	using namespace cv;
	
	for (const auto& match : matches) {
		const auto& kpA = kpsA[match.queryIdx];
		const auto& kpB = kpsB[match.trainIdx];

		const float distance = norm(kpA.pt - kpB.pt);
		auto col = Scalar(255,255,255);

		// highlight movementin range of around 20 px per image
		if (distance > min_dist_for_highlighting 
		 && distance < max_dist_for_highlighting){
			col = Scalar(0,255,0);
		}

		cv::arrowedLine(bg_img, kpA.pt, kpB.pt, col);
	}
}
