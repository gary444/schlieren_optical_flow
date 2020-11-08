#include <opencv2/core.hpp>


#include "Descriptors.hpp"


std::vector <cv::Point> GetLocalMaxima(const cv::Mat& Src,int MatchingSize, int Threshold, int GaussKernel  )
{  
  using namespace cv;
  using namespace std;

  vector <Point> vMaxLoc(0); 

  if ((MatchingSize % 2 == 0) || (GaussKernel % 2 == 0)) // MatchingSize and GaussKernel have to be "odd" and > 0
  {
    throw std::logic_error(" MatchingSize and GaussKernel have to be odd and > 0");
  }

  vMaxLoc.reserve(100); // Reserve place for fast access 
  Mat ProcessImg = Src.clone();
  int W = Src.cols;
  int H = Src.rows;
  int SearchWidth  = W - MatchingSize;
  int SearchHeight = H - MatchingSize;
  int MatchingSquareCenter = MatchingSize/2;

  if(GaussKernel > 1) // If You need a smoothing
  {
    GaussianBlur(ProcessImg,ProcessImg,Size(GaussKernel,GaussKernel),0,0,4);
  }
  uchar* pProcess = (uchar *) ProcessImg.data; // The Point2fer to image Data 

  int Shift = MatchingSquareCenter * ( W + 1);
  int k = 0;

  for(int y=0; y < SearchHeight; ++y)
  { 
    int m = k + Shift;
    for(int x=0;x < SearchWidth ; ++x)
    {
      if (pProcess[m++] >= Threshold)
      {
        Point LocMax;
        Mat mROI(ProcessImg, Rect(x,y,MatchingSize,MatchingSize));
        minMaxLoc(mROI,NULL,NULL,NULL,&LocMax);
        if (LocMax.x == MatchingSquareCenter && LocMax.y == MatchingSquareCenter)
        { 
          vMaxLoc.push_back(Point2f( x+LocMax.x,y + LocMax.y )); 
          // imshow("W1",mROI);cvWaitKey(0); //For gebug              
        }
      }
    }
    k += W;
  }
  return vMaxLoc; 
}

std::vector<cv::Point> find_min_max_keypoints_in_image (const cv::Mat& img){

	using namespace cv;
	using namespace std;

	if (img.channels() != 1) throw ("input to find_min_max_keypoints_in_image must be greyscale!");

    Mat img_inv = 255 - img;

	//MIN MAX DETECTION
	std::cout << "finding min and max points..." << std::endl; 

	// GetLocalMaxima(const cv::Mat Src,int MatchingSize, int Threshold, int GaussKernel  )
	std::vector <Point> points = GetLocalMaxima(img, 21, 20, 1 );
	std::vector <Point> inv_points = GetLocalMaxima(img_inv, 21, 20, 1 );


	const bool VIEW_MIN_MAX_POINTS = false;
	if (VIEW_MIN_MAX_POINTS){
		Mat test;
    	cvtColor(img, test, cv::COLOR_GRAY2RGB);
		for (auto l : points) circle(test,l,3,(255),1,8);
		for (auto l : inv_points) circle(test,l,3,(255,0),1,8);
		imwrite("show_min_max_points.png", test);
	}

	//concat points to one vector
	points.insert(points.end(), inv_points.begin(), inv_points.end());

	std::cout << "Found " << points.size() << " min/max points" << std::endl; 

	return points;
}

void calculate_descriptors_at_points(const std::vector<cv::Point>& points, const cv::Mat& img, cv::Mat& descriptors, const std::string& outpath){

	// convert to key points
	std::vector<cv::KeyPoint> keypoints;
	for(auto p : points){
		keypoints.push_back(cv::KeyPoint(p.x, p.y, 21));
	}
	
	std::cout << "Computing SIFT descriptors " << std::endl;
	cv::Ptr< cv::SIFT> sift =  cv::SIFT::create(100, 10, 0, 50.0, 0.5);
	sift->compute( img, keypoints, descriptors);

	//If path provided, draw points to image and save.
	if ("" != outpath){
		std::cout << "Drawing SIFT descriptors " << std::endl;
		cv::Mat output;
		cv::drawKeypoints(img, keypoints, output);
		imwrite(outpath,output);
	}


}

