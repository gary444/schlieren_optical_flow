
#include <dirent.h>
#include <fstream>
#include <stdio.h>


#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <opencv2/optflow.hpp>

#include <opencv2/video/tracking.hpp>


#include "OpenCVHelper.hpp"
#include "optical_flow_helpers.hpp"
#include "util.hpp"


using namespace cv;
using namespace std;


vector <Point> GetLocalMaxima(const cv::Mat Src,int MatchingSize, int Threshold, int GaussKernel  )
{  
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

float vdist(Point a, Point b) {
	return sqrt( pow(a.x-b.x, 2) + pow(a.y-b.y, 2));
}

//https://stackoverflow.com/questions/5550290/find-local-maxima-in-grayscale-image-using-opencv
int imregionalmax(Mat input, int nLocMax, float threshold, float minDistBtwLocMax, 
		// Mat locations
		std::vector<Point>& locations
		)
{

    Mat scratch = input.clone();
    int nFoundLocMax = 0;
    for (int i = 0; i < nLocMax; i++) {
        Point location;
        double maxVal;
        minMaxLoc(scratch, NULL, &maxVal, NULL, &location);
        if (maxVal > threshold) {
            nFoundLocMax += 1;
            int row = location.y;
            int col = location.x;
            // locations.at<int>(i,0) = row;
            // locations.at<int>(i,1) = col;
            locations.push_back(Point(row,col));

            int r0 = (row-minDistBtwLocMax > -1 ? row-minDistBtwLocMax : 0);
            int r1 = (row+minDistBtwLocMax < scratch.rows ? row+minDistBtwLocMax : scratch.rows-1);
            int c0 = (col-minDistBtwLocMax > -1 ? col-minDistBtwLocMax : 0);
            int c1 = (col+minDistBtwLocMax < scratch.cols ? col+minDistBtwLocMax : scratch.cols-1);
            for (int r = r0; r <= r1; r++) {
                for (int c = c0; c <= c1; c++) {
                    if (vdist(Point(r, c),Point(row, col)) <= minDistBtwLocMax) {
                        scratch.at<float>(r,c) = 0.0;
                    }
                }
            }
        } else {
            break;
        }
    }
    return nFoundLocMax;
}

void non_maxima_suppression(const cv::Mat& src, cv::Mat& mask, const bool remove_plateaus)
{
    // find pixels that are equal to the local neighborhood not maximum (including 'plateaus')
    cv::dilate(src, mask, cv::Mat());
    cv::compare(src, mask, mask, cv::CMP_GE);

    // optionally filter out pixels that are equal to the local minimum ('plateaus')
    if (remove_plateaus) {
        cv::Mat non_plateau_mask;
        cv::erode(src, non_plateau_mask, cv::Mat());
        cv::compare(src, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
        cv::bitwise_and(mask, non_plateau_mask, mask);
    }
}


int main(int argc, char* argv[] )
{


    //capture properties
    const float real_width_mm = 1481.f;
    const float time_between_imgs_ms = 20.0f;

    std::string outpath = "../output/maxima2.png";

    std::vector<std::vector<float> > all_path_vals;

    if (cmd_option_exists(argv, argv+argc, "-h")
        || !cmd_option_exists(argv, argv+argc, "-dir")
        || argc < 3){
        std::cout << "Optical flow batch processing app\n\n"
                  << "flags:\n" 
                  << "-dir: directory containing a sequence of images to be processed to find mean optical flow:\n" 
                  << "-o: output file path\n" 
                  << "-n: number of images from directory to use \n"
                  << "-m: mask path \n"
                  << std::endl; 

        return 0;
    } 

    if (cmd_option_exists(argv, argv+argc, "-o")){
        outpath = get_cmd_option(argv, argv+argc, "-o");
    } 
    else {
        std::cout << "\nWARNING: no output path specified (use -o)\n\n";
    }

    // load images
    const std::string inDirectory = get_cmd_option(argv, argv+argc, "-dir");
    std::cout << "Loading images from directory:" << inDirectory << std::endl;

    //check directory for image files
    std::vector<std::string> img_paths;
    if (auto dir = opendir( inDirectory.c_str() )) {
        while (auto f = readdir(dir)) {
            if (!f->d_name || f->d_name[0] == '.')
                continue; // Skip everything that starts with a dot
            std::string fname = inDirectory + "/" + f->d_name;
            
            img_paths.push_back(fname);
        }
        closedir(dir);
        std::sort( img_paths.begin(), img_paths.end() );
    }

    if (0 == img_paths.size()){
        std::cout << "Error: no files found in this folder\n";
        return 0;
    }

    std::cout << "Images for comparison, in order: \n";
    for (auto path : img_paths) std::cout << path << std::endl;
    std::cout << "Total: " << img_paths.size() << " images " << std::endl;

	// std::string mask_path = "";
	// Mat mask;
	// if (cmd_option_exists(argv, argv+argc, "-m")){
 //        mask_path = get_cmd_option(argv, argv+argc, "-m");
 //        mask = imread(mask_path);
 //    	cvtColor(mask, mask, cv::COLOR_RGB2GRAY);

 //    } 


    // create grey images for min/max detection
    Mat test = imread(img_paths[0]);
    Mat input;
    cvtColor(test, input, cv::COLOR_RGB2GRAY);
    Mat input_inv = 255 - input;


    printMatInfo(input, "input");

 	//MIN MAX DETECTION
  std::cout << "finding min and max points..." << std::endl; 
	// GetLocalMaxima(const cv::Mat Src,int MatchingSize, int Threshold, int GaussKernel  )
	vector <Point> points = GetLocalMaxima(input, 21, 20, 1 );
	vector <Point> inv_points = GetLocalMaxima(input_inv, 21, 20, 1 );


  const bool VIEW_MIN_MAX_POINTS = false;
  if (VIEW_MIN_MAX_POINTS){

  	for (auto l : points) {
  		circle(test,l,3,(255),1,8);
  	}
  	for (auto l : inv_points) {
  		circle(test,l,3,(255,0),1,8);
   	}
    imwrite(outpath, test);

    return 0;
  }


  std::cout << "converting points to SIFT keypoints..." << std::endl; 
  //concat points to one vector
  points.insert(points.end(), inv_points.begin(), inv_points.end());

  std::vector<cv::KeyPoint> keypoints;
  for(auto p : points){
    keypoints.push_back(KeyPoint(p.x, p.y, 21));
  }
  std::cout << "Found " << keypoints.size() << " keypoints" << std::endl; 

  std::cout << "Computing SIFT descriptors " << std::endl;
  Mat descriptors;
  Ptr< cv::SIFT> sift =  cv::SIFT::create(100, 10, 0, 50.0, 0.5);
  sift->compute( test, keypoints, descriptors);

  // Add results to image and save.
  std::cout << "Drawing SIFT descriptors " << std::endl;
  cv::Mat output;
  cv::drawKeypoints(test, keypoints, output);


  imwrite(outpath,output);





	return 0;

}