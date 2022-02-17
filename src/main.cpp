#include <libsocketcan.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include "ComputerVision/computervision.h"

namespace fs = std::filesystem;

int main( int argc, char** argv )
{
    cv::samples::addSamplesDataSearchPath("/home/robinvanwijk/Projects/SDC/SDC-HANZE-2022/images/");
    cv::Mat src = cv::imread( cv::samples::findFile( "test.jpg" ) );
    if( src.empty() )
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }
    cv::namedWindow("View");
    imshow("View", src);    

    ComputerVision cVision;
    cv::Mat denoisedImage = cVision.BlurImage(src);
    cv::Mat edgeMapImage = cVision.DetectEdges(denoisedImage);

    cv::namedWindow("Edge Map");
    imshow("Edge Map", edgeMapImage);
   
    cv::Mat maskedImage = cVision.MaskImage(edgeMapImage);

    cv::namedWindow("Mask");
    imshow("Mask", maskedImage);

    std::vector<std::vector<cv::Vec4i>> houghLines = cVision.HoughLines(edgeMapImage);

    cv::waitKey(0);
    return 0;
}