// #include <libsocketcan.h>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "ComputorVision/computorvision.h"

namespace fs = std::filesystem;

int main( int argc, char** argv )
{
    cv::samples::addSamplesDataSearchPath("E:\\Development\\Stage\\SDC-HANZE-2022\\images");
    cv::Mat src = cv::imread( cv::samples::findFile( "highway.jpg" ) );
    if( src.empty() )
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }

    ComputorVision cVision;
    cv::Mat grayScaleImage;
    cv::cvtColor(src, grayScaleImage, cv::COLOR_RGB2GRAY);

    cv::Mat denoisedImage = cVision.BlurImage(grayScaleImage);
    cv::Mat edgeMapImage = cVision.DetectEdges(denoisedImage);

    cv::namedWindow("Edge Map");
    imshow("Edge Map", edgeMapImage);
   
    cv::Mat maskedImage = cVision.MaskImage(edgeMapImage);

    cv::namedWindow("Mask");
    imshow("Mask", maskedImage);

    std::vector<cv::Vec4i> houghLines = cVision.HoughLines(maskedImage);
    std::vector<cv::Vec4i> averagedLines = cVision.AverageLines(src, houghLines);

    cv::Mat linesImage = cVision.PlotLaneLines(src, averagedLines);

    cv::namedWindow("Lanes");
    imshow("Lanes", linesImage);
   
    cv::waitKey(0);
    return 0;
}