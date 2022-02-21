// #include <libsocketcan.h>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "ComputorVision/computorvision.h"

namespace fs = std::filesystem;

int main( int argc, char** argv )
{
    cv::samples::addSamplesDataSearchPath("/home/robinvanwijk/Projects/SDC/SDC-HANZE-2022/images");

    cv::Mat src;
    cv::VideoCapture cap(cv::samples::findFile( "testvid.mp4" ));

    ComputorVision cVision;

    if (!cap.isOpened()) {
        return -1;
    }
    for (;;)
    {
        cap.read(src);
        if (src.empty()) {
            break;
        }

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
    

        if (cv::waitKey(1000/60) >= 0)
            break;
    }
    return 0;
}