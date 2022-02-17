#ifndef COMPUTER_VISION_H
#define COMPUTER_VISION_H

#include <opencv2/opencv.hpp>

class ComputerVision{
    public:
        cv::Mat BlurImage(cv::Mat src);
        cv::Mat DetectEdges(cv::Mat src);
        cv::Mat MaskImage(cv::Mat src);
        std::vector<std::vector<cv::Vec4i>> HoughLines(cv::Mat src);
};

#endif