#ifndef COMPUTOR_VISION_H
#define COMPUTOR_VISION_H

#include <opencv2/opencv.hpp>

class ComputorVision{
    private:
        cv::Vec2f averageVec2Vector(std::vector<cv::Vec2f> vectors);
        cv::Vec4i GeneratePoints(cv::Mat src, cv::Vec2f average);

    public:
        cv::Mat BlurImage(cv::Mat src);
        cv::Mat DetectEdges(cv::Mat src);
        cv::Mat MaskImage(cv::Mat src);
        std::vector<cv::Vec4i> HoughLines(cv::Mat src);
        std::vector<cv::Vec4i> AverageLines(cv::Mat src, std::vector<cv::Vec4i> lines);
        cv::Mat PlotLaneLines(cv::Mat src, std::vector<cv::Vec4i> lines);
};

#endif