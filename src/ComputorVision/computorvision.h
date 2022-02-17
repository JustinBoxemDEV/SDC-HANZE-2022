#ifndef COMPUTOR_VISION_H
#define COMPUTOR_VISION_H

#include <opencv2/opencv.hpp>

class ComputorVision{
    private:
        double img_size;
        double img_center;
        bool left_flag = false;  // Tells us if there's left boundary of lane detected
        bool right_flag = false;  // Tells us if there's right boundary of lane detected
        cv::Point right_b;  // Members of both line equations of the lane boundaries:
        double right_m;  // y = m*x + b
        cv::Point left_b;  //
        double left_m;  //
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