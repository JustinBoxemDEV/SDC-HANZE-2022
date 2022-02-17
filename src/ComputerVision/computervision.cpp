#include "computervision.h"

cv::Mat ComputerVision::BlurImage(cv::Mat src){
    cv::Mat result;
    cv::GaussianBlur(src, result, cv::Size(3,3), 0, 0);
    return result;
}

cv::Mat ComputerVision::DetectEdges(cv::Mat src){
    cv::Mat result;
    cv::Canny(src, result, 250, 3*3, 3 );
    return result;
}

cv::Mat ComputerVision::MaskImage(cv::Mat src){
    cv::Mat result;
    cv::Mat mask = cv::Mat::zeros(src.size(), src.type());
    cv::Point pts[4] = {
        cv::Point(0, 720),
        cv::Point(420, 580),
        cv::Point(1060, 580),
        cv::Point(1280, 720)
    };

    cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255, 0, 0));
    cv::bitwise_and(src, mask, result);
    return result;
}

std::vector<std::vector<cv::Vec4i>> ComputerVision::HoughLines(cv::Mat src){
    std::vector<std::vector<cv::Vec4i>> line;
    cv::HoughLinesP(src, line, 1, CV_PI, 20, 20, 30);
    return line;
}
