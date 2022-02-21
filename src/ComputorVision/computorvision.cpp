#include "computorvision.h"
#include <numeric>

cv::Mat ComputorVision::BlurImage(cv::Mat src){
    cv::Mat result;
    cv::GaussianBlur(src, result, cv::Size(3,3), 0, 0);
    return result;
}

cv::Mat ComputorVision::DetectEdges(cv::Mat src){
    cv::Mat result;
    cv::Canny(src, result, 100, 3*3, 3 );
    return result;
}

cv::Mat ComputorVision::MaskImage(cv::Mat src){
    cv::Mat result;
    cv::Mat mask = cv::Mat::zeros(src.size(), src.type());
    cv::Point pts[4] = {
        cv::Point(0, 720),
        cv::Point(1280-768, 720-350),
        cv::Point(1280-300, 720-350),
        cv::Point(1280, 720),
    };

    cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255, 0,0));
    cv::bitwise_and(mask, src , result);
    return result;
}

std::vector<cv::Vec4i> ComputorVision::HoughLines(cv::Mat src){
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(src, lines, 2, CV_PI/180, 100, 50, 5);
    return lines;
}

std::vector<cv::Vec4i> ComputorVision::AverageLines(cv::Mat src, std::vector<cv::Vec4i> lines){
    std::vector<cv::Vec2f> left;
    std::vector<cv::Vec2f> right;

    for (auto line : lines)
    {
        cv::Point start = cv::Point(line[0], line[1]);
        cv::Point end = cv::Point(line[2], line[3]);

        if(start.x == end.x){
            continue;
        }

        double slope = (static_cast<double>(end.y) - static_cast<double>(start.y))/ (static_cast<double>(end.x) - static_cast<double>(start.x) + 0.00001);
        double yIntercept = static_cast<double>(start.y) - (slope * static_cast<double>(start.x));
        
        if(slope < 0){
          left.push_back(cv::Vec2f(slope, yIntercept));
        }else{
          right.push_back(cv::Vec2f(slope, yIntercept));
        }
    }
        
    cv::Vec2f rightAverage = averageVec2Vector(right);
    cv::Vec2f leftAverage = averageVec2Vector(left);
    cv::Vec4i leftLine = GeneratePoints(src, leftAverage);
    cv::Vec4i rightLine = GeneratePoints(src, rightAverage);

    std::vector<cv::Vec4i> result(2);
    result[0] = leftLine; 
    result[1] = rightLine; 
    return result;
}

cv::Vec2f ComputorVision::averageVec2Vector(std::vector<cv::Vec2f> vectors){
    cv::Vec2f sum;

    for(auto vect2 : vectors){
        sum += vect2;
    }
    sum[0] = sum[0] / vectors.size();
    sum[1] = sum[1] / vectors.size();
    return sum;
}

cv::Vec4i ComputorVision::GeneratePoints(cv::Mat src, cv::Vec2f average){
    float slope = average[0];
    float y_int = average[1];
  
    int y1 = src.rows;
    int y2 = int(y1 * (float(2)/5)); //this defines height in image (inversed)
    int x1 = int((y1 - y_int) / slope);
    int x2 = int((y2 - y_int) / slope);
    return cv::Vec4i(x1, y1, x2, y2);
}

cv::Mat ComputorVision::PlotLaneLines(cv::Mat src, std::vector<cv::Vec4i> lines){
  for(auto line : lines){
    cv::Point start = cv::Point(line[0], line[1]);
    cv::Point end = cv::Point(line[2], line[3]);

    cv::line(src, start, end, cv::Scalar(0,0,255), 3, cv::LINE_AA);
  }
  return src;
}