#include "computorvision.h"
#include <numeric>
#include "../Math/Polynomial.h"

void ComputorVision::SetFrame(cv::Mat src){
    frame = src;
    dstP[0] = cv::Point2f(frame.cols * 0.2, 0);
    dstP[1] = cv::Point2f(frame.cols * 0.8, 0);
    dstP[2] = cv::Point2f(frame.cols * 0.8, frame.rows);
    dstP[3] = cv::Point2f(frame.cols * 0.2, frame.rows);
}

cv::Mat ComputorVision::BlurImage(cv::Mat src){
    cv::GaussianBlur(src, blurred, cv::Size(3,3), 0, 0);
    return blurred;
}

cv::Mat ComputorVision::DetectEdges(cv::Mat src){
    cv::Canny(src, edgeMap, 100, 3*3, 3 );
    return edgeMap;
}

cv::Mat ComputorVision::MaskImage(cv::Mat src){
    mask = cv::Mat::zeros(src.size(), src.type());
    cv::Point pts[4] = {
        cv::Point(0, src.rows * 0.7),
        cv::Point(0, src.rows * 0.30),
        cv::Point(src.cols, src.rows * 0.30),
        cv::Point(src.cols, src.rows * 0.7),
    };
    cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255, 0,0));
    cv::bitwise_and(mask, src , masked);
    return masked;
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
    int y2 = int(y1 * 0.30); //this defines height in image (inversed)
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

    cv::Point startL1 = cv::Point(lines[0][0], lines[0][1]);
    cv::Point endL1 = cv::Point( lines[0][2],  lines[0][3]);

    cv::Point startL2 = cv::Point(lines[1][0], lines[1][1]);
    cv::Point endL2 = cv::Point( lines[1][2],  lines[1][3]);
    
    cv::Point centerLineStart = (startL1 + startL2) / 2;
    cv::Point centerLineEnd = (endL1 + endL2) / 2;

    cv::line(src, centerLineStart, centerLineEnd, cv::Scalar(255,0,0) , 3, cv::LINE_AA);
    return src;
}

std::vector<int> ComputorVision::Histogram(cv::Mat src){
    std::vector<int> points;
    for(int i = 0; i < src.cols; i++){
        points.push_back(cv::countNonZero(src.col(i)));
    }
    return points;
}

std::vector<cv::Point2f> ComputorVision::SlidingWindow(cv::Mat image, cv::Rect window){
    std::vector<cv::Point2f> points;
    const cv::Size imgSize = image.size();
    
    while (window.y >= 0){
        float currentX = window.x + window.width * 0.5f;
        cv::Mat roi = image(window);         
        std::vector<cv::Point2f> locations;
        
        findNonZero(roi, locations);      
        float avgX = 0.0f;
        
        for (int i = 0; i < locations.size(); ++i) {
            float x = locations[i].x;
            avgX += window.x + x;
            cv::Point point(window.x + locations[i].x, window.y + locations[i].y);
            // points.push_back(point);
        }
        
        avgX = locations.empty() ? currentX : avgX / locations.size();
        cv::Point point(avgX, window.y + window.height * 0.5f);
        points.push_back(point);

        window.y -= window.height;
        window.x += (point.x - currentX);
        
        if (window.x < 0)
            window.x = 0;
        if (window.x + window.width >= imgSize.width)
            window.x = imgSize.width - window.width - 1;
    }
    return points;
}

cv::Mat ComputorVision::CreateBinaryImage(cv::Mat src){
    denoisedImage = BlurImage(src);

    cv::cvtColor(denoisedImage, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(hMin, sMin, vMin), cv::Scalar(hMax, sMax, vMax), hsvFilter); // cv::Scalar(0, 10, 28), cv::Scalar(38, 255, 255)
    // cv::inRange(hsv, cv::Scalar(0, 0, 36), cv::Scalar(179, 65, 154), hsvFilter); // cv::Scalar(0, 10, 28), cv::Scalar(38, 255, 255)

    cv::erode(hsvFilter, hsvFilter, structuringElement);
    cv::dilate(hsvFilter, hsvFilter, structuringElement);

    cv::dilate(hsvFilter, hsvFilter, structuringElement);
    cv::erode(hsvFilter, hsvFilter, structuringElement);
    
    cv::Mat sobelx;
    cv::Mat sobely;
    cv::Mat sobelxy;

    cv::Mat gray;
    cv::cvtColor(denoisedImage, gray, cv::COLOR_BGR2GRAY);
    Sobel(gray, sobelx, CV_64F, 1, 0);
    Sobel(gray, sobely, CV_64F, 0, 1);
    Sobel(gray, sobelxy, CV_64F, 1, 1);

    convertScaleAbs(sobelx, sobelx);
    convertScaleAbs(sobely, sobely);
    convertScaleAbs(sobelxy, sobelxy);
    // imshow("sobx'", sobelx);
    // imshow("soby'", sobely);
    // imshow("sobxy'", sobelxy);
    cv::Mat hsl;

    cv::cvtColor(src, hsl, cv::COLOR_BGR2HLS);
    std::vector<cv::Mat> hslChannels(3);
    cv::split(hsl, hslChannels);

    imshow("h", hslChannels[0]);
    imshow("s", hslChannels[1]);
    imshow("l", hslChannels[2]);

    cv::Mat sobel;
    cv::bitwise_or(sobelx, sobely, sobel);
    cv::bitwise_or(sobel, sobelxy, sobel);
    // imshow("sobel'", sobel);

    cv::Mat mask;
    cv::inRange(hslChannels[0], 100,255, mask);
    
    // imshow("hsvfilter", hsvFilter);
    binaryImage = DetectEdges(mask);
    imshow("binary", binaryImage);

    return binaryImage;
}

std::vector<cv::Vec4i> ComputorVision::GenerateLines(cv::Mat src){
    std::vector<cv::Vec4i> houghLines = HoughLines(src);
    std::vector<cv::Vec4i> averagedLines = AverageLines(src, houghLines);

    int imageCenter = src.cols / 2.0f;
    int laneCenterX = (averagedLines[0][0] + averagedLines[1][0]) / 2;
    laneOffset = imageCenter - laneCenterX;
    normalisedLaneOffset = 2 * (double(laneOffset - averagedLines[0][0]) / double(averagedLines[1][0] - averagedLines[0][0])) - 1;
    return averagedLines;
}

void ComputorVision::PredictTurn(cv::Mat src, std::vector<cv::Vec4i> edgeLines){
    cv::Point2f srcP[4] = { //NOTE: This could be hard coded using markers during a test day
        cv::Point2f(edgeLines[0][2], edgeLines[0][3]),
        cv::Point2f(edgeLines[1][2], edgeLines[1][3]),
        cv::Point2f(edgeLines[1][0], edgeLines[1][1]),
        cv::Point2f(edgeLines[0][0], edgeLines[0][1]),
    };

    homography = cv::getPerspectiveTransform(srcP, dstP);
    
    invert(homography, invertedPerspectiveMatrix);

    cv::warpPerspective(src, warped, homography, cv::Size(src.cols, src.rows));

    int rectHeight = 120;
    int rectwidth = 60;
    int rectY = src.rows - rectHeight;

    std::vector<cv::Point2f> rightLinePixels = SlidingWindow(warped, cv::Rect(dstP[2].x - rectwidth, rectY, rectHeight, rectwidth));
    std::vector<cv::Point2f> leftLinePixels = SlidingWindow(warped, cv::Rect(dstP[3].x - rectwidth, rectY, rectHeight, rectwidth));

    std::vector<double> fitR = Polynomial::Polyfit(rightLinePixels, 2);
    std::vector<double> fitL = Polynomial::Polyfit(leftLinePixels, 2);

    std::vector<cv::Point2f> rightLanePoints;

    for (auto pts : rightLinePixels)
    {
        cv::Point2f position;
        position.x = pts.x;
        position.y = (fitR[2] * pow(pts.x, 2) + (fitR[1] * pts.x) + fitR[0]);
        rightLanePoints.push_back(position);
    }

    curveRadiusR = Polynomial::Curvature(fitR, edgeLines[0][1]);
    curveRadiusL = Polynomial::Curvature(fitL, edgeLines[0][1]);

    // cv::putText(frame, "Curvature left edge: " + std::to_string(curveRadiusL), cv::Point(10, 75), 1, 1.2, cv::Scalar(255, 255, 0));
    // cv::putText(frame, "Curvature right edge: " + std::to_string(curveRadiusR), cv::Point(10, 100), 1, 1.2, cv::Scalar(255, 255, 0));

    double vertexRX = Polynomial::Vertex(fitR);
    double vertexLX = Polynomial::Vertex(fitL);

    std::string roadType = "";
    int turnThreshold = 3000;

    if(vertexLX < turnThreshold){
        roadType = "Left Turn";
    }else if(vertexRX < turnThreshold){
        roadType = "Right Turn";
    }else{
        roadType = "Straight";
    }

    //----DRAW STUF -----

    cv::putText(frame, roadType, cv::Point(src.cols/2 - 100, 175), 1, 1.5, cv::Scalar(255, 128, 255));

    std::vector<cv::Point2f> outPts;
    std::vector<cv::Point> allPts;

    cv::perspectiveTransform(rightLinePixels, outPts, invertedPerspectiveMatrix);
    // cv::line(frame, cv::Point(edgeLines[1][0], edgeLines[1][1]), outPts[0], cv::Scalar(0, 255, 0), 3);
    // allPts.push_back(cv::Point(edgeLines[1][0], edgeLines[1][1]));

    for (int i = 0; i < outPts.size() - 1; ++i)
    {
        cv::line(frame, outPts[i], outPts[i + 1], cv::Scalar(0, 255, 0), 3);
        allPts.push_back(cv::Point(outPts[i].x, outPts[i].y));
    }

    allPts.push_back(cv::Point(outPts[outPts.size() - 1].x, outPts[outPts.size() - 1].y));

    cv::perspectiveTransform(leftLinePixels, outPts, invertedPerspectiveMatrix);

    for (int i = 0; i < outPts.size() - 1; ++i)
    {
        cv::line(frame, outPts[i], outPts[i + 1], cv::Scalar(0, 255, 0), 3);
        allPts.push_back(cv::Point(outPts[outPts.size() - i - 1].x, outPts[outPts.size() - i - 1].y));
    }

    allPts.push_back(cv::Point(outPts[0].x - (outPts.size() - 1), outPts[0].y));
    // cv::line(frame, cv::Point(edgeLines[0][0], edgeLines[0][1]), outPts[outPts.size() -1], cv::Scalar(0, 255, 0), 3);
    // allPts.push_back(cv::Point(edgeLines[0][0], edgeLines[0][1]));

    std::vector<std::vector<cv::Point>> arr;
    arr.push_back(allPts);
    
    cv::Mat overlay = cv::Mat::zeros(frame.size(), frame.type());
    cv::fillPoly(overlay, arr, cv::Scalar(0, 255, 100));
    cv::addWeighted(frame, 1, overlay, 0.5, 0, frame);

    cv::namedWindow("Turn");
    imshow("Turn", frame);
}