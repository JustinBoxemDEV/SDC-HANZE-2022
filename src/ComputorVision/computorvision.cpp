#include "computorvision.h"
#include <numeric>
#include "../Math/Polynomial.h"

void ComputorVision::SetFrame(cv::Mat src){
    frame = src;
    dstP[0] = cv::Point2f(frame.cols * 0.1, 0);
    dstP[1] = cv::Point2f(frame.cols * 0.9, 0);
    dstP[2] = cv::Point2f(frame.cols * 0.65, frame.rows);
    dstP[3] = cv::Point2f(frame.cols * 0.35, frame.rows);
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
        cv::Point(0, src.rows * 0.8),
        cv::Point(0, src.rows * 0.6),
        cv::Point(src.cols, src.rows * 0.6),
        cv::Point(src.cols, src.rows * 0.8),
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
    int angleThreshold = 5;

    for (auto line : lines)
    {
        cv::Point start = cv::Point(line[0], line[1]);
        cv::Point end = cv::Point(line[2], line[3]);

        if(start.x == end.x){
            continue;
        }

        double slope = (static_cast<double>(end.y) - static_cast<double>(start.y))/ (static_cast<double>(end.x) - static_cast<double>(start.x) + 0.00001);
        double yIntercept = static_cast<double>(start.y) - (slope * static_cast<double>(start.x));
        double angle = atan2(end.y - start.y, end.x - start.x) * 180.0 / CV_PI;

        if((angle < angleThreshold && angle >  -angleThreshold) || (angle > 180 - angleThreshold && angle < 180 + angleThreshold)){
            continue;
        }

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
    int y2 = int(y1 * 0.6); //this defines height in image (inversed)
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
        if (window.x < 0)
            window.x = 0;
        if (window.x + window.width >= imgSize.width)
            window.x = imgSize.width - window.width - 1;
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
        if(!locations.empty()){
            points.push_back(point);
        }

        cv::rectangle(warped,window, (255,255,255));
        window.y -= window.height;
        window.x += (point.x - currentX);
        
    }
    return points;
}

cv::Mat ComputorVision::CreateBinaryImage(cv::Mat src){
    denoisedImage = BlurImage(src);

    // cv::cvtColor(denoisedImage, hsv, cv::COLOR_BGR2HSV);
    // cv::inRange(hsv, cv::Scalar(hMin, sMin, vMin), cv::Scalar(hMax, sMax, vMax), hsvFilter); // cv::Scalar(0, 10, 28), cv::Scalar(38, 255, 255)
    // cv::inRange(hsv, cv::Scalar(0, 0, 36), cv::Scalar(179, 65, 154), hsvFilter); // cv::Scalar(0, 10, 28), cv::Scalar(38, 255, 255)

    // cv::erode(hsvFilter, hsvFilter, structuringElement);
    // cv::dilate(hsvFilter, hsvFilter, structuringElement);

    // cv::dilate(hsvFilter, hsvFilter, structuringElement);
    // cv::erode(hsvFilter, hsvFilter, structuringElement);
    
    cv::Mat sobelx;
    cv::Mat sobely;
    cv::Mat sobelxy;

    cv::Mat gray;
    cv::cvtColor(denoisedImage, gray, cv::COLOR_BGR2GRAY);
    Sobel(gray, sobelx, CV_64F, 1, 0);
    Sobel(gray, sobely, CV_64F, 0, 1);
    Sobel(gray, sobelxy, CV_64F, 1, 1);
    cv::inRange(sobely, 75,255, sobely);
    cv::inRange(sobelx, 20,70, sobelx);
    imshow("soby'", sobely);
    imshow("sobx'", sobelx);

    convertScaleAbs(sobelx, sobelx);
    convertScaleAbs(sobely, sobely);
    convertScaleAbs(sobelxy, sobelxy);
    imshow("sobxy'", sobelxy);

    // cv::Mat rgb;
    // cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
    // std::vector<cv::Mat> rgbChannels(3);
    // cv::split(rgb, rgbChannels);

    // imshow("rgb - r", rgbChannels[0]);
    // imshow("rgb - g", rgbChannels[1]);
    // imshow("rgb - b", rgbChannels[2]);

    cv::Mat hls;
    cv::cvtColor(src, hls, cv::COLOR_BGR2HLS);
    std::vector<cv::Mat> hlsChannels(3);
    cv::split(hls, hlsChannels);

    // imshow("hls - h", hlsChannels[0]);
    // imshow("hls - l", hlsChannels[1]);
    // imshow("hls - s", hlsChannels[2]);

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels(3);
    cv::split(hsv, hsvChannels);

    // imshow("hsv - h", hsvChannels[0]);
    // imshow("hsv - s", hsvChannels[1]);
    // imshow("hsv - v", hsvChannels[2]);

    // cv::Mat lab;
    // cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    // std::vector<cv::Mat> labChannels(3);
    // cv::split(lab, labChannels);

    // imshow("lab - l", labChannels[0]);
    // imshow("lab - a", labChannels[1]);
    // imshow("lab - b", labChannels[2]);

    // cv::Mat lux;
    // cv::cvtColor(src, lux, cv::COLOR_BGR2Luv);
    // std::vector<cv::Mat> luxChannels(3);
    // cv::split(lux, luxChannels);

    // imshow("lux - l", luxChannels[0]);
    // imshow("lux - u", luxChannels[1]);
    // imshow("lux - x", luxChannels[2]);


    // cv::Mat sobel;
    // cv::bitwise_or(sobelx, sobely, sobel);
    // cv::bitwise_or(sobel, sobelxy, sobel);
    // imshow("sobel'", sobel);

    cv::inRange(hsvChannels[1], 105,255, hsvChannels[1]);
    cv::dilate(hsvChannels[1], hsvChannels[1], structuringElement);
    cv::erode(hsvChannels[1], hsvChannels[1], structuringElement);

    // imshow("hsv s'", hsvChannels[1]);

    cv::inRange(hlsChannels[2], 70,255, hlsChannels[2]);
    cv::dilate(hlsChannels[2], hlsChannels[2], structuringElement);
    cv::erode(hlsChannels[2], hlsChannels[2], structuringElement);
    // imshow("hls s'", hlsChannels[2]);

    cv::inRange(hlsChannels[1], 155,255, hlsChannels[1]);
    // imshow("hls l'", hlsChannels[1]);

    cv::Mat mask;
    cv::bitwise_or(hlsChannels[2], hlsChannels[1], mask);
    cv::bitwise_or(mask, hsvChannels[1], mask);
    cv::bitwise_or(sobely, sobelx, sobely);
    cv::bitwise_and(mask, sobely, mask);

    cv::Mat edges = DetectEdges(mask);
    cv::bitwise_or(edges, mask, binaryImage);

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
    cv::Point2f srcP[4] = { 
        cv::Point2f(src.cols * 0.35, src.rows * 0.6),
        cv::Point2f(src.cols * 0.85, src.rows * 0.6),
        cv::Point2f(src.cols, src.rows * 0.8),
        cv::Point2f(0, src.rows * 0.8),
    };

    homography = cv::getPerspectiveTransform(srcP, dstP);
    
    invert(homography, invertedPerspectiveMatrix);

    cv::warpPerspective(src, warped, homography, cv::Size(src.cols, src.rows));

    int rectHeight = 80;
    int rectwidth = 30;
    int rectY = src.rows - rectHeight;

    std::vector<int> histogram = Histogram(warped);
    std::vector<int> leftHist(histogram.begin(), histogram.begin() + src.cols * 0.5);
    std::vector<int> rightHist(histogram.begin() + src.cols * 0.5, histogram.end());
    int leftMaxX = std::max_element(leftHist.begin(), leftHist.end()) - leftHist.begin();
    int rightMaxX = std::max_element(rightHist.begin(), rightHist.end()) - rightHist.begin() + src.cols * 0.5;

    std::vector<cv::Point2f> rightLinePixels = SlidingWindow(warped, cv::Rect(rightMaxX - rectHeight, rectY, rectHeight, rectwidth));
    std::vector<cv::Point2f> leftLinePixels = SlidingWindow(warped, cv::Rect(leftMaxX - rectHeight, rectY, rectHeight, rectwidth));
    imshow("warped", warped);

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
    
    std::vector<cv::Point2f> leftLanePoints;

    for (auto pts : leftLinePixels)
    {
        cv::Point2f position;
        position.x = pts.x;
        position.y = (fitL[2] * pow(pts.x, 2) + (fitL[1] * pts.x) + fitL[0]);
        leftLanePoints.push_back(position);
    }

    curveRadiusR = Polynomial::Curvature(fitR, src.rows * 0.7);
    curveRadiusL = Polynomial::Curvature(fitL, src.rows * 0.7);

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

    if(rightLinePixels.size() >0){
        cv::perspectiveTransform(rightLinePixels, outPts, invertedPerspectiveMatrix);
        for (int i = 0; i < outPts.size() - 1; ++i)
        {
            cv::line(frame, outPts[i], outPts[i + 1], cv::Scalar(0, 255, 0), 3);
            allPts.push_back(cv::Point(outPts[i].x, outPts[i].y));
        }

        allPts.push_back(cv::Point(outPts[outPts.size() - 1].x, outPts[outPts.size() - 1].y));
    }
    
    if(leftLinePixels.size() >0){
        cv::perspectiveTransform(leftLinePixels, outPts, invertedPerspectiveMatrix);

        for (int i = 0; i < outPts.size() - 1; ++i)
        {
            cv::line(frame, outPts[i], outPts[i + 1], cv::Scalar(0, 255, 0), 3);
            allPts.push_back(cv::Point(outPts[outPts.size() - i - 1].x, outPts[outPts.size() - i - 1].y));
        }
        allPts.push_back(cv::Point(outPts[0].x - (outPts.size() - 1), outPts[0].y));
    }

    if(allPts.size() > 0){
        std::vector<std::vector<cv::Point>> arr;
        arr.push_back(allPts);
        
        cv::Mat overlay = cv::Mat::zeros(frame.size(), frame.type());
        cv::fillPoly(overlay, arr, cv::Scalar(0, 255, 100));
        cv::addWeighted(frame, 1, overlay, 0.5, 0, frame);
    }

    cv::namedWindow("Turn");
    imshow("Turn", frame);
}