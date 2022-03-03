#include <stdint.h>
#include <iostream>
#include "mediaCapture.h"
#include <time.h>
#include <string>
#include <filesystem>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../Math/Polynomial.h"
#include <string>

namespace fs = std::filesystem;

void MediaCapture::ProcessFeed(int cameraID, std::string filename)
{
    cv::VideoCapture *capture;

    if (cameraID != 0)
    {
        capture = new cv::VideoCapture(cameraID);
        capture->set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        capture->set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    }
    else if (filename != "")
    {
        std::cout << filename << std::endl;
        capture = new cv::VideoCapture(filename);
    }
    else
    {
        capture = new cv::VideoCapture(0);

        // Camera detection check
        if (!capture->isOpened())
        {
            std::cout << "NO CAMERA DETECTED!" << std::endl;
            return;
        }
    }

    cv::Mat frame;
    std::cout << "Camera selected: " << cameraID << std::endl;

    // Define total frames and start of a counter for FPS calculation
    int totalFrames = 0;

    time_t start, end;
    time(&start);

    // Camera feed
    while (capture->read(frame))
    {
        totalFrames++;

        ProcessImage(frame);

        if (cv::waitKey(1000 / 60) >= 0)
        {
            break;
        }
    }

    // End the time counter
    time(&end);

    // Time elapsed
    double seconds = difftime(end, start);
    std::cout << "Time taken : " << seconds << " seconds" << std::endl;

    // Estimate the FPS based on frames / elapsed time in seconds
    int fps = totalFrames / seconds;
    std::cout << "Estimated frames per second : " << fps << std::endl;
}

cv::Mat MediaCapture::LoadImage(std::string filepath)
{
    std::string path = fs::current_path().string() + "/assets/images/" + std::string(filepath);
    cv::Mat img = imread(path, cv::IMREAD_COLOR);
    if (!fs::exists(path))
    {
        std::cout << "The requested file cannot be found in /assets/images/!" << std::endl;
        return img;
    }

    if (img.empty())
    {
        std::cout << "Could not read the image: " << path << std::endl;
        return img;
    }
    return img;
}

void MediaCapture::ProcessImage(cv::Mat src)
{
    cv::Mat grayScaleImage;
    cv::Mat wipImage;
    src.copyTo(wipImage);
    cv::Mat denoisedImage = cVision.BlurImage(wipImage);

    cv::Mat hsv;
    cv::cvtColor(denoisedImage, hsv, cv::COLOR_BGR2HSV);

    cv::Mat hsvFilter;
    cv::inRange(hsv, cv::Scalar(0, 0, 36), cv::Scalar(179, 65, 154), hsvFilter); // cv::Scalar(0, 10, 28), cv::Scalar(38, 255, 255)

    cv::erode(hsvFilter, hsvFilter, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(12, 12)));
    cv::dilate(hsvFilter, hsvFilter, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(12, 12)));

    cv::dilate(hsvFilter, hsvFilter, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(12, 12)));
    cv::erode(hsvFilter, hsvFilter, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(12, 12)));

    cv::Mat edgeMapImage = cVision.DetectEdges(hsvFilter);
    cv::Mat maskedImage = cVision.MaskImage(edgeMapImage);

    // cv::Mat hsvMask;
    // cv::bitwise_and(maskedImage, maskedImage, hsvMask, hsvFilter);

    // imshow("Thresholded Image", hsvFilter);
    // imshow("mask", hsvMask);
    // imshow("maskiamge", maskedImage);

    std::vector<cv::Vec4i> houghLines = cVision.HoughLines(maskedImage);
    std::vector<cv::Vec4i> averagedLines = cVision.AverageLines(wipImage, houghLines);

    int imageCenter = src.cols / 2.0f;
    int laneCenterX = (averagedLines[0][0] + averagedLines[1][0]) / 2;
    int centerDelta = imageCenter - laneCenterX;
    float normalisedDelta = 2 * (float(centerDelta - averagedLines[0][0]) / float(averagedLines[1][0] - averagedLines[0][0])) - 1;
    cv::putText(src, "Center deviation: " + std::to_string(centerDelta), cv::Point(10, 25), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Center deviation (N): " + std::to_string(normalisedDelta), cv::Point(10, 50), 1, 1.2, cv::Scalar(255, 255, 0));

    cv::Mat linesImage = cVision.PlotLaneLines(wipImage, averagedLines);

    cv::Mat warped;
    cv::Point2f srcP[4] = {
        cv::Point2f(averagedLines[0][2], averagedLines[0][3]),
        cv::Point2f(averagedLines[1][2], averagedLines[1][3]),
        cv::Point2f(averagedLines[1][0], averagedLines[1][1]),
        cv::Point2f(averagedLines[0][0], averagedLines[0][1]),
    };

    cv::Point2f dstP[4] = {
        cv::Point2f(src.cols * 0.2, 0),
        cv::Point2f(src.cols * 0.8, 0),
        cv::Point2f(src.cols * 0.8, src.rows),
        cv::Point2f(src.cols * 0.2, src.rows),
    };

    cv::Mat homography = cv::getPerspectiveTransform(srcP, dstP);
    cv::Mat invertedPerspectiveMatrix;
    invert(homography, invertedPerspectiveMatrix);

    cv::warpPerspective(maskedImage, warped, homography, cv::Size(src.cols, src.rows));

    // cv::namedWindow("Warped");
    // imshow("Warped", warped);

    // cv::namedWindow("Lanes");
    // imshow("Lanes", linesImage);

    std::vector<int> hist = cVision.Histogram(warped);

    int rectHeight = 120;
    int rectwidth = 60;
    int rectY = src.rows - rectHeight;

    std::vector<cv::Point2f> rightLinePixels = cVision.SlidingWindow(warped, cv::Rect(dstP[2].x - rectwidth, rectY, rectHeight, rectwidth));
    std::vector<cv::Point2f> leftLinePixels = cVision.SlidingWindow(warped, cv::Rect(dstP[3].x - rectwidth, rectY, rectHeight, rectwidth));

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

    double curveRadiusR = pow(1 + pow((2 * fitR[2] * averagedLines[0][1] + fitR[2]), 2), 1.5) / abs(2 * fitR[1]);
    double curveRadiusL = pow(1 + pow((2 * fitL[2] * averagedLines[0][1] + fitL[2]), 2), 1.5) / abs(2 * fitL[1]);

    cv::putText(src, "Curvature left edge: " + std::to_string(curveRadiusL), cv::Point(10, 75), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Curvature right edge: " + std::to_string(curveRadiusR), cv::Point(10, 100), 1, 1.2, cv::Scalar(255, 255, 0));

    double vertexRX = (-fitR[1]) / 2 * fitR[2];
    double vertexLX = (-fitL[1]) / 2 * fitL[2];

    std::string roadType = "";
    int turnThreshold = 200;

    if(vertexLX < turnThreshold){
        roadType = "Left Turn";
    }else if(vertexRX < turnThreshold){
        roadType = "Right Turn";
    }else{
        roadType = "Straight";
    }

    cv::putText(src, roadType, cv::Point(src.cols/2 - 100, 175), 1, 1.5, cv::Scalar(255, 128, 255));

    std::vector<cv::Point2f> outPts;
    std::vector<cv::Point> allPts;

    cv::perspectiveTransform(rightLinePixels, outPts, invertedPerspectiveMatrix);
    cv::line(src, cv::Point(averagedLines[1][0], averagedLines[1][1]), outPts[0], cv::Scalar(0, 255, 0), 3);
    allPts.push_back(cv::Point(averagedLines[1][0], averagedLines[1][1]));

    for (int i = 0; i < outPts.size() - 1; ++i)
    {
        cv::line(src, outPts[i], outPts[i + 1], cv::Scalar(0, 255, 0), 3);
        allPts.push_back(cv::Point(outPts[i].x, outPts[i].y));
    }

    allPts.push_back(cv::Point(outPts[outPts.size() - 1].x, outPts[outPts.size() - 1].y));

    cv::perspectiveTransform(leftLinePixels, outPts, invertedPerspectiveMatrix);

    for (int i = 0; i < outPts.size() - 1; ++i)
    {
        cv::line(src, outPts[i], outPts[i + 1], cv::Scalar(0, 255, 0), 3);
        allPts.push_back(cv::Point(outPts[outPts.size() - i - 1].x, outPts[outPts.size() - i - 1].y));
    }

    allPts.push_back(cv::Point(outPts[0].x - (outPts.size() - 1), outPts[0].y));
    cv::line(src, cv::Point(averagedLines[0][0], averagedLines[0][1]), outPts[outPts.size() -1], cv::Scalar(0, 255, 0), 3);
    allPts.push_back(cv::Point(averagedLines[0][0], averagedLines[0][1]));

    std::vector<std::vector<cv::Point>> arr;
    arr.push_back(allPts);
    cv::Mat overlay = cv::Mat::zeros(src.size(), src.type());
    cv::fillPoly(overlay, arr, cv::Scalar(0, 255, 100));
    cv::addWeighted(src, 1, overlay, 0.5, 0, src);

    cv::namedWindow("Turn");
    imshow("Turn", src);
}