#include <stdint.h>
#include <iostream>
#include "mediaCapture.h"
#include <time.h>
#include <string>
#include <filesystem>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../Math/Polynomial.h"
#include <thread>
namespace fs = std::filesystem;
void MediaCapture::ProcessFeed(int cameraID, std::string filename)
{
    if (cameraID != 0)
    {
        capture = new cv::VideoCapture(cameraID);
        capture->set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture->set(cv::CAP_PROP_FRAME_WIDTH, 848);
    }
    else if (filename != "")
    {
        std::cout << filename << std::endl;
        capture = new cv::VideoCapture(filename);
    }
    else
    {
        capture = new cv::VideoCapture(0);
        capture->set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture->set(cv::CAP_PROP_FRAME_WIDTH, 848);

        // Camera detection check
        if (!capture->isOpened())
        {
            std::cout << "NO CAMERA DETECTED!" << std::endl;
            return;
        }
    }
    std::cout << "Camera selected: " << cameraID << std::endl;
    pid.PIDController_Init();

    std::thread tr([&](){ execute();});
    tr.join();
}

void MediaCapture::execute(){
    cv::Mat frame;


    // Define total frames and start of a counter for FPS calculation
    int totalFrames = 0;

    time_t start, end;
    time(&start);

    // Camera feed
    while (capture->read(frame))
    {
        totalFrames++;

        ProcessImage(frame);

        if (cv::waitKey(1) >= 0)
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

cv::Mat MediaCapture::LoadTestImage(std::string filepath)
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
    cv::setTrackbarPos("gamma", "Control", gamma);

    cv::Mat gammaCorrected = cVision.GammaCorrection(src, gamma );
    cVision.SetFrame(gammaCorrected);
    // cv::Mat wipImage;
    // src.copyTo(wipImage);

    cv::Mat binaryImage = cVision.CreateBinaryImage(gammaCorrected);
    cv::Mat maskedImage = cVision.MaskImage(binaryImage);

    std::vector<cv::Vec4i> averagedLines = cVision.GenerateLines(maskedImage);

    double laneOffset = cVision.getLaneOffset();
    double normalisedLaneOffset = cVision.getNormalisedLaneOffset();
    cv::putText(src, "Center Offset: " + std::to_string(laneOffset), cv::Point(10, 25), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Center Offset (N): " + std::to_string(normalisedLaneOffset), cv::Point(10, 50), 1, 1.2, cv::Scalar(255, 255, 0));

    double pidout = pid.PIDController_update(normalisedLaneOffset);
    cv::putText(src, "PID output: " + std::to_string(pidout), cv::Point(10, 125), 1, 1.2, cv::Scalar(255, 255, 0));

    imshow("masked", maskedImage);
    cVision.PredictTurn(maskedImage, averagedLines);
    
    double curveRadiusR = cVision.getRightEdgeCurvature();
    double curveRadiusL = cVision.getLeftEdgeCurvature();
    cv::putText(src, "Curvature left edge: " + std::to_string(curveRadiusL), cv::Point(10, 75), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Curvature right edge: " + std::to_string(curveRadiusR), cv::Point(10, 100), 1, 1.2, cv::Scalar(255, 255, 0));
}