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
    cv::Mat wipImage;
    src.copyTo(wipImage);

    cv::Mat binaryImage = cVision.CreateBinaryImage(wipImage);
    cv::Mat maskedImage = cVision.MaskImage(binaryImage);

    std::vector<cv::Vec4i> averagedLines = cVision.GenerateLines(maskedImage);
    std::vector<std::vector<cv::Point>> arr = cVision.PredictTurn(maskedImage, averagedLines);


    cv::Mat overlay = cv::Mat::zeros(src.size(), src.type());
    cv::fillPoly(overlay, arr, cv::Scalar(0, 255, 100));
    cv::addWeighted(src, 1, overlay, 0.5, 0, src);

    cv::namedWindow("Turn");
    imshow("Turn", src);
}