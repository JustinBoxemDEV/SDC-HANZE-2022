
#pragma once

#include <stdint.h>
#include <iostream>
#include <time.h>
#include <string>
#include <filesystem>
#include <thread>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../ComputorVision/computorvision.h"
#include "../PID/PID.h"
#include "../VehicleControl/communicationstrategy.h"
#include "../VehicleControl/strategies/acstrategy.h"
#include "../Math/Polynomial.h"
#include "../MediaCapture/screenCaptureWindows.h"
#include "../MediaCapture/CameraCapture.h"

class MediaCapture {
    private:
        ComputorVision cVision;
        cv::VideoCapture *capture; 
        CommunicationStrategy* strategy;
    public:
        MediaCapture(){}
        MediaCapture(CommunicationStrategy *strategy){
            this->strategy = strategy;
        }

        void execute();
        PIDController pid{0.6, 1.2, 2};
        void ProcessFeed(bool screenCapture = false, int cameraID = 0, std::string filepath = "");
        cv::Mat LoadImg(std::string filename);
        void ProcessImage(cv::Mat src);
};