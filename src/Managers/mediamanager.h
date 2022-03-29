#pragma once

#include <filesystem>

#include "../ComputorVision/computorvision.h"
#include "../MediaCapture/CameraCapture.h"
#include "../MediaCapture/VidCapture.h"
#include "../MediaCapture/screenCaptureWindows.h"
#include "../VehicleControl/strategies/acstrategy.h"
#include "../VehicleControl/strategies/canstrategy.h"

class MediaManager {
    private:
        ComputorVision cVision;
        CommunicationStrategy* strategy;
    public:
        MediaManager(){};
        MediaManager(CommunicationStrategy *strategy) {
            this->strategy = strategy;
        };
        void execute();
        PIDController pid{0.6, 1.2, 2};
        void ProcessFeed(bool screenCapture = false, int cameraID = 0, std::string filepath = "");
        cv::Mat LoadImg(std::string filename);
        void ProcessImage(cv::Mat src);
};