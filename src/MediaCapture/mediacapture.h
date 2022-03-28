#pragma once
#include "../VehicleControl/communicationstrategy.h"
#include "opencv2/opencv.hpp"
#include "../PID/PID.h"

class MediaCapture{
    public:
        cv::VideoCapture *capture;
        // CommunicationStrategy *strategy;
        int run(CommunicationStrategy *strategy);
    private:
};