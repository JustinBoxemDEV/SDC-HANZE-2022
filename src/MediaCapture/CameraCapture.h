#pragma once
#include <stdint.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "../ComputorVision/computorvision.h"
#include "../PID/PID.h"
#include "../VehicleControl/communicationstrategy.h"
#include "../VehicleControl/strategies/canstrategy.h"
#include "mediaCapture.h"

class CameraCapture {
    public:
        int run(int cameraID);
    private:
        cv::VideoCapture *capture; 
        void getCamera(int cameraID);
};
