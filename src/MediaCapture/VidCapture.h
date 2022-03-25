#pragma once
#include <stdint.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "../ComputorVision/computorvision.h"
#include "../PID/PID.h"
#include "../VehicleControl/communicationstrategy.h"
#include "../VehicleControl/strategies/canstrategy.h"
#include "MediaCapture.h"

class VidCapture {
    public:
        int run(std::string filename);
    private:
        cv::VideoCapture *capture; 
};
