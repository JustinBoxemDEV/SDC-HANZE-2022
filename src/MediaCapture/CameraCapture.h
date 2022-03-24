#pragma once
#include <stdint.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "../ComputorVision/computorvision.h"
#include "../PID/PID.h"
#include "../VehicleControl/CommunicationStrategy.h"

class CameraCapture {
    public:
        int run(int cameraID);
    private:
        cv::VideoCapture *capture; 
        void getCamera(int cameraID);
};
