#pragma once
#include <stdint.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "../ComputorVision/computorvision.h"
#include "../VehicleControl/communicationstrategy.h"
#include "mediacapture.h"

class CameraCapture : public MediaCapture{
    public:
        CameraCapture(int cameraID);
    private:
        void getCamera(int cameraID);
};
