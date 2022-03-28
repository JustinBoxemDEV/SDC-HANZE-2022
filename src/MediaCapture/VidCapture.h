#pragma once
#include <stdint.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "../ComputorVision/computorvision.h"
#include "../PID/PID.h"
#include "../VehicleControl/communicationstrategy.h"
#include "../VehicleControl/strategies/canstrategy.h"
#include "../Managers/mediamanager.h"
#include "mediacapture.h"

class VidCapture : public MediaCapture {
    public:
        VidCapture(std::string filepath);
    private:
};
