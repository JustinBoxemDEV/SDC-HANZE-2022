#ifndef CV_PROCESS_H
#define CV_PROCESS_H

#include "process.h"
#include "opencv2/opencv.hpp"
#include "VehicleControl/communicationstrategy.h"
#include "ComputorVision/computorvision.h"
#include "PID/PID.h"
#include <filesystem>

namespace fs = std::filesystem;

class CVProcess : public Process
{
    private:
        cv::VideoCapture *capture;
        ComputorVision cVision;
        PIDController pid{1.0, 0.1, 0.1};
        void ProcessFrame(cv::Mat src);
    public:
        CVProcess(MediaInput *input);
        float gamma = 2; 
        void Run() override;
        void Terminate() override;
};

#endif
