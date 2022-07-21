#ifndef CV_PROCESS_H
#define CV_PROCESS_H

#include "process.h"
#include "opencv2/opencv.hpp"
#include "../VehicleControl/communicationstrategy.h"
#include "../ComputerVision/computervision.h"
#include "../PID/PID.h"
#include "../MediaSources/streamsource.h"
#include <filesystem>

#ifdef linux
#include "../MachineLearning/model.h"
#endif

namespace fs = std::filesystem;

class CVProcess : public Process
{
    private:
        cv::VideoCapture *capture;
        StreamSource *streamSource;
        ComputerVision cVision;
        PIDController pid{0.3,0.2,0.4};
        void ProcessFrame(cv::Mat src);
        #ifdef linux
        Model *model;
        #endif
    public:
        CVProcess(MediaInput *input);
        float gamma = 2; 
        void Run() override;
        void Terminate() override;
};

#endif
