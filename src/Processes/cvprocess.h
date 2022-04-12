#ifndef CV_PROCESS_H
#define CV_PROCESS_H

#include "process.h"
#include "opencv2/opencv.hpp"
#include "../VehicleControl/communicationstrategy.h"
#include "../ComputerVision/computervision.h"
#include "../PID/PID.h"
#include "../MediaSources/streamsource.h"
#include <filesystem>
#include "../Processes/canprocess.h"

namespace fs = std::filesystem;

class CVProcess : public Process
{
    private:
        cv::VideoCapture *capture;
        StreamSource *streamSource;
        ComputerVision cVision;
        PIDController pid{0.20,0.4211,0.0414};
        void ProcessFrame(cv::Mat src);
    public:
        void setCanProcess(CanProcess *_canProcess);
        CVProcess(MediaInput *input);
        float gamma = 2; 
        void Run() override;
        void Terminate() override;
};

#endif
