#ifndef MEDIA_CAPTURE_H
#define MEDIA_CAPTURE_H

#include <stdint.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "../ComputorVision/computorvision.h"
#include "../PID/PID.h"
#include "../VehicleControl/CommunicationStrategy.h"

class MediaCapture
{
    private:
        ComputorVision cVision;
        cv::VideoCapture *capture; 
        CommunicationStrategy* strategy;
    public:
        MediaCapture(){}
        MediaCapture(CommunicationStrategy *strategy){
            this->strategy = strategy;
        }

        void execute();
        PIDController pid{0.6, 1.2, 2};
        void ProcessFeed(int cameraID, std::string filename);
        cv::Mat LoadImage(std::string filename);
        void ProcessImage(cv::Mat src);
};

#endif
