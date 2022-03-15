#ifndef MEDIA_CAPTURE_H
#define MEDIA_CAPTURE_H

#include <stdint.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "../ComputorVision/computorvision.h"


class MediaCapture
{
    private:
        ComputorVision cVision;
        // void 
    public:
        void ProcessFeed(int cameraID, std::string filename);
        cv::Mat LoadImage(std::string filename);
        void ProcessImage(cv::Mat src);
};

#endif