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

    public:
        void ProcessFeed(int cameraID=0, std::string filename="");
        cv::Mat LoadImage(std::string filename);
        void ProcessImage(cv::Mat src);
};

#endif