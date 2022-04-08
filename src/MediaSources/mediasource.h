#ifndef MEDIASOURCE_H
#define MEDIASOURCE_H
#include <opencv2/opencv.hpp>

class MediaSource
{
    private:
    public:
        virtual void Setup() = 0;
        virtual cv::Mat GetFrameMat() = 0;
};

#endif