#ifndef MEDIASTREAM_H
#define MEDIASTREAM_H
#include <opencv2/opencv.hpp>

class MediaStream
{
    private:
    public:
        virtual void Setup() = 0;
        virtual cv::Mat GetFrameMat() = 0;
};

#endif