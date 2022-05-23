#ifndef STREAMSOURCE_H
#define STREAMSOURCE_H
#include <opencv2/opencv.hpp>

class StreamSource
{
    private:
    public:
        virtual void Setup() = 0;
        virtual cv::Mat GetFrameMat() = 0;
        std::string currentImg = "";
        bool outOfImages = false;
};

#endif