#ifndef VIDEOSOURCE_H
#define VIDEOSOURCE_H
#include "streamSource.h"
#include <string>

class VideoSource : public StreamSource
{
    private:
        cv::VideoCapture *capture;
        cv::Mat frame;
    public:
        VideoSource(std::string filepath);
        VideoSource(int cameraID);
        void Setup() override;
        cv::Mat GetFrameMat() override;
};

#endif