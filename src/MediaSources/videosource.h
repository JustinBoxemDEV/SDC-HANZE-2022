#ifndef VIDEOSOURCE_H
#define VIDEOSOURCE_H
#include "streamsource.h"
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

class VideoSource : public StreamSource
{
    private:
        cv::VideoCapture *capture;
        cv::Mat frame;

        cv::String dir;
        int imgIndex = 0;
        bool outOfImgs = false;
    public:
        VideoSource(std::string path);
        VideoSource(int cameraID);
        void Setup() override;
        cv::Mat GetFrameMat() override;
        // std::string currentImg;
};

#endif