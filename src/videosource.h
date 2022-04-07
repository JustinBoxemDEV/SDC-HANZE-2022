#ifdef __WIN32__
#ifndef VIDEOSOURCE_H
#define VIDEOSOURCE_H
#include "mediastream.h"

#include <Windows.h>
#include <string>

class VideoSource : public MediaStream
{
    private:
        HMONITOR GetPrimaryMonitorHandle();
        HWND* hwndDesktop;
        cv::VideoCapture *capture;
        cv::Mat frame;
    public:
        VideoSource(std::string filepath);
        VideoSource(int cameraID);
        void Setup() override;
        cv::Mat GetFrameMat() override;
};

#endif
#endif