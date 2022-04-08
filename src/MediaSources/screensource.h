#ifdef __WIN32__
#ifndef SCREENSOURCE_H
#define SCREENSOURCE_H
#include "streamsource.h"

#include <Windows.h>

class ScreenSource : public StreamSource
{
    private:
        HMONITOR GetPrimaryMonitorHandle();
        HWND hwndDesktop;
    public:
        void Setup() override;
        cv::Mat GetFrameMat() override;
};

#endif
#endif