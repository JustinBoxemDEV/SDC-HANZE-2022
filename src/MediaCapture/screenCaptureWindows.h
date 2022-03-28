#ifdef __WIN32__

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <Windows.h>

// source: https://github.com/CasualCoder91/ZumaOpenCVBot

class ScreenCaptureWindows {
    public:
        int run();
    private:
        cv::Mat getMat(HWND hwnd);
        HMONITOR GetPrimaryMonitorHandle();
};
#endif