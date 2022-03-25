#ifdef __WIN32__

#pragma once

#include "mediaCapture.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <Windows.h>

// source: https://github.com/CasualCoder91/ZumaOpenCVBot

class ScreenCapture {
    public:
        int run();
    private:
        HMONITOR GetPrimaryMonitorHandle();
        cv::Mat getMat(HWND hwnd);
};

#endif