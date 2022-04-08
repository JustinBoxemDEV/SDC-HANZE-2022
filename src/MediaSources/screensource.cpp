#ifdef __WIN32__
#include "screensource.h"

void ScreenSource::Setup(){
	hwndDesktop = GetDesktopWindow();
}

cv::Mat ScreenSource::GetFrameMat(){
    MONITORINFO target;
	target.cbSize = sizeof(MONITORINFO);
	HMONITOR pHM = GetPrimaryMonitorHandle();
	GetMonitorInfo(pHM, &target);

    int width = target.rcMonitor.right;
    int height = target.rcMonitor.bottom;

    cv::Mat src;
    src.create(height, width, CV_8UC4);

    HDC hdc = GetDC(hwndDesktop);
    HDC memdc = CreateCompatibleDC(hdc);
    HBITMAP hbitmap = CreateCompatibleBitmap(hdc, width, height);
    HBITMAP oldbmp = (HBITMAP)SelectObject(memdc, hbitmap);

    BitBlt(memdc, 0, 0, width, height, hdc, 0, 0, SRCCOPY);
    SelectObject(memdc, oldbmp);

    BITMAPINFOHEADER  bi = { sizeof(BITMAPINFOHEADER), width, -height, 1, 32, BI_RGB };
    GetDIBits(hdc, hbitmap, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    DeleteObject(hbitmap);
    DeleteDC(memdc);
    ReleaseDC(hwndDesktop, hdc);

    return src;
}

HMONITOR ScreenSource::GetPrimaryMonitorHandle(){
	const POINT ptZero = { 0, 0 };
	return MonitorFromPoint(ptZero, MONITOR_DEFAULTTOPRIMARY);
}

#endif