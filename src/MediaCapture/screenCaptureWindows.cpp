#ifdef __WIN32__

#include "screenCaptureWindows.h"

HMONITOR ScreenCapture::GetPrimaryMonitorHandle() {
	const POINT ptZero = { 0, 0 };
	return MonitorFromPoint(ptZero, MONITOR_DEFAULTTOPRIMARY);
}

cv::Mat ScreenCapture::getMat(HWND hwnd) {
	// get the screen size from your primary window
	MONITORINFO target;
	target.cbSize = sizeof(MONITORINFO);
	HMONITOR pHM = GetPrimaryMonitorHandle();
	GetMonitorInfo(pHM, &target);

    int width = target.rcMonitor.right;
    int height = target.rcMonitor.bottom;

    cv::Mat src;
    src.create(height, width, CV_8UC4);

    HDC hdc = GetDC(hwnd);
    HDC memdc = CreateCompatibleDC(hdc);
    HBITMAP hbitmap = CreateCompatibleBitmap(hdc, width, height);
    HBITMAP oldbmp = (HBITMAP)SelectObject(memdc, hbitmap);

    BitBlt(memdc, 0, 0, width, height, hdc, 0, 0, SRCCOPY);
    SelectObject(memdc, oldbmp);

    BITMAPINFOHEADER  bi = { sizeof(BITMAPINFOHEADER), width, -height, 1, 32, BI_RGB };
    GetDIBits(hdc, hbitmap, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    DeleteObject(hbitmap);
    DeleteDC(memdc);
    ReleaseDC(hwnd, hdc);

    return src;
}

int ScreenCapture::run() {
    HWND hwndDesktop;
	hwndDesktop = GetDesktopWindow();
    int key = 0;
    cv::Mat src;
	MediaCapture mediaCapture;

    while (key != 27) {
        src = getMat(hwndDesktop);
		mediaCapture.ProcessImage(src);
        key = cv::waitKey(1);
    }
	return 0;
}

#endif