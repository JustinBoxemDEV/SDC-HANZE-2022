#ifdef __WIN32__

#include "../VehicleControl/strategies/ACStrategy.h"

#include "screenCaptureWindows.h"
#include "mediaCapture.h"

HMONITOR GetPrimaryMonitorHandle() {
	const POINT ptZero = { 0, 0 };
	return MonitorFromPoint(ptZero, MONITOR_DEFAULTTOPRIMARY);
}

cv::Mat hwnd2mat(HWND hwnd) {
    RECT rc;
    GetClientRect(hwnd, &rc);
    int width = rc.right;//rc.right;
    int height = rc.bottom;

	// //testssssssssss
	// std::cout << "all windows bottom:"  << std::endl;
	// std::cout << rc.bottom  << std::endl;
	// std::cout << "all windows right:"  << std::endl;
	// std::cout << rc.right  << std::endl;

	// // LPMONITORINFO target;
	// // HMONITOR pMH = GetPrimaryMonitorHandle();
	// // GetMonitorInfo(pMH, target);

	// MONITORINFO target;
	// target.cbSize = sizeof(MONITORINFO);

	// HMONITOR pHM = GetPrimaryMonitorHandle();
	// GetMonitorInfo(pHM, &target);

	// std::cout << "primary window bottom:"  << std::endl;
	// std::cout << target.rcMonitor.bottom << std::endl;
	// std::cout << "primary window right:"  << std::endl;
	// std::cout << target.rcMonitor.right << std::endl;


	//end testtttttttttttts

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

    //Hardcoded VehicleStrategy
    // ACStrategy assettocorsa;
    MediaCapture mediacapture;
    mediacapture.pid.PIDController_Init();

    // AC Specific things here


    while (key != 27) {
        src = hwnd2mat(hwndDesktop);
        // you can do some image processing here
        // imshow("output", src);

        mediacapture.ProcessImage(src);
        // mediacapture.execute();


        key = cv::waitKey(1); // you can change wait time
    }

	return 0;
}

#endif