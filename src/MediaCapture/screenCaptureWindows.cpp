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
ACStrategy assettocorsa;

// Scuffed fix for scheduler
void steer(){
    assettocorsa.steer();
}

void brake(){
    assettocorsa.brake();
}

void throttle(){
    assettocorsa.throttle();
}

int ScreenCapture::run() {
    HWND hwndDesktop;
	hwndDesktop = GetDesktopWindow();
    int key = 0;
    cv::Mat src;

    MediaCapture mediacapture(&assettocorsa);
    mediacapture.pid.PIDController_Init();

    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
    // Wait 2 seconds so you can tab back into the game
    Sleep(2000);
    assettocorsa.gearShiftUp();
    #endif

    assettocorsa.actuators.throttlePercentage = 80;
    assettocorsa.taskScheduler.SCH_Add_Task(throttle, 0, 0.04);
    assettocorsa.taskScheduler.SCH_Add_Task(steer, 0.02, 0.04);
    assettocorsa.taskScheduler.SCH_Start();
    
    while (key != 27) {
        src = hwnd2mat(hwndDesktop);

        mediacapture.ProcessImage(src);

        assettocorsa.taskScheduler.SCH_Dispatch_Tasks();

        key = cv::waitKey(1);
    }

	return 0;
}

#endif