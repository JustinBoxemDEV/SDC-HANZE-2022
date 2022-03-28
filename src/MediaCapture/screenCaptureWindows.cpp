#ifdef __WIN32__

#include "../VehicleControl/strategies/ACStrategy.h"

#include "screenCaptureWindows.h"
#include "../Managers/mediamanager.h"

ACStrategy assettocorsa;

HMONITOR ScreenCaptureWindows::GetPrimaryMonitorHandle() {
	const POINT ptZero = { 0, 0 };
	return MonitorFromPoint(ptZero, MONITOR_DEFAULTTOPRIMARY);
}

cv::Mat ScreenCaptureWindows::getMat(HWND hwnd) {
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

// Scuffed fix for scheduler
void sc_steer(){
    assettocorsa.steer();
}

void sc_brake(){
    assettocorsa.brake();
}

void sc_forward(){
    assettocorsa.forward();
}

int ScreenCaptureWindows::run() {
    HWND hwndDesktop;
	hwndDesktop = GetDesktopWindow();
    int key = 0;
    cv::Mat src;

    MediaManager mediamanager(&assettocorsa);
    mediamanager.pid.PIDController_Init();

    // Wait 2 seconds so you can tab back into the game
    Sleep(2000);
    assettocorsa.gearShiftUp();

    assettocorsa.actuators.throttlePercentage = 80;
    assettocorsa.taskScheduler.SCH_Add_Task(sc_forward, 0, 0.04);
    assettocorsa.taskScheduler.SCH_Add_Task(sc_steer, 0.02, 0.04);
    assettocorsa.taskScheduler.SCH_Start();
    
    while (key != 27) {
        src = getMat(hwndDesktop);

        mediamanager.ProcessImage(src);

        assettocorsa.taskScheduler.SCH_Dispatch_Tasks();

        key = cv::waitKey(1);
    }

	return 0;
}
#endif
