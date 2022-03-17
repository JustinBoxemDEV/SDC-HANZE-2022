#ifdef __WIN32__

#include "screenCaptureWindows.h"

// #include <iostream>
// #include <sstream>
// #include <chrono>

cv::Mat screenCapture::getMat(HWND hWND) {
	HDC deviceContext = GetDC(hWND);
	HDC memoryDeviceContext = CreateCompatibleDC(deviceContext);

	RECT windowRect;
	GetClientRect(hWND, &windowRect);

	int height = windowRect.bottom;
	int width = windowRect.right;

	HBITMAP bitmap = CreateCompatibleBitmap(deviceContext, width, height);

	SelectObject(memoryDeviceContext, bitmap);

	//copy data into bitmap
	BitBlt(memoryDeviceContext, 0, 0, width, height, deviceContext, 0, 0, SRCCOPY);

	//specify format by using bitmapinfoheader!
	BITMAPINFOHEADER bi;
	bi.biSize = sizeof(BITMAPINFOHEADER);
	bi.biWidth = width;
	bi.biHeight = -height;
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0; //because no compression
	bi.biXPelsPerMeter = 1; //we
	bi.biYPelsPerMeter = 2; //we
	bi.biClrUsed = 3; //we ^^
	bi.biClrImportant = 4; //still we

	cv::Mat mat = cv::Mat(height, width, CV_8UC4); // 8 bit unsigned ints 4 Channels -> RGBA

	//transform data and store into mat.data
	GetDIBits(memoryDeviceContext, bitmap, 0, height, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    //clean up!
	DeleteObject(bitmap);
	DeleteDC(memoryDeviceContext); //delete not release!
	ReleaseDC(hWND, deviceContext);

	return mat;
}

int screenCapture::run() {
    LPCSTR window_title = LPCSTR("Assetto Corsa Launcher");
	HWND hWND = FindWindow(NULL, window_title);
	cv::namedWindow("Assetto Corsa screen capture", cv::WINDOW_NORMAL);
	int key = 0;

	while (key != 27) {
		HWND temp = GetForegroundWindow();
		if (temp != hWND) {
			Sleep(10);
			continue;
		}

		cv::Mat target = getMat(hWND);
		cv::Mat background;
		target.copyTo(background);
		cv::cvtColor(target, target, cv::COLOR_BGR2HSV); // Convert the image into HSV image
		cv::rectangle(target, cv::Point(0, 0), cv::Point(640, 30), CV_RGB(0, 0, 0), cv::FILLED); //set top menue black

		cv::imshow("output", background);
		key = cv::waitKey(30);
	}
	
    // while(!hWND) {
    //     std::system("cls");
    //     std::cout << "Start Asetto Corsa..." << std::endl;
    //     Sleep(100);
    // }

    // cv::namedWindow("Asetto Corsa Screen Capture", cv::WINDOW_NORMAL);

    // while (true) {
    //     cv::Mat target = getMat(hWND);

	// 	cv::Mat background;
	// 	target.copyTo(background);	
	// 	cv::cvtColor(target, target, cv::COLOR_BGR2HSV); // Convert the image into HSV image
	
    //     cv::imshow("Asetto Corsa Screen Capture", background);
    //     cv::waitKey();
    // }

    return 1;
}

#endif