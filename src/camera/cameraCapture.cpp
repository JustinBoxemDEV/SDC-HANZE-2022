#include "opencv2/opencv.hpp"
#include <stdint.h>
#include "cameraCapture.h"
#include <time.h>

void CameraCapture::ProcessFeed(){

    cv::Mat frame;
    cv::VideoCapture videoCapture(0);

    videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);

    // Camera detection check
    if(!videoCapture.isOpened()){
        std::cout << "NO CAMERA DETECTED!" << std::endl;
        return;
    }

    // Define total frames and start of a counter for FPS calculation
    int totalFrames = 0;
     
    time_t start, end;

    time(&start);

    // Camera feed
    while (videoCapture.read(frame))
    {
        imshow("Video Feed", frame);

        totalFrames++;

        if(cv::waitKey(1000/60)>=0){
            break;
        }
    }

    // End the time counter
    time(&end);

    // Time elapsed
    double seconds = difftime (end, start);
    std::cout << "Time taken : " << seconds << " seconds" << std::endl;

    // Estimate the FPS based on frames / elapsed time in seconds
    int fps  = totalFrames / seconds;
    std::cout << "Estimated frames per second : " << fps << std::endl;
    
}