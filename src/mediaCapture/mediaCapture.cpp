#include "opencv2/opencv.hpp"
#include <stdint.h>
#include <iostream>
#include "mediaCapture.h"
#include <time.h>
#include <string>

void MediaCapture::ProcessFeed(int cameraID, char* filename){
   cv::VideoCapture* capture;

    if(cameraID!=0)
    {
        capture = new cv::VideoCapture(cameraID);
        capture->set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        capture->set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    }
    if(filename!="")
    {
        capture = new cv::VideoCapture(filename);
    }
    else
    {
        capture = new cv::VideoCapture(0);

        // Camera detection check
        if(!capture->isOpened())
        {
            std::cout << "NO CAMERA DETECTED!" << std::endl;
            return;
        }
    }

    cv::Mat frame;
    std::cout << "Camera selected: " << cameraID << std::endl;

    // Define total frames and start of a counter for FPS calculation
    int totalFrames = 0;
     
    time_t start, end;

    time(&start);

    // Camera feed
    while (capture->read(frame))
    {
        //call 2 robin's function
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