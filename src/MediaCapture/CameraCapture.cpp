#include "CameraCapture.h"

CameraCapture::CameraCapture(int cameraID){
    getCamera(cameraID);
}

void CameraCapture::getCamera(int cameraID){
    capture = new cv::VideoCapture(cameraID);

    // Comment this out when not using the extended camera (CameraID 4)
    // capture->set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // capture->set(cv::CAP_PROP_FRAME_WIDTH, 848);
    
    // Camera detection check
    if (!capture->isOpened()){
        std::cout << "NO CAMERA DETECTED!" << std::endl;
    }else{
        std::cout << "Camera selected: " << cameraID << std::endl;
    }
}
