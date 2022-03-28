#include "VidCapture.h"

VidCapture::VidCapture(std::string filepath) {
    capture = new cv::VideoCapture(filepath);

    if (!capture->isOpened()){
        std::cout << "NO File detected!" << std::endl;
    }else{
        capture->set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture->set(cv::CAP_PROP_FRAME_WIDTH, 848);
        std::cout << "File selected: " << std::endl;
    };
};