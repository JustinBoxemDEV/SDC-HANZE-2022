#include "VidCapture.h"

VidCapture::VidCapture(std::string filepath) {
    capture = new cv::VideoCapture(filepath);

    if (!capture->isOpened()){
        std::cout << "NO File detected!" << std::endl;
    }else{
        std::cout << "File selected: " << std::endl;
    };
};