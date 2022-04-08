#include "videosource.h"

VideoSource::VideoSource(int cameraID){
    capture = new cv::VideoCapture(cameraID);
}

VideoSource::VideoSource(std::string filepath){
    capture = new cv::VideoCapture(filepath);
}

void VideoSource::Setup(){
}

cv::Mat VideoSource::GetFrameMat(){
    capture->read(frame);
    return frame;
}
