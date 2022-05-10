#include "videosource.h"

VideoSource::VideoSource(int cameraID){
    capture = new cv::VideoCapture(cameraID);
    capture->set(cv::CAP_PROP_FRAME_WIDTH, 640.0);
    capture->set(cv::CAP_PROP_FRAME_HEIGHT, 480.0);
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
