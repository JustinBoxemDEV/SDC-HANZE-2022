#include "videosource.h"

VideoSource::VideoSource(int cameraID){
    capture = new cv::VideoCapture(cameraID);
}

VideoSource::VideoSource(std::string path){
    // path is a mp4
    if (path.substr(path.find_last_of(".") + 1) == "mp4") {
        capture = new cv::VideoCapture(path);
    } 
    // path is a folder
    else if (fs::is_directory(path)) {
        dir = path;
    } else {
        std::cerr << "path is not a mp4 or directory";
    }
}

void VideoSource::Setup(){
}

cv::Mat VideoSource::GetFrameMat(){
    if (!dir.empty()) {
        // frame =
        cv::String path(dir+"/*.jpg");
        std::vector<cv::String> fn;
        std::vector<cv::Mat> data;
        cv::glob(path,fn,true); // recurse
        if (imgIndex < fn.size()){
            frame = cv::imread(fn[imgIndex]);
            std::string newCurrentImg = fn[imgIndex].substr(fn[imgIndex].find_last_of("/\\") + 1);
            currentImg = newCurrentImg;
            imgIndex++;
        } else {
            if (!outOfImages) {
                std::cerr << "no more images found" << std::endl;
                outOfImages = true;
            }
        }
    } else {
        capture->read(frame);
    }
    return frame;
}
