#include "cvprocess.h"
#include <iostream>
#include "../MediaSources/screensource.h"
#include "../MediaSources/videosource.h"

CanProcess *canProcess;

void CVProcess::setCanProcess(CanProcess *_canProcess) {
    canProcess = _canProcess;
};

CVProcess::CVProcess(MediaInput *input){
    mediaInput = input;
    pid.PIDController_Init();

    switch (input->mediaType)
    {
        case MediaSource::video:
            if(!fs::exists(input->filepath)){
                std::cout << input->filepath << std::endl;
                break;
            }
            streamSource = new VideoSource(input->filepath);
            break;
        case MediaSource::realtime:
            streamSource = new VideoSource(input->cameraID);
            break;
        case MediaSource::assetto:
            #if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
                streamSource = new ScreenSource();
            #endif
            break;
    }
}

void CVProcess::Run(){
    cv::Mat frame;
    
    if(streamSource == nullptr){
        std::cout << "Could not set stream source!" << std::endl;
        return;
    }

    while (true){
        frame = streamSource->GetFrameMat();
        
        if(frame.empty()){
            break;
        }

        ProcessFrame(frame);
        if (cv::waitKey(100/60)>0){
            break;
        };
    };
}

void CVProcess::ProcessFrame(cv::Mat src){
    cv::Mat gammaCorrected = cVision.GammaCorrection(src, gamma );
    cVision.SetFrame(gammaCorrected);

    cv::Mat binaryImage = cVision.CreateBinaryImage(gammaCorrected);
    cv::Mat maskedImage = cVision.MaskImage(binaryImage);

    std::vector<cv::Vec4i> averagedLines = cVision.GenerateLines(maskedImage);

    double laneOffset = cVision.getLaneOffset();
    double normalisedLaneOffset = cVision.getNormalisedLaneOffset();
    cv::putText(src, "Center Offset: " + std::to_string(laneOffset), cv::Point(10, 25), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Center Offset (N): " + std::to_string(normalisedLaneOffset), cv::Point(10, 50), 1, 1.2, cv::Scalar(255, 255, 0));

    double pidout = pid.PIDController_update(normalisedLaneOffset);

    // This needs to be fixed to set steeringAngle
    canProcess->actuators.steeringAngle = pidout;

    cv::putText(src, "PID output: " + std::to_string(pidout), cv::Point(10, 125), 1, 1.2, cv::Scalar(255, 255, 0));

    imshow("masked", maskedImage);
    cVision.PredictTurn(maskedImage, averagedLines);
    
    double curveRadiusR = cVision.getRightEdgeCurvature();
    double curveRadiusL = cVision.getLeftEdgeCurvature();
    cv::putText(src, "Curvature left edge: " + std::to_string(curveRadiusL), cv::Point(10, 75), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Curvature right edge: " + std::to_string(curveRadiusR), cv::Point(10, 100), 1, 1.2, cv::Scalar(255, 255, 0));
}

void CVProcess::Terminate(){
    
}