#include "cvprocess.h"
#include <iostream>


void CVProcess::Init(MediaInput *input){
    Process::Init(input);

    pid.PIDController_Init();

    switch (input->mediaType)
    {
        case MediaSource::video:
            if(!fs::exists(input->filepath)){
                break;
            }
            capture = new cv::VideoCapture(input->filepath);
            break;
        case MediaSource::realtime:
            capture = new cv::VideoCapture(input->cameraID);
            break;
        case MediaSource::assetto:
            capture = new cv::VideoCapture(input->cameraID);
            break;
    }
}

void CVProcess::Run(){
    cv::Mat frame;

    while (capture->read(frame)){
        std::cout << "Processing Frame" << std::endl;

        ProcessFrame(frame);
        if (cv::waitKey(1)>0){
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