#include "cvprocess.h"
#include <iostream>
#include "../MediaSources/screensource.h"
#include "../MediaSources/videosource.h"

#ifdef __WIN32__
CommunicationStrategy::Actuators CommunicationStrategy::actuators;
#endif

#define MODEL_PATH "/assets/models/traced_SLSelfDriveModel_2022-05-23_00-55-20_Adam_0.00001.pt"

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
        case MediaSource::realtime_ml:
            streamSource = new VideoSource(input->cameraID);
            #ifdef linux
            model = new Model(input->filepath);
            #endif
            break;
        case MediaSource::images_ml:
            streamSource = new VideoSource(input->filepath);
            #ifdef linux
            model = new Model(fs::current_path().string()+MODEL_PATH);
            #endif
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

void CVProcess::ProcessFrame(cv::Mat src) {
    // if(mediaInput->mediaType == MediaSource::realtime || mediaInput->mediaType == MediaSource::video) {
    //     cv::Mat cameraMatrix = (cv::Mat1d(3,3) << 792.13574, 0, 319.5, 0, 792.13574, 239.5, 0, 0, 1);
    //     cv::Mat distortionCoefficients = (cv::Mat1d(1, 5) << 0.0905006, -0.55128, 0, 0, 0);

    //     cv::Mat temp = src.clone();
    //     cv::undistort(temp, src, cameraMatrix, distortionCoefficients);

    //     cv::imshow("distorted image", temp);
    //     cv::imshow("undistorted image", src);
    // }

    cv::Mat gammaCorrected = cVision.GammaCorrection(src, gamma );
    cVision.SetFrame(gammaCorrected);

    cv::Mat binaryImage = cVision.CreateBinaryImage(gammaCorrected);
    cv::Mat maskedImage = cVision.MaskImage(binaryImage);
    
    cVision.PredictTurn(maskedImage);
    
    double curveRadiusR = cVision.getRightEdgeCurvature();
    double curveRadiusL = cVision.getLeftEdgeCurvature();
    cv::putText(src, "Curvature left edge: " + std::to_string(curveRadiusL), cv::Point(10, 75), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Curvature right edge: " + std::to_string(curveRadiusR), cv::Point(10, 100), 1, 1.2, cv::Scalar(255, 255, 0));

    double laneOffset = cVision.getLaneOffset();
    double normalisedLaneOffset =  cVision.getNormalisedLaneOffset();
    cv::putText(src, "Center Offset: " + std::to_string(laneOffset), cv::Point(10, 25), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Center Offset (N): " + std::to_string(normalisedLaneOffset), cv::Point(10, 50), 1, 1.2, cv::Scalar(255, 255, 0));

        double pidout = pid.PIDController_update(normalisedLaneOffset);

    // std::cout << "pidout = "<< pidout << std::endl;
    // std::cout << "normilised lane offset = "<< normalisedLaneOffset << std::endl;

    if(pidout >= -1 && pidout <=1) {
        CommunicationStrategy::actuators.steeringAngle = static_cast<float>(pidout);
    };

    cv::putText(src, "PID output: " + std::to_string(pidout), cv::Point(10, 125), 1, 1.2, cv::Scalar(255, 255, 0));
    imshow("masked", maskedImage);
}

void CVProcess::Terminate(){
    
}