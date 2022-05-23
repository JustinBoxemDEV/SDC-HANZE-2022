#include "cvprocess.h"
#include <iostream>
#include "../MediaSources/screensource.h"
#include "../MediaSources/videosource.h"

#ifdef __WIN32__
CommunicationStrategy::Actuators CommunicationStrategy::actuators;
#endif

#define MODEL_PATH "/assets/models/model_traced_90p.pt"

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
            model = new Model(fs::current_path().string()+MODEL_PATH);
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

void CVProcess::ProcessFrame(cv::Mat src){
    if(mediaInput->mediaType == MediaSource::realtime_ml) {
        imshow("src", src);
        #ifdef __linux__
        model->Inference(src);
        #endif
    } else if (mediaInput->mediaType == MediaSource::images_ml) {
        imshow("src", src);
        std::string img = streamSource->currentImg;
        #ifdef __linux__
        model->EnableCSV();
        model->Inference(src, img);
        if (streamSource->outOfImages)
            model->closeCSV();
        #endif
    } else {
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

        if(pidout >= -1 && pidout <=1) {
            CommunicationStrategy::actuators.steeringAngle = pidout;
        };

        cv::putText(src, "PID output: " + std::to_string(pidout), cv::Point(10, 125), 1, 1.2, cv::Scalar(255, 255, 0));

        // imshow("masked", maskedImage);
        cVision.PredictTurn(maskedImage, averagedLines);
        
        double curveRadiusR = cVision.getRightEdgeCurvature();
        double curveRadiusL = cVision.getLeftEdgeCurvature();
        cv::putText(src, "Curvature left edge: " + std::to_string(curveRadiusL), cv::Point(10, 75), 1, 1.2, cv::Scalar(255, 255, 0));
        cv::putText(src, "Curvature right edge: " + std::to_string(curveRadiusR), cv::Point(10, 100), 1, 1.2, cv::Scalar(255, 255, 0));
    }
}

void CVProcess::Terminate(){
    
}