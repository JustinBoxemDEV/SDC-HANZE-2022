#include "MediaCapture.h"
#include "../MediaCapture/VidCapture.h"

namespace fs = std::filesystem;

void MediaCapture::ProcessFeed(bool screenCapture, int cameraID, std::string filepath){
    if (screenCapture){
        #ifdef __WIN32__
        ScreenCaptureWindows screenCaptureWindows;
        screenCaptureWindows.run();
        #else
        std::cout << "Screen capture currently not available on linux!" << std::endl;
        #endif
    } else if (filepath != ""){
        if(!fs::exists(filepath)){
            std::cout << "Cant find file in path: " << filepath << std::endl;
        }else{
            std::cout << "File found: " << filepath << std::endl;
            VidCapture vidcapture;
            vidcapture.run(filepath);
        }
    } else{
        #ifdef linux
        CameraCapture cameraCapture;
        cameraCapture.run(cameraID);
        #endif
    }
}

void MediaCapture::ProcessImage(cv::Mat src){
    cVision.SetFrame(src);
    // cv::Mat wipImage;
    // src.copyTo(wipImage);

    cv::Mat binaryImage = cVision.CreateBinaryImage(src);
    cv::Mat maskedImage = cVision.MaskImage(binaryImage);

    std::vector<cv::Vec4i> averagedLines = cVision.GenerateLines(maskedImage);

    double laneOffset = cVision.getLaneOffset();
    double normalisedLaneOffset = cVision.getNormalisedLaneOffset();
    cv::putText(src, "Center Offset: " + std::to_string(laneOffset), cv::Point(10, 25), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Center Offset (N): " + std::to_string(normalisedLaneOffset), cv::Point(10, 50), 1, 1.2, cv::Scalar(255, 255, 0));

    double pidout = pid.PIDController_update(normalisedLaneOffset);
    
    cv::putText(src, "PID output: " + std::to_string(pidout), cv::Point(10, 125), 1, 1.2, cv::Scalar(255, 255, 0));

    if(strategy != nullptr && !isnan(pidout)){
        strategy->actuators.steeringAngle = pidout;
        std::cout << "Steering with: " << pidout << std::endl;
    }
    
    cVision.PredictTurn(maskedImage, averagedLines);
    double curveRadiusR = cVision.getRightEdgeCurvature();
    double curveRadiusL = cVision.getLeftEdgeCurvature();
    cv::putText(src, "Curvature left edge: " + std::to_string(curveRadiusL), cv::Point(10, 75), 1, 1.2, cv::Scalar(255, 255, 0));
    cv::putText(src, "Curvature right edge: " + std::to_string(curveRadiusR), cv::Point(10, 100), 1, 1.2, cv::Scalar(255, 255, 0));

}

// WE DONT USE ANYTHING BELOW THIS RIGHT NOW
void MediaCapture::execute(){
    // std::cout << "EXECUTING " << std::endl;
    // cv::Mat frame;

    // // Define total frames and start of a counter for FPS calculation
    // int totalFrames = 0;

    // time_t start, end;
    // time(&start);

    // // Camera feed
    // while (capture->read(frame)){
    //     totalFrames++;
    //     ProcessImage(frame);

    //     if (cv::waitKey(1000 / 60) >= 0){
    //         break;
    //     }
    // }

    // // End the time counter
    // time(&end);

    // // Time elapsed
    // double seconds = difftime(end, start);
    // std::cout << "Time taken : " << seconds << " seconds" << std::endl;

    // // Estimate the FPS based on frames / elapsed time in seconds
    // int fps = totalFrames / seconds;
    // std::cout << "Estimated frames per second : " << fps << std::endl;
}

cv::Mat MediaCapture::LoadImg(std::string filepath){
    std::string path = fs::current_path().string() + "/assets/images/" + std::string(filepath);
    cv::Mat img = imread(path, cv::IMREAD_COLOR);
    if (!fs::exists(path)){
        std::cout << "The requested file cannot be found in /assets/images/!" << std::endl;
        return img;
    }

    if (img.empty())
    {
        std::cout << "Could not read the image: " << path << std::endl;
        return img;
    }
    return img;
}