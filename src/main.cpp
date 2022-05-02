#include "application.h"
#include "Processes/canprocess.h"
#include "Processes/cvprocess.h"
#include "Processes/readprocess.h"
#include "Processes/terminalprocess.h"
#include "ComputerVision/cameracalibration.hpp"

// namespace fs = std::filesystem;

// int main(int argc, char** argv) {
//     // std::string path = "/home/douwe/Projects/SDC-HANZE-2022/assets/images/calibration_dsh/";
//     // std::cout << path << std::endl;
//     // CameraCalibration calib(path, 20, 14, 20, 480, 640);
//   std::string path = "/home/douwe/Projects/SDC-HANZE-2022/assets/images/fuzz/";
//   std::vector<cv::String> fileNames;

//   std::vector<std::vector<cv::Point2f>> q(fileNames.size());
//   std::vector<std::vector<cv::Point3f>> Q;

//   cv::glob(path, fileNames, false);

//   std::cout << "Calibrating" << std::endl;

//   // Show lens corrected images
//   for (auto const &f : fileNames) {
//       std::cout << std::string(f) << std::endl;

//       cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
//       cv::Mat temp = img.clone();

//       // cv::Mat cameraMatrix = (cv::Mat1d(3,3) << 960.48218, 0, 319.5, 0, 960.48218, 239.5, 0, 0, 1); // OLD VALUES
//       // cv::Mat cameraMatrix = (cv::Mat1d(3,3) << 675.47607, 0, 319.5, 0, 675.47607, 239.5, 0, 0, 1);
//       cv::Mat cameraMatrix = (cv::Mat1d(3,3) << 1556.9927, 0, 239.5,0, 1556.9927, 319.5, 0, 0, 1); // 0.367196 reprojection error - flipped resolution


//       // cv::Mat distortionCoefficients = (cv::Mat1d(1, 5) << 0.207945, -1.80821, 0, 0, 0); // OLD VALUES
//       // cv::Mat distortionCoefficients = (cv::Mat1d(1, 5) << -0.0649378, -0.0861269, 0, 0, 0);
//         cv::Mat distortionCoefficients = (cv::Mat1d(1, 5) << -0.13394, 0.641369, 0, 0, 0); // 0.367196 reprojection error - flipped resolution


//       cv::undistort(temp, img, cameraMatrix, distortionCoefficients);

//       // Display
//       cv::imshow("distorted image", temp);
//       cv::imshow("undistorted image", img);
//       cv::waitKey(0);
//   }
// }

// #include "application.h"
// #include "Processes/canprocess.h"
// #include "Processes/cvprocess.h"
// #include "Processes/readprocess.h"
// #include "Processes/terminalprocess.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    int cursor = 1;
    Process::MediaInput mediaInput;
    Application application;

    std::cout << "before while" << std::endl;
    std::string arg;
    while(cursor < argc){
        arg = argv[cursor]; 
        if(arg == "-video"){
            std::cout << "video" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::video;
            cursor++;
            std::string path = fs::current_path().string() + "/assets/videos/" + argv[cursor];
            std::cout << path << std::endl;
            if(!fs::exists(path)){
                std::cout << "file does not exists!" << std::endl;
                return 1;
            }
            mediaInput.filepath = path;
            CVProcess *cvprocess = new CVProcess(&mediaInput);
            application.RegisterProcess(cvprocess);
        }else if(arg == "-realtime"){
            std::cout << "realtime" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::realtime;
        }else if(arg == "-assetto"){
            std::cout << "assetto" << std::endl;
            std::cout << arg << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::assetto;
        }else if(arg == "-terminal") {
            std::cout << "terminal" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::terminal;
        }
        cursor++;
    }
    CanProcess *canprocess = new CanProcess(&mediaInput);
        if(arg == "-realtime" || arg == "") {
            CVProcess *cvprocess = new CVProcess(&mediaInput);
            application.RegisterProcess(cvprocess);
            #ifdef linux
            ReadProcess *readcan = new ReadProcess();
            canprocess->setReadProcess(readcan);
            application.RegisterProcess(readcan);
            #endif
        } else if(arg == "-assetto") {
            CVProcess *cvprocess = new CVProcess(&mediaInput);
            application.RegisterProcess(cvprocess);
        } else if(arg == "-terminal") {
            #ifdef linux
            TerminalProcess *terminal = new TerminalProcess();
            canprocess->setTerminalProcess(terminal);
            application.RegisterProcess(terminal);
            #endif
        }
        application.RegisterProcess(canprocess);

        application.Run();
}