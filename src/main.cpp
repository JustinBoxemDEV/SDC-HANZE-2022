#include "application.h"
#include "Processes/canprocess.h"
#include "Processes/cvprocess.h"
#include "Processes/readprocess.h"
#include "Processes/terminalprocess.h"
#include "ComputerVision/cameracalibration.hpp"

namespace fs = std::filesystem;

// int main(int argc, char** argv) {
//     // std::string path = "/home/douwe/Projects/SDC-HANZE-2022/assets/images/calibration_dsh/";
//     // std::cout << path << std::endl;
//     // CameraCalibration calib(path, 20, 14, 20, 640, 480);
//   std::string path = "/home/douwe/Projects/SDC-HANZE-2022/assets/images/fuzz/";
//   std::vector<cv::String> fileNames;

//   std::vector<std::vector<cv::Point2f>> q(fileNames.size());
//   std::vector<std::vector<cv::Point3f>> Q;

//   cv::glob(path, fileNames, false);

//   std::cout << "Calibrating" << std::endl;

//   // Show lens corrected images
//   for (auto const &f : fileNames) {
//     std::cout << std::string(f) << std::endl;

//     cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
//     cv::Mat img2 = cv::imread(f, cv::IMREAD_COLOR);
//     cv::Mat img3 = cv::imread(f, cv::IMREAD_COLOR);
//     cv::Mat img4 = cv::imread(f, cv::IMREAD_COLOR);

//     cv::Mat temp = img.clone();
//     cv::Mat temp2 = img2.clone();
//     cv::Mat temp3 = img3.clone();
//     cv::Mat temp4 = img4.clone();

//     cv::Mat cameraMatrix2 = (cv::Mat1d(3,3) << 675.47607, 0, 319.5, 0, 675.47607, 239.5, 0, 0, 1);
//     cv::Mat distortionCoefficients2 = (cv::Mat1d(1, 5) << -0.0649378, -0.0861269, 0, 0, 0);

//     cv::Mat cameraMatrix4 = (cv::Mat1d(3,3) << 800.55762, 0, 319.5, 0, 800.55762, 239.5, 0, 0, 1); // new values without certain images 0.344804 reprojection error IMG 4
//     cv::Mat distortionCoefficients4 = (cv::Mat1d(1, 5) << 0.075056, -0.421619, 0, 0, 0); // new values without certain images 0.344804 reprojection error IMG 4

//     cv::Mat cameraMatrix3 = (cv::Mat1d(3,3) << 792.13574, 0, 319.5, 0, 792.13574, 239.5, 0, 0, 1); // reprojection error = 0.178943 only plane calibrating images IMG 3
//     cv::Mat distortionCoefficients3 = (cv::Mat1d(1, 5) << 0.0905006, -0.55128, 0, 0, 0); // reprojection error = 0.178943 only plane calibrating images IMG 3

//     cv::Mat cameraMatrix1 = (cv::Mat1d(3,3) << 674.10211, 0, 319.5, 0, 674.10211, 239.5, 0, 0, 1); // reprojection error = 2.25489 calibrating images with curves IMG 1
//     cv::Mat distortionCoefficients1 = (cv::Mat1d(1, 5) << -0.0565066, -0.151204, 0, 0, 0); // reprojection error = 2.25489 calibrating images with curves IMG 1

//     cv::undistort(temp, img, cameraMatrix1, distortionCoefficients1);
//     cv::undistort(temp3, img3, cameraMatrix3, distortionCoefficients3);
//     cv::undistort(temp2, img2, cameraMatrix2, distortionCoefficients2);
//     cv::undistort(temp4, img4, cameraMatrix4, distortionCoefficients4);

//     cv::Mat flipped;
//     cv::flip(img3, flipped, 1);

//     // Display
//     cv::imshow("distorted image", temp);
//     cv::imshow("undistorted image 1", img);
//     cv::imshow("undistorted image 2", img2);
//     cv::imshow("undisorted image 3", img3);
//     cv::imshow("undisorted image 4", img4);
//     //cv::imshow("undistorted image 4", flipped);

//     // Get dimension of final image
//     int rows = std::max(img2.rows, flipped.rows);
//     int cols = img2.cols + flipped.cols;

//     // Create a black image
//     cv::Mat3b res(rows, cols, cv::Vec3b(0,0,0));

//     // Copy images in correct position
//     img2.copyTo(res(cv::Rect(0, 0, img2.cols, img2.rows)));
//     flipped.copyTo(res(cv::Rect(img2.cols, 0, flipped.cols, flipped.rows)));

//     imshow("Result", res);

//     cv::waitKey(0);
//   }
// }

// #include "application.h"
// #include "Processes/canprocess.h"
// #include "Processes/cvprocess.h"
// #include "Processes/readprocess.h"
// #include "Processes/terminalprocess.h"

// namespace fs = std::filesystem;

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