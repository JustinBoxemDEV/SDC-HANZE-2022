// THIS CURRENT MAIN IS A MESS THAT JUST TESTS THE SCHEDULER I APOLOGISE
#include "VehicleControl/strategies/canstrategy.h"
#include <iostream>
#include <string.h>
#include <unistd.h>

#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include "ComputorVision/computorvision.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <filesystem>
#include <string>
#include "MediaCapture/MediaCapture.h"
#ifdef __WIN32__
#include "MediaCapture/screenCaptureWindows.h"
#else
#include "MediaCapture/screenCaptureLinux.h"
#endif

namespace fs = std::filesystem;
using namespace std;

int screenCaptureCommand(int argc, char** argv);
int cameraCaptureCommand(int argc, char** argv);
int videoCommand(int argc, char** argv);

int main(int argc, char** argv) {
    if (argv[1] == NULL) {
        // return screenCaptureCommand(argc, argv);
        return cameraCaptureCommand(argc, argv);
        // return videoCommand(argc, argv);
    } 
}
// TEST AC
int screenCaptureCommand(int argc, char** argv) {
    #ifdef __WIN32__
    MediaCapture mediacapture;
    mediacapture.ProcessFeed(true); // screenCapture=true
    return 0;
    #else
    cout << "ERROR: screen capture is currently not working for linux!" << endl;
    return -1;
    #endif
}

// TEST CANBUS
int cameraCaptureCommand(int argc, char** argv) {
    #ifdef linux
    MediaCapture mediacapture;
    mediacapture.ProcessFeed(false, 0); // cameraID=4 for webcam, cameraID=0 for built in laptop cam

    return 0;
    #else
    cout << "ERROR: This camera capture does not work for windows!" << endl;
    return -1;
    #endif
}

// TEST VIDEO
int videoCommand(int argc, char** argv) {
    MediaCapture mediacapture;
    mediacapture.ProcessFeed(false, 0, "../assets/videos/testvid.mp4"); // give file path
    return 0;
}


// int main( int argc, char** argv ){
//     // --help Output, describing basic usage to the user
//     if(argc==1){
//         MediaCapture mediaCapture;
//         mediaCapture.ProcessFeed(0,"");
//         return 0;
//     }
//     if(std::string(argv[1])=="-help" or std::string(argv[1])=="-h"){
//         std::cout << "Usage: SPECIFY RESOURCE TO USE" << std::endl;
//         std::cout << "-video -camera [CAMERA_ID]" << std::endl;
//         std::cout << "-video -filename [FILE]" << std::endl;
//         std::cout << "-image [FILE]" << std::endl;
//         return -1;
//     }else{
//         // The user has told us he wants to use media feed
//         if(std::string(argv[1])=="-video"){
//             if(argc==2){
//                 std::cout << "Usage:" << std::endl;
//                 std::cout << "-video -camera [CAMERA_ID]" << std::endl;
//                 std::cout << "-video -filename [FILE]" << std::endl;
//                 return -1;
//             }if(argc==3){
//                 std::cout << "Usage:" << std::endl;
//                 if(std::string(argv[2])=="-camera"){
//                     std::cout << "-video -camera [CAMERA_ID]" << std::endl;
//                     return -1;
//                 }else if(std::string(argv[2])=="-filename"){
//                     // No video file was provided to look for, so we are going to present a list of names
//                     std::cout << "Available videos to load using -filename [FILE]" << std::endl;
//                     std::string path = fs::current_path().string() + "/assets/videos/";
//                     for (const auto & file : fs::directory_iterator(path))
//                         std::cout << fs::path(file).filename().string() << std::endl;
//                     return -1;
//                 }
//             }if(argc==4){
//                 if(std::string(argv[2])=="-filename"){
//                     std::string path = fs::current_path().string() + "/assets/videos/" + std::string(argv[3]);
//                     if(!fs::exists(path)){
//                         std::cout << "The requested file cannot be found in /assets/videos/!" << std::endl;
//                         return -1;
//                     }
//                     MediaCapture mediaCapture;
//                     mediaCapture.ProcessFeed(0,path);
//                     return 0;
//                 }else if(std::string(argv[2])=="-camera"){
//                     MediaCapture mediaCapture;
//                     mediaCapture.ProcessFeed(std::stoi(argv[3]),"");
//                     return 0;
//                 }else{
//                     MediaCapture mediaCapture;
//                     mediaCapture.ProcessFeed(0,"");
//                     return 0;
//                 }
//             }
//         }else if(std::string(argv[1])=="-image"){
//             // An image was provided to look for
//             if(argc==3){
//                 MediaCapture mediaCapture;
//                 cv::Mat img = mediaCapture.LoadImg(std::string(argv[2]));
//                 mediaCapture.ProcessImage(img);
//                 cv::waitKey(0);
//                 return 0;
//             }

//             // No image was provided to look for, so we are going to present a list of names
//             std::cout << "Available images to load using -image [NAME]" << std::endl;
//             std::string path = fs::current_path().string() + "/assets/images/";
//             for (const auto & file : fs::directory_iterator(path))
//                 std::cout << fs::path(file).filename().string() << std::endl;
//             return -1;
//         }
//             // The parameter that the user provided is not compatible with our program | Provide error + help message
//         else{
//             std::cout << "ERROR: " << std::string(argv[1]) << " is not recognised. Use -help for information" << std::endl;
//             return -1;
//         }
//     }
// }
