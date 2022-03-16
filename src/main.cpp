// #include <libsocketcan.h>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "ComputorVision/computorvision.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <filesystem>
#include <string>
#include "MediaCapture/mediaCapture.h"
#include "MediaLoader/captureScreen.h"
#include "MediaLoader/screenshot.cpp"

namespace fs = std::filesystem;
using namespace std;

int helpCommand(int argc, char** argv);
int videoCommand(int argc, char** argv);
int imageCommand(int argc, char** argv);
int invalidCommand(int argc, char** argv);

int main(int argc, char** argv) {
    if (argv[1] == NULL) {
        /**
         * CaptureScreen
         */
        CaptureScreen captureScreen;
        captureScreen.runMain();

        // /**
        //  * Screenshot
        //  */
        // ScreenShot screen(0,0,1920,1080);
        // cv::Mat img;

        // clock_t current_ticks, delta_ticks;
        // clock_t fps = 0;
        
        // while(true) 
        // {
        // current_ticks = clock();
        //     screen(img);

        //     cv::imshow("img", img);
        //     char k = cv::waitKey(1);
        //     if (k == 'q')
        //         break;

        //     delta_ticks = clock() - current_ticks; //the time, in ms, that took to render the scene
        //     if(delta_ticks > 0)
        //         fps = CLOCKS_PER_SEC / delta_ticks;
        //     cout << fps << endl;
        // }
    } 
    
    else if (string(argv[1]) == "-help" || 
             string(argv[1]) == "-h") {
        return helpCommand(argc, argv);
    } 
    
    else if (string(argv[1])=="-video") {
        return videoCommand(argc, argv);
    }

    else if(string(argv[1])=="-image") {
        return imageCommand(argc, argv);
    }

    else {
        return invalidCommand(argc, argv);
    }
}

int helpCommand(int argc, char** argv) {
    cout << "Usage: SPECIFY RESOURCE TO USE" << endl;
    cout << "\t-video -camera [CAMERA_ID]" << endl;
    cout << "\t-video -filename [FILE]" << endl;
    cout << "\t-image [FILE]" << endl;
    return -1;
}

int videoCommand(int argc, char** argv) {
    if (argc==2) {
        std::cout << "Usage:" << std::endl; 
        std::cout << "-video -camera [CAMERA_ID]" << std::endl;
        std::cout << "-video -filename [FILE]" << std::endl;
        return -1;
    } if (argc==3) {
        std::cout << "Usage:" << std::endl;
        if (std::string(argv[2])=="-camera") {
            std::cout << "-video -camera [CAMERA_ID]" << std::endl;
            return -1;
        } else if (std::string(argv[2])=="-filename"){
            // No video file was provided to look for, so we are going to present a list of names
            std::cout << "Available videos to load using -filename [FILE]" << std::endl;
            std::string path = fs::current_path().string() + "/assets/videos/";
            for (const auto & file : fs::directory_iterator(path))
                std::cout << fs::path(file).filename().string() << std::endl;
            return -1;
        }
    } if (argc==4) {   
        if (std::string(argv[2])=="-filename") {
            std::string path = fs::current_path().string() + "/assets/videos/" + std::string(argv[3]);
            if (!fs::exists(path)) {
                std::cout << "The requested file cannot be found in /assets/videos/!" << std::endl;
                return -1;
            }
            MediaCapture mediaCapture;
            mediaCapture.ProcessFeed(0,path);
            return 0;
        } else if (std::string(argv[2])=="-camera") {
            MediaCapture mediaCapture;
            mediaCapture.ProcessFeed(std::stoi(argv[3]),"");
            return 0;
        } else {
            MediaCapture mediaCapture;
            mediaCapture.ProcessFeed(0,"");
            return 0;
        }
    }
    return -1;
}

int imageCommand(int argc, char** argv) {
    // An image was provided to look for
    if(argc==3){
        MediaCapture mediaCapture;
        cv::Mat img = mediaCapture.LoadImage(std::string(argv[2]));
        mediaCapture.ProcessImage(img);
        cv::waitKey(0);
        return 0;
    }

    // No image was provided to look for, so we are going to present a list of names
    std::cout << "Available images to load using -image [NAME]" << std::endl;
    std::string path = fs::current_path().string() + "/assets/images/";
    for (const auto & file : fs::directory_iterator(path))
        std::cout << fs::path(file).filename().string() << std::endl;
    return -1;
}

int invalidCommand(int argc, char** argv) {
    std::cout << "ERROR: " << std::string(argv[1]) << " is not recognised. Use -help for information" << std::endl;
    return -1;
}
