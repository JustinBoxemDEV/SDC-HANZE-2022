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

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // --help Output, describing basic usage to the user
    if (argc == 1) {
        MediaCapture mediaCapture;
        mediaCapture.ProcessFeed(0, "");
        return 0;
    }
    if (std::string(argv[1]) == "-help" or std::string(argv[1]) == "-h") {
        std::cout << "Usage: SPECIFY RESOURCE TO USE" << std::endl;
        std::cout << "-video -camera [CAMERA_ID]" << std::endl;
        std::cout << "-video -filename [FILE]" << std::endl;
        std::cout << "-image [FILE]" << std::endl;
        return -1;
    }
    else {
        // The user has told us he wants to use media feed
        if (std::string(argv[1]) == "-video") {
            if (argc == 2) {
                std::cout << "Usage:" << std::endl;
                std::cout << "-video -camera [CAMERA_ID]" << std::endl;
                std::cout << "-video -filename [FILE]" << std::endl;
                return -1;
            }if (argc == 3) {
                std::cout << "Usage:" << std::endl;
                if (std::string(argv[2]) == "-camera") {
                    std::cout << "-video -camera [CAMERA_ID]" << std::endl;
                    return -1;
                }
                else if (std::string(argv[2]) == "-filename") {
                    // No video file was provided to look for, so we are going to present a list of names
                    std::cout << "Available videos to load using -filename [FILE]" << std::endl;
                    std::string path = fs::current_path().string() + "/assets/videos/";
                    for (const auto& file : fs::directory_iterator(path))
                        std::cout << fs::path(file).filename().string() << std::endl;
                    return -1;
                }
            }if (argc == 4) {
                if (std::string(argv[2]) == "-filename") {
                    std::string path = fs::current_path().string() + "/assets/videos/" + std::string(argv[3]);
                    if (!fs::exists(path)) {
                        std::cout << "The requested file cannot be found in /assets/videos/!" << std::endl;
                        return -1;
                    }
                    MediaCapture mediaCapture;
                    mediaCapture.ProcessFeed(0, path);
                    return 0;
                }
                else if (std::string(argv[2]) == "-camera") {
                    MediaCapture mediaCapture;
                    mediaCapture.ProcessFeed(std::stoi(argv[3]), "");
                    return 0;
                }
                else {
                    MediaCapture mediaCapture;
                    mediaCapture.ProcessFeed(0, "");
                    return 0;
                }
            }
        }
        else if (std::string(argv[1]) == "-image") {
            // An image was provided to look for
            if (argc == 3) {
                MediaCapture mediaCapture;
                cv::Mat img = mediaCapture.LoadImage(std::string(argv[2]));
                mediaCapture.ProcessImage(img);
                cv::waitKey(0);
                return 0;
            }

            // No image was provided to look for, so we are going to present a list of names
            std::cout << "Available images to load using -image [NAME]" << std::endl;
            std::string path = fs::current_path().string() + "/assets/images/";
            for (const auto& file : fs::directory_iterator(path))
                std::cout << fs::path(file).filename().string() << std::endl;
            return -1;
        }
        // The parameter that the user provided is not compatible with our program | Provide error + help message
        else {
            std::cout << "ERROR: " << std::string(argv[1]) << " is not recognised. Use -help for information" << std::endl;
            return -1;
        }
    }
}
