// TEST CODE 1 (WITHOUT SCHEDULER)

#include "VehicleControl/strategies/CANStrategy.h"
#include <iostream>
#include <string.h>
#include <unistd.h>

CANStrategy canStrategy;

//TODO: Not functional yet, throttle function not called
void recursive() {

    std::cout << "What do you want to do? Type throttle, brake, steer or exit" << std::endl;

    char input[25];
    std::cin.get(input, 25);
    std::cin.ignore(256, '\n');

    if (strcmp(input, "throttle") == 0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE SPEED (PERCENTAGE BETWEEN 0-100) >" << std::endl;
        char speed[25];
        std::cin.get(speed, 25);
        std::cin.ignore(256, '\n');
        std::cout << "Speed is: " << speed << std::endl;

        for(int i = 0; i < 100; i++) {
           canStrategy.forward(std::stoi(speed)); 
           sleep(0.04);
        };
        recursive();
    } else if (strcmp(input, "brake")==0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE BRAKE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
        char brakePercentage[25];
        std::cin.get(brakePercentage, 25);
        std::cin.ignore(256, '\n');
        std::cout << "Brake percentage is: " << brakePercentage << std::endl;

        for(int i = 0; i < 100; i++) {
            canStrategy.brake(std::stoi(brakePercentage));
            sleep(0.04);
        };
        recursive();
    } else if (strcmp(input, "steer") == 0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE STEERING AMOUNT (BETWEEN -1.0 AND 1.0) >" << std::endl;
        char steeringamount[25];
        std::cin.get(steeringamount, 25);  
        std::cin.ignore(256, '\n');
        std::cout << "Steering amount is: " << steeringamount << std::endl;
        
        for(int i = 0; i < 100; i++) {
            canStrategy.steer(std::stof(steeringamount));
            sleep(0.04);
        };
        recursive();
    }
    else if (strcmp(input, "exit") == 0) {
        //CANController::closeCANController();
        std::cout << "Bye!" << std::endl;
    } else {
        std::cout << input << " is not a valid command." << std::endl;
        memset(input, 0, 25);
        recursive();
    };
};

int main() {
    std::cout << "test" << std::endl;
    recursive();
};

// CODE TO RUN AC WITH SCREENCAPTURE

// #include "opencv2/opencv.hpp"
// #include <opencv2/imgproc.hpp>
// #include <iostream>
// #include "ComputorVision/computorvision.h"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/imgproc.hpp"
// #include <filesystem>
// #include <string>
// #include "MediaCapture/mediaCapture.h"
// #ifdef __WIN32__
// #include "MediaCapture/screenCaptureWindows.h"
// #else
// #include "MediaCapture/screenCaptureLinux.h"
// #endif

// namespace fs = std::filesystem;
// using namespace std;

// int helpCommand(int argc, char** argv);
// int videoCommand(int argc, char** argv);
// int imageCommand(int argc, char** argv);
// int invalidCommand(int argc, char** argv);
// int screenCaptureCommand(int argc, char** argv);

// int main(int argc, char** argv) {
//     if (argv[1] == NULL) {
//         return screenCaptureCommand(argc, argv);
//     } 
    
//     else if (string(argv[1]) == "-help" || 
//              string(argv[1]) == "-h") {
//         return helpCommand(argc, argv);
//     } 
    
//     else if (string(argv[1])=="-video") {
//         return videoCommand(argc, argv);
//     }

//     else if (string(argv[1])=="-image") {
//         return imageCommand(argc, argv);
//     }

//     else {
//         return invalidCommand(argc, argv);
//     }
// }

// int helpCommand(int argc, char** argv) {
//     cout << "Usage: SPECIFY RESOURCE TO USE" << endl;
//     cout << "\t-video -camera [CAMERA_ID]" << endl;
//     cout << "\t-video -filename [FILE]" << endl;
//     cout << "\t-image [FILE]" << endl;
//     return -1;
// }

// int videoCommand(int argc, char** argv) {
//     if (argc==2) {
//         std::cout << "Usage:" << std::endl; 
//         std::cout << "-video -camera [CAMERA_ID]" << std::endl;
//         std::cout << "-video -filename [FILE]" << std::endl;
//         return -1;
//     } if (argc==3) {
//         std::cout << "Usage:" << std::endl;
//         if (std::string(argv[2])=="-camera") {
//             std::cout << "-video -camera [CAMERA_ID]" << std::endl;
//             return -1;
//         } else if (std::string(argv[2])=="-filename"){
//             // No video file was provided to look for, so we are going to present a list of names
//             std::cout << "Available videos to load using -filename [FILE]" << std::endl;
//             std::string path = fs::current_path().string() + "/assets/videos/";
//             for (const auto & file : fs::directory_iterator(path))
//                 std::cout << fs::path(file).filename().string() << std::endl;
//             return -1;
//         }
//     } if (argc==4) {   
//         if (std::string(argv[2])=="-filename") {
//             std::string path = fs::current_path().string() + "/assets/videos/" + std::string(argv[3]);
//             if (!fs::exists(path)) {
//                 std::cout << "The requested file cannot be found in /assets/videos/!" << std::endl;
//                 return -1;
//             }
//             MediaCapture mediaCapture;
//             mediaCapture.ProcessFeed(0,path);
//             return 0;
//         } else if (std::string(argv[2])=="-camera") {
//             MediaCapture mediaCapture;
//             mediaCapture.ProcessFeed(std::stoi(argv[3]),"");
//             return 0;
//         } else {
//             MediaCapture mediaCapture;
//             mediaCapture.ProcessFeed(0,"");
//             return 0;
//         }
//     }
//     return -1;
// }

// int imageCommand(int argc, char** argv) {
//     // An image was provided to look for
//     if(argc==3){
//         // MediaCapture mediaCapture;
//         // cv::Mat img = mediaCapture.LoadImage(std::string(argv[2]));
//         // mediaCapture.ProcessImage(img);
//         // cv::waitKey(0);
//         // return 0;
//     }

//     // No image was provided to look for, so we are going to present a list of names
//     std::cout << "Available images to load using -image [NAME]" << std::endl;
//     std::string path = fs::current_path().string() + "/assets/images/";
//     for (const auto & file : fs::directory_iterator(path))
//         std::cout << fs::path(file).filename().string() << std::endl;
//     return -1;
// }

// int invalidCommand(int argc, char** argv) {
//     std::cout << "ERROR: " << std::string(argv[1]) << " is not recognised. Use -help for information" << std::endl;
//     return -1;
// }

// int screenCaptureCommand(int argc, char** argv) {
//     #ifdef __WIN32__
//     ScreenCapture screenCapture;
//     screenCapture.run();
//     return 0;
//     #else
//     cout << "ERROR: screen capture is currently not working for linux!" << endl;
//     return -1;
//     #endif
// }

// CODE TO MANUAL CONTROL AC --- U need to comment above section to make this work

// #include "VehicleControl/strategies/ACStrategy.h"

// ACStrategy assettocorsa2;

// //TODO: Not functional yet, throttle function not called
// void recursive() {

//     std::cout << "What do you want to do? Type throttle, brake, steer, shift up, shift down, neutral or exit" << std::endl;

//     char input[25];
//     std::cin.get(input, 25);
//     std::cin.ignore(256, '\n');

//     bool sending = false;

//     if (strcmp(input, "throttle") == 0) {
//         std::cout << "Executing: " << input << std::endl;

//         std::cout << "GIVE SPEED (PERCENTAGE BETWEEN 0-100) >" << std::endl;
//         char speed[25];
//         std::cin.get(speed, 25);
//         std::cin.ignore(256, '\n');
//         std::cout << "Speed is: " << speed << std::endl;
//         sending = true;
//         char loopinput[25];
//         assettocorsa2.forward(std::stoi(speed));
//         recursive();
//     } else if (strcmp(input, "brake")==0) {
//         std::cout << "Executing: " << input << std::endl;

//         std::cout << "GIVE BRAKE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
//         char brakePercentage[25];
//         std::cin.get(brakePercentage, 25);
//         std::cin.ignore(256, '\n');
//         std::cout << "Brake percentage is: " << brakePercentage << std::endl;

//         sending = true;
//         assettocorsa2.brake(std::stoi(brakePercentage));
//         recursive();
//     } else if (strcmp(input, "steer") == 0) {
//         std::cout << "Executing: " << input << std::endl;

//         std::cout << "GIVE STEERING AMOUNT (BETWEEN -1.0 AND 1.0) >" << std::endl;
//         char steeringamount[25];
//         std::cin.get(steeringamount, 25);  
//         std::cin.ignore(256, '\n');
//         std::cout << "Steering amount is: " << steeringamount << std::endl;
        
//         sending = true;
//         assettocorsa2.steer(std::stof(steeringamount));
//         recursive();
//     } else if (strcmp(input, "shift up") == 0) {
//         std::cout << "Executing: " << input << std::endl;
//         sending = true;
//         assettocorsa2.gearShiftUp();
//         recursive();
//     } else if (strcmp(input, "shift down") == 0) {
//         std::cout << "Executing: " << input << std::endl;
//         sending = true;
//         assettocorsa2.gearShiftDown();
//         recursive();
//     } else if (strcmp(input, "neutral") == 0) {
//         std::cout << "Executing: " << input << std::endl;
//         sending = true;
//         assettocorsa2.neutral();
//         recursive();
//     }
//     else if (strcmp(input, "exit") == 0) {
//         sending = false;
//         //CANController::closeCANController();
//         std::cout << "Bye!" << std::endl;
//     } else {
//         std::cout << input << " is not a valid command." << std::endl;
//         memset(input, 0, 25);
//         recursive();
//     };
// };

// int main() {
//     std::cout << "test" << std::endl;
//     recursive();
// };