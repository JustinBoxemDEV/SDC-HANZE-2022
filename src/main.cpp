// THIS CURRENT MAIN IS A MESS THAT JUST TESTS THE SCHEDULER I APOLOGISE

#include "VehicleControl/strategies/CANStrategy.h"
#include <iostream>
#include <string.h>
#include <unistd.h>

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
#ifdef __WIN32__
#include "MediaCapture/screenCaptureWindows.h"
#else
#include "MediaCapture/screenCaptureLinux.h"
#endif

namespace fs = std::filesystem;
using namespace std;

int screenCaptureCommand(int argc, char** argv);

int main(int argc, char** argv) {
    if (argv[1] == NULL) {
        return screenCaptureCommand(argc, argv);
    } 
}

int screenCaptureCommand(int argc, char** argv) {
    #ifdef __WIN32__
    ScreenCaptureWindows screenCaptureWindows;
    screenCaptureWindows.run();
    return 0;
    #else
    cout << "ERROR: screen capture is currently not working for linux!" << endl;
    return -1;
    #endif
}

// TEST CANBUS
// int main(int argc, char** argv){
//     if(argc == 1){
//         MediaCapture mediacapture;
//         mediacapture.ProcessFeed(0, "");
//         return 0;
//     }
// }


// CODE TO MANUAL CONTROL AC --- U need to comment above section to make this work

// #include "VehicleControl/strategies/ACStrategy.h"

// ACStrategy assettocorsa2;

// void recursive() {
//     int send_amount = 100; // Could make this dynamic from input I guess

//     std::cout << "(MANUAL VER) What do you want to do? ;) Type throttle, brake, steer, shift up, shift down, neutral or exit" << std::endl;

//     char input[25];
//     std::cin.get(input, 25);
//     std::cin.ignore(256, '\n');

//     if (strcmp(input, "throttle") == 0) {
//         std::cout << "Executing: " << input << std::endl;

//         std::cout << "GIVE SPEED PERCENTAGE (BETWEEN 0-100) >" << std::endl;
//         char speed[25];
//         std::cin.get(speed, 25);
//         std::cin.ignore(256, '\n');
//         std::cout << "Speed is: " << speed << std::endl;
        
//         std::cout << "Sending " << send_amount << " times" << std::endl;

//         for(int i = 0; i < 100; i++) {
//             assettocorsa2.forward(std::stoi(speed));
//             sleep(0.04);
//         };
//         recursive();
//     } else if (strcmp(input, "brake")==0) {
//         std::cout << "Executing: " << input << std::endl;

//         std::cout << "GIVE BRAKE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
//         char brakePercentage[25];
//         std::cin.get(brakePercentage, 25);
//         std::cin.ignore(256, '\n');
//         std::cout << "Brake percentage is: " << brakePercentage << std::endl;

//         std::cout << "Sending " << send_amount << " times" << std::endl;

//         for(int i = 0; i < 100; i++) {
//             assettocorsa2.brake(std::stoi(brakePercentage));
//             sleep(0.04);
//         };
//         recursive();
//     } else if (strcmp(input, "steer") == 0) {
//         std::cout << "Executing: " << input << std::endl;

//         std::cout << "GIVE STEERING AMOUNT (BETWEEN -1.0 AND 1.0) >" << std::endl;
//         char steeringamount[25];
//         std::cin.get(steeringamount, 25);  
//         std::cin.ignore(256, '\n');
//         std::cout << "Steering amount is: " << steeringamount << std::endl;

//         std::cout << "Sending " << send_amount << " times" << std::endl;

//         for(int i = 0; i < send_amount; i++) {
//            assettocorsa2.steer(std::stof(steeringamount));
//            sleep(0.04);
//         };
//         recursive();
//     } else if (strcmp(input, "shift up") == 0) {
//         std::cout << "Executing: " << input << std::endl;
//         assettocorsa2.gearShiftUp();
//         recursive();
//     } else if (strcmp(input, "shift down") == 0) {
//         std::cout << "Executing: " << input << std::endl;
//         assettocorsa2.gearShiftDown();
//         recursive();
//     } else if (strcmp(input, "neutral") == 0) {
//         std::cout << "Executing: " << input << std::endl;
//         assettocorsa2.neutral();
//         recursive();
//     }
//     else if (strcmp(input, "exit") == 0) {
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
