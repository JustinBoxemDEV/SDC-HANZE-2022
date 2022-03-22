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
#include "VehicleControl/strategies/CANStrategy.h"
#include "./utils/TaskScheduler/TaskScheduler.h"
#ifdef __WIN32__
#include "MediaCapture/screenCaptureWindows.h"
#else
#include "MediaCapture/screenCaptureLinux.h"
#endif

namespace fs = std::filesystem;
using namespace std;

// [BUILD 1] Without scheduler (For loop send x amount of times)
CANStrategy canStrategy;

void recursive() {
    int send_amount = 1000; // Could make this dynamic from input I guess
    int sleep_time = 0.04;

    std::cout << "(LOOP VERSION) Hi there! ;) What do you want to do? Type throttle, brake, steer or exit" << std::endl;

    char input[25];
    std::cin.get(input, 25);
    std::cin.ignore(256, '\n');

    if (strcmp(input, "throttle") == 0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE THROTTLE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
        char speed[25];
        std::cin.get(speed, 25);
        std::cin.ignore(256, '\n');
        std::cout << "Speed is: " << speed << std::endl;

        std::cout << "Sending " << send_amount << " times" << std::endl;

        for(int i = 0; i < send_amount; i++) {
           canStrategy.forward(std::stoi(speed)); 
        //    canStrategy.steer(0.9);
           sleep(sleep_time);
        };
    
        recursive();
    } else if (strcmp(input, "brake")==0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE BRAKE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
        char brakePercentage[25];
        std::cin.get(brakePercentage, 25);
        std::cin.ignore(256, '\n');
        std::cout << "Brake percentage is: " << brakePercentage << std::endl;

        std::cout << "Sending " << send_amount << " times" << std::endl;

        for(int i = 0; i < send_amount; i++) {
            canStrategy.brake(std::stoi(brakePercentage));
            sleep(sleep_time);
        };
        recursive();
    } else if (strcmp(input, "steer") == 0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE STEERING ANGLE (BETWEEN -1.0 AND 1.0) >" << std::endl;
        char steeringamount[25];
        std::cin.get(steeringamount, 25);  
        std::cin.ignore(256, '\n');
        std::cout << "Steering amount is: " << steeringamount << std::endl;

        std::cout << "Sending " << send_amount << " times" << std::endl;
        
        for(int i = 0; i < send_amount; i++) {
            canStrategy.steer(std::stof(steeringamount));
            sleep(sleep_time);
        };
        recursive();
    }
    else if (strcmp(input, "exit") == 0) {
        // Close things?
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


// [BUILD 2] With scheduler (TODO: refactor. This wont work anymore with the new scheduler)
// TaskScheduler taskScheduler;

// void plswork() {
//     taskScheduler.SCH_Init();
//     taskScheduler.SCH_Start();
//     int speed = 0;
//     int brakePercentage = 0;
//     int steeringamount = 0;

//     // auto test = [](){canStrategy.forward(speed);};
//     taskScheduler.SCH_Add_Task(canStrategy.forward(std::stoi(speed)), 1, 4);
//     taskScheduler.SCH_Add_Task(canStrategy.brake(std::stoi(brakePercentage)), 1, 4);
//     taskScheduler.SCH_Add_Task(canStrategy.steer(std::stof(steeringamount)), 1, 4);
//     while(true){
//         taskScheduler.SCH_Dispatch_Tasks();
//     }
// };

// int main() {
//     plswork();
// };

// [BUILD 3] Full build (CAN + Computer vision + PID)

// int main(int argc, char** argv) {
//     if (argc == 1) {
//         MediaCapture mediaCapture;
//         mediaCapture.ProcessFeed(0, "");
//         return 0;
//     } 
// }

// [BUILD 4] Read CAN messages (steering angle)
// CANStrategy canStrategy;

// int main(int argc, char** argv) {
//     while(true) {
//         canStrategy.readCANMessages();
//     }
// }
