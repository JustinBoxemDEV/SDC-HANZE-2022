// TEST CODE 1 (WITHOUT SCHEDULER)

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
#include "/utils/TaskScheduler/TaskScheduler.h"
#ifdef __WIN32__
#include "MediaCapture/screenCaptureWindows.h"
#else
#include "MediaCapture/screenCaptureLinux.h"
#endif

namespace fs = std::filesystem;
using namespace std;

// WITHOUT SCHEDULER (FOR LOOP SEND X AMOUNT OF TIMES)
// CANStrategy canStrategy;

// void recursive() {
//     int send_amount = 100; // Could make this dynamic from input I guess

//     std::cout << "Hi there! ;) What do you want to do? Type throttle, brake, steer or exit" << std::endl;

//     char input[25];
//     std::cin.get(input, 25);
//     std::cin.ignore(256, '\n');

//     if (strcmp(input, "throttle") == 0) {
//         std::cout << "Executing: " << input << std::endl;

//         std::cout << "GIVE THROTTLE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
//         char speed[25];
//         std::cin.get(speed, 25);
//         std::cin.ignore(256, '\n');
//         std::cout << "Speed is: " << speed << std::endl;

//         std::cout << "Sending " << send_amount << " times" << std::endl;

//         for(int i = 0; i < send_amount; i++) {
//            canStrategy.forward(std::stoi(speed)); 
//            sleep(0.04);
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
//             canStrategy.brake(std::stoi(brakePercentage));
//             sleep(0.04);
//         };
//         recursive();
//     } else if (strcmp(input, "steer") == 0) {
//         std::cout << "Executing: " << input << std::endl;

//         std::cout << "GIVE STEERING ANGLE (BETWEEN -1.0 AND 1.0) >" << std::endl;
//         char steeringamount[25];
//         std::cin.get(steeringamount, 25);  
//         std::cin.ignore(256, '\n');
//         std::cout << "Steering amount is: " << steeringamount << std::endl;

//         std::cout << "Sending " << send_amount << " times" << std::endl;
        
//         for(int i = 0; i < 100; i++) {
//             canStrategy.steer(std::stof(steeringamount));
//             sleep(0.04);
//         };
//         recursive();
//     }
//     else if (strcmp(input, "exit") == 0) {
//         // Close things?
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


// WITH SCHEDULER 
TaskScheduler taskScheduler;

void recursive() {
    std::cout << "Hi there! ;) What do you want to do? Type throttle, brake, steer or exit" << std::endl;

    char input[25];
    std::cin.get(input, 25);
    std::cin.ignore(256, '\n');

    if (strcmp(input, "throttle") == 0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE THROTTLE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
        char speed[25];
        std::cin.get(speed, 25);
        std::cin.ignore(256, '\n');

        // add scheduler
        canStrategy.forward(std::stoi(speed)); 

        recursive();
    } else if (strcmp(input, "brake")==0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE BRAKE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
        char brakePercentage[25];
        std::cin.get(brakePercentage, 25);
        std::cin.ignore(256, '\n');
        std::cout << "Brake percentage is: " << brakePercentage << std::endl;
        
        // add scheduler
        canStrategy.brake(std::stoi(brakePercentage));

        recursive();
    } else if (strcmp(input, "steer") == 0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE STEERING ANGLE (BETWEEN -1.0 AND 1.0) >" << std::endl;
        char steeringamount[25];
        std::cin.get(steeringamount, 25);  
        std::cin.ignore(256, '\n');
        std::cout << "Steering amount is: " << steeringamount << std::endl;

        // add scheduler
        canStrategy.steer(std::stof(steeringamount));

        recursive();
    } else if (strcmp(input, "exit") == 0) {
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