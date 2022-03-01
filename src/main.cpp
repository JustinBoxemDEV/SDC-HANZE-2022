#include <libsocketcan.h>
#include <iostream>
#include "CANController/cancontroller.h"
#include <stdio.h>

void recursive() {
    std::cout << "What do you want to do? Type throttle, brake or steer" << std::endl;

    char test[25];
    std::cin.get(test, 25);
    std::cin.ignore(256, '\n');

    if (strcmp(test, "throttle") == 0) {
        std::cout << "Executing: " << test << std::endl;

        std::cout << "GIVE SPEED (BETWEEN 0-100) >" << std::endl;
        char speed[25];
        std::cin.get(speed, 25);
        std::cin.ignore(256, '\n');
        std::cout << "Speed is: " << speed << std::endl;

        std::cout << "GIVE DRIVING DIRECTION (0=NEUTRAL, 1=FORWARDS, 2=BACKWARDS) >" << std::endl;
        char direction[25];
        std::cin.get(direction, 25);
        std::cin.ignore(256, '\n');
        std::cout << "Direction is: " << direction << std::endl;

        CANController::throttle(std::stoi(speed), std::stoi(direction));
    };

    if (strcmp(test, "brake")==0) {
        std::cout << "Executing: " << test << std::endl;

        std::cout << "GIVE BRAKE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
        char brakePercentage[25];
        std::cin.get(brakePercentage, 25);
        std::cin.ignore(256, '\n');
        std::cout << "Brake percentage is: " << brakePercentage << std::endl;
        
        CANController::brake(std::stoi(brakePercentage));
    };

    if (strcmp(test, "steer") == 0) {
        std::cout << "Executing: " << test << std::endl;

        std::cout << "GIVE STEERING AMOUNT (BETWEEN -1.0 AND 1.0) >" << std::endl;
        char steeringamount[25];
        std::cin.get(steeringamount, 25);  
        std::cin.ignore(256, '\n');
        std::cout << "Steering amount is: " << steeringamount << std::endl;
        
        CANController::steer(std::stof(steeringamount));
    };

    if (strcmp(test, "exit") == 0) {
        CANController::closeCANController();
        std::cout << "Bye!" << std::endl;
    };
    memset(test, 0, 25);
    recursive();
};

int main( int argc, char** argv ) {
    std::cout << "Hi there! ;) Type initbus to start initializing the bus >" << std::endl;
    char input[25];
    std::cin.get(input, 25);
    std::cin.ignore(256, '\n');
    if (strcmp(input, "initbus") == 0) {
        std::cout << "Executing: " << input << std::endl;

        CANController::init("vcan0", "vcan");
        std::cout << "Init bus succesful" << std::endl;

        recursive();
    };
};