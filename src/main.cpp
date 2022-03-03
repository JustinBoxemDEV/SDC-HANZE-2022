#include <libsocketcan/.h>
#include <iostream>
#include "CANController/cancontroller.h"
#include <stdio.h>
// #include <conio.h>

void recursive() {
    std::cout << "What do you want to do? Type throttle, brake, steer or exit" << std::endl;

    char input[25];
    std::cin.get(input, 25);
    std::cin.ignore(256, '\n');

    bool sending = false; 

    if (strcmp(input, "throttle") == 0) {
        std::cout << "Executing: " << input << std::endl;

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

        sending = true;
        char stopinput[25];
        while (sending){ // && strcmp(loopinput, "stop")
            std::cout << "Throttling with direction " << direction << " and speed " << speed << std::endl;
            CANController::throttle(std::stoi(speed), std::stoi(direction));

            // TODO: Look at this
            // char ch = getch();
            // if (strcmp(cc, "stop")==0){
            //     sending = false;
            // }
        }
    } else if (strcmp(input, "brake")==0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE BRAKE PERCENTAGE (BETWEEN 0-100) >" << std::endl;
        char brakePercentage[25];
        std::cin.get(brakePercentage, 25);
        std::cin.ignore(256, '\n');
        std::cout << "Brake percentage is: " << brakePercentage << std::endl;

        sending = true;
        while (sending){
            std::cout << "Braking with percentage: " << brakePercentage << std::endl;
            CANController::brake(std::stoi(brakePercentage));
        }
    } else if (strcmp(input, "steer") == 0) {
        std::cout << "Executing: " << input << std::endl;

        std::cout << "GIVE STEERING AMOUNT (BETWEEN -1.0 AND 1.0) >" << std::endl;
        char steeringamount[25];
        std::cin.get(steeringamount, 25);  
        std::cin.ignore(256, '\n');
        std::cout << "Steering amount is: " << steeringamount << std::endl;
        
        sending = true;
        while (sending){
            std::cout << "Steering with amount: " << steeringamount << std::endl;
            CANController::steer(std::stof(steeringamount));
        }
    } else if (strcmp(input, "exit") == 0) {
        sending = false;
        CANController::closeCANController();
        std::cout << "Bye!" << std::endl;
    } else {
        std::cout << input << " is not a valid command." << std::endl;
        memset(input, 0, 25);
        recursive();
    };
};

int main( int argc, char** argv ) {
    std::cout << "Hi there! ;) Type initbus to start initializing the bus. Type skip to skip the setup sequence." << std::endl; 

    char initInput[25];
    std::cin.get(initInput, 25);
    std::cin.ignore(256, '\n');
    if (strcmp(initInput, "initbus") == 0) {
        std::cout << "Executing: " << initInput << std::endl;

        CANController::init("vcan");
        std::cout << "Init bus done" << std::endl;

        recursive();
    } if (strcmp(initInput, "skip") == 0){
        recursive();
    } else{
        std::cout << "Please enter a valid command: initbus skip" << std::endl;
    }
};