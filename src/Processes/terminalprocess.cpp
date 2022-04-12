#ifdef linux
#include "terminalprocess.h"

CommunicationStrategy *terminalStrategy;

void TerminalProcess::setStrategy(CommunicationStrategy *strategy) {
    std::cout << "setting strategy" << std::endl;
    terminalStrategy = strategy;
}

void TerminalProcess::terminal() {
    std::cout << "Type a command: [throttle] [brake] [steer]" << std::endl;
    char input[25];
    std::cin.get(input, 25);
    std::cin.ignore(256, '\n');

    if (strcmp(input, "throttle") == 0) {
        std::cout << "Execute throttle" << std::endl;
        std::cout << "Give speed percentage (0-100)" << std::endl;
        char speed[25];
        std::cin.get(speed, 25);
        std::cin.ignore(256, '\n');

        if(std::stoi(speed) >= 0 && std::stoi(speed) <= 100) {
            terminalStrategy->actuators.throttlePercentage = std::stoi(speed);
        } else {
            std::cout << "Please give value between 0-100" << std::endl;
        }
    } else if(strcmp(input, "steer") == 0) {
        std::cout << "Executing steer" << std::endl;
        std::cout << "Give steering angle (-1-1)" << std::endl;
        char angle[25];
        std::cin.get(angle, 25);
        std::cin.ignore(256, '\n');

        if(std::stof(angle) >= -1 && std::stof(angle) <= 1) {
            terminalStrategy->actuators.steeringAngle = std::stof(angle);
        } else {
            std::cout << "Please give value between -1 and 1" << std::endl;
        }
    } else if(strcmp(input, "brake") == 0) {
        std::cout << "Executing brake" << std::endl;
        std::cout << "Give brake percentage (0-100)" << std::endl;
        char brake[25];
        std::cin.get(brake, 25);
        std::cin.ignore(256, '\n');

        if(std::stoi(brake) == -1 || (std::stoi(brake) >= 0 && std::stoi(brake) <= 100)) {
            terminalStrategy->actuators.brakePercentage = std::stoi(brake);
        } else {
            std::cout << "Please give value between 0-100 or -1 to stop braking" << std::endl;
        }
    } else {
        std::cout << "Wrong command" << std::endl;
    }
    terminal();
}

void TerminalProcess::Run() {
    std::cout << "Terminal is running" << std::endl;
    terminal();
}

void TerminalProcess::Terminate() {

}

#endif