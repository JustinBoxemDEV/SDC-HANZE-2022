#ifdef linux
#include "terminalprocess.h"

CommunicationStrategy *terminalStrategy;

void TerminalProcess::setStrategy(CommunicationStrategy *strategy) {
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

        terminalStrategy->actuators.throttlePercentage = std::stoi(speed);
    } else if(strcmp(input, "steer") == 0) {
        std::cout << "Executing steer" << std::endl;
        std::cout << "Give steering angle (-1-1)" << std::endl;
        char angle[25];
        std::cin.get(angle, 25);
        std::cin.ignore(256, '\n');

        terminalStrategy->actuators.steeringAngle = std::stof(angle);
    } else if(strcmp(input, "brake") == 0) {
        std::cout << "Executing brake" << std::endl;
        std::cout << "Give brake percentage" << std::endl;
        char brake[25];
        std::cin.get(brake, 25);
        std::cin.ignore(256, '\n');

        terminalStrategy->actuators.brakePercentage = std::stoi(brake);
    } else {
        std::cout << "Wrong command" << std::endl;
    }
    terminal();
}

void TerminalProcess::Run() {
    terminal();
}

void TerminalProcess::Terminate() {

}

#endif