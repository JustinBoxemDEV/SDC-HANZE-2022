#ifdef linux
#include "terminalprocess.h"

CommunicationStrategy *terminalStrategy;

void TerminalProcess::setStrategy(CommunicationStrategy *strategy) {
    terminalStrategy = strategy;
}

void TerminalProcess::terminal() {
    std::cout << "Test terminal" << std::endl;
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
    }
    terminal();
}

void TerminalProcess::Run() {
    terminal();
}

void TerminalProcess::Terminate() {

}

#endif