#include "terminalprocess.h"

CommunicationStrategy *terminalStrategy;

void setStrategy(CommunicationStrategy *strategy) {
    terminalStrategy = strategy;
}

void TerminalProcess::terminal() {
    char input[25];
    std::cin.get(input, 25);
    std::cin.ignore(256, '\n');

    if (strcmp(input, "throttle") == 0) {
        std::cout << "Execute throttle" << std::endl;
        std::cout << "Give speed percentage (0-100)" << std::endl;
        char speed[25];
        std::cin.get(speed, 25);
        std::cin.ignore(256, '\n');

        terminalStrategy->actuators.throttlePercentage = 20;
    }
    terminal();
}

void TerminalProcess::Run() {
    terminal();
}

void TerminalProcess::Terminate() {

}