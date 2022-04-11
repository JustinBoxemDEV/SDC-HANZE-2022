#ifdef linux
#include "readprocess.h"

CommunicationStrategy *readStrategy;

void ReadProcess::setStrategy(CommunicationStrategy *_strategy) {
    readStrategy = _strategy;
};

void ReadProcess::Run() {
    std::cout << "Readprocess is running" << std::endl;
    while(true) {
        ((CANStrategy*) readStrategy)->readCANMessages();
    };
};

void ReadProcess::Terminate() {

};

#endif