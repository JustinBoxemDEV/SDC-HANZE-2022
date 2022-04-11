#if linux
#include "readprocess.h"

CommunicationStrategy *strategy2;

void ReadProcess::setStrategy(CommunicationStrategy *_strategy) {
    strategy2 = _strategy;
};

void ReadProcess::Run() {
    std::cout << "Readprocess is running" << std::endl;
    while(true) {
        ((CANStrategy*) strategy2)->readCANMessages();
    };
};

void ReadProcess::Terminate() {

};

#endif