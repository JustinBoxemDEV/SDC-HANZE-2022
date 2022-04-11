#ifdef linux
#include "readprocess.h"

CommunicationStrategy *strategy;

void setStrategy(CommunicationStrategy *_strategy) {
    strategy = _strategy;
};

void run() {
    ( (CANStrategy) strategy)->readCANMessages();
};
#endif