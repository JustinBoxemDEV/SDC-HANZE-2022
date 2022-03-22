#include "CANMessageTask.h"
#include <iostream>

// For real environment
CANMessageTask::CANMessageTask(const char* canMessage){
    _message = canMessage;
}

void CANMessageTask::execute(){
    // if (write(cansocket, &canMessage, sizeof(canMessage)) != sizeof(canMessage)) {
    //             perror("Write");
    //         };

    // std::cout << "Executed message task " << std::endl;
}
