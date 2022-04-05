#ifndef CAN_PROCESS_H
#define CAN_PROCESS_H

#include "process.h"

class CanProcess : public Process
{
    private:
    public:
        void Init();
        void Run();
};

#endif