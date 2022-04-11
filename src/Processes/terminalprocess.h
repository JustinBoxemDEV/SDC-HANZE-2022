#pragma once

#include "process.h"
#include "../utils/TaskScheduler/TaskScheduler.h"
#include "../VehicleControl/communicationstrategy.h"
#include "../VehicleControl/strategies/canstrategy.h"
#include <iostream>
#include <string.h>

class TerminalProcess : public Process
{
    public:
        void setStrategy(CommunicationStrategy *strategy);
        void Run() override;
        void Terminate() override;
        void terminal();
};
