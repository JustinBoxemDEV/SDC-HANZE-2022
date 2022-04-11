#pragma once
#include "readprocess.h"
#include "terminalprocess.h"
#include "../utils/TaskScheduler/TaskScheduler.h"
#include "../VehicleControl/communicationstrategy.h"

class CanProcess : public Process
{
    private:
        //CommunicationStrategy* strategy;
        TaskScheduler taskScheduler;
    public:
        CanProcess(MediaInput * input);
        void setReadProcess(ReadProcess *_readProcess);
        void setTerminalProcess(TerminalProcess *_terminalProcess);
        void Run() override;
        void Terminate() override;
};