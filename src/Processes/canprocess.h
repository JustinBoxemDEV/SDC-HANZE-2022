#ifndef CAN_PROCESS_H
#define CAN_PROCESS_H

#include "readprocess.h"
#include "../utils/TaskScheduler/TaskScheduler.h"
#include "../VehicleControl/communicationstrategy.h"

class CanProcess : public Process
{
    private:
        //CommunicationStrategy* strategy;
        TaskScheduler taskScheduler;
    public:
        struct Actuators {
            float steeringAngle = 0;
            int throttlePercentage = 0;
            int brakePercentage = 0;
            float steeringFeedback = 0;
        };
        CanProcess(MediaInput * input);
        static Actuators actuators;
        void setReadProcess(ReadProcess *_readProcess);
        void Run() override;
        void Terminate() override;
};

#endif