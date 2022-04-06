#ifndef CAN_PROCESS_H
#define CAN_PROCESS_H

#include "process.h"
#include "utils/TaskScheduler/TaskScheduler.h"
#include "VehicleControl/communicationstrategy.h"


class CanProcess : public Process
{
    private:
        CommunicationStrategy* strategy;
        TaskScheduler taskScheduler;
        struct Actuators {
            float steeringAngle = 0;
            int throttlePercentage = 0;
            int brakePercentage = 0;
            float steeringFeedback = 0;
        };

    public:
        static Actuators actuators;
        void Init(MediaInput* input) override;
        void Run() override;
        void Terminate() override;
};

#endif