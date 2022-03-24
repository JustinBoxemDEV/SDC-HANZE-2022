#pragma once
#include "../VehicleControl/communicationstrategy.h"

class VehicleControlManager {
    public:
        CommunicationStrategy *vehicleStrategy;
        VehicleControlManager(CommunicationStrategy *vehicleStrategy__);
        void throttle();
        void brake();
        void steer();
        void stop();
        void neutral();
};