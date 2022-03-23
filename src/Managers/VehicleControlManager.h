#pragma once
#include "../VehicleControl/CommunicationStrategy.h"

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