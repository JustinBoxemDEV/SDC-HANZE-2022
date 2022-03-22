#pragma once
#include "../VehicleControl/CommunicationStrategy.h"

class VehicleControlManager {
    public:
        CommunicationStrategy *vehicleStrategy;
        VehicleControlManager(CommunicationStrategy *vehicleStrategy__);
        void forward(int amount);
        void brake(int amount);
        void steer(float amount);
        void stop();
        void neutral();
};