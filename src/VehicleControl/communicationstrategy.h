#pragma once
#include "../utils/TaskScheduler/TaskScheduler.h"

class CommunicationStrategy {
    public:
        virtual void steer() = 0;
        virtual void brake() = 0;
        virtual void forward() = 0;
        virtual void neutral() = 0;
        virtual void stop() = 0;
        // TaskScheduler TaskScheduler;
        struct Actuators {
            float steeringAngle = 0;
            float throttlePercentage = 0;
            float brakePercentage = 0;
            float steeringFeedback = 0;
        };
        Actuators actuators;
    private:
        template<class T>
        void sendCanMessage(T & canMessage);
};