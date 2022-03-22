#pragma once
#include "../utils/TaskScheduler/TaskScheduler.h"

class CommunicationStrategy {
    public:
        virtual void steer(float amount) = 0;
        virtual void brake(int amount) = 0;
        virtual void forward(int amount) = 0;
        virtual void neutral() = 0;
        virtual void stop() = 0;
        // TaskScheduler TaskScheduler;
    private:
        virtual void throttle(int amount, int direction) = 0;
        template<class T>
        void sendCanMessage(T & canMessage);
};