#pragma once
// TODO: finding correct canbus library for C++ (import can in python)

class Gas {
    public:
        const double CAN_MSG_SENDING_SPEED = .04; // 100Hz
        void run();
};