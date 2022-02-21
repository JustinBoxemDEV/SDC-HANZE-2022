#pragma once

class Brake {
    public:
        const double CAN_MSG_SENDING_SPEED = .04; // 100Hz
        void run();
};