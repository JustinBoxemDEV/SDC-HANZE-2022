#pragma once
#include "../CANBus.h"

class CANStrategy : public CANBus {
    public:
        CANStrategy() {
            system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 type can bitrate 500000");
            system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 up");
        
            CANBus::init("can0");
        };
};

