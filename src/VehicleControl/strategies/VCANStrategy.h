#pragma once
#include "../CANBus.h"

class VCANStrategy : public CANBus {
    public:
        VCANStrategy() {
            system("sudo ip link del dev vcan0 type vcan");
            system("sudo ip link add dev vcan0 type vcan");
            system("sudo ip link set vcan0 type vcan");
            system("sudo ip link set vcan0 up");
        
            CANBus::init("vcan0");
        };
};

