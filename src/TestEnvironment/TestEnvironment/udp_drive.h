#pragma once
#include <stdio.h>
#include <winsock2.h>
#include <iostream>

class UDP_DRIVE {
    public:
        static SOCKET s;
        static void drive();
        static void throttle(int speedPercentage);
        static void brake(int brakePercentage);
        static void steer(float steeringAngle);
        static void gearShiftUp();
        static void gearShiftDown();
        static void init();
};