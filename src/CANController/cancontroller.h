#pragma once
#include <iostream>
#include <sys/socket.h>
#include <linux/can.h>
#include <string.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>

class CANController {
    public:
        static int cansocket;
        static void create();
        static void closeCANController();
        static void throttle(int speed, int direction);
        static void brake(int brakePercentage);
        static void steer(float amount);
        static void init(std::string canName="can0", std::string canType="can");
};