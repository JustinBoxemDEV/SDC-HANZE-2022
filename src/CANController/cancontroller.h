#ifdef linux
#pragma once
#include <iostream>
#include <sys/socket.h>
#include <linux/can.h>
#include <string.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>

class CANController {
    public:
        static int cansocket;
        static void create();
        static void closeCANController(std::string canType="can");
        static void throttle(int speed, int direction);
        static void brake(int brakePercentage);
        static void steer();
        static void init(std::string canType="can");
        static void readCANMessages();
};

#endif