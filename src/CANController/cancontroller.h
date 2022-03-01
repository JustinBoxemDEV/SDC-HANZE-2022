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
        static void readFrame();
        static void sendFrame();
        static void throttle(short speed, std::byte direction);
        static void brake();
        static void steer();
        static void setup();
};