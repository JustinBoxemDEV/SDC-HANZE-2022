#pragma once
#include <iostream>
#include <sys/socket.h>
#include <linux/can.h>
#include <string.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>

class CANSocket {
    public:
        static int cansocket;
        static void create();
        static void closeCANSocket();
        static void readFrame();
        static void sendFrame();
};