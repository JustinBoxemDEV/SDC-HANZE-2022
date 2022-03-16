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

        static void init(std::string canType="can");

        static void throttle(int speedPercentage, int direction);
        static void brake(int brakePercentage);
        static void steer(float amount);

        static void readCANMessages();
        static void closeCANController(std::string canType="can");

    private:
        template<class T>
        static void send(T & canMessage) {
            if (write(CANController::cansocket, &canMessage, sizeof(canMessage)) != sizeof(canMessage)) {
                perror("Write");
            };
        };
};