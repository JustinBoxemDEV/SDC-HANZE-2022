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
        template<class T>
        static void send(short arbitration_id, T & data) {
            char combined[sizeof arbitration_id + sizeof data];

            memcpy(combined, &arbitration_id, sizeof arbitration_id);
            memcpy(combined+sizeof arbitration_id, &data, sizeof data);

            const char *canMessage = (const char*) combined;

            if(send(s, canMessage, sizeof(canMessage), 0) < 0) {
                puts("send failed");
            };

            puts("Data send");
        };
};