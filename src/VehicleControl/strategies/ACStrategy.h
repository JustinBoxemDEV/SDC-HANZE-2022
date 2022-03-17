#ifdef false
#pragma once
#include <stdio.h>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include <winsock2.h>
#endif
#include <iostream>
#include "../CommunicationStrategy.h"

class ACStrategy : CommunicationStrategy {
    public:
        SOCKET s;
        ACStrategy();
        void steer(float amount);
        void brake(int amount);
        void forward(int amount);
        void neutral();
        void stop();
        void gearShiftUp();
        void gearShiftDown();
     private:
        void throttle(int amount, int direction);
        template<class T>
        void sendCanMessage(T & canMessage) {
            const char* socketMessage = (const char*) canMessage;
            if(send(s, socketMessage, sizeof(socketMessage), 0) < 0) {
                puts("send failed");
            };

            puts("Data send");
        };
        template<class T>
        const char* merge(short arbitration_id, T & data) {
            char combined[sizeof arbitration_id + sizeof data];

            memcpy(combined, &arbitration_id, sizeof arbitration_id);
            memcpy(combined+sizeof arbitration_id, &data, sizeof data);

            const char* merged = combined;
            return merged;
        };
};
#endif