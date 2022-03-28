#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#pragma once
#include <stdio.h>
#include <winsock2.h>
#include <iostream>
#include "../CommunicationStrategy.h"
#include "../../utils/TaskScheduler/TaskScheduler.h"
#include "../../utils/TaskScheduler/MessageTask.h"

class ACStrategy : public CommunicationStrategy {
    public:
        SOCKET s;
        ACStrategy();
        void steer() override;
        void brake() override;
        void forward() override;
        void neutral();
        void stop();
        void gearShiftUp();
        void gearShiftDown();
     private:
        template<class T>
        const char* merge(short arbitration_id, T & data) {
            char combined[sizeof arbitration_id + sizeof data];

            memcpy(combined, &arbitration_id, sizeof arbitration_id);
            memcpy(combined+sizeof arbitration_id, &data, sizeof data);

            const char* merged = combined;
            return merged;
        };
        void sendCanMessage(const char* canMessage) {
            const char* socketMessage = (const char*) canMessage;
            if(send(s, socketMessage, sizeof(socketMessage), 0) < 0) {
                puts("send failed");
            };
            puts("Sent can-message");
        };
};
#endif