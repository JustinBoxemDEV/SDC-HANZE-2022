#ifdef linux
#pragma once
#include "../communicationstrategy.h"
#include <string.h>
#include <linux/can.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include "../../Logger/logger.h"
#include <cstdio>
#include <iostream>
#include <ctime>
#include <mutex>

class CANStrategy : public CommunicationStrategy {
    public:
        int cansocket;
        std::mutex loggerMutex;
        std::string timestamp;
        CANStrategy();
        void steer() override;
        void brake() override;
        void forward() override;
        void neutral() override;
        void stop() override;
        void readCANMessages();
        void homing();
        void backward();
        void init(const char* canType);
        
        template <class T>
        struct frame {
            canid_t     can_id;
            __u8        can_dlc;
            __u8        __pad; 
            __u8        __res0; 
            __u8        __res1;
            T           data;
            __u_int     trailer;
        };
    private:
        void throttle(int amount, int direction);
        template<typename T>
        void sendCanMessage(T & canMessage) {
            if (write(cansocket, &canMessage, sizeof(canMessage)) != sizeof(canMessage)) {
                perror("Write");
            };
        };
};
#endif