#ifdef linux
#pragma once
#include "../CommunicationStrategy.h"
#include <iostream>
#include <string.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>

class CANStrategy : public CommunicationStrategy {
    public:
        int cansocket;
        std::string timestamp;
        CANStrategy();
        void steer();
        void brake(int amount);
        void throttle(int amount, int direction);
        void neutral();
        void stop();
        void readCANMessages();
        void forward(int amount);
        void backward(int amount)
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
        
        template<typename T>
        void sendCanMessage(T & canMessage) {
            if (write(cansocket, &canMessage, sizeof(canMessage)) != sizeof(canMessage)) {
                perror("Write");
            };
        };
};
#endif