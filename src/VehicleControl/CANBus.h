#pragma once
#include "CommunicationStrategy.h"
#include <iostream>
#include <string.h>
#ifdef linux
#include <sys/socket.h>
#include <linux/can.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#endif

struct CANBus : public CommunicationStrategy {
    public:
        int cansocket;
        virtual void throttle(int amount, int direction) {
            typedef frame<std::byte[4]> throttleFrame;

            throttleFrame canMessage;

            canMessage.can_id = 0x120;
            canMessage.can_dlc = 8;
            canMessage.data[0] = (std::byte) amount;
            canMessage.data[1] = (std::byte) 0x00;
            canMessage.data[2] = (std::byte) direction;
            canMessage.data[3] = (std::byte) 0x00;
            canMessage.trailer =   0x00000000;

            sendCanMessage(canMessage);
        };
        virtual void steer(float amount) {
            typedef frame<float> steerFrame;

            steerFrame canMessage;

            canMessage.can_id = 0x12c;
            canMessage.can_dlc = 8;
            canMessage.data = amount;
            canMessage.trailer = 0x00000000;

            sendCanMessage(canMessage);
        };

        virtual void brake(int amount) {
            typedef frame<std::byte[4]> brakeFrame;

            brakeFrame canMessage;

            canMessage.can_id = 0x126;
            canMessage.can_dlc = 8;
            canMessage.data[0] = (std::byte) amount;
            canMessage.data[1] = (std::byte) 0x00;
            canMessage.data[2] = (std::byte) 0x00;
            canMessage.data[3] = (std::byte) 0x00;
            canMessage.trailer =  0x00000000;

            sendCanMessage(canMessage);
        };

        virtual void forward(int amount) {
            throttle(amount, 1);
        };

        virtual void neutral() {
            throttle(0, 0);
        };

        virtual void stop() {
            // stop gas, break and set to neutral
        };
        
        virtual void readCANMessages() {
            int nbytes;
            struct can_frame frame;
            nbytes = read(cansocket, &frame, sizeof(struct can_frame));
            if (nbytes < 0) {
            perror("Read");
                
            };
            printf("0x%03X [%d] ",frame.can_id, frame.can_dlc);
            for (int i = 0; i < frame.can_dlc; i++)
                printf("%02X ",frame.data[i]);
                printf("\r\n");
        };
        virtual void init(const char* canType) {
            if ((cansocket = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
                perror("Socket");
            };
            
            // Set up the bind
            struct ifreq ifr;
            strcpy(ifr.ifr_name, canType);

            //  if you use zero as the interface index, you can retrieve packets from all CAN interfaces.
            ioctl(cansocket, SIOCGIFINDEX, &ifr);
            ioctl(cansocket, SIOCSIFPFLAGS, &ifr);

            struct sockaddr_can addr;

            memset(&addr, 0, sizeof(addr));
            addr.can_family = AF_CAN;
            addr.can_ifindex = ifr.ifr_ifindex;   

            if(bind(cansocket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
                perror("Bind");
            };
            
            // When starting the kart

            // Wait 15 seconds after kart is turned on, set the kart to drive (forwards) using message: can0 0x0000000120 50 00 01 00 00 00 00 00
            throttle(0, 1);
            sleep(0.1);
            // Make sure the brake won't activate while accelerating. Set brakes to 0 using message: can0 0x0000000126 00 00 00 00 00 00 00 00
            brake(0);
            // Homing message: can0 0x0000006F1 00 00 00 00 00 00 00 00 (correct wheels, can last between 1-20 seconds)
            steer(0.00);    
        };
    private:
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
        template <class T>
        void sendCanMessage(T & canMessage) {
            if (write(cansocket, &canMessage, sizeof(canMessage)) != sizeof(canMessage)) {
                perror("Write");
            };
        };
};