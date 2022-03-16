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
        ACStrategy() {
            WSADATA wsa;
            struct sockaddr_in server;

            printf("\nInitialising Winsock...\n");
            if (WSAStartup(MAKEWORD(2,2),&wsa) != 0)
            {
                printf("Failed. Error Code : %d",WSAGetLastError());
            };

            printf("Initialised.\n");

            s = socket(AF_INET, SOCK_DGRAM, 0);

            server.sin_addr.s_addr  =   inet_addr("127.0.0.1");
            server.sin_family       =   AF_INET;
            server.sin_port         =   htons(5454);

            if (connect(s, (struct sockaddr *)&server, sizeof(server)) < 0) {
                puts("connect error");
            };

            puts("connected");    
        };
        void steer(float amount) {
            const char* data = merge(__builtin_bswap16(0x12c), amount);
            sendCanMessage(data);
        };
        void brake(int amount) {
            const char* data = merge(__builtin_bswap16(0x126), amount);
            sendCanMessage(data);
        };
        void forward(int amount) {
            throttle(amount, 1);
        };
        void neutral() {
            // TODO: get current gear and reduce it to 0
        };
        void stop() {
            throttle(0, 0);
            brake(100);
            neutral();
        };
        void gearShiftUp() {
            int dummy = 0;
            const char* data = merge(__builtin_bswap16(0x121), dummy);
            sendCanMessage(data);
        };
        void gearShiftDown() {
            int dummy = 0;
            const char* data = merge(__builtin_bswap16(0x122), dummy);
            sendCanMessage(data);
        };
     private:
        void throttle(int amount, int direction) {
            const char* data = merge(__builtin_bswap16(0x120), amount);
            sendCanMessage(data);
        };
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

