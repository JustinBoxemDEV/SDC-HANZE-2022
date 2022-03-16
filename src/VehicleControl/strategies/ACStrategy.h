#pragma once
#include <stdio.h>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include <winsock2.h>
#endif
#include <iostream>
#include "CommunicationStrategy.h"

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
            send(merge(__builtin_bswap16(0x12c), amount));
        };
        void brake(int amount) {
            send(merge(__builtin_bswap16(0x126), amount));
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
            send(merge(__builtin_bswap16(0x121), 0));
        };
        void gearShiftDown() {
            send(merge(__builtin_bswap16(0x122), 0));
        };
     private:
        void throttle(int amount, int direction) {
            send(merge(__builtin_bswap16(0x120), amount));
        };
        template<class T>
        void send(T & canMessage) {
            if(send(s, canMessage, sizeof(canMessage), 0) < 0) {
                puts("send failed");
            };

            puts("Data send");
        };
        template<class T>
        const char* merge(short arbitration_id, T & data) {
            char combined[sizeof arbitration_id + sizeof data];

            memcpy(combined, &arbitration_id, sizeof arbitration_id);
            memcpy(combined+sizeof arbitration_id, &data, sizeof data);

            return (const char*) combined;
        };
};

