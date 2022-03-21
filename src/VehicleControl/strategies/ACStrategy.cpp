#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include "ACStrategy.h"
#include <stdio.h>
#include <winsock2.h>
#include <iostream>

ACStrategy::ACStrategy() {
    WSADATA wsa;
    struct sockaddr_in server;

    printf("\nInitialising Winsock...\n");
    if (WSAStartup(MAKEWORD(2,2),&wsa) != 0)
    {
        printf("Failed. Error Code : %d",WSAGetLastError());
    };

    printf("Initialised.\n");

    ACStrategy::s = socket(AF_INET, SOCK_DGRAM, 0);

    server.sin_addr.s_addr  =   inet_addr("127.0.0.1");
    server.sin_family       =   AF_INET;
    server.sin_port         =   htons(5454);

    if (connect(ACStrategy::s, (struct sockaddr *)&server, sizeof(server)) < 0) {
        puts("connect error");
    };

    // taskScheduler.SCH_Init();

    puts("connected");    
};

void ACStrategy::steer(float amount) {
    const char* data = ACStrategy::merge(__builtin_bswap16(0x12c), amount);
    // taskScheduler.SCH_Add_Task([=](){ACStrategy::sendCanMessage(data);}, 0, 0);
    ACStrategy::sendCanMessage(data);
};

void ACStrategy::brake(int amount) {
    const char* data = ACStrategy::merge(__builtin_bswap16(0x126), amount);
    // taskScheduler.SCH_Add_Task(ACStrategy::sendCanMessage(data), 0, 0);
    ACStrategy::sendCanMessage(data);
};

void ACStrategy::throttle(int amount, int direction) {
    const char* data = ACStrategy::merge(__builtin_bswap16(0x120), amount);
    // taskScheduler.SCH_Add_Task(ACStrategy::sendCanMessage(data), 0, 0);
    ACStrategy::sendCanMessage(data);
};

void ACStrategy::forward(int amount) {
    int a = amount;
    // taskScheduler.SCH_Add_Task([=, &amount](){ACStrategy::throttle(a, 1);}, 0, 0);
    ACStrategy::throttle(amount, 1);
};

void ACStrategy::neutral() {
    // TODO: get current gear and reduce it to 0
};

void ACStrategy::stop() {
    ACStrategy::throttle(0, 0);
    ACStrategy::brake(100);
    ACStrategy::neutral();
};

void ACStrategy::gearShiftUp() {
    int dummy = 0;
    const char* data = ACStrategy::merge(__builtin_bswap16(0x121), dummy);
    // taskScheduler.SCH_Add_Task(ACStrategy::sendCanMessage(data), 0, 0);
    ACStrategy::sendCanMessage(data);
};

void ACStrategy::gearShiftDown() {
    int dummy = 0;
    const char* data = ACStrategy::merge(__builtin_bswap16(0x122), dummy);
    // taskScheduler.SCH_Add_Task(ACStrategy::sendCanMessage(data), 0, 0);
    ACStrategy::sendCanMessage(data);
};

#endif
