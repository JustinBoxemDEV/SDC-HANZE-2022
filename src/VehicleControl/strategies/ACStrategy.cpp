#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include "ACStrategy.h"
#include "..\..\utils\TaskScheduler\MessageTask.h"
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
        puts("Connect error");
    };

    taskScheduler.SCH_Init();

    puts("Connected");    
};

void ACStrategy::steer() {
    const char* data = ACStrategy::merge(__builtin_bswap16(0x12c), actuators.steeringAngle);
    ACStrategy::sendCanMessage(data);
};

void ACStrategy::brake() {
    const char* data = ACStrategy::merge(__builtin_bswap16(0x126), actuators.brakePercentage);
    ACStrategy::sendCanMessage(data);
};

void ACStrategy::forward() {
    const char* data = ACStrategy::merge(__builtin_bswap16(0x120), actuators.throttlePercentage);
    ACStrategy::sendCanMessage(data);
};

void ACStrategy::neutral() {
    // Temporary fix
    gearShiftDown();
    gearShiftDown();
    gearShiftDown();
    gearShiftDown();
};

void ACStrategy::stop() {
    actuators.brakePercentage = 0;
    actuators.throttlePercentage = 0;
    actuators.steeringAngle = 0;
    
    forward();
    brake();
    neutral();
};

void ACStrategy::gearShiftUp() {
    int dummy = 0;
    const char* data = ACStrategy::merge(__builtin_bswap16(0x121), dummy);
    ACStrategy::sendCanMessage(data);
};

void ACStrategy::gearShiftDown() {
    int dummy = 0;
    const char* data = ACStrategy::merge(__builtin_bswap16(0x122), dummy);
    ACStrategy::sendCanMessage(data);
};

#endif
