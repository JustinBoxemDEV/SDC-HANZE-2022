#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include "acstrategy.h"

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

void ACStrategy::reset() {
    //Sleep(5000);
    INPUT controlAndODown[2] = {};
    ZeroMemory(controlAndODown, sizeof(controlAndODown));

    // std::cout << "Control has been pressed" << std::endl;

    controlAndODown[0].type = INPUT_KEYBOARD;
    controlAndODown[0].ki.wVk = 0x11;

    controlAndODown[1].type = INPUT_KEYBOARD;
    controlAndODown[1].ki.wVk = 0x4f;

    // std::cout << "O has been pressed" << std::endl;

    SendInput(ARRAYSIZE(controlAndODown), controlAndODown, sizeof(INPUT));

    Sleep(1);

    INPUT controlAndOUp[2] = {};
    ZeroMemory(controlAndOUp, sizeof(controlAndOUp));

    controlAndOUp[0].type = INPUT_KEYBOARD;
    controlAndOUp[0].ki.wVk = 0x11;
    controlAndOUp[0].ki.dwFlags = KEYEVENTF_KEYUP;

    controlAndOUp[1].type = INPUT_KEYBOARD;
    controlAndOUp[1].ki.wVk = 0x4f;
    controlAndOUp[1].ki.dwFlags = KEYEVENTF_KEYUP;

    SendInput(ARRAYSIZE(controlAndOUp), controlAndOUp, sizeof(INPUT));

    INPUT controlAndYDown[2] = {};
    ZeroMemory(controlAndYDown, sizeof(controlAndYDown));

    controlAndYDown[0].type = INPUT_KEYBOARD;
    controlAndYDown[0].ki.wVk = 0x11;

    // std::cout << "Control has been pressed" << std::endl;

    controlAndYDown[1].type = INPUT_KEYBOARD;
    controlAndYDown[1].ki.wVk = 0x59;

    // std::cout << "Y has been pressed" << std::endl;

    SendInput(ARRAYSIZE(controlAndYDown), controlAndYDown, sizeof(INPUT));

    Sleep(1);

    INPUT controlAndYUp[2] = {};
    ZeroMemory(controlAndYUp, sizeof(controlAndYUp));

    controlAndYUp[0].type = INPUT_KEYBOARD;
    controlAndYUp[0].ki.wVk = 0x11;
    controlAndYUp[0].ki.dwFlags = KEYEVENTF_KEYUP;

    controlAndYUp[1].type = INPUT_KEYBOARD;
    controlAndYUp[1].ki.wVk = 0x59;
    controlAndYUp[1].ki.dwFlags = KEYEVENTF_KEYUP;

    SendInput(ARRAYSIZE(controlAndYUp), controlAndYUp, sizeof(INPUT));
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
