#ifdef linux
#include "CANStrategy.h"
#include <iostream>
#include <string.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include "../../Logger/logger.h"
#include "../../utils/Time/time.h"

std::string timestamp;

CANStrategy::CANStrategy() {
    // system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 type can bitrate 500000");
    // system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 up");
    std::cout << "Initialize bus" << std::endl;
    // Vcan
    system("sudo ip link del dev vcan0 type vcan");
    system("sudo ip link add dev vcan0 type vcan");
    system("sudo ip link set vcan0 type vcan");
    system("sudo ip link set vcan0 up");
    timestamp = Time::currentDateTime();
    Logger::createFile("send " + timestamp);
    Logger::createFile("receive " + timestamp);
    CANStrategy::init("vcan0");
};

void CANStrategy::throttle(int amount, int direction) {
    typedef CANStrategy::frame<std::byte[4]> throttleFrame;

    throttleFrame canMessage;

    canMessage.can_id = 0x120;
    canMessage.can_dlc = 8;
    canMessage.data[0] = (std::byte) amount;
    canMessage.data[1] = (std::byte) 0x00;
    canMessage.data[2] = (std::byte) direction;
    canMessage.data[3] = (std::byte) 0x00;
    canMessage.trailer =  0x00000000;

    Logger::setActiveFile("send " + timestamp);
    Logger::info("Throttle : speed = " + std::to_string(amount) + " : direction = " + std::to_string(direction));

    CANStrategy::sendCanMessage(canMessage);
};

void CANStrategy::steer(float amount) {
    typedef CANStrategy::frame<float> steerFrame;

    steerFrame canMessage;

    canMessage.can_id = 0x12c;
    canMessage.can_dlc = 8;
    canMessage.data = amount;
    canMessage.trailer = 0x00000000;

    Logger::setActiveFile("send " + timestamp);
    Logger::info("Steering : angle = " + std::to_string(amount));

    CANStrategy::sendCanMessage<steerFrame>(canMessage);
};

void CANStrategy::brake(int amount) {
    typedef CANStrategy::frame<std::byte[4]> brakeFrame;

    brakeFrame canMessage;

    canMessage.can_id = 0x126;
    canMessage.can_dlc = 8;
    canMessage.data[0] = (std::byte) amount;
    canMessage.data[1] = (std::byte) 0x00;
    canMessage.data[2] = (std::byte) 0x00;
    canMessage.data[3] = (std::byte) 0x00;
    canMessage.trailer =  0x00000000;

    Logger::setActiveFile("send " + timestamp);
    Logger::info("Brake : amount = " + std::to_string(amount));

    CANStrategy::sendCanMessage(canMessage);
};

void CANStrategy::forward(int amount) {
    CANStrategy::throttle(amount, 1);
};

void CANStrategy::neutral() {
    CANStrategy::throttle(0, 0);
};

void CANStrategy::stop() {
    Logger::setActiveFile("send " + timestamp);
    Logger::info("Force stopping");

    // stop gas, break and set to neutral
    CANStrategy::throttle(0, 1);
    sleep(0.04);
    CANStrategy::brake(100);
};

void CANStrategy::readCANMessages() {
    int nbytes;
    struct can_frame frame;
    nbytes = read(CANStrategy::cansocket, &frame, sizeof(struct can_frame));
    if (nbytes < 0) {
    perror("Read");
        
    };
    printf("0x%03X [%d] ",frame.can_id, frame.can_dlc);
    for (int i = 0; i < frame.can_dlc; i++) {
        Logger::setActiveFile("receive " + timestamp);
        Logger::info(frame.data[i]);
        printf("%02X ",frame.data[i]);
        printf("\r\n");
    };
};

void CANStrategy::init(const char* canType) {
    if ((CANStrategy::cansocket = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
        perror("Socket");
    };
    
    // Set up the bind
    struct ifreq ifr;
    strcpy(ifr.ifr_name, canType);

    //  if you use zero as the interface index, you can retrieve packets from all CAN interfaces.
    ioctl(CANStrategy::cansocket, SIOCGIFINDEX, &ifr);
    ioctl(CANStrategy::cansocket, SIOCSIFPFLAGS, &ifr);

    struct sockaddr_can addr;

    memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;   

    if(bind(CANStrategy::cansocket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Bind");
    };

    // When starting the kart
    sleep(15);
    // Wait 15 seconds after kart is turned on, set the kart to drive (forwards) using message: can0 0x0000000120 50 00 01 00 00 00 00 00
    throttle(0, 1);
    sleep(0.1);
    // Make sure the brake won't activate while accelerating. Set brakes to 0 using message: can0 0x0000000126 00 00 00 00 00 00 00 00
    brake(0);
    // Homing message: can0 0x0000006F1 00 00 00 00 00 00 00 00 (correct wheels, can last between 1-20 seconds)
    steer(0.00);    
};
#endif