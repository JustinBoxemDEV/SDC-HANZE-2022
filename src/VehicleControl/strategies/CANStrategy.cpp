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
    timestamp = Time::currentDateTime();

    Logger::createFile("send " + timestamp);
    Logger::createFile("receive " + timestamp);

    CANStrategy::init("vcan0"); // use vcan0 for testing, can0 for real kart.
};

void CANStrategy::init(const char* canType) {
    if (strcmp(canType, "can0")){
        std::cout << "Initializing canbus" << std::endl;
        system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 type can bitrate 500000");
        system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 up");
    }else if (strcmp(canType, "vcan0")){
        std::cout << "Initializing virtual canbus" << std::endl;
        system("sudo ip link del dev vcan0 type vcan");
        system("sudo ip link add dev vcan0 type vcan");
        system("sudo ip link set vcan0 type vcan");
        system("sudo ip link set vcan0 up");
    } else{
        std::cout << "Wrong can type" << std::endl;
    }

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
    sleep(5);
    // Wait 15 seconds after kart is turned on, set the kart to drive (forward) using message: can0 0x0000000120 50 00 01 00 00 00 00 00
    actuators.throttlePercentage = 0;
    sleep(0.04);
    // Make sure the brake won't activate while accelerating. Set brakes to 0 using message: can0 0x0000000126 00 00 00 00 00 00 00 00
    actuators.brakePercentage = 0;
    sleep(0.04);
    // Homing message: can0 0x0000006F1 00 00 00 00 00 00 00 00 (correct wheels, can last between 1-20 seconds)
    actuators.steeringAngle = 0;
    sleep(0.04);
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

void CANStrategy::steer() {
    typedef CANStrategy::frame<float> steerFrame;
    int amount = 0;
    steerFrame canMessage;

    canMessage.can_id = 0x12c;
    canMessage.can_dlc = 8;
    canMessage.data = actuators.steeringAngle;
    canMessage.trailer = 0x00000000;

    Logger::setActiveFile("send " + timestamp);
    Logger::info("Steering : angle = " + std::to_string(actuators.steeringAngle));

    CANStrategy::sendCanMessage<steerFrame>(canMessage);
};

void CANStrategy::brake() {
    typedef CANStrategy::frame<std::byte[4]> brakeFrame;

    brakeFrame canMessage;

    canMessage.can_id = 0x126;
    canMessage.can_dlc = 8;
    canMessage.data[0] = (std::byte) actuators.brakePercentage;
    canMessage.data[1] = (std::byte) 0x00;
    canMessage.data[2] = (std::byte) 0x00;
    canMessage.data[3] = (std::byte) 0x00;
    canMessage.trailer =  0x00000000;

    Logger::setActiveFile("send " + timestamp);
    Logger::info("Brake : amount = " + std::to_string(actuators.brakePercentage));

    CANStrategy::sendCanMessage(canMessage);
};

void CANStrategy::forward() {
    CANStrategy::throttle(actuators.throttlePercentage, 1);
};

void CANStrategy::backward() {
    CANStrategy::throttle(actuators.throttlePercentage, 2);
};

void CANStrategy::neutral() {
    CANStrategy::throttle(0, 0);
};

void CANStrategy::stop() {
    // Logger::setActiveFile("send " + timestamp);
    // Logger::info("Force stopping");

    // // stop gas, break and set to neutral
    // CANStrategy::throttle(0, 0);
    // sleep(0.04);
    // CANStrategy::brake(100);
};

void CANStrategy::readCANMessages() {
    int nbytes;
    struct can_frame frame;
    nbytes = read(CANStrategy::cansocket, &frame, sizeof(struct can_frame));
    if (nbytes < 0) {
    perror("Read");
        
    };
    // std::string s(frame.can_id, sizeof(frame.can_id));
    // std::cout << s << ": bytes conversion\n";
    // std::string data(frame.can_dlc, sizeof(frame.can_dlc));
    // std::cout << data << ": data conversion\n";

    // std::string receivedMessage = s+data;
    // std::cout << receivedMessage << " : the received message\n";

    std::string s = std::to_string(frame.can_id);
    std::cout << "The id: " << s << std::endl;

    std::stringstream stream;
    stream << "0x" << std::hex << frame.can_id;
    std::string id(stream.str());
    std::cout << "id: " << id << std::endl;

    stream.clear();

    std::string dlc = "[" + std::to_string(frame.can_dlc) + "]";
    std::cout << "The dlc: " << dlc << std::endl;

    // std::string data = std::to_string(frame.data);
    // std::cout << "The data: " << data << std::endl;

    int dataLength = std::stoi(std::to_string(frame.can_dlc));
    std::string data;
    std::stringstream dataStream;    
    //printf("0x%03X [%d] ",frame.can_id, frame.can_dlc);
    for (int i = 0; i < frame.can_dlc; i++) {
        //Logger::setActiveFile("receive " + this->timestamp);
        //Logger::info(frame.data[i]);

        dataStream << std::hex << std::stoi(std::to_string(frame.data[i]));
        dataStream << " ";
        std::string hex(dataStream.str());

        std::cout << "data " << i << ": " << hex << "\n";

        //std::cout << "frame data: " << std::hex << (int) std::to_string(frame.data[i]) << std::endl;
        // printf("%02X ",frame.data[i]);
        // printf("\r\n");
        std::cout << i << " -> i " << dataLength << " -> dlc " << std::endl;
        if(i == dataLength-1) {
            std::cout << "STOP THE FOR LOOP" << std::endl;
            data.append(hex);
        }
    };
    
    std::string canMessage = id + " " + dlc + " " + data;
    std::cout << "canMessage: " << canMessage << std::endl;
    
    Logger::setActiveFile("receive " + this->timestamp);
    Logger::info(canMessage);

    std::cout << "test the data: " << data << std::endl;
    data.clear();
};

#endif