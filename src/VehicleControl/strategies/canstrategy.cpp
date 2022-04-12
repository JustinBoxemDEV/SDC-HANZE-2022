#ifdef linux
#include "canstrategy.h"

CommunicationStrategy::Actuators CommunicationStrategy::actuators;

bool zeroThrottle = false;

CANStrategy::CANStrategy() {
    timestamp = Time::currentDateTime();

    Logger::createFile("send " + timestamp);
    Logger::createFile("receive " + timestamp);

    CANStrategy::init("vcan0"); // use vcan0 for testing, can0 for real kart.
};

void CANStrategy::init(const char* canType) {
    if (strcmp(canType, "can0") == 0) {
        std::cout << "Initializing canbus" << std::endl;
        system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 type can bitrate 500000");
        system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 up");
        system("echo wijgaanwinnen22 |sudo -S sudo ifconfig can0 txqueuelen 1000");
    } else if (strcmp(canType, "vcan0") == 0) {
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
   
    // set the kart to drive (forward) using message: can0 0x0000000120 50 00 01 00 00 00 00 00
    actuators.throttlePercentage = 0;
    CANStrategy::forward();

    float delay = 0.1;
    delay *= CLOCKS_PER_SEC;
    std::cout << delay << std::endl;
    clock_t now = clock();
    while(clock() - now < delay);

    // Make sure the brake won't activate while accelerating. Set brakes to 0 using message: can0 0x0000000126 00 00 00 00 00 00 00 00
    actuators.brakePercentage = 0;
    CANStrategy::brake();

    std::cout << delay << std::endl;
    now = clock();
    while(clock() - now < delay);

    actuators.steeringAngle = 0.0;
    CANStrategy::steer();
    std::cout << "Test steering" << std::endl;
};

void CANStrategy::throttle(int amount, int direction) {
    if((actuators.throttlePercentage == 0 && zeroThrottle) || actuators.brakePercentage >= 0) {
        return;
    }
    if(actuators.throttlePercentage == 0) {
        zeroThrottle = true;
    } else {
        zeroThrottle = false;
    }
    typedef CANStrategy::frame<std::byte[4]> throttleFrame;

    throttleFrame canMessage;

    canMessage.can_id = 0x120;
    canMessage.can_dlc = 8;
    canMessage.data[0] = (std::byte) amount;
    canMessage.data[1] = (std::byte) 0x00;
    canMessage.data[2] = (std::byte) direction;
    canMessage.data[3] = (std::byte) 0x00;
    canMessage.trailer =  0x00000000;

    std::unique_lock<std::mutex> locker(loggerMutex);
    Logger::info("Throttle : speed = " + std::to_string(amount) + " : direction = " + std::to_string(direction), "send " + timestamp);

    CANStrategy::sendCanMessage<throttleFrame>(canMessage);
    locker.unlock();
};

void CANStrategy::steer() {
    typedef CANStrategy::frame<float> steerFrame;
    int amount = 0;
    steerFrame canMessage;

    canMessage.can_id = 0x12c;
    canMessage.can_dlc = 8;
    canMessage.data = actuators.steeringAngle;
    canMessage.trailer = 0x00000000;

    std::unique_lock<std::mutex> locker(loggerMutex);
    Logger::info("Steering : angle = " + std::to_string(actuators.steeringAngle), "send " + timestamp);

    CANStrategy::sendCanMessage<steerFrame>(canMessage);
    locker.unlock();
};

void CANStrategy::brake() {
    if(actuators.brakePercentage == -1) {
        return;
    }
    typedef CANStrategy::frame<std::byte[4]> brakeFrame;

    brakeFrame canMessage;

    canMessage.can_id = 0x126;
    canMessage.can_dlc = 8;
    canMessage.data[0] = (std::byte) actuators.brakePercentage;
    canMessage.data[1] = (std::byte) 0x00;
    canMessage.data[2] = (std::byte) 0x00;
    canMessage.data[3] = (std::byte) 0x00;
    canMessage.trailer =  0x00000000;

    std::unique_lock<std::mutex> locker(loggerMutex);
    Logger::info("Brake : amount = " + std::to_string(actuators.brakePercentage), "send " + timestamp);

    CANStrategy::sendCanMessage<brakeFrame>(canMessage);
    if(actuators.brakePercentage == 0) {
        actuators.brakePercentage = -1;
    };
    locker.unlock();
};

void CANStrategy::forward() {
    CANStrategy::throttle(actuators.throttlePercentage, 1);
};

void CANStrategy::homing() {
    typedef CANStrategy::frame<std::byte[4]> homingFrame;

    homingFrame canMessage;

    canMessage.can_id = 0x6f1;
    canMessage.can_dlc = 8;
    canMessage.data[0] = (std::byte) 0x00;
    canMessage.data[1] = (std::byte) 0x00;
    canMessage.data[2] = (std::byte) 0x00;
    canMessage.data[3] = (std::byte) 0x00;
    canMessage.trailer =  0x00000000;

    std::unique_lock<std::mutex> locker(loggerMutex);
    Logger::info("Homing", "send " + timestamp);

    CANStrategy::sendCanMessage<homingFrame>(canMessage);
    locker.unlock();
};

void CANStrategy::backward() {
    CANStrategy::throttle(actuators.throttlePercentage, 2);
};

void CANStrategy::neutral() {
    CANStrategy::throttle(0, 0);
};

void CANStrategy::stop() {
    // Logger::info("Force stopping", "send " + timestamp);

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

    std::stringstream canMessage;
    canMessage << "0x" << std::hex << frame.can_id << " [" + std::to_string(frame.can_dlc) + "] ";

    for (int i = 0; i < frame.can_dlc; i++) {
        canMessage << std::hex << std::stoi(std::to_string(frame.data[i])) << " ";
    };

    std::unique_lock<std::mutex> locker(loggerMutex);
    Logger::info(canMessage.str(), "receive " + this->timestamp);
    locker.unlock();
};
#endif
