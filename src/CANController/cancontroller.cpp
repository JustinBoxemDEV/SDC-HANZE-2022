#include "cancontroller.h"

int CANController::cansocket;

struct defaultFrame {
    canid_t     can_id;
    __u8        can_dlc;
    __u8        __pad; 
    __u8        __res0; 
    __u8        __res1;
    std::byte   amount;
    std::byte   space;
    std::byte   direction;
    std::byte   trailer1;
    __u_int     trailer2;
}; 

struct steeringFrame {
    canid_t     can_id;
    __u8        can_dlc;
    __u8        __pad;
    __u8        __res0;
    __u8        __res1;
    float       steering;
    __u_int     trailer;
};

/**
    Initialize the bus.
    @param canName The name of the can
    @param canType The type of can that is used: vcan or can

    This method sets up a link for canName with canType. After this, set up the bind with the cansocket.
    Before the kart starts driving, apply homing to the steering wheel, set the kart to drive (forward) and release the brakes.
*/
void CANController::init(std::string canName, std::string canType) {
    // First delete (existing) CAN
    system(("sudo ip link del dev "+canName+" type "+canType).c_str());
    
    // Add CAN
    system(("sudo ip link add dev "+canName+" type "+canType+" bitrate 50000").c_str());
    //system(("sudo ip link set "+canName+" type "+canType+" bitrate 500000").c_str());
    
    // Setup the CAN network
    system(("sudo ifconfig "+canName+" up").c_str());
    
    if ((CANController::cansocket = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
        perror("Socket");
    };
    
    // Set up the bind
    struct ifreq ifr;
    strcpy(ifr.ifr_name, canName.c_str());

    //  if you use zero as the interface index, you can retrieve packets from all CAN interfaces.
    ioctl(CANController::cansocket, SIOCGIFINDEX, &ifr);
    
    struct sockaddr_can addr;

    memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if(bind(CANController::cansocket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Bind");
    };
    
    // When starting the kart
    // Homing message: can0 0x0000006F1 00 00 00 00 00 00 00 00 (correct wheels, can last between 1-20 seconds)
    CANController::steer(0.00);

    // Wait 15 seconds after kart is turned on, set the kart to drive (forwards) using message: can0 0x0000000120 50 00 01 00 00 00 00 00
    sleep(15);
    CANController::throttle(5, 1);
    // Make sure the brake won't activate while accelerating. Set brakes to 0 using message: can0 0x0000000126 00 00 00 00 00 00 00 00
    CANController::brake(0);
};

/**
    Send a throttle message to the canbus. Throttle corresponds to id 0x125.
    @param speed The speed as integer between 0 and 100.
    @param direction The driving direction as integer: 0 = neutral, 1 = gas, 2 = reverse.

    Additional info:
    Message example: Arb ID: 0x00000120	Data: 50 00 01 00 00 00 00 00
    Send in intervals of 40ms
*/
void CANController::throttle(int speed, int direction) {
    struct defaultFrame frame;
    
    frame.can_id = 0x000125;
    frame.can_dlc = 8;

    frame.amount    =   (std::byte) speed;
    frame.space     =   (std::byte) 0x00;
    frame.direction =   (std::byte) direction;
    frame.trailer1  =   (std::byte) 0x00;
    frame.trailer2  =   0x00000000;

    if (write(CANController::cansocket, &frame, sizeof(struct defaultFrame)) != sizeof(struct defaultFrame)) {
        perror("Write");
    };
};

/**
    Send a brake message to the canbus. Throttle corresponds to id 0x126.
    @param brakePercentage The brake percentage as an int between 0-100.

    Additional info:
    Only allowed when braking -> engine will shutdown (to prevent braking when accelerating)
    Only send in intervals of 40m
    Message example: Arb ID: 0x00000126	Data: 50 00 00 00 00 00 00 00
    
*/
void CANController::brake(int brakePercentage) {
    struct defaultFrame frame;
    
    frame.can_id = 0x000126;
    frame.can_dlc = 8;

    frame.amount    =   (std::byte) brakePercentage;
    frame.space     =   (std::byte) 0x00;
    frame.direction =   (std::byte) 0x00;
    frame.trailer1  =   (std::byte) 0x00;
    frame.trailer2  =   0x00000000;

    if (write(CANController::cansocket, &frame, sizeof(struct defaultFrame)) != sizeof(struct defaultFrame)) {
        perror("Write");
    };
};

/**
    Send a steer message to the canbus. Throttle corresponds to id 0x6F1.
    @param amount The amount to be steered as float between -1.0 and 1.0. 

    Additional info:
    Only send in intervals of 40ms
    An amount parameter of -1.0 represents steering all the way left, 1.0 represents steering all the way right, and 0 will centre the steering wheel.
    Message example: Arb ID: 0x000006F1	Data: 00 00 00 00 00 00 00 00
*/
void CANController::steer(float amount) {
    struct steeringFrame frame;
    
    frame.can_id    = 0x6F1;
    frame.can_dlc   = 8;

    frame.steering  = amount;
    frame.trailer   = 0x00000000;

    if (write(CANController::cansocket, &frame, sizeof(struct steeringFrame)) != sizeof(struct steeringFrame)) {
        perror("Write");
    };
};

/**
    Close the existing CANBus socket.
*/
void CANController::closeCANController(std::string canName, std::string canType) {
    system(("sudo ip link del dev "+canName+" type "+canType).c_str());

    if (close(CANController::cansocket) < 0) {
        perror("Close");
    };
};