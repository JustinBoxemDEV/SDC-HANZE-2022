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

    This method sets up a link for a cantype (vcan or can). After this, set up the bind with the cansocket.
    Before the kart starts driving, set the kart to drive (forward), release the brakes and apply homing to the steering wheel.
*/
void CANController::init(std::string canType) {
    // For real can
    if(strcmp(canType.c_str(), "can") == 0) {
        system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 type can bitrate 500000");
        system("echo wijgaanwinnen22 |sudo -S sudo ip link set can0 up");
    // For virtual can
    } else if (strcmp(canType.c_str(), "vcan") == 0) {
        // First delete (existing) vcan
        system("sudo ip link del dev vcan0 type vcan");
        // Add new vcan
        system("sudo ip link add dev vcan0 type vcan");
        system("sudo ip link set vcan0 type vcan");
        system("sudo ip link set vcan0 up");
    }else{
        std::cout << "Could not find inserted CAN type. Please try again"  << std::endl;
    }
    
    if ((CANController::cansocket = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
        perror("Socket");
    };
    
    // Set up the bind
    struct ifreq ifr;
    strcpy(ifr.ifr_name, (canType +"0").c_str());

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

    // Wait 15 seconds after kart is turned on, set the kart to drive (forwards) using message: can0 0x0000000120 50 00 01 00 00 00 00 00
    CANController::throttle(0, 1);
    sleep(0.1);
    // Make sure the brake won't activate while accelerating. Set brakes to 0 using message: can0 0x0000000126 00 00 00 00 00 00 00 00
    CANController::brake(0);
    // Homing message: can0 0x0000006F1 00 00 00 00 00 00 00 00 (correct wheels, can last between 1-20 seconds)
    CANController::steer(0.00);
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
    std::cout << "VROOM VROOM" << std::endl;
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
void CANController::closeCANController(std::string canType) {
    if(strcmp(canType.c_str(), "vcan") == 0) {
        system("sudo ip link del dev vcan0 type vcan");
        std::cout << "Deleted vcan" << std::endl;
    }

    if (close(CANController::cansocket) < 0) {
        perror("Close");
        std::cout << "Disconnected from can succesfully. See you!" << std::endl;
    };
};