#include "cancontroller.h"

int CANController::cansocket;

void CANController::setup() {
     //system("sudo ip link set can0 type can bitrate 500000");
    system("sudo ip link add dev vcan0 type vcan");
     //system("sudo ip link set can0 up");
    system("sudo ifconfig vcan0 up");
    
    // Creating a CAN frame (basic structure)
    struct can_frame {
        canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
        __u8        can_dlc; /* frame payload length in byte (0 .. 8) */
        __u8        __pad;   /* padding */
        __u8        __res0;  /* reserved / padding */
        __u8        __res1;  /* reserved / padding */
        std::byte       speed; // speed 0-100
        std::byte       space;
        std::byte       direction; // 0 = neutral, 1 = gas, 2 = reverse
        std::byte       trailer1;
        __u_short       trailer2;
        __u_short       trailer3;
    };

    // Homing (correct wheel)
    // Message: can0 0x0000006F1 00 00 00 00 00 00 00 00
    // can last between 1-20 seconds

    // wait 15 seconds after kart is turned on
    // Send can0 0x0000006120 50 00 01 00 00 00 00 00

    // Make sure the brake wont activate while accelerating
    // can0 0x0000006126 00 00 00 00 00 00 00 00

};

// Speed int between 0-100? (first byte)
// Third byte indicates direction: 0 = neutral, 1 = gas, 2 = reverse
// Last 5 bytes are most likely unused
// Send in intervals of 40ms
// Message example: Arb ID: 0x00000120	Data: 50 00 01 00 00 00 00 00 
void CANController::throttle(short speed, std::byte direction) {
    // Creating a CAN frame (basic structure)
    struct can_frame {
        canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
        __u8        can_dlc; /* frame payload length in byte (0 .. 8) */
        __u8        __pad;   /* padding */
        __u8        __res0;  /* reserved / padding */
        __u8        __res1;  /* reserved / padding */
        std::byte       speed; // speed 0-100
        std::byte       space;
        std::byte       direction; // 0 = neutral, 1 = gas, 2 = reverse
        std::byte       trailer1;
        std::byte       trailer2;
        std::byte       trailer3;
        __u_short       trailer4;
    };
    
    struct can_frame frame;
    
    frame.can_id = 0x125;
    frame.can_dlc = 8;

    frame.speed     =   (std::byte) speed;
    frame.space     =   (std::byte) 0x00;
    frame.direction =   (std::byte) direction;
    frame.trailer1  =   (std::byte) 0x00;
    frame.trailer2  =   (std::byte) 0x00;
    frame.trailer3  =   (std::byte) 0x00;
    frame.trailer4  =   __builtin_bswap16(0x0000);

    if (write(CANController::cansocket, &frame, sizeof(struct can_frame)) != sizeof(struct can_frame)) {
        perror("Write");
    };
};

// Brake percentage int between 0-100%
// Only allowed when braking -> engine will shutdown (to prevent braking when accelerating)
// Last 5 bytes are most likely unused
// Message example: Arb ID: 0x00000126	Data: 50 00 00 00 00 00 00 00 
void CANController::brake() {

};

// Steering angle float between -1.0 and 1.0 (first 4 bytes)
// Send in intervals of 40ms
// Last 5 bytes are most likely unused
// Message example: Arb ID: 0x000006F1	Data: 00 00 00 00 00 00 00 00 
void CANController::steer() {
    // Creating a CAN frame (basic structure)
    struct can_frame {
        canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
        __u8        can_dlc; /* frame payload length in byte (0 .. 8) */
        __u8        __pad;   /* padding */
        __u8        __res0;  /* reserved / padding */
        __u8        __res1;  /* reserved / padding */
        float           steering;
        std::byte       trailer2;
        std::byte       trailer3;
        __u_short       trailer4;
    };

    struct can_frame frame;
    
    frame.can_id = 0x125;
    frame.can_dlc = 8;

    frame.steering = __builtin_bswap32((float)-3.63693e+31f);
    frame.trailer2 = (std::byte) 0x00;
    frame.trailer3 = (std::byte) 0x00;
    frame.trailer4 = 0x0000;
};

void CANController::create() {
    if ((CANController::cansocket = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
        perror("Socket");
    };

    struct ifreq ifr;
    strcpy(ifr.ifr_name, "vcan0");
    ioctl(CANController::cansocket, SIOCGIFINDEX, &ifr);
    //  if you use zero as the interface index, you can retrieve packets from all CAN interfaces.

    struct sockaddr_can addr;

    memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if(bind(CANController::cansocket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Bind");
    };
};

void CANController::closeCANController() {
    if (close(CANController::cansocket) < 0) {
        perror("Close");
    };
};

void CANController::readFrame() {
    int nbytes;
    struct can_frame frame;
    nbytes = read(CANController::cansocket, &frame, sizeof(struct can_frame));
    
    if (nbytes < 0) {
        perror("Read");
    };

    printf("0x%03X [%d] ",frame.can_id, frame.can_dlc);
    for (int i = 0; i < frame.can_dlc; i++)
        printf("%02X ",frame.data[i]);
        printf("\r\n");
};

void CANController::sendFrame() {
    // Creating a CAN frame (basic structure)
    struct can_frame {
        canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
        __u8    can_dlc; /* frame payload length in byte (0 .. 8) */
        __u8    __pad;   /* padding */
        __u8    __res0;  /* reserved / padding */
        __u8    __res1;  /* reserved / padding */
        float  value; 
        std::byte   direction;
        std::byte   trailer1;
        __u_int     trailer;
    };

    struct can_frame frame;

    frame.can_id = __builtin_bswap32(0x00000125);
    frame.can_dlc = 8;

    //frame.trailer = __builtin_bswap32(0xFFABCDEF);
    //frame.trailer1 = (std::byte) 0x00;
    frame.direction = (std::byte) 0x00;
    frame.value = -200;

    // frame.data = (const unsigned char*)&value;
    // sprintf((char*)frame.data, 0, (const char *)&value);
    std::cout << "HELLO!" << std::endl;
    if (write(CANController::cansocket, &frame, sizeof(struct can_frame)) != sizeof(struct can_frame)) {
        perror("Write");
    };
    std::cout << "BYE" << std::endl;
};