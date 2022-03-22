#ifdef linux
#include "cansocket.h"

int CANSocket::cansocket;

void CANSocket::create() {
    if ((CANSocket::cansocket = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
        perror("Socket");
    };

    struct ifreq ifr;
    strcpy(ifr.ifr_name, "vcan0");
    ioctl(CANSocket::cansocket, SIOCGIFINDEX, &ifr);
    //  if you use zero as the interface index, you can retrieve packets from all CAN interfaces.

    struct sockaddr_can addr;

    memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if(bind(CANSocket::cansocket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Bind");
    };
};

void CANSocket::closeCANSocket() {
    if (close(CANSocket::cansocket) < 0) {
        perror("Close");
    };
}

void CANSocket::readFrame() {
    int nbytes;
    struct can_frame frame;
    nbytes = read(CANSocket::cansocket, &frame, sizeof(struct can_frame));
    
    if (nbytes < 0) {
        perror("Read");
    };

    printf("0x%03X [%d] ",frame.can_id, frame.can_dlc);
    for (int i = 0; i < frame.can_dlc; i++)
        printf("%02X ",frame.data[i]);
        printf("\r\n");
};

void CANSocket::sendFrame() {
    // Creating a CAN frame (basic structure)
    struct can_frame {
        canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
        __u8    can_dlc; /* frame payload length in byte (0 .. 8) */
        __u8    __pad;   /* padding */
        __u8    __res0;  /* reserved / padding */
        __u8    __res1;  /* reserved / padding */
        float  value; 
        // std::byte   direction;
        // std::byte   trailer1;
        __u_int     trailer;
    };

    struct can_frame frame;

    frame.can_id = 0x00000124;
    frame.can_dlc = 8;
    
    frame.trailer = __builtin_bswap32(0xFFABCDEF);
    // frame.trailer1 = (std::byte) 0x00;
    // frame.direction = (std::byte) 0x00;
    frame.value = -200;

    

    // frame.data = (const unsigned char*)&value;
    // sprintf((char*)frame.data, 0, (const char *)&value);

    //system("cansend vcan0 123#DEADBEEF");

    if (write(CANSocket::cansocket, &frame, sizeof(struct can_frame)) != sizeof(struct can_frame)) {
        perror("Write");
    };
};

#endif