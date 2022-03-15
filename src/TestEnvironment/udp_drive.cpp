#include "udp_drive.h"

/**
 * @brief The (client) socket
 * 
 */
SOCKET UDP_DRIVE::s;

/**
 * @brief function to initialize the socket for sending fake CAN-messages to the socket server.
 * 
 */
void UDP_DRIVE::init() {
    WSADATA wsa;
    struct sockaddr_in server;

    printf("\nInitialising Winsock...\n");
	if (WSAStartup(MAKEWORD(2,2),&wsa) != 0)
	{
		printf("Failed. Error Code : %d",WSAGetLastError());
	}

	printf("Initialised.\n");

    s = socket(AF_INET, SOCK_DGRAM, 0);

        server.sin_addr.s_addr  =   inet_addr("127.0.0.1");
        server.sin_family       =   AF_INET;
        server.sin_port         =   htons(5454);

        if (connect(s, (struct sockaddr *)&server, sizeof(server)) < 0) {
            puts("connect error");
        };

        puts("connected");
};

/**
 * @brief throttle function which takes integer values between 0 and 100. These values represent the speedpercentage.
 * 
 * @param speedPercentage 
 */
void UDP_DRIVE::throttle(int speedPercentage) {
    short arbitration_id = __builtin_bswap16(0x120);
    UDP_DRIVE::send(arbitration_id, speedPercentage); 
};

/**
 * @brief braking function which takes integer values between 0 and 100. These values represent the brakepercentage.
 * 
 * @param brakePercentage 
 */
void UDP_DRIVE::brake(int brakePercentage) {
    short arbitration_id = __builtin_bswap16(0x126);
    UDP_DRIVE::send(arbitration_id, brakePercentage); 
};

/**
 * @brief steering function which takes floating values between -1 and 1
 * 
 * @param steeringAngle The steering angle
 */
void UDP_DRIVE::steer(float steeringAngle) {
    short arbitration_id = __builtin_bswap16(0x12c);
    UDP_DRIVE::send(arbitration_id, steeringAngle);  
};

/**
 * @brief function to shift up one gear
 * 
 */
void UDP_DRIVE::gearShiftUp() {
    short arbitration_id = __builtin_bswap16(0x121);
    int32_t data = 0;
    UDP_DRIVE::send(arbitration_id, data);
};

/**
 * @brief function to shift down one gear
 * 
 */
void UDP_DRIVE::gearShiftDown() {
    short arbitration_id = __builtin_bswap16(0x122);
    int32_t data = 0;
    UDP_DRIVE::send(arbitration_id, data);
};