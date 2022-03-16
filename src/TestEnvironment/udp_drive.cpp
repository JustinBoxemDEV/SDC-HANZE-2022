// #include "udp_drive.h"

// /**
//  * @brief The (client) socket
//  * 
//  */
// SOCKET UDP_DRIVE::s;

// /**
//  * @brief function to initialize the socket for sending fake CAN-messages to the socket server.
//  * 
//  */
// void UDP_DRIVE::init() {
//     WSADATA wsa;
//     struct sockaddr_in server;

//     printf("\nInitialising Winsock...\n");
// 	if (WSAStartup(MAKEWORD(2,2),&wsa) != 0)
// 	{
// 		printf("Failed. Error Code : %d",WSAGetLastError());
// 	};

// 	printf("Initialised.\n");

//     s = socket(AF_INET, SOCK_DGRAM, 0);

//     server.sin_addr.s_addr  =   inet_addr("127.0.0.1");
//     server.sin_family       =   AF_INET;
//     server.sin_port         =   htons(5454);

//     if (connect(s, (struct sockaddr *)&server, sizeof(server)) < 0) {
//         puts("connect error");
//     };

//     puts("connected");
// };

// /**
//  * @brief throttle function which takes integer values between 0 and 100. These values represent the speedpercentage.
//  * 
//  * @param speedPercentage 
//  */
// void UDP_DRIVE::throttle(int speedPercentage, int direction) {
//     UDP_DRIVE::send(UDP_DRIVE::merge(__builtin_bswap16(0x120), speedPercentage));
// };

// /**
//  * @brief braking function which takes integer values between 0 and 100. These values represent the brakepercentage.
//  * 
//  * @param brakePercentage 
//  */
// void UDP_DRIVE::brake(int brakePercentage) {
//     UDP_DRIVE::send(UDP_DRIVE::merge(__builtin_bswap16(0x126), brakePercentage));
// };

// /**
//  * @brief steering function which takes floating values between -1 and 1
//  * 
//  * @param amount The amount to be steered as float between -1.0 and 1.0.
//  */
// void UDP_DRIVE::steer(float amount) {
//     UDP_DRIVE::send(UDP_DRIVE::merge(__builtin_bswap16(0x12c), amount));
// };

// /**
//  * @brief function to shift up one gear
//  * 
//  */
// void UDP_DRIVE::gearShiftUp() {
//     UDP_DRIVE::send(UDP_DRIVE::merge(__builtin_bswap16(0x121), 0));
// };

// /**
//  * @brief function to shift down one gear
//  * 
//  */
// void UDP_DRIVE::gearShiftDown() {
//     UDP_DRIVE::send(UDP_DRIVE::merge(__builtin_bswap16(0x122), 0));
// };