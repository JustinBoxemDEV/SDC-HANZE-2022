#include "udp_drive.h"

std::string ip = "127.0.0.1";
float sending_speed = .040;

SOCKET UDP_DRIVE::s;

void UDP_DRIVE::init() {
    WSADATA wsa;
    struct sockaddr_in server;

    printf("\nInitialising Winsock...");
	if (WSAStartup(MAKEWORD(2,2),&wsa) != 0)
	{
		printf("Failed. Error Code : %d",WSAGetLastError());
	}

	printf("Initialised.\n");

    s = socket(AF_INET, SOCK_DGRAM, 0);

        float steering_angle    =   0.5;
        float throttle          =   0.6;
        float brake             =   0;

        server.sin_addr.s_addr  =   inet_addr("127.0.0.1");
        server.sin_family       =   AF_INET;
        server.sin_port         =   htons(5454);

        if (connect(s, (struct sockaddr *)&server, sizeof(server)) < 0) {
            puts("connect error");
        };

        puts("connected");
};

void UDP_DRIVE::throttle(int speedPercentage) {
    short arbitration_id = __builtin_bswap16(0x120);
    char combined[sizeof arbitration_id + sizeof speedPercentage];

        memcpy(combined, &arbitration_id, sizeof arbitration_id);
        memcpy(combined+sizeof arbitration_id, &speedPercentage, sizeof speedPercentage);

        const char *canMessage = (const char*) combined;

        if(send(s, canMessage, sizeof(canMessage), 0) < 0) {
            puts("send failed");
        };

        puts("Data send");
};

void UDP_DRIVE::brake(int brakePercentage) {
    short arbitration_id = __builtin_bswap16(0x126);
    char combined[sizeof arbitration_id + sizeof brakePercentage];

    memcpy(combined, &arbitration_id, sizeof arbitration_id);
    memcpy(combined+sizeof arbitration_id, &brakePercentage, sizeof brakePercentage);

    const char *canMessage = (const char*) combined;

    if(send(s, canMessage, sizeof(canMessage), 0) < 0) {
        puts("send failed");
    };

    puts("Data send");
};

void UDP_DRIVE::steer(float steeringAngle) {
    short arbitration_id = __builtin_bswap16(0x12c);
    char combined[sizeof arbitration_id + sizeof steeringAngle];

    memcpy(combined, &arbitration_id, sizeof arbitration_id);
    memcpy(combined+sizeof arbitration_id, &steeringAngle, sizeof steeringAngle);

    const char *canMessage = (const char*) combined;

    if(send(s, canMessage, sizeof(canMessage), 0) < 0) {
        puts("send failed");
    };

    puts("Data send");
};

void UDP_DRIVE::gearShiftUp() {
    short arbitration_id = __builtin_bswap16(0x121);
    int32_t data = 0;
    char combined[sizeof arbitration_id + sizeof data];

        memcpy(combined, &arbitration_id, sizeof arbitration_id);
        memcpy(combined+sizeof arbitration_id, &data, sizeof data);

        const char *canMessage = (const char*) combined;

        if(send(s, canMessage, sizeof(canMessage), 0) < 0) {
            puts("send failed");
        };

        puts("Data send");
};

void UDP_DRIVE::gearShiftDown() {
    short arbitration_id = __builtin_bswap16(0x122);
    int32_t data = 0;
    char combined[sizeof arbitration_id + sizeof data];

        memcpy(combined, &arbitration_id, sizeof arbitration_id);
        memcpy(combined+sizeof arbitration_id, &data, sizeof data);

        const char *canMessage = (const char*) combined;

        if(send(s, canMessage, sizeof(canMessage), 0) < 0) {
            puts("send failed");
        };

        puts("Data send");
};


void UDP_DRIVE::drive() {
        short arbitration_id = __builtin_bswap16(0x120);
        int32_t data = 0;
        //float i = -1.0;

        char combined[sizeof arbitration_id + sizeof data];

        memcpy(combined, &arbitration_id, sizeof arbitration_id);
        memcpy(combined+sizeof arbitration_id, &data, sizeof data);

        const char *canMessage = (const char*) combined;

        if(send(s, canMessage, sizeof(canMessage), 0) < 0) {
            puts("send failed");
        };

        puts("Data send");
};