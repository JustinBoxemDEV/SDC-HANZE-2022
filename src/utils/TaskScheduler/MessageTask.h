#include <winsock2.h>
#ifndef MESSAGE_TASK_H
#define MESSAGE_TASK_H

class MessageTask {
    private:
        const char* _message;
        SOCKET *_socket;

    public:
        MessageTask(const char* message, SOCKET* socket);
        void execute();
};
#endif