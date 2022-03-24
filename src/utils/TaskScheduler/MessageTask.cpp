// #include "MessageTask.h"
// #include <winsock2.h>
// #include <iostream>

// // For virtual environment
// MessageTask::MessageTask(const char* message, SOCKET* socket){
//     _message = message;
//     _socket = socket;
// }

// void MessageTask::execute(){
//     if(send(*_socket, _message, sizeof(_message), 0) < 0) {
//         puts("send failed");
//     };
//     puts("Data sent");

//     std::cout << "Executed message task " << std::endl;
// }