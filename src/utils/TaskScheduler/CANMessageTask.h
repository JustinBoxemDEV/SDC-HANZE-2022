#ifndef CAN_MESSAGE_TASK_H
#define CAN_MESSAGE_TASK_H

class CANMessageTask {
    private:
        const char* _message;
    public:
        CANMessageTask(const char* canMessage);
        void execute();
};
#endif