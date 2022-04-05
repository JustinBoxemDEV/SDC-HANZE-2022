#ifndef PROCESS_H
#define PROCESS_H

class Process
{
    private:
    public:
        virtual void Init() = 0;
        virtual void Run() = 0;
};

#endif