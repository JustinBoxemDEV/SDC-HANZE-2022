#ifndef APPLICATION_H
#define APPLICATION_H

#include "process.h"
#include "cvprocess.h"
#include "canprocess.h"
#include <string>
#include <vector>
#include <thread>

class Application
{

    private:
        std::vector<Process*> processes;
        std::vector<std::thread*> threads;
    public:
        void RegisterProcess(Process *process);
        void TerminateProcess(int processID);
        void Run();

};


#endif