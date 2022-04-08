#ifndef APPLICATION_H
#define APPLICATION_H

#include "process.h"
#include "Processes/cvprocess.h"
#include "Processes/canprocess.h"
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