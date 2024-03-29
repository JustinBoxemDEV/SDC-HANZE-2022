#ifndef APPLICATION_H
#define APPLICATION_H

#include "Processes/process.h"
#include "Processes/cvprocess.h"
#include "Processes/canprocess.h"
#include "Processes/terminalprocess.h"
#include "Processes/readprocess.h"
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