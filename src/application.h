#ifndef APPLICATION_H
#define APPLICATION_H

#include "process.h"
#include "cvprocess.h"
#include "canprocess.h"
#include <string>
#include <vector>

class Application
{

    private:
        std::vector<Process*> processes;
        Process::MediaInput* mediaInput;
    public:
        Application(Process::MediaInput* mediaInput):mediaInput(mediaInput){};
        void RegisterProcess(Process *process);
        void TerminateProcess(int processID);
        void Run();

};


#endif