#include "application.h"
#include <iostream>
#include <thread>

void Application::RegisterProcess(Process *process){
    process->Init(mediaInput);
    processes.push_back(process);
}

void Application::TerminateProcess(int processID){
    if(processID > 0 && processID < processes.size()){
        Process *process = processes.at(processID);
        process->Terminate();
        processes.erase(processes.begin() + processID);
    }
}

void Application::Run(){
    for(Process *process : processes){
        std::thread thread(&Process::Run, process);
    }
}