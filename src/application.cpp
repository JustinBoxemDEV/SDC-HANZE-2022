#include "application.h"
#include <iostream>

void Application::RegisterProcess(Process *process){
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
        threads.push_back(new std::thread(&Process::Run, process));
    }

    for(std::thread *thread : threads){
        thread->join();
    }
}