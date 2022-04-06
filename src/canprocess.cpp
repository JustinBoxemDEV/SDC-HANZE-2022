#include "canprocess.h"
#include <iostream>
#include "VehicleControl/strategies/canstrategy.h"
#include "VehicleControl/strategies/acstrategy.h"

CommunicationStrategy* strategy;

// Scuffed fix for scheduler
void Steer(){
    strategy->steer();
};

void Brake(){
    strategy->brake();
};

void Forward(){
    strategy->forward();
};

void CanProcess::Init(MediaInput *input){
    Process::Init(input);
    switch (input->mediaType)
    {
        case MediaSource::realtime:{
            strategy = new CANStrategy();
            break;
        }
        case MediaSource::assetto: case MediaSource::video:{

            #if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
                strategy = new ACStrategy();
            #endif
            break;
        }
    }
    if(strategy == nullptr){
        std::cout << "Could not set strategy" << std::endl;
        return;
    }
    taskScheduler.SCH_Add_Task(Forward, 0, 0.04);
    taskScheduler.SCH_Add_Task(Steer, 0.02, 0.04);
    taskScheduler.SCH_Start();
}

void CanProcess::Run(){
    while(true){
        taskScheduler.SCH_Dispatch_Tasks();
    }
}

void CanProcess::Terminate(){
    
}