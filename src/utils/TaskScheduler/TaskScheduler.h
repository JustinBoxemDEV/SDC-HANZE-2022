#include "MessageTask.h"
#ifndef TASK_SCHEDULER_H
#define TASK_SCHEDULER_H
#define SCH_MAX_TASKS 10
#include <chrono>

class TaskScheduler{
    private:
        std::chrono::high_resolution_clock clock;
        std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
    public:
        struct sTask{
            void (* pTask) (void);
            double Delay;
            double Period;
        };
        // Data structure for storing task data
        void SCH_Init();
        void SCH_Start();

        // Core scheduler functions
        void SCH_Dispatch_Tasks();
        unsigned char SCH_Add_Task(void (*pFunction)(), const float DELAY, const float PERIOD);
        unsigned char SCH_Delete_Task(const unsigned char);
        sTask SCH_tasks_G[SCH_MAX_TASKS] = {0};
};


#endif