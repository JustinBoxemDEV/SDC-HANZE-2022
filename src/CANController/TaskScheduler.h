#ifndef TASK_SCHEDULER_H
#define TASK_SCHEDULER_H
#define SCH_MAX_TASKS 10
#include <chrono>

class TaskScheduler
{
    private:
        struct sTask{
            void (* pTask)(void);
            float Delay;
            float Period;
        };
        sTask SCH_tasks_G[SCH_MAX_TASKS];
        std::chrono::time_point<std::chrono::steady_clock> lastTime;
    public:
        // Data structure for storing task data
        void SCH_Init();
        // void SCH_Start();

        // Core scheduler functions
        void SCH_Dispatch_Tasks();
        unsigned char SCH_Add_Task(void (*)(void), const float, const float);
        unsigned char SCH_Delete_Task(const unsigned char);
};


#endif