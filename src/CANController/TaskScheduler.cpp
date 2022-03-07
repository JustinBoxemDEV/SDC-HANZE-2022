#include "TaskScheduler.h"
#include <iostream>

TaskScheduler::sTask SCH_tasks_G[SCH_MAX_TASKS];

void TaskScheduler::SCH_Dispatch_Tasks()
{
   unsigned char Index = 0;

   auto currentTime = std::chrono::steady_clock::now();

   float deletatime = std::chrono::duration_cast<std::chrono::duration<float,std::milli>>(lastTime - currentTime).count();

   // Dispatches (runs) the next task (if one is ready)
   for(Index = 0; Index < SCH_MAX_TASKS; Index++)
   {
      // std::cout << "for loop" << std::endl;
      if((SCH_tasks_G[Index].Delay <= 0)) 
      {
         std::cout << SCH_tasks_G[Index].Period << std::endl;
         if((SCH_tasks_G[Index].pTask != 0)){
            std::cout << "ifje function" << std::endl;
            (*SCH_tasks_G[Index].pTask)();  // Run the task

            if(SCH_tasks_G[Index].Period)
            {
               // Schedule periodic tasks to run again
               SCH_tasks_G[Index].Delay = SCH_tasks_G[Index].Period;
               SCH_tasks_G[Index].Delay -= deletatime; 
            }

            // Periodic tasks will automatically run again
            // - if this is a 'one shot' task, remove it from the array
            if(SCH_tasks_G[Index].Period == 0)
            {
               SCH_Delete_Task(Index);
            }
         }
      }
      else
      {
         // Not yet ready to run: just decrement the delay
         SCH_tasks_G[Index].Delay -= deletatime; 
      }
   }

   //Set last frame naar current frame
   lastTime = currentTime;
}

unsigned char TaskScheduler::SCH_Add_Task(void (*pFunction)(void), const float DELAY, const float PERIOD)
{
   
   unsigned char Index = 0;

   // First find a gap in the array (if there is one)
   while((SCH_tasks_G[Index].pTask != 0) && (Index < SCH_MAX_TASKS))
   {
      Index++;
   }

   // Have we reached the end of the list?   
   if(Index == SCH_MAX_TASKS)
   {
      // Task list is full, return an error code
      return SCH_MAX_TASKS;  
   }

   // IF there is space in the task array
   std::cout << "There is space" << std::endl;
   SCH_tasks_G[Index].pTask = pFunction;
   SCH_tasks_G[Index].Delay = DELAY;
   SCH_tasks_G[Index].Period = PERIOD;
   // std::cout << SCH_tasks_G[Index].pTask << std::endl;
   // return position of task (to allow later deletion)
   return Index;
}

unsigned char TaskScheduler::SCH_Delete_Task(const unsigned char TASK_INDEX)
{
   // Return_code can be used for error reporting, NOT USED HERE THOUGH!
   unsigned char Return_code = 0;

   SCH_tasks_G[TASK_INDEX].pTask = 0;
   SCH_tasks_G[TASK_INDEX].Delay = 0;
   SCH_tasks_G[TASK_INDEX].Period = 0;

   return Return_code;
}


void TaskScheduler::SCH_Init()
{
   unsigned char i;

   for(i = 0; i < SCH_MAX_TASKS; i++)
   {
      SCH_Delete_Task(i);
   }

   // take timestamp of start (first lasttime)
   lastTime = std::chrono::steady_clock::now();
}
