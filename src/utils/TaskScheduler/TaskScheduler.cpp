#include "TaskScheduler.h"
#include <iostream>

void TaskScheduler::SCH_Dispatch_Tasks(){
   unsigned char Index = 0;

   auto currentTime = clock.now();

   double deltatime = std::chrono::duration_cast<std::chrono::duration<double>>(currentTime - lastTime).count();

   //Set last frame naar current frame
   lastTime = currentTime;

   // Dispatches (runs) the next task (if one is ready)
   std::cout << "Dispatching tasks " << std::endl;
   for(Index = 0; Index < SCH_MAX_TASKS; Index++){
      std::cout << "Running task " << SCH_tasks_G[Index].Task << std::endl;
      if((SCH_tasks_G[Index].Task != 0)){
         if((SCH_tasks_G[Index].Delay < 0)) 
         {
            std::cout << "Running task " << SCH_tasks_G[Index].Task << std::endl;
            SCH_tasks_G[Index].Task->execute();  // Run the task
         
            if(SCH_tasks_G[Index].Period != 0)
            {
               // Schedule periodic tasks to run again
               SCH_tasks_G[Index].Delay = SCH_tasks_G[Index].Period;
               SCH_tasks_G[Index].Delay -= deltatime; 

            }else{
               SCH_Delete_Task(Index);
            }
         }
         else
         {
            // Not yet ready to run: just decrement the delay
            SCH_tasks_G[Index].Delay -= deltatime; 
         }
      }
   }
}

unsigned char TaskScheduler::SCH_Add_Task(CANMessageTask* task, const float DELAY, const float PERIOD){
   
   unsigned char Index = 0;

   // First find a gap in the array (if there is one)
   while((SCH_tasks_G[Index].Task != 0) && (Index < SCH_MAX_TASKS))
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
   SCH_tasks_G[Index].Task = task;
   SCH_tasks_G[Index].Delay = DELAY;
   SCH_tasks_G[Index].Period = PERIOD;
   // return position of task (to allow later deletion)

   std::cout << "Added task " << task << std::endl;
   return Index;
}

unsigned char TaskScheduler::SCH_Delete_Task(const unsigned char TASK_INDEX){
   unsigned char Return_code = 0;

   SCH_tasks_G[TASK_INDEX].Task = 0;
   SCH_tasks_G[TASK_INDEX].Delay = 0;
   SCH_tasks_G[TASK_INDEX].Period = 0;

   return Return_code;
}


void TaskScheduler::SCH_Init(){
   unsigned char i;

   for(i = 0; i < SCH_MAX_TASKS; i++)
   {
      SCH_Delete_Task(i);
   }
}

void TaskScheduler::SCH_Start(){
   // take timestamp of start (first lasttime)
   lastTime = clock.now();
}
