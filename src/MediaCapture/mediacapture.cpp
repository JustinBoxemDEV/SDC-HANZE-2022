#include "mediacapture.h"
#include "../Managers/mediamanager.h"

CommunicationStrategy *communicationStrategy;

// Scuffed fix for scheduler
void steer(){
    communicationStrategy->steer();
};

void brake(){
    communicationStrategy->brake();
};

void forward(){
    communicationStrategy->forward();
};

void read() {
    #ifdef linux
    // strategy->readCANMessages();
    #endif
};

int MediaCapture::run(CommunicationStrategy *strategy) {
    communicationStrategy = strategy;

    cv::Mat frame;
    int key = 0;

    // Define total frames and start of a counter for FPS calculation
    int totalFrames = 0;

    time_t start, end;
    time(&start);

    MediaManager mediamanager(strategy);
    mediamanager.pid.PIDController_Init();

    // communicationStrategy->taskScheduler.SCH_Add_Task(brake, 0, 0.04);  // zelfs wanneer het bericht de instructie bevat om niet te remmen, zal de motorcontroller tijdelijk worden uitgeschakeld als een soort failsafe
    communicationStrategy->taskScheduler.SCH_Add_Task(brake, 0, 0.04);
    communicationStrategy->taskScheduler.SCH_Add_Task(steer, 0.02, 0.04);
    // communicationStrategy->taskScheduler.SCH_Add_Task(read, 0, 0.04);
    communicationStrategy->taskScheduler.SCH_Start();

    while (capture->read(frame)){
        totalFrames++;
        mediamanager.ProcessImage(frame);

        communicationStrategy->taskScheduler.SCH_Dispatch_Tasks();
        
        if (cv::waitKey(100/60)>0){
            break;
        };
    };

    // End the time counter
    time(&end);

    // Time elapsed
    double seconds = difftime(end, start);
    std::cout << "Time taken : " << seconds << " seconds" << std::endl;

    // Estimate the FPS based on frames / elapsed time in seconds
    int fps = totalFrames / seconds;
    std::cout << "Estimated frames per second : " << fps << std::endl;

	return 0;
};