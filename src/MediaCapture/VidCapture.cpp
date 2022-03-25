#include "VidCapture.h"
#ifdef linux
CANStrategy vidstrategy;
#else
ACStrategy vidstrategy;
#endif

// Scuffed fix for scheduler
void vid_steer(){
    vidstrategy.steer();
}

void vid_brake(){
    vidstrategy.brake();
}

void vid_forward(){
    vidstrategy.forward();
}

#ifdef linux
void vid_read() {
    vidstrategy.readCANMessages();
}
#endif

int VidCapture::run(std::string filename) {
    capture = new cv::VideoCapture(filename);

    if (!capture->isOpened()){
        std::cout << "NO File detected!" << std::endl;
    }else{
        std::cout << "File selected: " << std::endl;
    }

    cv::Mat frame;

    // Define total frames and start of a counter for FPS calculation
    int totalFrames = 0;

    time_t start, end;
    time(&start);

    MediaCapture mediacapture(&vidstrategy);
    mediacapture.pid.PIDController_Init();

    // canStrategy.taskScheduler.SCH_Add_Task(brake, 0, 0.04);  // zelfs wanneer het bericht de instructie bevat om niet te remmen, zal de motorcontroller tijdelijk worden uitgeschakeld als een soort failsafe
    vidstrategy.taskScheduler.SCH_Add_Task(vid_forward, 0, 0.04);
    vidstrategy.taskScheduler.SCH_Add_Task(vid_steer, 0.02, 0.04);

    #ifdef linux
    // canStrategy.taskScheduler.SCH_Add_Task(vid_read, 0, 0.04);
    #endif

    vidstrategy.taskScheduler.SCH_Start();
    std::cout << "HELLO" << std::endl;
    while (capture->read(frame)){
        std::cout << "DO WE GET HERE " << std::endl;
        totalFrames++;
        // std::cout << "Frame : " << frame << std::endl;
        mediacapture.ProcessImage(frame);

        vidstrategy.taskScheduler.SCH_Dispatch_Tasks();
        
        if (cv::waitKey(100/60)>0){
            break;
        }
    }

    // End the time counter
    time(&end);

    // Time elapsed
    double seconds = difftime(end, start);
    std::cout << "Time taken : " << seconds << " seconds" << std::endl;

    // Estimate the FPS based on frames / elapsed time in seconds
    int fps = totalFrames / seconds;
    std::cout << "Estimated frames per second : " << fps << std::endl;

	return 0;
}