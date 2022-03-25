#ifdef linux
#include "VidCapture.h"
CANStrategy canStrategy2;

// Scuffed fix for scheduler
void steer(){
    canStrategy2.steer();
}

void brake(){
    canStrategy2.brake();
}

void forward(){
    canStrategy2.forward();
}

void read() {
    canStrategy2.readCANMessages();
}

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

    MediaCapture mediacapture(&canStrategy2);
    mediacapture.pid.PIDController_Init();

    // canStrategy.taskScheduler.SCH_Add_Task(brake, 0, 0.04);  // zelfs wanneer het bericht de instructie bevat om niet te remmen, zal de motorcontroller tijdelijk worden uitgeschakeld als een soort failsafe
    canStrategy2.taskScheduler.SCH_Add_Task(forward, 0, 0.04);
    canStrategy2.taskScheduler.SCH_Add_Task(steer, 0.02, 0.04);
    // canStrategy.taskScheduler.SCH_Add_Task(read, 0, 0.04);
    canStrategy2.taskScheduler.SCH_Start();
    std::cout << "HELLO" << std::endl;
    while (capture->read(frame)){
        std::cout << "DO WE GET HERE " << std::endl;
        totalFrames++;
        // std::cout << "Frame : " << frame << std::endl;
        mediacapture.ProcessImage(frame);

        canStrategy2.taskScheduler.SCH_Dispatch_Tasks();
        
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
#else
std::cout << "Please use linux or change the strategy in vidcapture.cpp to ACStrategy" << std::endl;
#endif