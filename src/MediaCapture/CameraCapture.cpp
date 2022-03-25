#ifdef linux
#include "CameraCapture.h"

CANStrategy canStrategy;

void CameraCapture::getCamera(int cameraID){
    capture = new cv::VideoCapture(cameraID);
    capture->set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture->set(cv::CAP_PROP_FRAME_WIDTH, 848);
    
    // Camera detection check
    if (!capture->isOpened()){
        std::cout << "NO CAMERA DETECTED!" << std::endl;
    }else{
        std::cout << "Camera selected: " << cameraID << std::endl;
    }
}

// Scuffed fix for scheduler
void steer(){
    canStrategy.steer();
}

void brake(){
    canStrategy.brake();
}

void forward(){
    canStrategy.forward();
}

void read() {
    canStrategy.readCANMessages();
}

int CameraCapture::run(int cameraID) {
    getCamera(cameraID);

    cv::Mat frame;
    int key = 0;

    // Define total frames and start of a counter for FPS calculation
    int totalFrames = 0;

    time_t start, end;
    time(&start);

    MediaCapture mediacapture(&canStrategy);
    mediacapture.pid.PIDController_Init();

    // canStrategy.taskScheduler.SCH_Add_Task(brake, 0, 0.04);  // zelfs wanneer het bericht de instructie bevat om niet te remmen, zal de motorcontroller tijdelijk worden uitgeschakeld als een soort failsafe
    canStrategy.taskScheduler.SCH_Add_Task(forward, 0, 0.04);
    canStrategy.taskScheduler.SCH_Add_Task(steer, 0.02, 0.04);
    // canStrategy.taskScheduler.SCH_Add_Task(read, 0, 0.04);
    canStrategy.taskScheduler.SCH_Start();

    while (capture->read(frame)){
        totalFrames++;
        mediacapture.ProcessImage(frame);

        canStrategy.taskScheduler.SCH_Dispatch_Tasks();
        
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

#endif