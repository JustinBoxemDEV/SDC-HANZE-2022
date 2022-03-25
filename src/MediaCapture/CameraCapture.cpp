#include "CameraCapture.h"

#ifdef linux
CANStrategy camerastrategy;
#else
// scuffed
ACStrategy camerastrategy;
#endif

void CameraCapture::getCamera(int cameraID){
    capture = new cv::VideoCapture(cameraID);

    // Comment this out when not using the extended camera (CameraID 4)
    // capture->set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // capture->set(cv::CAP_PROP_FRAME_WIDTH, 848);
    
    // Camera detection check
    if (!capture->isOpened()){
        std::cout << "NO CAMERA DETECTED!" << std::endl;
    }else{
        std::cout << "Camera selected: " << cameraID << std::endl;
    }
}

// Scuffed fix for scheduler
void cam_steer(){
    camerastrategy.steer();
}

void cam_brake(){
    camerastrategy.brake();
}

void cam_forward(){
    camerastrategy.forward();
}

void cam_read() {
    #ifdef linux
    camerastrategy.readCANMessages();
    #endif
}

int CameraCapture::run(int cameraID) {
    getCamera(cameraID);

    cv::Mat frame;
    int key = 0;

    // Define total frames and start of a counter for FPS calculation
    int totalFrames = 0;

    time_t start, end;
    time(&start);

    MediaCapture mediacapture(&camerastrategy);
    mediacapture.pid.PIDController_Init();

    // canStrategy.taskScheduler.SCH_Add_Task(cam_brake, 0, 0.04);  // zelfs wanneer het bericht de instructie bevat om niet te remmen, zal de motorcontroller tijdelijk worden uitgeschakeld als een soort failsafe
    camerastrategy.taskScheduler.SCH_Add_Task(cam_forward, 0, 0.04);
    camerastrategy.taskScheduler.SCH_Add_Task(cam_steer, 0.02, 0.04);
    // canStrategy.taskScheduler.SCH_Add_Task(cam_read, 0, 0.04);
    camerastrategy.taskScheduler.SCH_Start();

    while (capture->read(frame)){
        totalFrames++;
        mediacapture.ProcessImage(frame);

        camerastrategy.taskScheduler.SCH_Dispatch_Tasks();
        
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