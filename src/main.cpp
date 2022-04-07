#include "Managers/mediamanager.h"
#ifdef __WIN32__
#include "MediaCapture/screenCaptureWindows.h"
#else
#include "MediaCapture/screenCaptureLinux.h"
#endif
#include "application.h"
#include "canprocess.h"
#include "cvprocess.h"
#include "VehicleControl/strategies/canstrategy.h"

namespace fs = std::filesystem;
using namespace std;

int screenCaptureCommand(int argc, char** argv);
int cameraCaptureCommand(int argc, char** argv);
int videoCommand(int argc, char** argv);

int main(int argc, char** argv) {
    // if (argv[1] == NULL) {
    //     // return screenCaptureCommand(argc, argv); // AC
    //     return cameraCaptureCommand(argc, argv); // Kart
    //     // return videoCommand(argc, argv); // Tests

    //     // TEST Receive log (steering angle)
    //     // CANStrategy canstrategy;
    //     // while(true){
    //     //     canstrategy.readCANMessages();
    //     // }
    // } 
    Process::MediaInput mediaInput;
    mediaInput.mediaType = CVProcess::MediaSource::video;
    mediaInput.filepath = "E:\\Development\\Stage\\SDC-HANZE-2022\\assets\\videos\\480p.mp4";
    Application application;
    application.RegisterProcess(new CVProcess(&mediaInput));
    application.RegisterProcess(new CanProcess(&mediaInput));
    application.Run();
}

// TEST AC (Virtual environment, AC, ONLY FOR WINDOWS)
int screenCaptureCommand(int argc, char** argv) {
    #ifdef __WIN32__
    MediaManager mediamanager;
    mediamanager.ProcessFeed(true); // screenCapture=true, the rest can be left on default
    return 0;
    #else
    cout << "ERROR: Screen capture is currently not working for linux!" << endl;
    return -1;
    #endif
}

// TEST CAMERA (Physical environment, CANBus, ONLY FOR LINUX)
int cameraCaptureCommand(int argc, char** argv) {
    #ifdef linux
    MediaManager mediamanager;
    mediamanager.ProcessFeed(false, 0); // cameraID=4 for webcam, cameraID=0 for built in laptop cam
    return 0;
    #else
        cout << "ERROR: Camera capture is currently not working for windows!" << endl;
    return -1;
    #endif
}

// TEST VIDEO (WINDOWS AND LINUX)
int videoCommand(int argc, char** argv) {
    MediaManager mediamanager;
    std::string path = fs::current_path().string() + "/../assets/videos/outpy.mp4";
    std::cout << path << std::endl;
    mediamanager.ProcessFeed(false, 0, path);
    return 0;
}

