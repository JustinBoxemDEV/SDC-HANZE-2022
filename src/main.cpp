#include "Managers/mediamanager.h"
#ifdef __WIN32__
#include "MediaCapture/screenCaptureWindows.h"
#else
#include "MediaCapture/screenCaptureLinux.h"
#endif
 
namespace fs = std::filesystem;
using namespace std;

int screenCaptureCommand(int argc, char** argv);
int cameraCaptureCommand(int argc, char** argv);
int videoCommand(int argc, char** argv);

int main(int argc, char** argv) {
    if (argv[1] == NULL) {
        //return screenCaptureCommand(argc, argv); // AC
        //return cameraCaptureCommand(argc, argv); // Kart
        //return videoCommand(argc, argv); // Tests

        // TEST Receive log (steering angle)
        // CANStrategy canstrategy;
        // while(true){
        //     canstrategy.readCANMessages();
        // }
    } 

    //testspul PID//
            PIDController pid{0.2,0.0,0.1456}; //0.2,0.2,0.1456
            pid.PIDController_Init();
            double testOffset = 500.0;
            double targetOffset = 500.0;
            int self =1;
            int I ;
            int x = 0;
            int prefX = x;
            double prefY = 500.0;
            cv::Mat drawing;
            drawing = cv::Mat::zeros(cv::Size(2048, 2048), CV_8UC1);
            
            double rech = 600.0;
            double link = 0.0;
            double oldrech = 600.0;
            double oldlink = 0.0;

            double lastDistanceTravelled = 0;
            double distanceTravelled =0;
            double totalhoek =0.0;
            double pref = 0;
            while ( x < 2000) {
                cv::line(drawing, cv::Point(prefX, oldlink+724), cv::Point(x, link+724), cv::Scalar(200), 5, 5, 0);
                cv::line(drawing, cv::Point(prefX, oldrech+724), cv::Point(x, rech+724), cv::Scalar(200), 5, 5, 0);
                cv::line(drawing, cv::Point(prefX, (oldlink + oldrech)/2 +724), cv::Point(x, (oldlink + oldrech)/2 +724), cv::Scalar(200), 5, 5, 0);
                cv::line(drawing, cv::Point(prefX, prefY+724), cv::Point(x, testOffset+724), cv::Scalar(255), 5, 5, 0);
                

                imshow("image", drawing);
                cv::waitKey(1);
                double normalTestOffset = 2.0* ((testOffset-link)/(rech-link))-1.0;
                double pidout = pid.PIDController_update(-normalTestOffset);
                pref = pidout;
                totalhoek = totalhoek+pidout;
                prefX = x;
                double xDiff = pid.calculateTest(totalhoek);
                distanceTravelled = distanceTravelled + abs(xDiff);
                prefY = testOffset;
                testOffset = testOffset + xDiff;
                //cv::line(drawing, cv::Point(lastDistanceTravelled, prefY+212), cv::Point(distanceTravelled, testOffset+212), cv::Scalar(200), 2, 2, 0);
                //cv::line(drawing, cv::Point(prefX, pref*100+212), cv::Point(x, pidout*100+212), cv::Scalar(200), 2, 2, 0);
                
                
                oldlink = link;
                oldrech = rech;
                lastDistanceTravelled = distanceTravelled;


                std::cout << "op " << x << std::endl;
                std::cout <<"Distance " << distanceTravelled << std::endl;
                std::cout <<"offset " << testOffset << std::endl;
                std::cout <<"normal " << normalTestOffset << std::endl;
                std::cout <<"pidout " << pidout << std::endl;
                std::cout <<"hoek van pid " << pidout *100.0/4.2 << std::endl;
                std::cout <<"aftand uit pid " << pid.calculateTest(pidout) << std::endl;
                std::cout <<"totale hoek " << totalhoek << std::endl;
                std::cout << "  " << std::endl;
                x++;
                if(testOffset <= targetOffset && x>10 && self == 1){
                    I = x;
                    self=0;
                }
                if (x >= 250 && x <= 700){
                    link = link +1;
                    rech = rech +1;
                }
                if (x >= 720){
                    link = link -1;
                    rech = rech -1;
                }
                if (x >= 1500){
                    link = link +4;
                    rech = rech +4;
                }
            }
            std::cout <<"ult I = "<< I << std::endl;
            cv::Mat scaledown;
            cv::resize(drawing, scaledown, cv::Size(1000,500));
            cv::imshow ("scale",scaledown);
            cv::waitKey();
    // eind test pid//
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
    std::string path = fs::current_path().string() + "/../assets/videos/highway.mp4";
    std::cout << path << std::endl;
    mediamanager.ProcessFeed(false, 0, path);
    return 0;
}

