#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "application.h"
#include "Processes/canprocess.h"
#include "Processes/cvprocess.h"
#include "Processes/readprocess.h"
#include "Processes/terminalprocess.h"

namespace fs = std::filesystem;

void my_handler(int s){
    printf("Caught signal %d\n",s);
    exit(1); 
}

int main(int argc, char** argv) {
    int cursor = 1;
    Process::MediaInput mediaInput;
    Application application;

    // catch for keyboard interupt
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    std::cout << "before while" << std::endl;
    std::string arg;
    while(cursor < argc){
        arg = argv[cursor]; 
        if(arg == "-video"){
            std::cout << "video" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::video;
            cursor++;
            std::string path = fs::current_path().string() + "/assets/videos/" + argv[cursor];
            std::cout << path << std::endl;
            if(!fs::exists(path)){
                std::cout << "file does not exists!" << std::endl;
                return 1;
            }
            mediaInput.filepath = path;
            CVProcess *cvprocess = new CVProcess(&mediaInput);
            application.RegisterProcess(cvprocess);
        }else if(arg == "-realtime"){
            std::cout << "realtime" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::realtime;
        }else if(arg == "-realtimeml"){
            std::cout << "realtime machine learning" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::realtime_ml;
            cursor++;
            std::string path = fs::current_path().string() + "/assets/models/" + argv[cursor];
            std::cout << path << std::endl;
            if(!fs::exists(path)){
                std::cout << "model does not exists!" << std::endl;
                return 1;
            }
            mediaInput.filepath = path;
        }else if(arg == "-imagesml"){
            std::cout << "images machine learning" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::images_ml;
            cursor++;
            std::string dir = argv[cursor];
            if (dir.back() != '/')
                dir = dir+"/";
            std::string path = fs::current_path().string() + "/assets/images/" + dir;
            // when de dir starts with a dot we presume that the user filled in the entire dir path
            if (dir.at(0) == '.')
                path = dir;
            if(!fs::exists(path)){
                std::cout << "file does not exists!" << std::endl;
                return 1;
            }
            mediaInput.filepath = path;
        }else if(arg == "-assetto"){
            std::cout << "assetto" << std::endl;
            std::cout << arg << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::assetto;
        }else if(arg == "-terminal") {
            std::cout << "terminal" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::terminal;
        }
        cursor++;
    }
    CanProcess *canprocess = new CanProcess(&mediaInput);
        if(arg == "-realtime" || arg == "-realtimeml" || arg == "-imagesml" || arg == "") {
            CVProcess *cvprocess = new CVProcess(&mediaInput);
            application.RegisterProcess(cvprocess);
            #ifdef linux
            ReadProcess *readcan = new ReadProcess();
            canprocess->setReadProcess(readcan);
            application.RegisterProcess(readcan);
            #endif
        } else if(arg == "-assetto") {
            CVProcess *cvprocess = new CVProcess(&mediaInput);
            application.RegisterProcess(cvprocess);
        } else if(arg == "-terminal") {
            #ifdef linux
            TerminalProcess *terminal = new TerminalProcess();
            canprocess->setTerminalProcess(terminal);
            application.RegisterProcess(terminal);
            #endif
        }
        application.RegisterProcess(canprocess);

        application.Run();
}
