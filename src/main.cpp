#include "application.h"
#include "Processes/canprocess.h"
#include "Processes/cvprocess.h"
#include "Processes/readprocess.h"
#include "Processes/terminalprocess.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    int cursor = 1;
    Process::MediaInput mediaInput;
    Application application;

    std::cout << "before while" << std::endl;
    std::string arg;
    while(cursor < argc){
        arg = argv[cursor]; 
        if(arg == "-video"){
            std::cout << "video" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::video;
            cursor++;
            std::string path = fs::current_path().string() + "/assets/videos/" + argv[cursor];
            if(!fs::exists(path)){
                std::cout << "file does not exists!" << std::endl;
                return 1;
            }
            mediaInput.filepath = path;
        }else if(arg == "-realtime"){
            std::cout << "realtime" << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::realtime;
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
        if(arg == "-realtime") {
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