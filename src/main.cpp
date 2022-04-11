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

    while(cursor < argc){
        std::string arg = argv[cursor];
        if(arg == "-video"){
            mediaInput.mediaType = CVProcess::MediaSource::video;
            cursor++;
            std::string path = fs::current_path().string() + "/assets/videos/" + argv[cursor];
            if(!fs::exists(path)){
                std::cout << "file does not exists!" << std::endl;
                return 1;
            }
            mediaInput.filepath = path;
        }else if(arg == "-realtime"){
            mediaInput.mediaType = CVProcess::MediaSource::realtime;
            CVProcess *cvprocess = new CVProcess(&mediaInput);
            application.RegisterProcess(cvprocess);
        }else if(arg == "-assetto"){
            std::cout << arg << std::endl;
            mediaInput.mediaType = CVProcess::MediaSource::assetto;
        }else if(arg == "-terminal") {
            mediaInput.mediaType = CVProcess::MediaSource::terminal;
        }
        cursor++;
    }

    CanProcess *canprocess = new CanProcess(&mediaInput);
    
    #ifdef linux
    //ReadProcess *readcan = new ReadProcess();
    TerminalProcess *terminal = new TerminalProcess();

    canprocess->setTerminalProcess(terminal);
    //canprocess->setReadProcess(readcan);
    
    //application.RegisterProcess(readcan);
    application.RegisterProcess(terminal);
    #endif

    application.RegisterProcess(canprocess);

    application.Run();
}