#include "application.h"
#include "Processes/canprocess.h"
#include "Processes/cvprocess.h"
#include "Processes/readprocess.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    int cursor = 1;
    Process::MediaInput mediaInput;

    mediaInput.mediaType = CVProcess::MediaSource::realtime;

    Application application;

    application.RegisterProcess(new CVProcess(&mediaInput));

    #if __WIN32__
    application.RegisterProcess(new CanProcess(&mediaInput));
    #endif

    #if linux
    ReadProcess *readcan = new ReadProcess();
    CanProcess *canprocess = new CanProcess(&mediaInput);
    canprocess->setReadProcess(readcan);
    application.RegisterProcess(canprocess);
    application.RegisterProcess(readcan);
    #endif
    application.Run();
}