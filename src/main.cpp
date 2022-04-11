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

    CanProcess *canprocess = new CanProcess(&mediaInput);
    CVProcess *cvprocess = new CVProcess(&mediaInput);

    cvprocess->setCanProcess(canprocess);
    application.RegisterProcess(cvprocess);

    #if linux
    ReadProcess *readcan = new ReadProcess();
    canprocess->setReadProcess(readcan);
    application.RegisterProcess(readcan);
    #endif

    application.RegisterProcess(canprocess);

    application.Run();
}