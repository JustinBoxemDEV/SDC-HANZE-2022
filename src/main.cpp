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

    ReadProcess *readcan = new ReadProcess();

    application.RegisterProcess(new CVProcess(&mediaInput));
    application.RegisterProcess(new CanProcess(&mediaInput, readcan));
    application.RegisterProcess(readcan);
    application.Run();
}