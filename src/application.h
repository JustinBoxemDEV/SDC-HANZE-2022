#ifndef APPLICATION_H
#define APPLICATION_H

#include "cvprocess.h"
#include "cvprocess.h"
#include "canprocess.h"
#include <string>

class Application
{
    private:
        Process processes[] = { new CvProcess(), new CanProcess()};
    public:
        void Initialize(MediaInput input);
        void Run();
        
    enum class MediaSource {
        image,
        video,
        realtime
    };

    struct {
        MediaSource mediaType = MediaSource::realtime;
        int cameraID = 0;
        std::string filepath = "";
    } MediaInput;
};


#endif