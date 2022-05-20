#ifndef PROCESS_H
#define PROCESS_H
#include <string>

class Process
{
    private:
    public:
        enum class MediaSource {
            video,
            realtime,
            realtime_ml,
            assetto,
            terminal
        };
        typedef struct {
            MediaSource mediaType = MediaSource::realtime;
            int cameraID = 0;
            std::string filepath = "";
        } MediaInput;
        MediaInput* mediaInput;
        virtual void Run() = 0;
        virtual void Terminate() = 0;
};

#endif