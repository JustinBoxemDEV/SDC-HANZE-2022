#ifndef MEDIA_CAPTURE_H
#define MEDIA_CAPTURE_H

#include <stdint.h>
#include <string>

class MediaCapture
{
    public:
        void ProcessFeed(int cameraID, char* filename);
};

#endif