#ifdef false // __linux__
#ifndef CAPTURE_SCREEN_H
#define CAPTURE_SCREEN_H

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <sys/shm.h>
#include <X11/extensions/XShm.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <sys/time.h>

// source: https://stackoverflow.com/questions/32972539/gdk-x11-screen-capture

class CaptureScreenLinux {
    public:
        void run();
    private:
        void initimage(struct shmimage* image);
        void destroyimage(Display* dsp, struct shmimage* image);
        int createimage(Display* dsp, struct shmimage* image, int width, int height);
        void getrootwindow(Display* dsp, struct shmimage* image);
        long timestamp();
        Window createwindow(Display* dsp, int width, int height);
        void destroywindow(Display* dsp, Window window);
        unsigned int getpixel(struct shmimage* src, struct shmimage* dst, int j, int i, int w, int h);
        int processimage(struct shmimage* src, struct shmimage* dst);
        int load(Display* dsp, Window window, struct shmimage* src, struct shmimage* dst);

};

#endif
#endif