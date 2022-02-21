#pragma once

// TODO: finding correct canbus library for C++ (import can in python)
#include <map>
#include <queue>

class Record {
    public:
        const double CAN_MSG_SENDING_SPEED = .040; // 25Hz

        void run();
        void initialize_can();
        void initialize_camera();

        class CanListener {

            public:
                // TODO: adding canbus variable parameter in CanListener constructor
                CanListener();
                void startListenting();
                void stopListening();
                void getNewValues();
                void listen();
                static std::map<short, std::string> idConversion;
        };

        class ImageWorker {
            
            public:
                // TODO: figuring out datatype for image queue
                ImageWorker(std::queue<int> imageQueue);
                void start();
                void stop();
                void put();
                void process();
        };

        class CanWorker {

            public:
                // TODO: figuring out datatype for can queue
                CanWorker(std::queue<int> canQueue, char* folderName);
                void start();
                void stop();
                void put();
                void process();
        };
};