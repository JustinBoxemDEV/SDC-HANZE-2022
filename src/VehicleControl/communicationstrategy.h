#pragma once

class CommunicationStrategy {
    public:
        virtual void steer() = 0;
        virtual void brake() = 0;
        virtual void forward() = 0;
        virtual void neutral() = 0;
        virtual void stop() = 0;
    private:
        template<class T>
        void sendCanMessage(T & canMessage);
};