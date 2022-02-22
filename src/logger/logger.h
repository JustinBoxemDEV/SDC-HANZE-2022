#pragma once

#include "../time/time.h"
#include <iostream>

class Logger {
    public:
        Logger();
        enum Level { DEBUG, INFO, WARNING, ERROR, SUCCESS, DEFAULT };
        void setLevel(enum Level);
        void log(std::string message);
        void createFile(std::string fileName = Time::currentDateTime());
        void writeToFile(std::string data, std::string fileName);
        void setActiveFile(std::string fileName);
        void resetActiveFile();
        void warning(std::string message);
        void error(std::string message);
        void success(std::string message);
        void info(std::string message);
        void debug(std::string message);
    private:
        Level level;
        std::string activeFile;
        std::string getCurrentPath();
        bool existsFile(std::string fileName);
};
