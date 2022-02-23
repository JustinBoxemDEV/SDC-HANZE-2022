#pragma once

#include "../time/time.h"
#include <iostream>

class Logger {
    public:
        Logger();
        enum Level { DEBUG, INFO, WARNING, ERROR, SUCCESS, DEFAULT };
        static void setLevel(enum Level);
        static void log(std::string message);
        static void createFile(std::string fileName = Time::currentDateTime());
        static void writeToFile(std::string data, std::string fileName);
        static void setActiveFile(std::string fileName);
        static void resetActiveFile();
        static void warning(std::string message);
        static void error(std::string message);
        static void success(std::string message);
        static void info(std::string message);
        static void debug(std::string message);
    private:
        static std::string getCurrentPath();
        static bool existsFile(std::string fileName);
};
