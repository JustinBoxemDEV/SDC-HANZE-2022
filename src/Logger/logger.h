#pragma once

#include "../utils/Time/time.h"
#include <iostream>
#include <sstream>
#include <chrono>

class Logger {
    public:
        static std::string activeFile;

        static void createFile(std::string fileName = Time::currentDateTime());
        static void writeToFile(std::string data, std::string fileName);
        static void setActiveFile(std::string fileName);
        static void resetActiveFile();

        template<class T>
        static std::stringstream getSS(const T & object) {
            std::stringstream ss;
            ss << object;
            return ss;
        };

        template<class T>
        static void log(const T & object, std::string type="", std::string ascii="", bool timeStamp=false, std::string tabs="") {
            std::string time;
            if(!type.empty()) type = "["+type+"]";
            if(timeStamp) time = Time::currentDateTime()+"\t\t ";
            if(!ascii.empty()) ascii = "\033["+ascii+"m";
            std::cout << ascii+type+tabs+"\033[0m"+time+getSS(object).str() << std::endl;
            writeToFile(type+tabs+time+getSS(object).str(), activeFile);
        };
        
        template<class T>
        static void warning(const T & object) {
            log(getSS(object).str(), "WARNING", "1;33", true, "\t");
        };

        template<class T>
        static void error(const T & object) {
            log(getSS(object).str(), "ERROR", "1;31", true, "\t\t");
        };

        template<class T>
        static void success(const T & object) {
            log(getSS(object).str(), "SUCCESS", "1;32", true, "\t");
        };

        template<class T>
        static void info(const T & object) {
            log(getSS(object).str(), "INFO", "1;34", true, "\t\t");
        };

        template<class T>
        static void debug(const T & object) {
            log(getSS(object).str(), "DEBUG", "1;37", true, "\t\t");
        };
    private:
        static std::string getCurrentPath();
        static bool existsFile(std::string fileName);
};
