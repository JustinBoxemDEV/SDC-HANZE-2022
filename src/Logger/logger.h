#pragma once

#include "../time/time.h"
#include <iostream>
#include <sstream>

class Logger {
    public:
        Logger();
        enum Level { DEBUG, INFO, WARNING, ERROR, SUCCESS, DEFAULT };
        static Level level;
        static std::string activeFile;
        static void setLevel(enum Level);
        static void createFile(std::string fileName = Time::currentDateTime());
        static void writeToFile(std::string data, std::string fileName);
        static void setActiveFile(std::string fileName);
        static void resetActiveFile();

        template<class T>
        static void log(const T & object) {
            
            std::stringstream ss;
            ss << object;

            std::string messageLevel = "";
            switch(level) {
                case DEBUG: {
                    std::cout << "\033[1;37m[DEBUG]\t\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
                    messageLevel = "[DEBUG]\t\t "+Time::currentDateTime()+": ";
                    break;
                };
                case INFO: {
                    std::cout << "\033[1;34m[INFO]\t\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
                    messageLevel = "[INFO]\t\t "+Time::currentDateTime()+": ";
                    break;
                };
                case WARNING: {
                    std::cout << "\033[1;33m[WARNING]\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
                    messageLevel = "[WARNING]\t "+Time::currentDateTime()+": ";
                    break;
                };
                case ERROR: {
                    std::cout << "\033[1;31m[ERROR]\t\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
                    messageLevel = "[ERROR]\t\t "+Time::currentDateTime()+": ";
                    break;
                };
                case SUCCESS: {
                    std::cout << "\033[1;32m[SUCCESS]\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
                    messageLevel = "[SUCCESS]\t "+Time::currentDateTime()+": ";
                    break;
                };
                default: {
                    std::cout << ss.str() << std::endl;
                    break; 
                };
            };
            if(!activeFile.empty()) {
                //std::cout << "Writing data to file "+activeFile << std::endl;
                writeToFile(messageLevel + ss.str(), activeFile);
            };
        };
        
        template<class T>
        static void log(const T (&object)[]) {
            std::cout << "This is an array!!" << std::endl;
        }

        template<class T>
        static void warning(const T & object) {
            level = WARNING;
            log(object);
            level = DEFAULT;
        };

        template<class T>
        static void error(const T & object) {
            level = ERROR;
            log(object);
            level = DEFAULT;
        };

        template<class T>
        static void success(const T & object) {
            level = SUCCESS;
            log(object);
            level = DEFAULT;
        };

        template<class T>
        static void info(const T & object) {
            level = INFO;
            log(object);
            level = DEFAULT;
        };

        template<class T>
        static void debug(const T & object) {
            level = DEBUG;
            log(object);
            level = DEFAULT;
        };
        
        Level getLevel();
        std::string getActiveFile();
    private:
        static std::string getCurrentPath();
        static bool existsFile(std::string fileName);
};
