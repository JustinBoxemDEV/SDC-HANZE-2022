#pragma once

#include "../utils/Time/time.h"
#include <iostream>
#include <sstream>

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
        static void log(const T & object) {
            std::stringstream ss = getSS(object);
            std::cout << ss.str() << std::endl;
            writeToFile(ss.str(), activeFile);
        };

        template<class T>
        static void warning(const T & object) {
            std::stringstream ss = getSS(object);
            std::cout << "\033[1;33m[WARNING]\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
            writeToFile("[WARNING]\t "+Time::currentDateTime()+": "+ss.str(), activeFile);
        };

        template<class T>
        static void error(const T & object) {
            std::stringstream ss = getSS(object);
            std::cout << "\033[1;31m[ERROR]\t\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
            writeToFile("[ERROR]\t "+Time::currentDateTime()+": "+ss.str(), activeFile);
        };

        template<class T>
        static void success(const T & object) {
            std::stringstream ss = getSS(object);
            std::cout << "\033[1;32m[SUCCESS]\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
            writeToFile("[SUCCESS]\t "+Time::currentDateTime()+": "+ss.str(), activeFile);
        };

        template<class T>
        static void info(const T & object) {
            std::stringstream ss = getSS(object);
            std::cout << "\033[1;34m[INFO]\t\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
            writeToFile("[SUCCESS]\t "+Time::currentDateTime()+": "+ss.str(), activeFile);
        };

        template<class T>
        static void debug(const T & object) {
            std::stringstream ss = getSS(object);
            std::cout << "\033[1;37m[DEBUG]\t\t\033[0m " << Time::currentDateTime() << "\t " << ss.str() << std::endl;
            writeToFile("[DEBUG]\t "+Time::currentDateTime()+": "+ss.str(), activeFile);
        };
    private:
        static std::string getCurrentPath();
        static bool existsFile(std::string fileName);
};
