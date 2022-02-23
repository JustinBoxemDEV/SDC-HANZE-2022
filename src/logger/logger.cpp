#include "logger.h"
#include <fstream>
#include <filesystem>
#include <unistd.h>

Logger::Level level;
std::string activeFile;

Logger::Logger() {
    level = DEFAULT;
};

void Logger::log(std::string message) {
    std::string messageLevel = "";
    switch(level) {
        case DEBUG: {
            std::cout << "\033[1;37m[DEBUG]\t\t\033[0m " << Time::currentDateTime() << "\t " + message << std::endl;
            messageLevel = "[DEBUG]\t\t "+Time::currentDateTime()+": ";
            break;
        };
        case INFO: {
            std::cout << "\033[1;34m[INFO]\t\t\033[0m " << Time::currentDateTime() << "\t " + message << std::endl;
            messageLevel = "[INFO]\t\t "+Time::currentDateTime()+": ";
            break;
        };
        case WARNING: {
            std::cout << "\033[1;33m[WARNING]\t\033[0m " << Time::currentDateTime() << "\t " + message << std::endl;
            messageLevel = "[WARNING]\t "+Time::currentDateTime()+": ";
            break;
        };
        case ERROR: {
            std::cout << "\033[1;31m[ERROR]\t\t\033[0m " << Time::currentDateTime() << "\t " + message << std::endl;
            messageLevel = "[ERROR]\t\t "+Time::currentDateTime()+": ";
            break;
        };
        case SUCCESS: {
            std::cout << "\033[1;32m[SUCCESS]\t\033[0m " << Time::currentDateTime() << "\t " + message << std::endl;
            messageLevel = "[SUCCESS]\t "+Time::currentDateTime()+": ";
            break;
        };
        default: {
            std::cout << message << std::endl;
            break; 
        };
    };
    if(!activeFile.empty()) {
        //std::cout << "Writing data to file "+activeFile << std::endl;
        writeToFile(messageLevel + message, activeFile);
    };
};

void Logger::setLevel(Logger::Level level) {
    switch(level) {
        case DEBUG:
            level = DEBUG;
            break;
        case INFO:
            level = INFO;
            break;
        case WARNING:
            level = WARNING;
            break;
        case ERROR:
            level = ERROR;
            break;
        case SUCCESS:
            level = SUCCESS;
            break;
        case DEFAULT:
            level = DEFAULT;
    };
};

std::string Logger::getCurrentPath() {
    return (std::string) get_current_dir_name()+"/../logs/";
};

bool Logger::existsFile(std::string fileName) {
    std::ifstream ifile;
    ifile.open(getCurrentPath()+fileName);
    return ifile ? true : false;
};

void Logger::createFile(std::string fileName) {
    std::string directoryPath = getCurrentPath();
    Logger::Level currentLevel = level;

    if(!std::filesystem::is_directory(directoryPath)) {
        std::filesystem::create_directory(directoryPath);
    };

    if(existsFile(fileName)) {
        Logger::setLevel(WARNING);
        Logger::log("File \033[1;37m"+fileName+"\033[0m does already exist!");
        Logger::setLevel(currentLevel);
    } else {
        std::string filePath = directoryPath + fileName;
        const char *path = const_cast<char*>(filePath.c_str());
        std::ofstream file(path);
        Logger::setLevel(SUCCESS);
        Logger::log("File \033[1;37m"+fileName+"\033[0m successfully created!");
        Logger::setLevel(currentLevel);
        activeFile = fileName;
    };
};

void Logger::writeToFile(std::string data, std::string fileName) {
    std::ofstream ofile;
    ofile.open(getCurrentPath()+fileName, std::ios_base::app);
    ofile << data+"\n";
    ofile.close();
};

void Logger::warning(std::string message) {
    level = WARNING;
    Logger::log(message);
    level = DEFAULT;
};

void Logger::error(std::string message) {
    level = ERROR;
    Logger::log(message);
    level = DEFAULT;
};

void Logger::info(std::string message) {
    level = INFO;
    Logger::log(message);
    level = DEFAULT;
};

void Logger::debug(std::string message) {
    level = DEBUG;
    Logger::log(message);
    level = DEFAULT;
};

void Logger::success(std::string message) {
    level = SUCCESS;
    Logger::log(message);
    level = DEFAULT;
};

void Logger::setActiveFile(std::string fileName) {
    Logger::Level currentLevel = level;
    if(existsFile(fileName)) {
        Logger::setLevel(SUCCESS);
        Logger::log("File \033[1;37m"+fileName+"\033[0m is active!");
        Logger::setLevel(currentLevel);
        activeFile = fileName;
    } else {
        Logger::setLevel(ERROR);
        Logger::log("File \033[1;37m"+fileName+"\033[0m does not exist!");
        Logger::setLevel(currentLevel);
    };
};

void Logger::resetActiveFile() {
    activeFile.clear();
};