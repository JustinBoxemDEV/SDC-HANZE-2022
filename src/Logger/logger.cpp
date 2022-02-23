#include "logger.h"
#include <fstream>
#include <filesystem>
#include <unistd.h>

Logger::Level Logger::level = Logger::DEFAULT;
std::string Logger::activeFile;

Logger::Logger() {
    level = DEFAULT;
};

Logger::Level Logger::getLevel() {
    return level;
}

std::string Logger::getActiveFile() {
    return activeFile;
}

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