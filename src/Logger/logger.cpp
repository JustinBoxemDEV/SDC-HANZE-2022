#include "logger.h"
#include <fstream>
#include <filesystem>
#include <unistd.h>

std::string Logger::activeFile;

std::string Logger::getCurrentPath() {
    return "";//(std::string) get_current_dir_name()+"/../logs/";
};

bool Logger::existsFile(std::string fileName) {
    std::ifstream ifile;
    ifile.open(getCurrentPath()+fileName);
    return ifile ? true : false;
};

void Logger::createFile(std::string fileName) {
    std::string directoryPath = getCurrentPath();

    if(!std::filesystem::is_directory(directoryPath)) {
        std::filesystem::create_directory(directoryPath);
    };

    if(existsFile(fileName)) {
        Logger::warning("File \033[1;37m"+fileName+"\033[0m does already exist!");
    } else {
        std::string filePath = directoryPath + fileName;
        const char *path = const_cast<char*>(filePath.c_str());
        std::ofstream file(path);
        Logger::success("File \033[1;37m"+fileName+"\033[0m successfully created!");
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
    if(existsFile(fileName)) {
        Logger::success("File \033[1;37m"+fileName+"\033[0m is active!");
        activeFile = fileName;
    } else {
        Logger::error("File \033[1;37m"+fileName+"\033[0m does not exist!");
    };
};

void Logger::resetActiveFile() {
    activeFile.clear();
};