#include "logger.h"
#include <fstream>
#include <filesystem>
#include <unistd.h>

std::string Logger::activeFile;
namespace fs = std::filesystem;

std::string Logger::getCurrentPath() {
    std::string path = fs::current_path().string();
    return (std::string) path+"/../logs/";
};

bool Logger::existsFile(std::string fileName) {
    std::ifstream ifile;
    ifile.open(getCurrentPath()+fileName);
    return ifile ? true : false;
};

void Logger::createFile(std::string fileName) {
    std::string directoryPath = getCurrentPath();
    std::cout << "Getting current path" << std::endl;
    std::cout << directoryPath << std::endl;
    if(!fs::is_directory(directoryPath)) {
        std::cout << "Creating the directory" << std::endl;
        fs::create_directory(directoryPath)?
            std::cout << "Succeeded creating directory" << std::endl :
            std::cout << "Failed creating directory" << std::endl;
    };
    if(existsFile(fileName)) {
        Logger::warning("File \033[1;37m"+fileName+"\033[0m does already exist!");
    } else {
        std::string filePath = directoryPath + fileName;
        const char *path = const_cast<char*>(filePath.c_str());
        std::cout << path << std::endl;
        std::ofstream file(path);
        file.is_open()? std::cout << "filed is opened" << std::endl : std::cout << "file is closed" << std::endl;
        file.flush();
        file.close();
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