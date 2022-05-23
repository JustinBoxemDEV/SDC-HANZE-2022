#pragma once
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;
namespace fs = filesystem;

class CSVLogger {
    public:
    CSVLogger();
    void Log(string line);
    void Close();
    private:
    std::ofstream csvFile;
};