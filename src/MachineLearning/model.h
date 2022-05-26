#ifdef linux
#ifndef LOAD_MODEL_H
#define LOAD_MODEL_H

#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <filesystem>

#include "../utils/Time/time.h"
#include "../VehicleControl/communicationstrategy.h"

using namespace std;
namespace fs = filesystem;

class Model
{
    private:
        torch::jit::script::Module module;
        int intStayBetween(int integer, int min, int max);
        torch::Tensor PreprocessImage(cv::Mat img);
        int LimitOutputInt(int value, int min = 0, int max = 100);
        float LimitOutputFloat(float value, int min = -1, int max = 1);
        bool CSVIsEnabled = false;
        std::ofstream csvFile;
        bool csvIsClosed = false;
    public:
        Model(string model);
        void EnableCSV();
        void closeCSV();
        void Inference(cv::Mat frame, string img = "");
};

#endif
#endif