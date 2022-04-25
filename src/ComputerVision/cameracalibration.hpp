#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class CameraCalibration {
    public:
        int checkerBoard[2];
        int fieldSize;

        std::vector<cv::String> fileNames;
        std::vector<std::vector<cv::Point3f>> Q;

        std::vector<cv::Point3f> objp;
        std::vector<cv::Point2f> imgPoint;

        CameraCalibration(std::string path, int chessLength, int chessWidth, int fieldSize, int frameLength, int frameHight);
};

#endif