#include "cameracalibration.hpp"

namespace fs = std::filesystem;

/**
 * @brief Construct a new Camera Calibration:: Camera Calibration object
 * 
 * @param path Path to distorted images
 * @param chessLength The amount of chess tiles in the length of the chessboard
 * @param chessWidth The amount of chess tiles in the width of the chessboard
 * @param fieldSize The size of the chess tiles in mm
 * @param frameLength The amount of pixels horizontally
 * @param frameHeight The amount of pixels vertically
 */
CameraCalibration::CameraCalibration(std::string path, int chessLength, int chessWidth, int fieldSize, int frameLength, int frameHeight) {
    std::string currentPath = fs::current_path().string();
    currentPath.append("\\assets\\images\\calibration\\results\\");

    cv::glob(path, CameraCalibration::fileNames, false);
    cv::Size patternSize(chessLength-1, chessWidth-1);

    std::vector<std::vector<cv::Point2f>> q(CameraCalibration::fileNames.size());
    std::vector<std::vector<cv::Point3f>> Q;

    CameraCalibration::checkerBoard[0] = chessLength;
    CameraCalibration::checkerBoard[1] = chessWidth;
    CameraCalibration::fieldSize = fieldSize;

    for(int i = 1; i<checkerBoard[1]; i++){
        for(int j = 1; j<checkerBoard[0]; j++) {
            CameraCalibration::objp.push_back(cv::Point3f(j*CameraCalibration::fieldSize, i*CameraCalibration::fieldSize, 0));
        }
    };

    std::size_t i = 0;
    for (auto const &f : CameraCalibration::fileNames) {
        std::cout << std::string(f) << std::endl;
    
        cv::Mat img = cv::imread(CameraCalibration::fileNames[i]);
        cv::Mat gray;

        cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

        bool patternFound = cv::findChessboardCorners(gray, patternSize, q[i], cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        if(patternFound) {
            cv::cornerSubPix(gray, q[i], cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
            Q.push_back(CameraCalibration::objp);
        }

        cv::drawChessboardCorners(img, patternSize, q[i], patternFound);
        cv::imshow("chessboard detection", img);
        std::string filename = currentPath+"calibration"+std::to_string(i)+".jpg";
        imwrite(filename, img);
        cv::waitKey(0);

        i++;
    }

    i = 0;

    cv::Matx33f K(cv::Matx33f::eye());
    cv::Vec<float, 5> k(0, 0, 0, 0, 0);

    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<double> stdIntrinsics, stdExtrinsics, perViewError;
    int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;
    cv::Size frameSize(640, 480);

    std::cout << "Calibrating" << std::endl;

    float error = cv::calibrateCamera(Q, q, frameSize, K, k, rvecs, tvecs, flags);

    std::cout << "Reprojection error = " << error << "\nK =\n"
        << K << "\nk=\n"
        << k << std::endl;

    cv::Mat mapX, mapY;
    cv::initUndistortRectifyMap(K, k, cv::Matx33f::eye(), K, frameSize, CV_32FC1,
                              mapX, mapY);

    // Show lens corrected images
    for (auto const &f : fileNames) {
        std::cout << std::string(f) << std::endl;

        cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);

        cv::Mat imgUndistorted;
        // 5. Remap the image using the precomputed interpolation maps.
        cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);

        // Display
        cv::imshow("undistorted image", imgUndistorted);
        std::string filename = currentPath+"undistorted"+std::to_string(i)+".jpg";
        imwrite(filename, imgUndistorted);
        cv::waitKey(0);
        i++;
  }
};