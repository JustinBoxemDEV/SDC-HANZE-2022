#ifndef COMPUTER_VISION_H
#define COMPUTER_VISION_H

#include <opencv2/opencv.hpp>

class ComputerVision{
    private:
        cv::Mat frame;
        cv::Mat structuringElement;
        cv::Point2f dstP[4];
        double normalisedLaneOffset = 0;
        double laneOffset = 0;
        double curveRadiusR = 0;
        double curveRadiusL = 0;

        cv::Mat masked;
        cv::Mat mask;
        cv::Mat edgeMap;
        cv::Mat blurred;

        cv::Mat denoisedImage;
        cv::Mat hsv;
        cv::Mat hsvFilter;
        cv::Mat binaryImage;

        cv::Mat warped;
        cv::Mat homography;
        cv::Mat invertedPerspectiveMatrix;

        cv::Mat warpedOverlay;

        std::vector<double> lastKnownAveragedFitR;
        std::vector<double> lastKnownAveragedFitL;
    private:
        cv::Vec2f averageVec2Vector(std::vector<cv::Vec2f> vectors);
        cv::Vec4i GeneratePoints(cv::Mat src, cv::Vec2f average);
        std::vector<double> ExponentalMovingAverage(std::vector<double> &lastAveragedFit, std::vector<double> fit, double beta);
    public:
        ComputerVision(){
            structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6, 6));
        }
        double getNormalisedLaneOffset(){ return normalisedLaneOffset; }
        double getLaneOffset(){ return laneOffset; }
        double getRightEdgeCurvature(){ return curveRadiusR; }
        double getLeftEdgeCurvature(){ return curveRadiusL; }
        cv::Mat GammaCorrection(const cv::Mat src, const float gamma);

        void SetFrame(cv::Mat src);
        cv::Mat BlurImage(cv::Mat src);
        cv::Mat DetectEdges(cv::Mat src);
        cv::Mat MaskImage(cv::Mat src);
        std::vector<cv::Vec4i> HoughLines(cv::Mat src);
        std::vector<cv::Vec4i> AverageLines(cv::Mat src, std::vector<cv::Vec4i> lines);
        cv::Mat PlotLaneLines(cv::Mat src, std::vector<cv::Vec4i> lines);
        std::vector<cv::Point2f> SlidingWindow(cv::Mat image, cv::Rect window);
        std::vector<int> Histogram(cv::Mat src);
        cv::Mat CreateBinaryImage(cv::Mat src);
        std::vector<cv::Vec4i> GenerateLines(cv::Mat src);
        void PredictTurn(cv::Mat src);
};

#endif