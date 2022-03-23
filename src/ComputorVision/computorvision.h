#ifndef COMPUTOR_VISION_H
#define COMPUTOR_VISION_H

#include <opencv2/opencv.hpp>

class ComputorVision{
    private:
        cv::Mat frame;
        cv::Mat structuringElement;
        cv::Point2f dstP[4];
        double normalisedLaneOffset = 0;
        double laneOffset = 0;
        double curveRadiusR = 0;
        double curveRadiusL = 0;


        int hMin = 0;
        int hMax = 178;
        int sMin = 0; //33
        int sMax = 255;
        int vMin = 0;
        int vMax = 255;


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
    private:
        cv::Vec2f averageVec2Vector(std::vector<cv::Vec2f> vectors);
        cv::Vec4i GeneratePoints(cv::Mat src, cv::Vec2f average);
    public:

        ComputorVision(){
            cv::namedWindow("Trackbars", 0); 
            cv::createTrackbar("Hue min", "Trackbars", &hMin, hMax);
            cv::createTrackbar("Hue max", "Trackbars", &hMax, hMax);
            cv::createTrackbar("Sat min", "Trackbars", &sMin, sMax);
            cv::createTrackbar("Sat max", "Trackbars", &sMax, sMax);
            cv::createTrackbar("Val min", "Trackbars", &vMin, vMax);
            cv::createTrackbar("Val max", "Trackbars", &vMax, vMax);
            structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
        }
        double getNormalisedLaneOffset(){ return normalisedLaneOffset; }
        double getLaneOffset(){ return laneOffset; }
        double getRightEdgeCurvature(){ return curveRadiusR; }
        double getLeftEdgeCurvature(){ return curveRadiusL; }

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
        void PredictTurn(cv::Mat src, std::vector<cv::Vec4i> edgeLines);
};

#endif