#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <opencv2/opencv.hpp>
class Polynomial
{
    private:
        static void gaussEliminationLS(int m, int n,  double ** a, double *x);
    public:
        static void Polyfit(std::vector<cv::Point2f> pts, int degree);

};

#endif