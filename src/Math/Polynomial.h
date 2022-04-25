#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <opencv2/opencv.hpp>
class Polynomial
{
    private:
        static void gaussEliminationLS(int m, int n,  double ** a, double *x);
    public:
        static std::vector<double> Polyfit(std::vector<cv::Point2f> pts, int degree);
        static double Curvature(std::vector<double> fit, int yEval);
        static double Vertex(std::vector<double> fit);
        static std::vector<double> QuadraticFormula(std::vector<double> fit);
};

#endif