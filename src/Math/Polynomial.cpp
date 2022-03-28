#include "Polynomial.h"
#include <iterator>
#include <math.h>

//https://www.bragitoff.com/2018/06/polynomial-fitting-c-program/

/*******
 Function that performs Gauss-Elimination and returns the Upper triangular matrix and solution of equations:
There are two options to do this in C.
1. Pass the augmented matrix (a) as the parameter, and calculate and store the upperTriangular(Gauss-Eliminated Matrix) in it.
2. Use malloc and make the function of pointer type and return the pointer.
This program uses the first option.
********/
void Polynomial::gaussEliminationLS(int m, int n, double **a, double *x){
    int i,j,k;
    for(i=0;i<m-1;i++){
        //Partial Pivoting
        for(k=i+1;k<m;k++){
            //If diagonal element(absolute vallue) is smaller than any of the terms below it
            if(fabs(a[i][i])<fabs(a[k][i])){
                //Swap the rows
                for(j=0;j<n;j++){                
                    double temp;
                    temp=a[i][j];
                    a[i][j]=a[k][j];
                    a[k][j]=temp;
                }
            }
        }
        //Begin Gauss Elimination
        for(k=i+1;k<m;k++){
            double  term=a[k][i]/ a[i][i];
            for(j=0;j<n;j++){
                a[k][j]=a[k][j]-term*a[i][j];
            }
        }
         
    }
    //Begin Back-substitution
    for(i=m-1;i>=0;i--){
        x[i]=a[i][n-1];
        for(j=i+1;j<n-1;j++){
            x[i]=x[i]-a[i][j]*x[j];
        }
        x[i]=x[i]/a[i][i];
    }
             
}

std::vector<double> Polynomial::Polyfit(std::vector<cv::Point2f> pts, int degree){
    int N = pts.size();  
    int n = degree;  
    double X[2*n+1];  

    for(int i=0;i<=2*n;i++){
        X[i]=0;
        for(int j=0;j<N;j++){
            X[i]=X[i]+pow(pts[j].x,i);
        }
    }

    //the normal augmented matrix
    double **B = new double*[n+1];//[n+1][n+2];  
    for(int i=0;i<n+1;i++)
    {
        B[i] = new double[n+2];
    }

    // rhs
    double Y[n+1];      
    for(int i=0;i<=n;i++){
        Y[i]=0;
        for(int j=0;j<N;j++){
            Y[i]=Y[i]+pow(pts[j].x,i)*pts[j].y;
        }
    }

    for(int i=0;i<=n;i++){
        for(int j=0;j<=n;j++){
            B[i][j]=X[i+j]; 
        }
    }

    for(int i=0;i<=n;i++){
        B[i][n+1]=Y[i];
    }

    double A[n+1];
    std::vector<double> result;
    gaussEliminationLS(n+1,n+2,B,A);
    for(int i=0;i<=n;i++){
        result.push_back(A[i]);
    }

    return result;
}

double Polynomial::Curvature(std::vector<double> fit, int yEval){
    return pow(1 + pow((2 * fit[2] * yEval + fit[2]), 2), 1.5) / abs(2 * fit[1]);
}

double Polynomial::Vertex(std::vector<double> fit){
    return (-fit[1]) / 2 * fit[2];
}
