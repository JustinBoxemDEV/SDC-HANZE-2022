// #include <stdio.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/core/utility.hpp>
// using namespace cv;

// int main(int argc, char** argv )
// {
//     Mat image;
//     samples::addSamplesDataSearchPath("/home/robinvanwijk/Projects/SDC/SDC-HANZE-2022/images");
//     image = imread( samples::findFile( "megamind.jpg" ), 1 );
//     if ( !image.data )
//     {
//         printf("No image data \n");
//         return -1;
//     }
//     namedWindow("Display Image", WINDOW_AUTOSIZE );
//     imshow("Display Image", image);
//     waitKey(0);
//     return 0;
// }

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <filesystem>
#include <string>
#include "camera/cameraCapture.h"
#include "PID/PID.h"
//#include <libsocketcan.h>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;

Mat src_gray;
int thresh = 100;
RNG rng(12345);
void thresh_callback(int, void* );

void testCAN(){
    int * result;
    //cout << can_get_state("can0", result) << endl; 	
}

int main( int argc, char** argv )
{
    // dit is test spul
    int x = 0;
    int prefX = x;
    double y = -0.5;//verander dit
    double prefY = y;
    double out = 0;
    PIDController pid{};
    pid.PIDController_Init(pid);
    Mat drawing;
    drawing = Mat::zeros(Size(160, 160), CV_8UC1);
    line(drawing, Point(0,80), Point(160,80), Scalar(150), 2, 2, 0);
    
    while ( x < 150) {
        line(drawing, Point(prefX, prefY*80+80), Point(x, y*80+80), Scalar(255), 2, 2, 0);
        imshow("image", drawing);
        waitKey(1000);

        out = pid.PIDController_update(pid, y);
        prefY = y;
        prefX = x;
        y = y - out;
        cout << "offset " << y << endl;
        cout <<"output " << out << endl;
        cout << "  " << endl;
        x++;
    }
    waitKey(10000);
    // eind test spul

/*
    CameraCapture cameraCapture;
    cameraCapture.ProcessFeed();

   // samples::addSamplesDataSearchPath(fs::current_path().string() + "/images");
    
    testCAN();

    Mat src = imread("C:/Users/ShandorPC/Documents/GitHub/SDC/images/megamind.jpg");
    if( src.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    blur( src_gray, src_gray, Size(3,3) );
    const char* source_window = "Source";
    namedWindow( source_window );
    imshow( source_window, src );
    const int max_thresh = 255;
    createTrackbar( "Canny thresh:", source_window, &thresh, max_thresh, thresh_callback );
    thresh_callback( 0, 0 );
    waitKey();
    return 0;
    */

}
void thresh_callback(int, void* )
{
    Mat canny_output;
    Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }
    imshow( "Contours", drawing );
}