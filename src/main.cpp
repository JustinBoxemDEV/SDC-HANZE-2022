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
#include "mediaCapture/mediaCapture.h"
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
    // --help Output, describing basic usage to the user
    if(argc==1)
    {
        MediaCapture mediaCapture;
        mediaCapture.ProcessFeed(0,"");
        return 0;
    }
    if(string(argv[1])=="-help" or string(argv[1])=="-h")
    {
        cout << "Usage: SPECIFY RESOURCE TO USE" << endl;
        cout << "-video -camera [CAMERA_ID]" << endl;
        cout << "-video -filename [FILE]" << endl;
        cout << "-image [FILE]" << endl;
        return -1;
    }
    else
    {
        // The user has told us he wants to use media feed
        if(string(argv[1])=="-video")
        {
            if(argc==2)
            {
                cout << "Usage:" << endl; 
                cout << "-video -camera [CAMERA_ID]" << endl;
                cout << "-video -filename [FILE]" << endl;
                return -1;
            }
            if(argc==3)
            {
                cout << "Usage:" << endl;
                if(string(argv[2])=="-camera")
                {
                    cout << "-video -camera [CAMERA_ID]" << endl;
                    return -1;
                }
                if(string(argv[2])=="-filename")
                {
                    // No video file was provided to look for, so we are going to present a list of names
                    cout << "Available videos to load using -filename [FILE]" << endl;
                    string path = fs::current_path().string() + "/assets/videos/";
                    for (const auto & file : fs::directory_iterator(path))
                        //cout << file << endl;
                        cout << fs::path(file).filename().string() << endl;
                    return -1;
                }
            }
            if(argc==4)
            {   
                if(string(argv[2])=="-filename")
                {
                    string path = fs::current_path().string() + "/assets/videos/" + string(argv[3]);
                    if(!fs::exists(path))
                    {
                        cout << "The requested file cannot be found in /assets/videos!" << endl;
                        return -1;
                    }
                    MediaCapture mediaCapture;
                    mediaCapture.ProcessFeed(0,argv[3]);
                    return 0;
                }
                if(string(argv[2])=="-camera")
                {
                    MediaCapture mediaCapture;
                    mediaCapture.ProcessFeed(stoi(argv[3]),"");
                    return 0;
                }
                else
                {
                    MediaCapture mediaCapture;
                    mediaCapture.ProcessFeed(0,argv[3]);
                    return 0;
                }
            }
        }
        if(string(argv[1])=="-image")
        {
            // An image was provided to look for
            if(argc==3)
            {
                samples::addSamplesDataSearchPath(fs::current_path().string() + "/assets/images/");
                Mat src = imread( samples::findFile(string(argv[2])));
                if( src.empty() )
                {
                    cout << "Could not open or find the image!\n" << endl;
                    cout << "Check the provided image name (include extension)" << endl;
                    return -1;
                }
               
                return 0;
            }
            // No image was provided to look for, so we are going to present a list of names
            cout << "Available images to load using -image [NAME]" << endl;
            string path = fs::current_path().string() + "/assets/images/";
            for (const auto & file : fs::directory_iterator(path))
                //cout << file << endl;
                cout << fs::path(file).filename().string() << endl;
            return -1;
        }
        // The parameter that the user provided is not compatible with our program | Provide error + help message
        else
        {
            cout << "ERROR: " << string(argv[1]) << " is not recognised. Use -help for information" << endl;
            return -1;
        }
    }
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