#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <stdint.h>

TEST(InitializationTests, VideoCaptureObjectTest)
{
    cv::VideoCapture* vidcap = nullptr;
    ASSERT_NE(vidcap, nullptr);
}