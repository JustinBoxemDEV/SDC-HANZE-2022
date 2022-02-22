#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <stdint.h>

TEST(InitializationTests, VideoCaptureObjectTest)
{
    cv::VideoCapture* vidcap = new cv::VideoCapture();
    ASSERT_NE(vidcap, nullptr);
}

/*
TEST(TestNameSuite, TestName)
{
    Test logic here + asserts (see test above)
    ASSET_NE = Assert not equal
    ASSER_EQ = Assert equal
}
*/