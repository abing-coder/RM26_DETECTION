#include <iostream>
#include "detection.h"
#include <thread>
#include <vector>
#include <functional>
#include "timeCounter.hpp"
#include <random>  // C++11的随机数库
#include <ctime> 
#include <opencv2/opencv.hpp>
#include <string>


int main(int, char**)
{
    std::string video_path = "/home/ubuntu/桌面/RM26_Detect_Ling/detection/video/2.mp4";
    std::string model_path = "/home/ubuntu/桌面/RM26_Detect_Ling/detection/model/new.onnx";
    detection::DetectionArmor detectionArmor(model_path, true,video_path);

    detectionArmor.start_detection();

    return 0;
}
