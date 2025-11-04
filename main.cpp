#include <iostream>
#include "yolo_detection.hpp"
#include <thread>
#include <vector>
#include <functional>
#include "timeCounter.hpp"
#include <random>  // C++11的随机数库
#include <ctime> 
#include <opencv2/opencv.hpp>
#include <string>
int main(int argc, char** argv)
{
    std::string video_path = "/Users/abing/detection/video/2.mp4";
    std::string model_path = "/Users/abing/detection/model/new.onnx";
    detection::DetectionArmor detectionArmor(model_path, true,video_path);

    detectionArmor.start_detection();

    return 0;
}
