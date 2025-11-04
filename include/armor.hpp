#ifndef ARMOR_HPP
#define ARMOR_HPP

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <string>

const int RED = 0;
const int BLUE = 1;

enum class ArmorType { SMALL, LARGE, INVALID };
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

class Light : public cv::RotatedRect
{
public:
    Light() = default;
    explicit Light(cv::RotatedRect box) : cv::RotatedRect(box)
    {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2; 
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom); //灯条长度
        width = cv::norm(p[0] - p[1]); //灯条宽度

        tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y)); //弧度制，与y轴夹角
        tilt_angle = tilt_angle / CV_PI * 180; //弧度转角度
    }


    int color;
    cv::Point2f top, bottom;
    double length;
    double width;
    float tilt_angle;
};

struct Armor
{
  Armor() = default;
  Armor(const Light & l1, const Light & l2) 
  {
    if (l1.center.x < l2.center.x) {
      left_light = l1, right_light = l2; 
    } else {
      left_light = l2, right_light = l1;
    }
    center = (left_light.center + right_light.center) / 2;
  }

  //灯条部分
  Light left_light, right_light;
  cv::Point2f center;
  ArmorType type;

  //数字识别部分
  cv::Mat number_img;
  std::string number;
  float confidence;
  std::string classfication_result;
};

#endif