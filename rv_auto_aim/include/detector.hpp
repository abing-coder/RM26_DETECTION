#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include "armor.hpp"
// #include "number_classifier.hpp"
#include <vector>

class Detector
{
public:
  struct LightParams
  {
    // width / height
    double min_ratio = 0.1;
    double max_ratio = 0.4;
    //垂直于y轴的最大倾斜角
    double max_angle = 40.0;
  };

  struct ArmorParams
  {
    double min_light_ratio = 0.7;
    //灯条间距与灯条高度比值
    double min_small_center_distance = 0.8;
    double max_small_center_distance = 3.2;
    double min_large_center_distance = 3.2;
    double max_large_center_distance = 5.5;
    //水平夹角最大值
    double max_angle = 40.0;
  };

  Detector(const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a); 

  std::vector<Armor> detect(const cv::Mat & input); 

  cv::Mat preprocessImage(const cv::Mat & input); //预处理，得到二值图
  std::vector<Light> findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img); //检测灯条
  std::vector<Armor> matchLights(const std::vector<Light> & lights); //匹配灯条，得到装甲板

  //调试
  cv::Mat getAllNumbersImage(); //获取所有装甲板数字图像的拼接图
  void drawResults(cv::Mat & img); //在图像上绘制检测结果

  int binary_thres;
  int detect_color;
  LightParams l; //灯条参数
  ArmorParams a; //装甲板参数

  //std::unique_ptr<NumberClassifier> classifier;

  

private:
  bool isLight(const Light & possible_light); //判断是否为灯条
  bool containLight(
    const Light & light_1, const Light & light_2, const std::vector<Light> & lights); 
  ArmorType isArmor(const Light & light_1, const Light & light_2); //判断是否为装甲板及类型

  std::vector<Light> lights_; //检测到的灯条
  std::vector<Armor> armors_; //检测到的装甲板

};




#endif