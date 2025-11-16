#ifndef __PNP_HPP_
#define __PNP_HPP_


#include <opencv2/core.hpp>
#include <array>
#include <vector>
#include "armor.hpp"


class PnPSolver
{
public:
  PnPSolver(const std::array<double, 9> & camera_matrix, //相机内参数矩阵
    const std::vector<double> & distortion_coefficients); //畸变系数

  // Get 3d position
  bool solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec); // 求解pnp

  // Calculate the distance between armor center and image center
  float calculateDistanceToCenter(const cv::Point2f & image_point); // 计算装甲板中心到图像中心的距离

private:
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;

  // Unit: mm
  static constexpr float SMALL_ARMOR_WIDTH = 135;
  static constexpr float SMALL_ARMOR_HEIGHT = 55;
  static constexpr float LARGE_ARMOR_WIDTH = 225;
  static constexpr float LARGE_ARMOR_HEIGHT = 55;

  // Four vertices of armor in 3d
  std::vector<cv::Point3f> small_armor_points_; // 小装甲板
  std::vector<cv::Point3f> large_armor_points_; // 大装甲板
};



#endif  