#include "pnp.hpp"

//相机内参矩阵
std::array<double, 9> camera_matrix;
//畸变系数
std::vector<double>dist_coeffs;

PnPSolver::PnPSolver(
  const std::array<double, 9> & camera_matrix, const std::vector<double> & dist_coeffs)
: camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
  dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone())
{
  // Unit: m
  constexpr double small_half_y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0; 
  constexpr double small_half_z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  constexpr double large_half_y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double large_half_z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // Start from bottom left in clockwise order
  // Model coordinate: x forward, y left, z up (右手坐标系)
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, -small_half_z)); 
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, -small_half_z));

  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, -large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, -large_half_z));
}

bool PnPSolver::solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec)
{
  std::vector<cv::Point2f> image_armor_points;

  // Fill in image points
  image_armor_points.emplace_back(armor.left_light.bottom);
  image_armor_points.emplace_back(armor.left_light.top);
  image_armor_points.emplace_back(armor.right_light.top);
  image_armor_points.emplace_back(armor.right_light.bottom);

  // Solve pnp
  auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
  return cv::solvePnP(
    object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);
}

float PnPSolver::calculateDistanceToCenter(const cv::Point2f & image_point)
{
  float cx = camera_matrix_.at<double>(0, 2);
  float cy = camera_matrix_.at<double>(1, 2);
  return cv::norm(image_point - cv::Point2f(cx, cy));
}

// 旋转向量转欧拉角
void PnPSolver::rvecToEuler(const cv::Mat& rvec, double& pitch, double& yaw, double& roll) {
    cv::Mat rotMat;
    cv::Rodrigues(rvec, rotMat); // 旋转向量 -> 旋转矩阵
    // 计算 yaw (绕 Y 轴)
    yaw = asin(-rotMat.at<double>(2, 0));
    // 计算 pitch (绕 X 轴) 和 roll (绕 Z 轴)
    if (cos(yaw) > 1e-6) { 
        pitch = atan2(rotMat.at<double>(2, 1), rotMat.at<double>(2, 2));
        roll = atan2(rotMat.at<double>(1, 0), rotMat.at<double>(0, 0));
    } else {
        // 特殊情况：当 yaw 为 ±90 度时
        pitch = 0.0;
        roll = atan2(-rotMat.at<double>(0, 1), rotMat.at<double>(1, 1));
    }
}
 
// 计算距离
double PnPSolver::calculateDistance(const cv::Mat& tvec) {
    double tx = tvec.at<double>(0);
    double ty = tvec.at<double>(1);
    double tz = tvec.at<double>(2);
    
    return sqrt(tx*tx + ty*ty + tz*tz); //单位：米
}
