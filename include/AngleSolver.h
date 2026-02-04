#pragma once
#include "ArmorBox.h"

// 姿态解算类
class AngleSolver{
public:
    AngleSolver();
    ~AngleSolver();

    // 通过xml文件设置相机内参和畸变参数
    bool setCameraParam(const std::string &filePath);

    // 设置大小装甲板尺寸，单位m
    void setArmorSize(ArmorType2 type, float width, float height);

    // PnP解算 点顺序：左上->右上->右下->左下
    void solve(ArmorBox &armor);

    // 将相机系下的装甲板三维点转成枪管系下的转角
    void getAngle(ArmorBox &armor);
    
    // 将相机系下的目标三维点转成枪管系下的转角
    void getAngle(cv::Point3f point, float &yaw, float &pitch);

    void getAngle(cv::Point2f point, float &yaw, float &pitch, 
                            const double &cx, const double &cy, 
                            const double &fx, const double &fy );

    // 将相机系下的三维点投影到像素系下的像素点
    cv::Point2f projectPoint(cv::Point3f point3D);

    cv::Mat CAMERA_MATRIX;    //IntrinsicMatrix		  fx,fy,cx,cy
    cv::Mat DISTORTION_COEFF; //DistortionCoefficients k1,k2,p1,p2,k3

private:
    //Object points in world coordinate
    std::vector<cv::Point3f> SMALL_ARMOR_POINTS_3D;
    std::vector<cv::Point3f> BIG_ARMOR_POINTS_3D;

    //distance between camera and barrel in xyz axis
    float GUN_CAM_DISTANCE_X;  // 前
    float GUN_CAM_DISTANCE_Y;  // 左
    float GUN_CAM_DISTANCE_Z;  // 上
};