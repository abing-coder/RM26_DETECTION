#pragma once

#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

// // 装甲板大小类型
enum ArmorType2{
    SMALL_ARMOR = 0,
    BIG_ARMOR = 1
};

// // 颜色B蓝 G绿 R红
// enum Color{
//     BLUE = 0,
//     GREEN = 1,
//     RED = 2
// };

// 识别状态
enum DetectorState{
	LIGHTS_NOT_FOUND = 0,
	LIGHTS_FOUND = 1,
	ARMOR_NOT_FOUND = 2,
	ARMOR_FOUND = 3
};

// 装甲板相关数据信息
class ArmorBox{
public:
	ArmorBox();

	~ArmorBox();
	
    int armorNum;                               // 装甲板上的数字（用SVM识别得到）
    // 供pnp解算和装甲板图像显示时使用
    void setArmor();                            // 提供给pnp解算时用，避免提前调用增加耗时
    bool aleadySetArmor = false;                // 用来判断pnp是否有调用setArmor，否的话识别器调试显示装甲板时也会调用
    std::vector<cv::Point2f> armorVertices_;    // tl->tr->br->bl; lightPoints: bl->tl->tr->br; PnP用
    ArmorType2 type;                             // 装甲板大小类型
    cv::Point2f center;                         // 装甲板中心
    cv::Vec3f rVec;                             // 相机系下的旋转向量
    cv::Vec3f tVec;                             // 相机系下的平移向量
    cv::Point3f tVec_;                          // 惯性系下的坐标
    float yaw;                                  // 惯性系下的装甲板朝向角（装甲板坐标系的z轴与惯性系的x轴形成的夹角）

};