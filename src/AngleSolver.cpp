#include "AngleSolver.h"

using namespace cv;
using namespace std;

AngleSolver::AngleSolver(){}

AngleSolver::~AngleSolver(){}

// 通过xml文件设置相机内参、畸变参数和枪管相机偏移
bool AngleSolver::setCameraParam(const std::string &filePath){
    FileStorage fsRead;
    fsRead.open(filePath, FileStorage::READ);
    if (!fsRead.isOpened()){
        cout << "Failed to open xml" << endl;
        return false;
    }

    // 相机系为base系（向前Z+、向右X+、向下Y+），摩擦轮与弹丸的接触点在base的哪个方向
    fsRead["X_GUN_AND_CAM"] >> GUN_CAM_DISTANCE_X;     //-left +right
    fsRead["Y_GUN_AND_CAM"] >> GUN_CAM_DISTANCE_Y;     //-up +down
    fsRead["Z_GUN_AND_CAM"] >> GUN_CAM_DISTANCE_Z;     //+front -back

    Mat camera_matrix;
    Mat distortion_coeff;
    fsRead["CAMERA_MATRIX"] >> camera_matrix;
    fsRead["DISTORTION_COEFF"] >> distortion_coeff;

    camera_matrix.copyTo(CAMERA_MATRIX);
    distortion_coeff.copyTo(DISTORTION_COEFF);
    fsRead.release();
    return true;
}

// 设置大小装甲板尺寸，单位m，保证方向与相机系一致，后期求欧拉角不容易混淆
void AngleSolver::setArmorSize(ArmorType2 type, float width, float height){
    float half_x = width / 2.0;
    float half_y = height / 2.0;
    switch (type){
        case SMALL_ARMOR:
            SMALL_ARMOR_POINTS_3D.emplace_back(Point3f(-half_x, -half_y, 0));  //tl top left
            SMALL_ARMOR_POINTS_3D.emplace_back(Point3f(half_x, -half_y, 0));   //tr top right
            SMALL_ARMOR_POINTS_3D.emplace_back(Point3f(half_x, half_y, 0));    //br below right
            SMALL_ARMOR_POINTS_3D.emplace_back(Point3f(-half_x, half_y, 0));   //bl below left
            break;
        case BIG_ARMOR:
            BIG_ARMOR_POINTS_3D.emplace_back(Point3f(-half_x, -half_y, 0));    //tl top left
            BIG_ARMOR_POINTS_3D.emplace_back(Point3f(half_x, -half_y, 0));	   //tr top right
            BIG_ARMOR_POINTS_3D.emplace_back(Point3f(half_x, half_y, 0));      //br below right
            BIG_ARMOR_POINTS_3D.emplace_back(Point3f(-half_x, half_y, 0));     //bl below left
            break;
        default: break;
    }
}

// PnP解算 点顺序：左上->右上->右下->左下
void AngleSolver::solve(ArmorBox &armor){
    armor.setArmor();
    switch (armor.type){
        case SMALL_ARMOR:
            solvePnP(SMALL_ARMOR_POINTS_3D, armor.armorVertices_, CAMERA_MATRIX, DISTORTION_COEFF, armor.rVec, armor.tVec, false, SOLVEPNP_IPPE);  // SOLVEPNP_ITERATIVE
            // 如果解出nan，则用迭代法求解
            if (armor.tVec[0] != armor.tVec[0]){
                solvePnP(SMALL_ARMOR_POINTS_3D, armor.armorVertices_, CAMERA_MATRIX, DISTORTION_COEFF, armor.rVec, armor.tVec, false, SOLVEPNP_ITERATIVE);
            }
            break;
        case BIG_ARMOR:
            solvePnP(BIG_ARMOR_POINTS_3D, armor.armorVertices_, CAMERA_MATRIX, DISTORTION_COEFF, armor.rVec, armor.tVec, false, SOLVEPNP_IPPE);
            if (armor.tVec[0] != armor.tVec[0]){
                solvePnP(BIG_ARMOR_POINTS_3D, armor.armorVertices_, CAMERA_MATRIX, DISTORTION_COEFF, armor.rVec, armor.tVec, false, SOLVEPNP_ITERATIVE);
            }
            break;
        default:break;
    }
}

// 将相机系下的目标三维点转成枪管系下的转角（弧度制）
void AngleSolver::getAngle(Point3f point, float &yaw, float &pitch){
    float x_pos = point.x;
    float y_pos = point.y;
    float z_pos = point.z;
    // 在相机系下枪管和相机的偏移补偿，单位m
    x_pos -= GUN_CAM_DISTANCE_X;
    y_pos -= GUN_CAM_DISTANCE_Y;
    z_pos -= GUN_CAM_DISTANCE_Z;
    // 转角转换
    float tan_pitch = y_pos / sqrt(x_pos * x_pos + z_pos * z_pos);
    float tan_yaw = x_pos / z_pos;
    pitch = atan(tan_pitch);
    yaw = atan(tan_yaw);
}

// 将相机系下的目标二维点转成枪管系下的转角（弧度制）
void AngleSolver::getAngle(cv::Point2f point, float &yaw, float &pitch, 
                            const double &cx, const double &cy, 
                            const double &fx, const double &fy ){

    float x_pos = (point.x-cx)/fx;
    float y_pos = (point.y-cy)/fy;
    float z_pos = 1;
    // 在相机系下枪管和相机的偏移补偿，单位m
    x_pos -= 0;
    y_pos -= 0;
    z_pos -= 0;
    // 转角转换
    float tan_pitch = y_pos / sqrt(x_pos * x_pos + z_pos * z_pos);
    float tan_yaw = x_pos / z_pos;
    pitch = atan(tan_pitch);
    yaw = atan(tan_yaw);
}

// 将相机系下的三维点投影到像素系下的像素点
Point2f AngleSolver::projectPoint(cv::Point3f point3D){
    std::vector<cv::Point3f> objectPoints;
    objectPoints.push_back(point3D);
    std::vector<cv::Point2f> imagePoints;
    // 由于是相机坐标系下的三维点，旋转和平移向量都定义为0
    Vec3f rVec(0,0,0), tVec(0,0,0);
    // 调用projectPoints
    projectPoints(objectPoints, rVec, tVec, CAMERA_MATRIX, DISTORTION_COEFF, imagePoints);
    return imagePoints[0];
}
