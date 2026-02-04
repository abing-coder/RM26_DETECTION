#ifndef TRACLER
#define TRACLER

#include "ArmorBox.h"
#include "AngleSolver.h"
#include "kalman_filter.h"

using namespace cv;
using namespace std;

namespace rm
{

// 跟踪状态
enum TrackState{
    LOST = 0,
    DETECTING = 1,
    TRACKING = 2,
    TEMPLOST = 3,
};

// 目标装甲板的数量
enum ArmorsNum{
    NORMAL_4 = 4,
    BALANCE_2 = 2,
    OUTPOST_3 = 3,
    BASE_1 = 1
};

//用于存储目标装甲板的信息
struct tar_pos
{
    float x;           // 装甲板在世界坐标系下的x
    float y;           // 装甲板在世界坐标系下的y
    float z;           // 装甲板在世界坐标系下的z
    float yaw;         // 装甲板坐标系相对于世界坐标系的yaw角
};

class Tracker{
public:
    Tracker();
    ~Tracker();

    // 设置弹丸速度mm/s
    void setBulletSpeed(int &bulletSpeed);
    // 操作手用，设置目标装甲板数字
    void setTargetNum(int &targetNum);

    bool run(cv::Mat src, std::vector<ArmorBox> armors, int64 timestamp, int64 cvTickCount, float imu_yaw, float imu_pitch, float &add_yaw, float &add_pitch, int &fire);

    cv::Mat show(float add_yaw = 0, float add_pitch = 0, int width = 0);

    AngleSolver solver;

    std::vector<ArmorBox> armors;
    std::vector<cv::Point3f> pre_armors;
    struct tar_pos tar_position[4];

    std::ofstream csv_file;

    cv::Rect ROI;
    cv::Point2f ROI_ratio;

private:
    cv::Mat srcImg;

    int targetNum;          // 操作手设定的目标装甲板数字
    int bulletSpeed;        // 弹丸速度m/s

    // ekf
    ExtendedKalmanFilter ekf;
    Eigen::VectorXd measurement;
    Eigen::VectorXd targetState;
    double s2qxyz_, s2qyaw_, s2qr_;
    double r_xyz_factor, r_yaw;

    // To store another pair of armors message
    double dz, another_r;

    int trackedID;          // 1英雄 2工程 3、4、5步兵 6哨兵 7基地 8前哨站
    TrackState state;
    ArmorsNum armorsNum;
    ArmorBox trackedArmor;

    double dt;
    float useTime;  // ms
    int64 lastTimestamp;
    float last_x, last_dx;

    int detectCount;
    int lostCount;
    int trackingThres;
    int lostThres;
    double lost_time_thres;

    double info_position_diff;
    double info_yaw_diff;
    double max_match_distance;
    double max_match_yaw_diff;

    void init();
    void update();
    void initEKF(ArmorBox &armor);
    void updateArmorsNum(ArmorBox &armor);
    void armorJump(ArmorBox &armor);
    float getArmorYaw(ArmorBox &armor);
    Eigen::Vector3d getArmorPositionFromState(Eigen::VectorXd &x);

    // 坐标变换
    float distance;
    float imu_yaw, imu_pitch, last_imu_yaw, last_yaw;
    Eigen::Matrix3f R;                                      // 旋转矩阵；惯性系到相机系；转置R.transpose()为相机系到惯性系；P_world = R.transpose() * P_cam
    void getR();                                            // 获取旋转矩阵R
    void cam2imu(ArmorBox &armor);                          // 相机系到惯性系
    void solve(ArmorBox &armor);                            // pnp解算并转到惯性系
    cv::Point3f imu2cam(cv::Point3f tVec);                  // 惯性系到相机系
    cv::Point2f imu2img(cv::Point3f tVec);                  // 惯性坐标转图像坐标
    void getAngle(Point3f point, float &yaw, float &pitch); // 将惯性系下的三维坐标转成发送给下位机的角度（弧度制）
};

}

#endif // TRACKER
