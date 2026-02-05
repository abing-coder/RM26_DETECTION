#include "tracker.h"

using namespace cv;
using namespace std;

namespace rm
{

bool armorCompare(const ArmorBox &a_armor, const ArmorBox &b_armor, const Mat &src, const int &targetNum);

Tracker::Tracker():
    measurement(Eigen::VectorXd::Zero(4)),
    targetState(Eigen::VectorXd::Zero(9))
{
    // init
    state = LOST;
    trackedID = 0;
    targetNum = 0;
    bulletSpeed = 30;
    lastTimestamp = 0;
    last_imu_yaw = 0;
    detectCount = 0;
    lostCount = 0;

    // ekf
    s2qxyz_ = 0.05;             // 0.05
    s2qyaw_ = 5.0;              // 5.0
    s2qr_ = 80.0;               // 80.0
    r_xyz_factor = 4e-4;        // 4e-4
    r_yaw = 5e-3;               // 5e-3

    // tracker
    max_match_distance = 0.5;   // 单位米，默认0.5
    max_match_yaw_diff = 0.1;   // 弧度制，默认1.0
    trackingThres = 5;          // 累加数，默认5
    lost_time_thres = 1.0;      // 单位秒，默认1.0

    // 初始化EKF
    // xa = x_armor, xc = x_robot_center
    // state: xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r
    // measurement: xa, ya, za, yaw
    // f - Process function
    auto f = [this](const Eigen::VectorXd & x) {
        Eigen::VectorXd x_new = x;
        x_new(0) += x(1) * dt;
        x_new(2) += x(3) * dt;
        x_new(4) += x(5) * dt;
        x_new(6) += x(7) * dt;
        return x_new;
    };
    // J_f - Jacobian of process function
    auto j_f = [this](const Eigen::VectorXd &) {
        Eigen::MatrixXd f(9, 9);
        // clang-format off
        f <<  1,   dt, 0,   0,   0,   0,   0,   0,   0,
              0,   1,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   1,   dt, 0,   0,   0,   0,   0,
              0,   0,   0,   1,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   1,   dt, 0,   0,   0,
              0,   0,   0,   0,   0,   1,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   1,   dt, 0,
              0,   0,   0,   0,   0,   0,   0,   1,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   1;
        // clang-format on
        return f;
    };
    // h - Observation function
    auto h = [](const Eigen::VectorXd & x) {
        Eigen::VectorXd z(4);
        double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
        z(0) = xc - r * cos(yaw);  // xa
        z(1) = yc - r * sin(yaw);  // ya
        z(2) = x(4);               // za
        z(3) = x(6);               // yaw
        return z;
    };
    // J_h - Jacobian of observation function
    auto j_h = [](const Eigen::VectorXd & x) {
        Eigen::MatrixXd h(4, 9);
        double yaw = x(6), r = x(8);
        // clang-format off
        //    xc   v_xc yc   v_yc za   v_za yaw         v_yaw r
        h <<  1,   0,   0,   0,   0,   0,   r*sin(yaw), 0,   -cos(yaw),
              0,   0,   1,   0,   0,   0,   -r*cos(yaw),0,   -sin(yaw),
              0,   0,   0,   0,   1,   0,   0,          0,   0,
              0,   0,   0,   0,   0,   0,   1,          0,   0;
        // clang-format on
        return h;
    };
    // update_Q - process noise covariance matrix
    auto u_q = [this]() {
        Eigen::MatrixXd q(9, 9);
        double t = dt, x = s2qxyz_, y = s2qyaw_, r = s2qr_;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * y;
        double q_r = pow(t, 4) / 4 * r;
        // clang-format off
        //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
        q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
              q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
              0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
              0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
              0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
              0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
              0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
              0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
              0,      0,      0,      0,      0,      0,      0,      0,      q_r;
        // clang-format on
        return q;
    };
    // update_R - measurement noise covariance matrix
    auto u_r = [this](const Eigen::VectorXd & z) {
        Eigen::DiagonalMatrix<double, 4> r;
        double x = r_xyz_factor;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw;
        return r;
    };
    // P - error estimate covariance matrix
    Eigen::DiagonalMatrix<double, 9> p0;
    p0.setIdentity();
    ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};
}

Tracker::~Tracker() {}

// 设置弹丸速度m/s
void Tracker::setBulletSpeed(int &bulletSpeed) {
    this->bulletSpeed = bulletSpeed;
}

// 操作手用，设置目标装甲板数字
void Tracker::setTargetNum(int &targetNum) {
    if (targetNum != 0 && this->targetNum != targetNum) {
        state = LOST;
    }
    this->targetNum = targetNum;
}

bool Tracker::run(Mat src, vector<ArmorBox> armors, int64 timestamp, int64 cvTickCount, float imu_yaw, float imu_pitch, float &add_yaw, float &add_pitch, int &fire) {
    src.copyTo(srcImg);
    this->armors = armors;
    this->imu_yaw = imu_yaw;
    this->imu_pitch = imu_pitch;
    add_yaw = 0;
    add_pitch = 0;
    fire = 0;

    // 存在装甲板时更新旋转矩阵R
    if (!armors.empty()) getR();

    if (state == LOST) {
        init();
    } else {
        dt = (timestamp - lastTimestamp) / 1000.0;
        // 限制dt的范围，防止异常值导致预测不稳定
        if (dt <= 0 || dt > 0.1) {
            // 如果dt异常，使用一个合理的默认值（例如16ms，对应60fps）
            dt = 0.016;
        }
        // 限制dt的最小值，避免除零错误
        if (dt < 0.001) {
            dt = 0.001;
        }
        lostThres = static_cast<int>(lost_time_thres / dt);
        // 更新
        update();

        if (state == TRACKING || state == TEMPLOST) {
            double yaw = targetState(6), v_yaw = targetState(7), r1 = targetState(8), r2 = another_r;
            double xc = targetState(0), yc = targetState(2), za = targetState(4);
            double vx = targetState(1), vy = targetState(3), vz = targetState(5);

            // std::cout << "EKF_yaw ---> " << yaw << endl;
            // std::cout << "EKF_xc ---> " << xc << endl;
            // std::cout << "EKF_yc ---> " << yc << endl;

            //计算四块装甲板的位置
            //装甲板id顺序，以四块装甲板为例，逆时针编号
            //      2
            //   3     1
            //      0
            double r = 0;
            bool is_current_pair = true;
            size_t a_n = armorsNum;
            pre_armors.clear();
            if (a_n != BASE_1) {
                Point3f centerPoint(xc, yc, za + dz / 2);
                pre_armors.emplace_back(centerPoint);
            }
            for (size_t i = 0; i < a_n; i++) {
                double tmp_yaw = yaw + i * (2 * CV_PI / a_n);
                Point3f p_a;
                // Only 4 armors has 2 radius and height
                if (a_n == 4) {
                    r = is_current_pair ? r1 : r2;
                    p_a.z = za + (is_current_pair ? 0 : dz);
                    is_current_pair = !is_current_pair;
                } else {
                    r = r1;
                    p_a.z = za;
                }
                p_a.x = xc - r * cos(tmp_yaw);
                p_a.y = yc - r * sin(tmp_yaw);
                pre_armors.emplace_back(p_a);
            }

            // 选择打击目标
            if (distance > 0.5) {
                double delay_time = distance / bulletSpeed + 0.08;    // 延迟时间s：子弹飞行时间+开火时间
                double new_yaw = yaw + v_yaw * delay_time;
                for (size_t i = 0; i < a_n; i++) {
                    double tmp_yaw = new_yaw + i * (2 * CV_PI / a_n);
                    Point3f p_a;
                    // Only 4 armors has 2 radius and height
                    if (a_n == 4) {
                        r = is_current_pair ? r1 : r2;
                        p_a.z = za + (is_current_pair ? 0 : dz);
                        is_current_pair = !is_current_pair;
                    } else {
                        r = r1;
                        p_a.z = za;
                    }
                    p_a.x = xc - r * cos(tmp_yaw);
                    p_a.y = yc - r * sin(tmp_yaw);
                    tar_position[i].x = p_a.x;
                    tar_position[i].y = p_a.y;
                    tar_position[i].z = p_a.z;
                    // 将yaw角转到[-pi, pi]范围内
                    tmp_yaw = fmod(tmp_yaw, CV_2PI);
                    if (tmp_yaw >= CV_PI) {
                        tmp_yaw -= CV_2PI;
                    } else if (tmp_yaw < -CV_PI) {
                        tmp_yaw += CV_2PI;
                    }
                    tar_position[i].yaw = tmp_yaw;
                }

                // 采集数据
                if (csv_file.is_open()) {
                    double tmp_yaw = fmod(yaw, CV_2PI);
                    if (tmp_yaw >= CV_PI) {
                        tmp_yaw -= CV_2PI;
                    } else if (tmp_yaw < -CV_PI) {
                        tmp_yaw += CV_2PI;
                    }
                    tmp_yaw -= imu_yaw;
                    Mat R;
                    Rodrigues(trackedArmor.rVec, R);
                    Mat mtxR, mtxQ, Qx, Qy, Qz;
                    RQDecomp3x3(R.t(), mtxR, mtxQ, Qx, Qy, Qz);
                    csv_file << timestamp << "," << asin(-Qy.at<float>(0, 2)) << "," << tmp_yaw << "\n";
                }

                // 选板
                int idx = 0;
                // 计算枪管到目标装甲板yaw最小的那个装甲板
                float yaw_diff_min = fabsf(tar_position[0].yaw - imu_yaw);
                for (size_t i = 1; i < a_n; i++) {
                    float temp_yaw_diff = fabsf(tar_position[i].yaw - imu_yaw);
                    if (temp_yaw_diff < yaw_diff_min) {
                        yaw_diff_min = temp_yaw_diff;
                        idx = i;
                    }
                }
                //计算距离最近的装甲板
                //float dis_diff_min = sqrt(tar_position[0].x * tar_position[0].x + tar_position[0].y * tar_position[0].y);
                //for (size_t i = 1; i < a_n; i++) {
                //    float temp_dis_diff = sqrt(tar_position[i].x * tar_position[0].x + tar_position[i].y * tar_position[0].y);
                //    if (temp_dis_diff < dis_diff_min) {
                //        dis_diff_min = temp_dis_diff;
                //        idx = i;
                //    }
                //}

                // 云台运动模式
                // 跟随模式
                Point3f aim_point(tar_position[idx].x, tar_position[idx].y, tar_position[idx].z);
                aim_point.x += vx * delay_time;
                aim_point.y += vy * delay_time;
                aim_point.z += vz * delay_time;
                getAngle(aim_point, add_yaw, add_pitch);
                if (abs(tar_position[idx].yaw) <= 0.01) {
                    fire = 1;
                }
                // 瞄中模式
                //Point3f aim_point = pre_armors[0];
                //aim_point.x += vx * delay_time;
                //aim_point.y += vy * delay_time;
                //aim_point.z += vz * delay_time;
                //getAngle(aim_point, add_yaw, add_pitch);
                //Point3f pre_point(tar_position[idx].x, tar_position[idx].y, tar_position[idx].z);
                //pre_point.x += vx * delay_time;
                //pre_point.y += vy * delay_time;
                //pre_point.z += vz * delay_time;
                //float pre_yaw, pre_pitch;
                //getAngle(pre_point, pre_yaw, pre_pitch);
                //add_pitch = pre_pitch;
                //if (abs(pre_yaw - add_yaw) < 0.003) {
                //    fire = 1;
                //}
            }

            // 打印信息
            // cout << "\n********************************************" << endl;
            // cout << "id: " << trackedID << endl;
            // cout << "pos_diff: " << info_position_diff << endl;
            // cout << "yaw_diff: " << info_yaw_diff << endl;
            // cout << "yaw: " << yaw * 180 / CV_PI << " r1: " << r1 << " r2: " << r2 << endl;
            // cout << "********************************************\n" << endl;
        }
    }
    lastTimestamp = timestamp;
    last_imu_yaw = imu_yaw;
    useTime = (getTickCount() - cvTickCount) / 1000000.0;
    return state == TRACKING;
}

void Tracker::init() {
    if (armors.empty()) return;
    // 选板
    armors[0].setArmor();
    ArmorBox mva = armors[0];
    for (size_t i = 1; i < armors.size(); i++) {
        armors[i].setArmor();
        if (armorCompare(armors[i], mva, srcImg, targetNum)) {mva = armors[i];}
    }
    // 更新targetArmor
    trackedArmor = mva;
    // 初始化卡尔曼
    initEKF(trackedArmor);
    // 更新状态
    trackedID = trackedArmor.armorNum;
    state = DETECTING;
    last_x = trackedArmor.tVec[0];
    last_dx = 0;
    distance = sqrt(trackedArmor.tVec[0] * trackedArmor.tVec[0] + trackedArmor.tVec[1] * trackedArmor.tVec[1] + trackedArmor.tVec[2] * trackedArmor.tVec[2]);
    // 更新目标的装甲板数量
    updateArmorsNum(trackedArmor);
}

void Tracker::update() {
    // KF predict
    Eigen::VectorXd ekf_prediction = ekf.predict();
    bool matched = false;
    // Use KF prediction as default target state if no matched armor is found
    targetState = ekf_prediction;
    auto predicted_position = getArmorPositionFromState(ekf_prediction);
    if (!armors.empty()) {
        // Find the closest armor with the same id
        ArmorBox same_id_armor;
        int same_id_armors_count = 0;
        double min_position_diff = DBL_MAX;
        double yaw_diff = DBL_MAX;
        for (ArmorBox &armor : armors) {
            // Only consider armors with the same id
            if (armor.armorNum == trackedID) {
                // 姿态解算
                solve(armor);
                same_id_armors_count++;
                // Calculate the difference between the predicted position and the current armor position
                auto p = armor.tVec_;
                Eigen::Vector3d position_vec(p.x, p.y, p.z);
                double position_diff = (predicted_position - position_vec).norm();
                if (position_diff < min_position_diff) {
                    // Find the closest armor
                    min_position_diff = position_diff;
                    yaw_diff = abs(getArmorYaw(armor) - ekf_prediction(6));
                    trackedArmor = armor;
                    if (same_id_armors_count == 1) {
                        same_id_armor = armor;
                    }
                } else {
                    same_id_armor = armor;
                }
            }
        }

        // Store tracker info
        info_position_diff = min_position_diff;
        info_yaw_diff = yaw_diff;

        // 设置ROI，只针对高度
        // if (same_id_armors_count > 1) {
        //     std::vector<cv::Point2f> ROI_;
        //     ROI_.resize(8);
        //     ROI_[0] = same_id_armor.armorVertices[0];
        //     ROI_[1] = same_id_armor.armorVertices[1];
        //     ROI_[2] = same_id_armor.armorVertices[2];
        //     ROI_[3] = same_id_armor.armorVertices[3];
        //     ROI_[4] = trackedArmor.armorVertices[0];
        //     ROI_[5] = trackedArmor.armorVertices[1];
        //     ROI_[6] = trackedArmor.armorVertices[2];
        //     ROI_[7] = trackedArmor.armorVertices[3];
        //     ROI = boundingRect(ROI_);
        //     float ratio = 0.8;
        //     ROI.x = 0;
        //     ROI.y -= ROI.height * ratio;
        //     ROI.width = srcImg.cols;
        //     ROI.height += ROI.height * ratio * 2;
        //     trackedArmor.armorRect = boundingRect(trackedArmor.armorVertices);
        //     ROI_ratio.y = ROI.height / trackedArmor.armorRect.height;
        // } else if (same_id_armors_count == 1) {
        //     trackedArmor.armorRect = boundingRect(trackedArmor.armorVertices);
        //     if (ROI_ratio.y == 0) {
        //         ROI.width = srcImg.cols;
        //         ROI.height = 3 * trackedArmor.armorRect.height;
        //         ROI.x = 0;
        //         ROI.y = trackedArmor.center.y - ROI.height / 2;
        //     } else {
        //         ROI.width = srcImg.cols;
        //         ROI.height = ROI_ratio.y * trackedArmor.armorRect.height;
        //         ROI.x = 0;
        //         ROI.y = trackedArmor.center.y - ROI.height / 2;
        //     }
        // }

        // 计算相机系下两帧间目标x方向移动的距离，单位m
        float dx = 0;
        if (same_id_armors_count >= 1) {
            dx = trackedArmor.tVec[0] - last_x;
            last_x = trackedArmor.tVec[0];
            distance = sqrt(trackedArmor.tVec[0] * trackedArmor.tVec[0] + trackedArmor.tVec[1] * trackedArmor.tVec[1] + trackedArmor.tVec[2] * trackedArmor.tVec[2]);
            // cout << same_id_armors_count << " dx: " << dx << " yaw_diff: " << yaw_diff << "\n" << endl;
        }

        // cout << "\n************************************************************" <<endl;
        // cout << "same_id_armors_count:" << same_id_armors_count  << "   bool: " << (same_id_armors_count >= 1) << endl;
        // cout << " dx: " << dx << " ---> last_dx: " << last_dx << "   bool: " << (abs(dx) > abs(last_dx)) << endl;
        // cout << " yaw_diff: " << yaw_diff << " ---> max_match_yaw_diff: " << max_match_yaw_diff << "   bool: " << (yaw_diff > max_match_yaw_diff)<< endl;
        // cout << "************************************************************\n" <<endl;

        if (same_id_armors_count >= 1 && abs(dx) > 0.15 && abs(dx) > abs(last_dx) && yaw_diff > max_match_yaw_diff) {
            // cout << " ####################### jumping! #######################" <<endl;
            armorJump(trackedArmor);
        } else if (min_position_diff < max_match_distance) {
            // Matched armor found
            matched = true;
            auto p = trackedArmor.tVec_; // 有可能是这里的问题
            // cout << "p --> " << p << endl;
            // Update EKF
            measurement = Eigen::Vector4d(p.x, p.y, p.z, trackedArmor.yaw);
            cout << "\tp.x: " << p.x << "\tp.y: " << p.y << "\tp.z: " << p.z<<"\ttrackedArmor.yaw: " <<trackedArmor.yaw<<endl;
            targetState = ekf.update(measurement);
            last_dx = dx;
            if (same_id_armors_count == 2) {
                dz = same_id_armor.tVec_.z - p.z;
            }
            // cout << "################ EKF update ################" << endl;
        } else {
            // cout << "############### No matched armor found! ##################" << endl;
        }
    }

    // Prevent radius from spreading
    if (targetState(8) < 0.12) {
        targetState(8) = 0.12;
        ekf.setState(targetState);
    } else if (targetState(8) > 0.4) {
        targetState(8) = 0.4;
        ekf.setState(targetState);
    }

    // Tracking state machine
    if (state == DETECTING) {
        if (matched) {
            detectCount++;
            if (detectCount > trackingThres) {
                detectCount = 0;
                state = TRACKING;
            }
        } else {
            detectCount = 0;
            state = LOST;
        }
    } else if (state == TRACKING) {
        if (!matched) {
            state = TEMPLOST;
            lostCount++;
        }
    } else if (state == TEMPLOST) {
        if (!matched) {
            lostCount++;
            if (lostCount > lostThres) {
                lostCount = 0;
                state = LOST;
                ROI = Rect(0, 0, 0, 0);
                ROI_ratio = Point2f(0, 0);
            }
        } else {
            state = TRACKING;
            lostCount = 0;
        }
    }
}

cv::Mat Tracker::show(float add_yaw, float add_pitch, int width) {
    if (state == TRACKING || state == TEMPLOST) {
        // 绘制预测的装甲板中心点
        for (size_t i = 0; i < armorsNum + 1; i++) {
            Point2f armor_point = imu2img(pre_armors[i]);
            // circle(srcImg, armor_point, 2, Scalar(255, 255, 0), 15);
            if (armorsNum == BASE_1) break;
        }
        
        // 可视化四块装甲板四边形（与上面黄色中心点 pre_armors 一致，不依赖 distance）
        if (armorsNum == NORMAL_4 && pre_armors.size() >= 5) {
            float armor_width = 0.135f;
            float armor_height = 0.055f;
            double yaw = targetState(6);  // 当前 EKF yaw，与 run() 里算 pre_armors 一致
            for (size_t i = 0; i < 4; i++) {
                double armor_x = pre_armors[i + 1].x;
                double armor_y = pre_armors[i + 1].y;
                double armor_z = pre_armors[i + 1].z;
                double armor_yaw = yaw + i * (2 * CV_PI / 4);  // 从圆心到该板的方向角
                double perp_x = -sin(armor_yaw);
                double perp_y = cos(armor_yaw);
                std::vector<Point3f> armor_corners_local(4);
                armor_corners_local[0] = Point3f(0, -armor_width/2, -armor_height/2);
                armor_corners_local[1] = Point3f(0, armor_width/2, -armor_height/2);
                armor_corners_local[2] = Point3f(0, armor_width/2, armor_height/2);
                armor_corners_local[3] = Point3f(0, -armor_width/2, armor_height/2);
                std::vector<Point2f> armor_corners_img(4);
                for (size_t j = 0; j < 4; j++) {
                    double local_y = armor_corners_local[j].y;
                    double local_z = armor_corners_local[j].z;
                    Point3f corner_3d;
                    corner_3d.x = armor_x + local_y * perp_x;
                    corner_3d.y = armor_y + local_y * perp_y;
                    corner_3d.z = armor_z + local_z;
                    armor_corners_img[j] = imu2img(corner_3d);
                }
                Scalar armor_color(0, 255, 255);
                int thickness = 1;
                for (size_t j = 0; j < 4; j++)
                    //circle(srcImg, armor_corners_img[j], 3, armor_color, -1);
                line(srcImg, armor_corners_img[0], armor_corners_img[1], armor_color, thickness);
                line(srcImg, armor_corners_img[1], armor_corners_img[2], armor_color, thickness);
                line(srcImg, armor_corners_img[2], armor_corners_img[3], armor_color, thickness);
                line(srcImg, armor_corners_img[3], armor_corners_img[0], armor_color, thickness);
                // line(srcImg, armor_corners_img[0], armor_corners_img[2], armor_color, 1);
                // line(srcImg, armor_corners_img[1], armor_corners_img[3], armor_color, 1);
                Point2f center_img = imu2img(Point3f(armor_x, armor_y, armor_z));
                putText(srcImg, "P" + to_string(i), center_img + Point2f(-8, 4),
                       FONT_HERSHEY_SIMPLEX, 0.5, armor_color, 1);
            }
        }
        
        // 将角度转成像素坐标
        Point2f aim_point = solver.projectPoint(Point3f(tan(add_yaw), tan(add_pitch), 1));
        circle(srcImg, aim_point, 2, Scalar(255, 255, 255), 20);
    }
    std::stringstream latency_ss;
    latency_ss << "Latency: " << std::fixed << std::setprecision(2) << useTime*100 << "ms";
    auto latency_s = latency_ss.str();
    putText(srcImg, latency_s, Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    putText(srcImg, "Time: " + to_string(lastTimestamp) + "ms", Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(srcImg, "Distance: " + to_string(distance) + "m", Point(10, 80), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(srcImg, "State:", Point(10, 110), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    if (state == LOST) {
        putText(srcImg, "LOST", Point(100, 110), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    } else if (state == DETECTING) {
        putText(srcImg, "DETECTING", Point(100, 110), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    } else if (state == TRACKING) {
        putText(srcImg, "TRACKING", Point(100, 110), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    } else if (state == TEMPLOST) {
        putText(srcImg, "TEMPLOST", Point(100, 110), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    }

    if (!armors.empty()) {
        for (auto armor : armors) {
            armor.setArmor();
            Scalar color(255, 255, 0);
            if (armor.center == trackedArmor.center) {
                color = Scalar(0, 255, 0);
            }
            circle(srcImg, armor.center, 2, color, 5);
            circle(srcImg, armor.armorVertices_[0], 1, color, 2);
            circle(srcImg, armor.armorVertices_[1], 1, color, 2);
            circle(srcImg, armor.armorVertices_[2], 1, color, 2);
            circle(srcImg, armor.armorVertices_[3], 1, color, 2);
            line(srcImg, armor.armorVertices_[0], armor.armorVertices_[2], color, 1, 8, 0);
            line(srcImg, armor.armorVertices_[1], armor.armorVertices_[3], color, 1, 8, 0);
            line(srcImg, armor.armorVertices_[0], armor.armorVertices_[3], color, 1, 8, 0);
            line(srcImg, armor.armorVertices_[1], armor.armorVertices_[2], color, 1, 8, 0);
            putText(srcImg, "x:" + to_string(int(armor.center.x)), armor.center + Point2f(-20, -15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1, 8, false);
            putText(srcImg, "y:" + to_string(int(armor.center.y)), armor.center + Point2f(-20, 0), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1, 8, false);
            putText(srcImg, "num:" + to_string(int(armor.armorNum)), armor.center + Point2f(-20, 15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1, 8, false);
            if (armor.type == 0) {
                putText(srcImg, "type:S", armor.center + Point2f(-20, 30), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1, 8, false);
            } else {
                putText(srcImg, "type:B", armor.center + Point2f(-20, 30), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1, 8, false);
            }
        }
    }
    drawMarker(srcImg, Point(srcImg.cols/2, srcImg.rows/2), Scalar(255, 255, 255), 0, 50, 2);
    if (width) {
        int height = srcImg.rows * width / srcImg.cols;
        resize(srcImg, srcImg, Size(width, height));
    }
    return srcImg;
}

void Tracker::initEKF(ArmorBox &armor) {
    solve(armor);
    auto p = armor.tVec_;
    last_yaw = 0;
    getArmorYaw(armor);
    // Set initial position at 0.2m behind the target
    targetState = Eigen::VectorXd::Zero(9);
    double r = 0.26;
    double xc = p.x + r * cos(armor.yaw);
    double yc = p.y + r * sin(armor.yaw);
    dz = 0, another_r = r;
    targetState << xc, 0, yc, 0, p.z, 0, armor.yaw, 0, r;
    ekf.setState(targetState);
}

// 更新目标的装甲板数量
void Tracker::updateArmorsNum(ArmorBox &armor) {
    if (armor.type == BIG_ARMOR && (trackedID == 3 || trackedID == 4 || trackedID == 5)) armorsNum = BALANCE_2;
    else if (trackedID == 7) armorsNum = BASE_1;
    else if (trackedID == 8) armorsNum = OUTPOST_3;
    else armorsNum = NORMAL_4;
}

// 处理装甲板跳变
void Tracker::armorJump(ArmorBox &armor) {
    auto p = armor.tVec_;
    targetState(6) = armor.yaw;
    updateArmorsNum(armor);
    // Only 4 armors has 2 radius and height
    if (armorsNum == NORMAL_4) {
        dz = targetState(4) - p.z;
        targetState(4) = p.z;
        std::swap(targetState(8), another_r);
    }
    // cout << getTickCount() / 1000000 << "ms: " << "Armor jump!" << endl;

    // If position difference is larger than max_match_distance_,
    // take this case as the ekf diverged, reset the state
    Eigen::Vector3d current_p(p.x, p.y, p.z);
    Eigen::Vector3d infer_p = getArmorPositionFromState(targetState);
    if ((current_p - infer_p).norm() > max_match_distance) {
        double r = targetState(8);
        targetState(0) = p.x + r * cos(armor.yaw);  // xc
        targetState(1) = 0;                         // vxc
        targetState(2) = p.y + r * sin(armor.yaw);  // yc
        targetState(3) = 0;                         // vyc
        targetState(4) = p.z;                       // za
        targetState(5) = 0;                         // vza
        // cout << "Reset State!" << endl;
    }
    ekf.setState(targetState);
}

// 获取旋转矩阵R
void Tracker::getR() {
    // 将角度转换为弧度
    float a1 = CV_PI / 2;               // 90度
    float a2 = -CV_PI / 2 + imu_yaw;    // 左+右-
    float a3 = imu_pitch;               // 上-下+
    // 绕x轴旋转a1度的旋转矩阵
    Eigen::Matrix3f Rx;
    Rx << 1, 0, 0,
          0, cos(a1), -sin(a1),
          0, sin(a1), cos(a1);
    // 绕y轴旋转a2度的旋转矩阵
    Eigen::Matrix3f Ry;
    Ry << cos(a2), 0, sin(a2),
          0, 1, 0,
          -sin(a2), 0, cos(a2);
    // 绕x轴旋转a3度的旋转矩阵
    Eigen::Matrix3f Rx1;
    Rx1 << 1, 0, 0,
           0, cos(a3), -sin(a3),
           0, sin(a3), cos(a3);
    // 按XYX顺序，绕参考系的固定轴旋转(外旋，左乘)
    R = Rx1 * Ry * Rx;
}

// 三维位置：相机系转惯性系
void Tracker::cam2imu(ArmorBox &armor) {
    Eigen::Vector3f p(armor.tVec[0], armor.tVec[1], armor.tVec[2]);
    Eigen::Vector3f p1 = R.transpose() * p;
    armor.tVec_ = cv::Point3f(p1(0), p1(1), p1(2));
}

// pnp解算
void Tracker::solve(ArmorBox &armor) {
    solver.solve(armor);
    cam2imu(armor);
}

// 三维位置：惯性系转相机系
cv::Point3f Tracker::imu2cam(cv::Point3f tVec) {
    Eigen::Vector3f p(tVec.x, tVec.y, tVec.z);
    Eigen::Vector3f p1 = R * p;
    return cv::Point3f(p1(0), p1(1), p1(2));
}

// 惯性坐标转图像坐标
cv::Point2f Tracker::imu2img(cv::Point3f tVec) {
    return solver.projectPoint(imu2cam(tVec));
}

// 将惯性系下的三维坐标转成发送给下位机的角度（弧度制）
void Tracker::getAngle(Point3f point, float &yaw, float &pitch) {
    solver.getAngle(imu2cam(point), yaw, pitch);
}

// 获取装甲板的Yaw朝向角，返回弧度值
float Tracker::getArmorYaw(ArmorBox &armor) {
    // 将旋转向量转换为旋转矩阵
    Mat R;
    Rodrigues(armor.rVec, R);
    // 提取相机系下装甲板的欧拉角（pitch, yaw, roll）
    Mat mtxR, mtxQ, Qx, Qy, Qz;
    RQDecomp3x3(R.t(), mtxR, mtxQ, Qx, Qy, Qz);
    //Vec3d eulerAngles = RQDecomp3x3(R.t(), mtxR, mtxQ, Qx, Qy, Qz);
    // 输出旋转矩阵和欧拉角
    //std::cout << "3x3 上三角矩阵 mtxR:\n" << mtxR << std::endl;
    //std::cout << "3x3 正交矩阵 mtxQ:\n" << mtxQ << std::endl;
    //std::cout << "绕 x 轴的旋转矩阵 Qx:\n" << Qx << std::endl;
    //std::cout << "绕 y 轴的旋转矩阵 Qy:\n" << Qy << std::endl;
    //std::cout << "绕 z 轴的旋转矩阵 Qz:\n" << Qz << std::endl;
    //std::cout << "欧拉角 (绕 x, y, z 轴的旋转角度，单位为度): " << eulerAngles << std::endl;
    // 转到惯性系下的欧拉角，并处理边界的过渡，使角度连续
    float yaw = imu_yaw + asin(-Qy.at<float>(0, 2));
    float diff = fmod(yaw - last_yaw, CV_2PI);
    if (diff < 0) diff += CV_2PI;
    if (diff > CV_PI) {
        yaw = last_yaw + diff - CV_2PI;
    } else {
        yaw = last_yaw + diff;
    }
    last_yaw = yaw;
    armor.yaw = yaw;
    // cout << "yaw: " << imu_yaw << " " << asin(-Qy.at<float>(0, 2)) * 180 / CV_PI << " " << yaw  * 180 / CV_PI << endl;
    return yaw;
}

Eigen::Vector3d Tracker::getArmorPositionFromState(Eigen::VectorXd &x) {
    // Calculate predicted position of the current armor
    double xc = x(0), yc = x(2), za = x(4);
    double yaw = x(6), r = x(8);
    double xa = xc - r * cos(yaw);
    double ya = yc - r * sin(yaw);
    return Eigen::Vector3d(xa, ya, za);
}

/**************************************************************************************/

// 获取两点之间的距离
float getPointsDistance(const Point2f &a, const Point2f &b) {
    float delta_x = a.x - b.x;
    float delta_y = a.y - b.y;
    return sqrt(delta_x * delta_x + delta_y * delta_y);
}

// 根据优先级增加装甲板打击度
void setNumScore(const int &armorNum, const int &targetNum, float &armorScore) {
    if (targetNum == 0 || armorNum != targetNum) {
        if (armorNum == 1) armorScore += 5000.0;          // 英雄
        else if (armorNum == 2) armorScore += 1000.0;     // 工程
        else if (armorNum == 3) armorScore += 4000.0;     // 步兵
        else if (armorNum == 4) armorScore += 3000.0;     // 步兵
        else if (armorNum == 5) armorScore += 2000.0;     // 步兵
        else if (armorNum == 6) armorScore += 6000.0;     // 哨兵
        else if (armorNum == 7) armorScore += 8000.0;     // 基地
        else if (armorNum == 8) armorScore += 7000.0;     // 前哨
    } else {
        armorScore += 10000.0;
    }
}

// 比较a_armor装甲板与b_armor装甲板的打击度，判断a_armor是否比b_armor更适合打击
bool armorCompare(const ArmorBox &a_armor, const ArmorBox &b_armor, const Mat &src, const int &targetNum) {
    float a_score = 0;  // a_armor的打击度
    float b_score = 0;  // b_armor的打击度
    // 设置a、b装甲板的分数
    setNumScore(a_armor.armorNum, targetNum, a_score);
    setNumScore(b_armor.armorNum, targetNum, b_score);
    // 与图像中心的距离得分
    Point2f center(src.cols/2, src.rows/2);
    float a_distance = getPointsDistance(a_armor.center, center); // 装甲板距离得分，算负分
    float b_distance = getPointsDistance(b_armor.center, center); // 装甲板距离得分，算负分
    a_score -= a_distance * 2;
    b_score -= b_distance * 2;
    return a_score > b_score; // 根据打击度判断a是否比b更适合打击
}

}
