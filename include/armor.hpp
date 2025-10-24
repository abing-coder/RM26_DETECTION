#ifndef ARMOR_HPP
#define ARMOR_HPP


//channel B0 G1 R2
#define BLUE_LIGHTBARS 0   //蓝色灯条
#define RED_LIGHTBARS 2     //红色灯条



#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>

class lightbars
{
public:
    lightbars() = default;
    lightbars(const cv::RotatedRect& minRect)
    {
        this->minRect = minRect;
        this->length = std::max(minRect.size.width, minRect.size.height); //灯条长度
        this->width = std::min(minRect.size.width, minRect.size.height);  //灯条宽度
        this->area = minRect.size.area(); //灯条面积
        this->angle = minRect.angle; //灯条倾斜角
    }





    cv::RotatedRect minRect; //最小外接矩形
    float length;  //灯条长度
    float width;  //灯条宽度
    float area; //灯条面积
    float angle; //灯条倾斜角
};

// 装甲板结构体
struct Armor {
    cv::Point2f center;
    cv::Rect boundingRect;
    float confidence;
    std::pair<lightbars, lightbars> lightbarPair;
    
    Armor(const lightbars& left, const lightbars& right) {
        lightbarPair = std::make_pair(left, right);
        center = (left.minRect.center + right.minRect.center) / 2.0f;
        
        // 计算边界框
        cv::Point2f left_vertices[4], right_vertices[4];
        left.minRect.points(left_vertices);
        right.minRect.points(right_vertices);
        
        float min_x = std::min({left_vertices[0].x, left_vertices[1].x, left_vertices[2].x, left_vertices[3].x,
                               right_vertices[0].x, right_vertices[1].x, right_vertices[2].x, right_vertices[3].x});
        float max_x = std::max({left_vertices[0].x, left_vertices[1].x, left_vertices[2].x, left_vertices[3].x,
                               right_vertices[0].x, right_vertices[1].x, right_vertices[2].x, right_vertices[3].x});
        float min_y = std::min({left_vertices[0].y, left_vertices[1].y, left_vertices[2].y, left_vertices[3].y,
                               right_vertices[0].y, right_vertices[1].y, right_vertices[2].y, right_vertices[3].y});
        float max_y = std::max({left_vertices[0].y, left_vertices[1].y, left_vertices[2].y, left_vertices[3].y,
                               right_vertices[0].y, right_vertices[1].y, right_vertices[2].y, right_vertices[3].y});
        
        boundingRect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
        confidence = 1.0f; // 可以根据匹配质量调整
    }
};

cv::Mat process_img(const cv::Mat& img);
void detect_lightbar(const cv::Mat& binary_img, const cv::Mat& img);
std::vector<Armor> detect_armors(const cv::Mat& binary_img, const cv::Mat& img);
std::vector<Armor> non_maximum_suppression(const std::vector<Armor>& armors, float overlap_threshold = 0.3);
//void draw_lightbars(std::vector<lightbars> lightbars_res, cv::Mat &img);




#endif