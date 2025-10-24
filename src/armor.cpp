#include "armor.hpp"



cv::Mat process_img(const cv::Mat &img)
{
    std::vector<cv::Mat> channels;
    cv::Mat frame,thresh_img,binary_img;
    cv::split(img, channels); //通道分离BGR     
    cv::Mat enhanced;
    channels[RED_LIGHTBARS].convertTo(enhanced, -1, 1.2, 10); // 增加对比度和亮度
    cv::GaussianBlur(enhanced, frame, cv::Size(3, 3), 0); 
    cv::threshold(frame, thresh_img, 100, 255, cv::THRESH_BINARY);   
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); 
    cv::morphologyEx(thresh_img, binary_img, cv::MORPH_OPEN, kernel); //开运算，去除灯条边缘噪点

    cv::imshow("binary_img",binary_img);
    return binary_img;
   
}
cv::Point2f getLineIntersection(const std::pair<cv::Point2f, cv::Point2f>& line1,
                            const std::pair<cv::Point2f, cv::Point2f>& line2) {
    // 提取直线1的两点坐标
    float x1 = line1.first.x, y1 = line1.first.y;
    float x2 = line1.second.x, y2 = line1.second.y;
    // 提取直线2的两点坐标
    float x3 = line2.first.x, y3 = line2.first.y;
    float x4 = line2.second.x, y4 = line2.second.y;

    // 直线1的一般式参数
    float A1 = y2 - y1;
    float B1 = x1 - x2;
    float C1 = x2 * y1 - x1 * y2;

    // 直线2的一般式参数
    float A2 = y4 - y3;
    float B2 = x3 - x4;
    float C2 = x4 * y3 - x3 * y4;

    // 计算分母 D
    float D = A1 * B2 - A2 * B1;

    // 平行或重合返回(-1, -1)
    if (fabs(D) < 1e-6) {
        return cv::Point2f(-1, -1);  // 用(-1,-1)表示无交点
    }

    // 计算并返回交点
    return cv::Point2f(
        (B1 * C2 - B2 * C1) / D,
        (A2 * C1 - A1 * C2) / D
    );
}


double Distance(cv::Point2f a, cv::Point2f b) {
		return sqrt((a.x - b.x) * (a.x - b.x) +
			(a.y - b.y) * (a.y - b.y));
	}

std::vector<Armor> non_maximum_suppression(const std::vector<Armor>& armors, float overlap_threshold) {
    if (armors.empty()) return {};
    
    std::vector<Armor> result;
    std::vector<bool> suppressed(armors.size(), false);
    
    // 按置信度排序
    std::vector<std::pair<float, int>> confidence_indices;
    for (int i = 0; i < armors.size(); i++) {
        confidence_indices.push_back({armors[i].confidence, i});
    }
    std::sort(confidence_indices.begin(), confidence_indices.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
    
    for (const auto& conf_idx : confidence_indices) {
        int idx = conf_idx.second;
        if (suppressed[idx]) continue;
        
        result.push_back(armors[idx]);
        
        // 抑制重叠的装甲板
        for (int j = 0; j < armors.size(); j++) {
            if (suppressed[j] || j == idx) continue;
            
            // 计算IoU
            cv::Rect intersection = armors[idx].boundingRect & armors[j].boundingRect;
            if (intersection.area() <= 0) continue;
            
            float intersection_area = intersection.area();
            float union_area = armors[idx].boundingRect.area() + armors[j].boundingRect.area() - intersection_area;
            float iou = intersection_area / union_area;
            
            if (iou > overlap_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}


std::vector<Armor> detect_armors(const cv::Mat& binary_img, const cv::Mat& img) {
    std::vector<lightbars> lightbars_box;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::cout << "contours:" << contours.size() << std::endl; 

    // 检测灯条
    for(const auto& contour : contours) {
        auto area = cv::contourArea(contour);
        if(area < 5) continue;

        cv::RotatedRect minRect = cv::minAreaRect(contour);
        if (minRect.size.width > minRect.size.height) {
            minRect.angle += 90; 
            float t = minRect.size.width;
            minRect.size.width = minRect.size.height;
            minRect.size.height = t;
        }

        float length = std::max(minRect.size.width, minRect.size.height);
        float width = std::min(minRect.size.width, minRect.size.height);
        float lw_ratio = length / width;

        if (lw_ratio < 2.0 || lw_ratio > 12.0) continue;
        lightbars_box.push_back(lightbars(minRect));
    }

    std::cout << "lightbars_box_size: " << lightbars_box.size() << std::endl;
    
    // 匹配灯条形成装甲板
    std::vector<Armor> armors;
    for(int i = 0; i < lightbars_box.size(); i++) {
        for(int j = i + 1; j < lightbars_box.size(); j++) {
            lightbars &left = lightbars_box[i];
            lightbars &right = lightbars_box[j];
            
            float angle_gap = std::abs(left.angle - right.angle);
            if(angle_gap > 30.0) {
                angle_gap = 180 - angle_gap;
            }
            
            float length_gap_ratio = std::abs(left.length - right.length) / std::max(left.length, right.length);
            float meanLen = (left.length + right.length) / 2;
            float lengap_ratio = std::abs(left.length - right.length) / meanLen;
            float yGap = abs(left.minRect.center.y - right.minRect.center.y);
            float yGap_ratio = yGap / meanLen;
            float xGap = abs(left.minRect.center.x - right.minRect.center.x);
            float xGap_ratio = xGap / meanLen;
            float dis = Distance(left.minRect.center, right.minRect.center);
            float ratio = dis / meanLen;
            
           
            if (angle_gap > 30.0 ||           
                length_gap_ratio > 2.0 ||     
                lengap_ratio > 0.6 ||         
                yGap_ratio > 1.5 ||           
                xGap_ratio > 8 || xGap_ratio < 1.0 ||  
                ratio > 4.0 || ratio < 0.8) {
                continue;
            }
            
            // 创建装甲板
            Armor armor(left, right);
            armors.push_back(armor);
        }
    }
    
    
    std::vector<Armor> filtered_armors = non_maximum_suppression(armors, 0.3);
    
    // 绘制结果
    for (const auto& armor : filtered_armors) {
        const auto& left = armor.lightbarPair.first;
        const auto& right = armor.lightbarPair.second;
        
        std::vector<cv::Point2f> left_vertices(4);
        std::vector<cv::Point2f> right_vertices(4);
        
        left.minRect.points(left_vertices.data());
        right.minRect.points(right_vertices.data());
        
        cv::Point left_top = (left_vertices[1] + left_vertices[2]) / 2;
        cv::Point left_bottom = (left_vertices[0] + left_vertices[3]) / 2;
        cv::Point right_top = (right_vertices[1] + right_vertices[2]) / 2;
        cv::Point right_bottom = (right_vertices[0] + right_vertices[3]) / 2;
        
        if(right_top.y > right_bottom.y) {
            std::swap(right_top, right_bottom);
        }
        if(left_top.y > left_bottom.y) {
            std::swap(left_top, left_bottom);
        }

        cv::circle(img, left_top, 3, cv::Scalar(255, 0, 0), -1);
        cv::circle(img, left_bottom, 3, cv::Scalar(255, 0, 0), -1);
        cv::circle(img, right_top, 3, cv::Scalar(255, 0, 0), -1);
        cv::circle(img, right_bottom, 3, cv::Scalar(255, 0, 0), -1);
        
        std::pair<cv::Point, cv::Point> line(left_top, right_bottom);   
        std::pair<cv::Point, cv::Point> line2(left_bottom, right_top);
        
        cv::line(img, left_top, right_bottom, cv::Scalar(0, 255, 0), 1);
        cv::line(img, left_bottom, right_top, cv::Scalar(0, 255, 0), 1);
        cv::Point2f center = getLineIntersection(line, line2);
        cv::circle(img, center, 3, cv::Scalar(255, 0, 0), -1);
        
        // 绘制边界框
        //cv::rectangle(img, armor.boundingRect, cv::Scalar(0, 0, 255), 2);
    }
    
    //std::cout << "检测到装甲板数量: " << filtered_armors.size() << std::endl;
    return filtered_armors;
}

void detect_lightbar(const cv::Mat& binary_img, const cv::Mat& img)
{
    // 使用新的装甲板检测函数
    detect_armors(binary_img, img);
}




// int main(int argc, char** argv)
// {

    
//     cv::VideoCapture video("/home/ubuntu/桌面/lightbar_derection/vedio/3.mp4");
//     cv::Mat frame;
//     while(true)
//     {
//         video >> frame;
//         if(frame.empty())
//         {
//             std::cout << "Could not read the frame" << std::endl;
//             break;
//         }
//         auto start_time = std::chrono::steady_clock::now();
//         cv::Mat processed_img = process_img(frame);
//         detect_lightbar(processed_img, frame);
//         auto end_time = std::chrono::steady_clock::now();
//         auto fps = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//         std::cout << "FPS: " << 1000.0 / fps << std::endl;
//         cv::imshow("Frame", frame);
//         if(cv::waitKey(100) >= 0) break;
//     }
//     video.release();
//     cv::destroyAllWindows();



//     return 0;
// }