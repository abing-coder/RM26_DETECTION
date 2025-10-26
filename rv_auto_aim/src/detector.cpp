#include "detector.hpp"

int color = 0;//0-红色，1-蓝色
Detector::Detector(const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a)
: binary_thres(bin_thres), detect_color(color), l(l), a(a)
{
}

cv::Mat Detector::preprocessImage(const cv::Mat & rgb_img)
{
  if(color==1)
  {
    cv::Mat gray_img;
    cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);

    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);
    cv::imshow("binary_img", binary_img);
    return binary_img;
  }
  
  if(color==0)
  {
    int flag = 2; // 0-蓝色通道，1-绿色通道，2-红色通道
    std::vector<cv::Mat> channels;
    cv::Mat frame,thresh_img,binary_img;
    cv::split(rgb_img, channels); //通道分离BGR
    cv::GaussianBlur(channels[flag], frame, cv::Size(5, 5), 0); //高斯模糊，去除细小噪点
    cv::threshold(frame, thresh_img, 100, 255, cv::THRESH_BINARY);   //二值化
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); 
    cv::morphologyEx(thresh_img, binary_img, cv::MORPH_OPEN, kernel); //开运算，去除灯条边缘噪点

    cv::imshow("binary_img",binary_img);
      
    return binary_img;
  }
  return cv::Mat();
  
}
bool Detector::isLight(const Light & light)
{
  // The ratio of light (short side / long side)
  float ratio = light.width / light.length;
  bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;

  bool angle_ok = light.tilt_angle < l.max_angle;

  bool is_light = ratio_ok && angle_ok;
  

  // Fill in debug information
//   auto_aim_interfaces::msg::DebugLight light_data;
//   light_data.center_x = light.center.x;
//   light_data.ratio = ratio;
//   light_data.angle = light.tilt_angle;
//   light_data.is_light = is_light;
//   this->debug_lights.data.emplace_back(light_data);

  return is_light;
}

std::vector<Light> Detector::findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img)
{
  using std::vector;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  vector<Light> lights;
  // this->debug_lights.data.clear();

  for (const auto & contour : contours) {
    if (contour.size() < 5) continue; //过滤轮廓

    auto r_rect = cv::minAreaRect(contour);
    auto light = Light(r_rect);

    if (isLight(light)) {
      auto rect = light.boundingRect();
      if (  // Avoid assertion failed
        0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols && 0 <= rect.y &&
        0 <= rect.height && rect.y + rect.height <= rbg_img.rows) {
        int sum_r = 0, sum_b = 0;
        auto roi = rbg_img(rect);
        // Iterate through the ROI
        for (int i = 0; i < roi.rows; i++) //行
        { 
          for (int j = 0; j < roi.cols; j++) //列
          {
            if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0) // pointPolygonTest判断点是否在轮廓内
            {
              // if point is inside contour
              if(color == 0)
              {
                sum_r += roi.at<cv::Vec3b>(i, j)[2]; //红色
                sum_b += roi.at<cv::Vec3b>(i, j)[0]; //蓝色
              }
              else{
                sum_r += roi.at<cv::Vec3b>(i, j)[0]; //红色
                sum_b += roi.at<cv::Vec3b>(i, j)[1]; //蓝色
              }
              
            }
          }
        }
        // Sum of red pixels > sum of blue pixels ?
        light.color = sum_r > sum_b ? RED : BLUE;
        lights.emplace_back(light);
      }
    }
  }

  return lights;
}

std::vector<Armor> Detector::matchLights(const std::vector<Light> & lights)
{
  std::vector<Armor> armors;
  // this->debug_armors.data.clear();

  // Loop all the pairing of lights
  for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
    for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
      if (light_1->color != detect_color || light_2->color != detect_color) continue;

      if (containLight(*light_1, *light_2, lights)) {
        continue;
      }

      auto type = isArmor(*light_1, *light_2);
      if (type != ArmorType::INVALID) {
        auto armor = Armor(*light_1, *light_2);
        armor.type = type;
        armors.emplace_back(armor);
      }
    }
  }
  std::cout << "Detected armors: " << armors.size() << std::endl;
  return armors;
}

bool Detector::containLight(
  const Light & light_1, const Light & light_2, const std::vector<Light> & lights)
{
  auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
  auto bounding_rect = cv::boundingRect(points);

  for (const auto & test_light : lights) {
    if (test_light.center == light_1.center || test_light.center == light_2.center) continue;

    if (
      bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
      bounding_rect.contains(test_light.center)) {
      return true;
    }
  }

  return false;
}

ArmorType Detector::isArmor(const Light & light_1, const Light & light_2)
{
  // Ratio of the length of 2 lights (short side / long side)
  float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                             : light_2.length / light_1.length;
  bool light_ratio_ok = light_length_ratio > a.min_light_ratio;

  // Distance between the center of 2 lights (unit : light length)
  float avg_light_length = (light_1.length + light_2.length) / 2;
  float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
  bool center_distance_ok = (a.min_small_center_distance <= center_distance &&
                             center_distance < a.max_small_center_distance) ||
                            (a.min_large_center_distance <= center_distance &&
                             center_distance < a.max_large_center_distance);

  // Angle of light center connection
  cv::Point2f diff = light_1.center - light_2.center;
  float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
  bool angle_ok = angle < a.max_angle;

  bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

  // Judge armor type
  ArmorType type;
  if (is_armor) {
    type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
  } else {
    type = ArmorType::INVALID;
  }

  // Fill in debug information
  //auto_aim_interfaces::msg::DebugArmor armor_data;
//   armor_data.type = ARMOR_TYPE_STR[static_cast<int>(type)];
//   armor_data.center_x = (light_1.center.x + light_2.center.x) / 2;
//   armor_data.light_ratio = light_length_ratio;
//   armor_data.center_distance = center_distance;
//   armor_data.angle = angle;
  //this->debug_armors.data.emplace_back(armor_data);

  return type;
}
cv::Mat Detector::getAllNumbersImage()
{
  if (armors_.empty()) {
    return cv::Mat(cv::Size(20, 28), CV_8UC1);
  } else {
    std::vector<cv::Mat> number_imgs;
    number_imgs.reserve(armors_.size());
    for (auto & armor : armors_) {
      number_imgs.emplace_back(armor.number_img);
    }
    cv::Mat all_num_img;
    cv::vconcat(number_imgs, all_num_img);
    return all_num_img;
  }
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
void Detector::drawResults(cv::Mat & img)
{
  // Draw Lights
  for (const auto & light : lights_) {
    cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
    auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
    cv::line(img, light.top, light.bottom, line_color, 5);
  }

  // Draw armors
  for (const auto & armor : armors_) {
    cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
    cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
    std::pair<cv::Point2f, cv::Point2f> line1 = {armor.left_light.top, armor.right_light.bottom};
    std::pair<cv::Point2f, cv::Point2f> line2 = {armor.left_light.bottom, armor.right_light.top};
    cv::Point2f center = getLineIntersection(line1, line2);
    cv::circle(img, center, 5, cv::Scalar(255, 0, 0), -1);
  }

  // Show numbers and confidence
  for (const auto & armor : armors_) {
    cv::putText(
      img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 255), 2);
  }
}
std::vector<Armor> Detector::detect(const cv::Mat & input)
{
    cv::Mat binary_img;
    binary_img = preprocessImage(input);
    lights_ = findLights(input, binary_img);
    armors_ = matchLights(lights_);

    // if (!armors_.empty()) {
    //     classifier->extractNumbers(input, armors_);
    //     classifier->classify(armors_);
    // }

    return armors_;
}
