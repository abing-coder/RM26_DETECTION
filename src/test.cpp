// // #include "armor.hpp"




// // cv::Mat processimg(const cv::Mat& img)
// // {
// //     cv::Mat img_gray,img_binary,img_morph,img_blur;
// //     cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
// //     cv::threshold(img_gray, img_binary, 55,160, cv::THRESH_BINARY); //二值化

// //     cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
// //     //cv::morphologyEx(img_binary, img_morph, cv::MORPH_CLOSE, kernel);  //闭运算，填充灯条内部空洞
// //     cv::morphologyEx(img_binary, img_morph, cv::MORPH_OPEN, kernel);   //开运算，去除小噪声点
// //     cv::GaussianBlur(img_morph, img_blur,cv::Size(5,5), 0); //高斯模糊，去除噪声
    

// //     return img_blur;
// //     //cv::imshow("Gray Image", img_gray);
// //     //cv::imshow("Binary Image", img_binary);
// //     //cv::imshow("Morphology Image", img_morph);
// //     //cv::imshow("Blur Image", img_blur);
// //     //cv::waitKey(0);
// // }

// // //std::vector<lightbar>
// // void  detect_lightbar(const cv::Mat& img, const cv::Mat& original_img)
// // {
// //     std::vector<std::vector<cv::Point>> contours;
// //     std::vector<cv::Vec4i> hierarchy;
// //     cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
// //     //cv::drawContours(original_img, contours, -1, cv::Scalar(255), 2);  //灰度图只能显示白色
// //     //cv::imshow("Contours", original_img);
// //     //cv::waitKey(0);


// //     for(const auto& contour : contours)
// //     {
// //         float lightbar_area = cv::contourArea(contour);
// //         if(lightbar_area < 20) continue; //面积太小，过滤掉

// //         cv::RotatedRect minRect = cv::minAreaRect(contour); //最小外接矩形
// //         //cv::Rect bounding_box = cv::boundingRect(contour); //外接矩形,这个其实可以不用
        
// //         if (minRect.size.width > minRect.size.height) {
// // 				minRect.angle += 90;
// // 				float t = minRect.size.width;
// // 				minRect.size.width = minRect.size.height;
// // 				minRect.size.height = t;   
// // 			} //转换矩形的width和height

// //         if ((minRect.size.width * 10 > minRect.size.height) && (minRect.size.width * 1 < minRect.size.height) && (abs(minRect.angle) < 30)) {
// // 				minRects.push_back(minRect);
// // 			}//太小的舍去
        
       

// //     }

// // }
   
   
// // int main()
// // {
// //     cv::Mat original_img = cv::imread("C:\\Users\\ASUS\\Desktop\\lightbar_derection\\armor.jpg");
// //     cv::Mat processed = processimg(original_img);
// //     detect_lightbar(processed, original_img);

// //     return 0;
// // }




// #include "stdio.h"
// #include<iostream> 
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// using namespace std;
// using namespace cv;
// //由于在识别中的核心物体以及相关的物理特性是灯条，所以建一个灯条类
// class LightDescriptor
// {	    //在识别以及匹配到灯条的功能中需要用到旋转矩形的长宽偏转角面积中心点坐标等
// public:float width, length, angle, area;
//       cv::Point2f center;
// public:
//     LightDescriptor() {};
//     //让得到的灯条套上一个旋转矩形，以方便之后对角度这个特殊因素作为匹配标准
//     LightDescriptor(const cv::RotatedRect& light)
//     {
//         width = light.size.width;
//         length = light.size.height;
//         center = light.center;
//         angle = light.angle;
//         area = light.size.area();
//     }
// };
 
// int main()
// {
//     VideoCapture video; //VC类对象化
//     video.open("C:\\Users\\ASUS\\Desktop\\lightbar_derection\\vedio\\4.mp4");
//     //变量集中定义
//     Mat frame, channels[3], binary, Gaussian, dilatee;
//     Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//     Rect boundRect;
//     RotatedRect box;
//     vector<vector<Point>> contours;
//     vector<Vec4i> hierarchy;
//     vector<Point2f> boxPts(4);
//     //图像预处理
//     for (;;) {
//         Rect point_array[20];
//         video >> frame;  //读取每帧
//         if (frame.empty()) {
//             break;
//         }
//         split(frame, channels); //通道分离BGR
//         threshold(channels[0], binary, 200, 255, 0);//二值化
//         GaussianBlur(binary, Gaussian, Size(5, 5), 0);//滤波
//         dilate(Gaussian, dilatee, element);

//         imshow("dilate", dilatee);

//         // dilate(Gaussian, dilate, element, Point(-1, -1));//膨胀，把滤波得到的细灯条变宽
//         findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//轮廓检测
//         vector<LightDescriptor> lightInfos;//创建一个灯条类的动态数组
//         //筛选灯条
//         for (int i = 0; i < contours.size(); i++) {
//             // 求轮廓面积
//             double area = contourArea(contours[i]);
//             // 去除较小轮廓&fitEllipse的限制条件
//             if (area < 5 || contours[i].size() <= 1)
//                 continue;//相当于就是把这段轮廓去除掉
//             // 用椭圆拟合区域得到外接矩形（特殊的处理方式：因为灯条是椭圆型的，所以用椭圆去拟合轮廓，再直接获取旋转外接矩形即可）
//             RotatedRect Light_Rec = fitEllipse(contours[i]);
 
//             // 长宽比和轮廓面积比限制（由于要考虑灯条的远近都被识别到，所以只需要看比例即可）
//             if (Light_Rec.size.width / Light_Rec.size.height > 4)
//                 continue;
//             lightInfos.push_back(LightDescriptor(Light_Rec));
//         }
//         //二重循环多条件匹配灯条
//         for (size_t i = 0; i < lightInfos.size(); i++) {
//             for (size_t j = i + 1; (j < lightInfos.size()); j++) {
//                 LightDescriptor& leftLight = lightInfos[i];
//                 LightDescriptor& rightLight = lightInfos[j];
//                 float angleGap_ = abs(leftLight.angle - rightLight.angle);
//                 //由于灯条长度会因为远近而受到影响，所以按照比值去匹配灯条
//                 float LenGap_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
//                 float dis = pow(pow((leftLight.center.x - rightLight.center.x), 2) + pow((leftLight.center.y - rightLight.center.y), 2), 0.5);
//                 //均长
//                 float meanLen = (leftLight.length + rightLight.length) / 2;
//                 float lengap_ratio = abs(leftLight.length - rightLight.length) / meanLen;
//                 float yGap = abs(leftLight.center.y - rightLight.center.y);
//                 float yGap_ratio = yGap / meanLen;
//                 float xGap = abs(leftLight.center.x - rightLight.center.x);
//                 float xGap_ratio = xGap / meanLen;
//                 float ratio = dis / meanLen; 
//                 //匹配不通过的条件
//                 if (angleGap_ > 15 ||
//                     LenGap_ratio > 1.0 ||
//                     lengap_ratio > 0.8 ||
//                     yGap_ratio > 1.5 ||
//                     xGap_ratio > 2.2 ||
//                     xGap_ratio < 0.8 ||
//                     ratio > 3 ||
//                     ratio < 0.8) {  
//                     continue;
//                 }
//                 //绘制矩形
//                 Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
//                 RotatedRect rect = RotatedRect(center, Size(dis, meanLen), (leftLight.angle + rightLight.angle) / 2);
//                 Point2f vertices[4];
//                 rect.points(vertices);
//                 for (int i = 0; i < 4; i++) {
//                     line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2.2);
//                     circle(frame, vertices[0], 5, Scalar(255, 0, 0), -1);
//                     circle(frame, vertices[1], 5, Scalar(255, 0, 0), -1);
//                     circle(frame, vertices[2], 5, Scalar(255, 0, 0), -1);
//                     circle(frame, vertices[3], 5, Scalar(255, 0, 0), -1);
//                 }
//             }
//         }
 
//         namedWindow("video", WINDOW_FREERATIO);
//         imshow("video", frame);
//         waitKey(30);
//     }
//     video.release();
//     cv::destroyAllWindows();
//     return 0;
// }