#ifndef __DETECTION_HPP__
#define __DETECTION_HPP__

#include "detection.h"
#include "STrack.h"
#include <thread>    // std::this_thread
#include <chrono>    // std::chrono::seconds
#include <system_error>
#include <pthread.h>
#include "armor.hpp"

// 声明 detect_lightbar 函数（根据实际参数类型进行调整）
void detect_lightbar(const cv::Mat& binary_img, const cv::Mat& img);


using namespace detection;

bool setThreadPriority(std::thread& thread, int priority) {
    pthread_t pthread = thread.native_handle();
    
    // 获取当前调度策略
    int policy;
    struct sched_param param;
    if (pthread_getschedparam(pthread, &policy, &param) != 0) {
        std::cerr << "获取线程调度参数失败" << std::endl;
        return false;
    }
    
    // 设置新优先级（macOS 和 Linux 通用）
    // 注意：优先级范围通常为 1-99，值越大优先级越高
    param.sched_priority = priority;
    
    // 应用新参数（使用 SCHED_RR 实时调度策略）
    if (pthread_setschedparam(pthread, SCHED_RR, &param) != 0) {
        std::cerr << "设置线程优先级失败，可能需要 root 权限" << std::endl;
        return false;
    }
    
    return true;
}

DetectionArmor::DetectionArmor(string& model_path, bool ifcountTime, string video_path)
    : ifCountTime(ifcountTime)
{
    cap = VideoCapture(video_path);

    ov::AnyMap config = {
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}, // 设置性能模式为延迟优化
        {ov::inference_num_threads(4)}, // 使用4个线程进行推理
        {ov::num_streams(1)}, // 允许同时执行1个推理流
        {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY)}, // 性能核心绑定
        {ov::hint::enable_hyper_threading(false)}, // 关闭超线程
        {ov::hint::enable_cpu_pinning(false)} // 关闭CPU固定
    };

    auto network = core.read_model(model_path);
    compiled = core.compile_model(network, "CPU", config);
    infer_request = compiled.create_infer_request();
    input_port = compiled.input();

    input_blob = Mat(640, 640, CV_32F, Scalar(0)); // 初始化输入blob

    //tracker = BYTETracker(10, 10); // 初始化BYTETracker

    // armorsDatas = new ArmorData[20]; // 最多装20个装甲板
}

DetectionArmor::~DetectionArmor() 
{
    // std::cout << "quit from detection" << std::endl;
    clearHeap();
}

void DetectionArmor::clearHeap()
{
    cap.release();
    cv::destroyAllWindows();
}

void DetectionArmor::drawObject(Mat& image, const ArmorData& d)
{
    // 绘制装甲板的边界框
    std::vector<Point> points = {d.p1, d.p2, d.p3, d.p4};
    // polylines(image, points, true, Scalar(0, 0, 255), 2);
    // cv::circle(image, d.p1, 5, Scalar(0, 255, 0), -1);
    // cv::circle(image, d.p2, 5, Scalar(0, 255, 0), -1);
    // cv::circle(image, d.p3, 5, Scalar(0, 255, 0), -1);
    // cv::circle(image, d.p4, 5, Scalar(0, 255, 0), -1);
    // // 绘制装甲板的中心点
    // cv::circle(image, d.center_point, 5, Scalar(255, 0, 0), -1);
    // 计算ROI边界，确保不超出图像范围
    int x1 = std::max(0, std::min(d.p1.x, d.p3.x) - 10);
    int y1 = std::max(0, std::min(d.p1.y, d.p3.y) - 10);
    int x2 = std::min(image.cols, std::max(d.p1.x, d.p3.x) + 10);
    int y2 = std::min(image.rows, std::max(d.p1.y, d.p3.y) + 10);
    
    cv::Point lt = cv::Point(x1, y1);
    cv::Point rb = cv::Point(x2, y2);
    cv::Mat roi = image(cv::Rect(lt, rb));
    cv::Mat processed_roi = process_img(roi);
    detect_lightbar(processed_roi, roi);



    cv::rectangle(image, lt, rb, Scalar(0, 255, 0), 2);






    // cv::rectangle(
    //     image, 
    //     Point(s_x, s_y), 
    //     Point(e_x, e_y), 
    //     Scalar(0, 0, 255),
    //     3
    // );
}

inline double DetectionArmor::sigmoid(double x) 
{
    return (1 / (1 + exp(-x)));
}

void DetectionArmor::run()
{
    size_t frame_count = 0;

    while (1) 
    {
        armorsDatas.clear(); // 清空当前帧的装甲板数据
        cap >> frame;

        resize(frame, img, Size(640, 640));
        
        // 推理
        infer();

        frame_count += 1;
        if (frame_count == int(cap.get(cv::CAP_PROP_FRAME_COUNT)))
        {
            frame_count = 0;
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        }

        showImage();

        if (cv::waitKey(120) == 27)
        {
            isRunning = false; // 设置线程停止标志
            clearHeap();
            break;
        } // 按下ESC键退出

        // cout << "Detected armors: " << getdata().size() << endl;

        // for (auto i : getdata()) 
        // {
        //     drawObject(img, i); // 绘制检测结果
        // }
        // imshow("Detection", img); // 显示图像

        // if (cv::waitKey(20) == 27)
        // {
        //     isRunning = false; // 设置线程停止标志
        //     break;
        // } // 按下ESC键退出
    }
}

void DetectionArmor::infer()
{

    Timer t(counter);

    // 归一化
    input_blob = blobFromImage(
        img, 
        1 / 255.0
    );

    // 固定八股
    Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), input_blob.data);
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();    
    auto outputs = compiled.outputs();
    Tensor output = infer_request.get_tensor(outputs[0]);
    ov::Shape output_shape = output.get_shape();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());
    
    float conf_threshold = 0.6;   
    float nms_threshold = 0.4;   
    
    // 存储临时结果
    std::vector<Rect> boxes;
    std::vector<int> num_class;
    std::vector<int> color_class;
    std::vector<float> confidences;     
    std::vector<int> indices;

    // 临时的四点
    std::vector<vector<Point>> fourPointModel;

    // 遍历所有的网络输出
    for (int i = 0; i < output_buffer.rows; ++i) 
    {
        // 获取当前的置信度
        float confidence = output_buffer.at<float>(i, 8);

        // 激活到0-1之间
        confidence = sigmoid(confidence);

        // 过滤低置信度检测框
        if (confidence < conf_threshold) continue;  

        // 检查出颜色和数字的类别
        cv::Mat color_scores = output_buffer.row(i).colRange(9, 13);   // 颜色概率（红/蓝等）
        cv::Mat classes_scores = output_buffer.row(i).colRange(13, 22);// 类别概率
        cv::Point class_id, color_id;
        cv::minMaxLoc(classes_scores, nullptr, nullptr, nullptr, &class_id);
        cv::minMaxLoc(color_scores, nullptr, nullptr, nullptr, &color_id);
        // 加入预测出来的数字和颜色
        num_class.push_back(class_id.x);
        color_class.push_back(color_id.x);

        // 检测颜色
        if ((detect_color == 0 && color_id.x == 1) || (detect_color == 1 && color_id.x == 0)) continue;
        
        // 获取第一个输出向量的指针
        float* f_ptr = output_buffer.ptr<float>(i);

        vector<Point> box_point(4);

        box_point[0].x = f_ptr[0];
        box_point[0].y = f_ptr[1];

        box_point[1].x = f_ptr[2];
        box_point[1].y = f_ptr[3];

        box_point[2].x = f_ptr[4];
        box_point[2].y = f_ptr[5];

        box_point[3].x = f_ptr[6];
        box_point[3].y = f_ptr[7];

        fourPointModel.push_back(box_point);

        // 创建rect
        cv::Rect rect(
            f_ptr[0], // x
            f_ptr[1], // y
            f_ptr[4] - f_ptr[0], // width
            f_ptr[5] - f_ptr[1]  // height
        );
        
        // 加入
        boxes.push_back(rect);
        confidences.push_back(confidence);
    }

    // 非什么几把极大值抑制
    cv::dnn::NMSBoxes(
        boxes,                // 输入边界框（std::vector<cv::Rect>）
        confidences,          // 输入置信度（std::vector<float>）
        conf_threshold,       // 得分阈值（如 0.5f）
        nms_threshold,        // NMS 阈值（如 0.4f）
        indices               // 输出索引（必须传入引用）
    );

    // 保留最终的数据
    std::vector<ArmorData> data;
    for (int valid_index = 0; valid_index < indices.size(); ++valid_index) 
    {
        ArmorData d;

        d.p1 = fourPointModel[valid_index][0];
        d.p2 = fourPointModel[valid_index][1];
        d.p3 = fourPointModel[valid_index][2];
        d.p4 = fourPointModel[valid_index][3];

        d.center_point.x = (d.p1.x + d.p2.x + d.p3.x + d.p4.x) / 4;
        d.center_point.y = (d.p1.y + d.p2.y + d.p3.y + d.p4.y) / 4;
        // d.length = boxes[indices[valid_index]].width;
        // d.width = boxes[indices[valid_index]].height;
        d.ID = num_class[indices[valid_index]];

        int color = color_class[indices[valid_index]];
        if (color == 0){ d.color = Color::RED; }
        else if (color == 1){ d.color = Color::BLUE; }
        else { d.color = Color::NONE; }

        armorsDatas.push_back(d);

        // 创建对象用于跟踪器
        Object dog;
        dog.rect = cv::Rect_<float>(
            boxes[indices[valid_index]].x,
            boxes[indices[valid_index]].y,

            boxes[indices[valid_index]].width,
            boxes[indices[valid_index]].height
        );
        dog.label = num_class[indices[valid_index]];  //从类别里面取
        dog.prob = confidences[indices[valid_index]];
        detection_objects.push_back(dog);

        
    }

    tracks_objects = tracker.update(detection_objects);
    detection_objects.clear();
}

void DetectionArmor::drawTracks(Mat& image)
{
    // 绘制跟踪轨迹
    for (const auto& track : tracks_objects) {
        if (track.is_activated) {  // 绘制已激活的轨迹
            auto tlwh = track.tlwh;
            Scalar color = tracker.get_color(track.track_id); // 获取跟踪ID对应的颜色
            
            // 绘制跟踪框
            Point tl = Point(tlwh[0], tlwh[1]); // 左上角   1
            Point br = Point(tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]); // 右下角  3


            //这是测试同时取跟踪四个点的效果(不太理想)
            //Point tr = Point(tlwh[0] + tlwh[2], tlwh[1]); // 右上角  2
            //Point bl = Point(tlwh[0], tlwh[1] + tlwh[3]); // 左下角  4
            
            //std::vector<Point> points = {tl, tr, br, bl}; // 1 2 3 4

            //cv::polylines(image, points, true, cv::Scalar(0, 255, 0), 2);


            Point center = Point(tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2);

            circle(image, center, 5, Scalar(0,255,0), -1);

            //rectangle(image, tl, br, color, 2);
            //rectangle(image, tl, br,Scalar(0, 255, 0), 2);
            
            // 绘制跟踪ID
            // putText(image, 
            //     "Track " + to_string(track.track_id), 
            //     Point(tlwh[0], tlwh[1] - 5), 
            //     FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
    }
}


inline const vector<ArmorData> DetectionArmor::getdata()
{
    return armorsDatas; // 返回当前帧的装甲板数据
}

void DetectionArmor::start_detection()
{
    this->isRunning = true; // 设置线程运行标志
    run();

    // run(); // 直接调用run函数
}

void __TEST__ DetectionArmor::showImage()
{
    if (!img.empty()) 
    {
        for (auto i : getdata()) 
        {
            drawObject(img, i); // 绘制检测结果
        }
        //drawTracks(img); // 绘制跟踪轨迹

        cv::imshow("Detection Armor", img); // 显示图像
        // format_print_data_test();
    }

    // std::lock_guard<std::mutex> lock(_mtx);
}

void __TEST__ DetectionArmor::format_print_data_test()
{
    cout << "armor Num: " << getdata().size() << endl;
    for (auto d : getdata())
    {
        // cout << "center X: " << d.center_x << " ";
        // cout << "center Y: " << d.center_y << endl;
    }
}

#endif // __DETECTION_HPP__