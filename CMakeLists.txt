cmake_minimum_required(VERSION 3.10.0)
project(detection VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(OpenCV_DIR "/Users/Opencv/opencv_build")
find_package(OpenCV REQUIRED)

find_package(OpenVINO REQUIRED)


# find_package(OpenCV REQUIRED COMPONENTS core dnn)
# find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
# find_package(OpenVINO REQUIRED COMPONENTS runtime)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${OpenVINO_INCLUDE_DIRS}
)

include_directories(./include)
include_directories(./TimeCounter)
include_directories(./TrackerHeader)
include_directories(./Eigen)

file(GLOB_RECURSE SOURCES "TrackerSource/*.cpp")
file(GLOB_RECURSE ARMOR_SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} main.cpp ${SOURCES} ${ARMOR_SOURCES})
# message(${SOURCES})


target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenVINO_LIBRARIES}  
    ${OpenCV_LIBS}               # OpenCV 库
    openvino::runtime
)
