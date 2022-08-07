cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
project(minibot)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

ament_export_include_directories(include)

