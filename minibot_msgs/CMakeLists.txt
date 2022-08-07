cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
project(minibot_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(ament_cmake REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
# Instead of message_generation
find_package(rosidl_default_generators REQUIRED)
# Instead of message_runtime
find_package(rosidl_default_runtime REQUIRED)

set(srv_files
"srv/lss_4dof_to_named.srv"
  "srv/lss_4dof_to_pose.srv"
  "srv/lss_4dof_to_pose_euler.srv"
  "srv/lss_4dof_pick.srv"
  "srv/set_url.srv"


)


rosidl_generate_interfaces(${PROJECT_NAME}
    DEPENDENCIES builtin_interfaces 
)


ament_export_dependencies(rosidl_default_runtime)


ament_export_include_directories(include)

