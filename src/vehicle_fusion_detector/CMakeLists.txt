cmake_minimum_required(VERSION 3.0.2)
project(vehicle_fusion_detector)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
# geometry_msgs # 如果 Detection.msg 中用到了 geometry_msgs 类型
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Detection.msg
  DetectionArray.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing dependencies for your messages
# geometry_msgs
)

## Declare catkin package
catkin_package(
#  INCLUDE_DIRS include                # 如果有 C++ 头文件
#  LIBRARIES ${PROJECT_NAME}           # 如果有 C++ 库
  CATKIN_DEPENDS rospy std_msgs message_runtime # geometry_msgs
#  DEPENDS system_lib
)

## Install Python executable scripts to ${CATKIN_PACKAGE_BIN_DESTINATION}
# Ensure scripts are executable (chmod +x scripts/your_script.py)
catkin_install_python(PROGRAMS
  scripts/own_sensor_detector.py
  scripts/other_vehicle_detector.py
  scripts/fusion_node.py
  scripts/matching_rpc_server.py
  scripts/detection_visualizer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 如果你有 launch 文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)