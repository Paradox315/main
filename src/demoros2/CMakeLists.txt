cmake_minimum_required(VERSION 3.0.2)
project(demoros2)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
  nav_msgs
  geometry_msgs
)

add_message_files(
  FILES
  Detections.msg
  DetectionsWithOdom.msg
  MatchResult.msg
  FusionResult.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  nav_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs nav_msgs geometry_msgs
)

catkin_install_python(PROGRAMS
  scripts/detection_listener.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)