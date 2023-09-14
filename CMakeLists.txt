cmake_minimum_required(VERSION 2.8.3)
project(kinova_apps)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-fpermissive")

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    pcl_ros
    geometry_msgs
    cv_bridge
    tf2
    tf2_ros
)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

catkin_python_setup()
catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

### EXECUTABLES ###############################################
add_executable(task_board_detector
  src/task_board_detector.cpp
)
target_link_libraries(task_board_detector
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
