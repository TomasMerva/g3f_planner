cmake_minimum_required(VERSION 3.0.2)
project(rollout_fabrics)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

add_executable(${PROJECT_NAME} rollout_fabrics.cpp)
target_link_libraries(${PROJECT_NAME} )