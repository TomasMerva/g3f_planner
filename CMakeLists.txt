cmake_minimum_required(VERSION 3.9)
project(fabrics_controller VERSION 1.0.0 DESCRIPTION "Shared libraries that provide compiled fabric controllers.")

set(CMAKE_BUILD_TYPE Release)

set(SRC_PATH ${PROJECT_SOURCE_DIR}/g3f_planner)

file(GLOB SRC_FILES "${SRC_PATH}/src/*.cpp")

foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(LIB_NAME ${SRC_FILE} NAME_WE)
    
    add_library(${LIB_NAME} SHARED ${SRC_FILE})

endforeach()


