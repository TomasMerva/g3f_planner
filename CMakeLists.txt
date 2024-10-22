cmake_minimum_required(VERSION 3.9)
project(fabrics_controller VERSION 1.0.0 DESCRIPTION "Shared library providing with compiled fabric controller.")

set(CMAKE_BUILD_TYPE Release)

set(SRC_PATH ${PROJECT_SOURCE_DIR}/fabrics_rollouts)

file(GLOB SRC_FILES "${SRC_PATH}/src/*.cpp")

foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(LIB_NAME ${SRC_FILE} NAME_WE)
    
    add_library(${LIB_NAME} SHARED ${SRC_FILE})

endforeach()


# cmake_minimum_required(VERSION 3.9)
# project(fabrics_controller VERSION 1.0.0 DESCRIPTION "Shared library providing with compiled fabric controller.")

# set(CMAKE_BUILD_TYPE Release)

# SET(SRC_PATH ${PROJECT_SOURCE_DIR}/fabrics_rollouts)

# add_library(${PROJECT_NAME} SHARED
#   ${SRC_PATH}/src/pure_controller.cpp
# )

# add_library(fabrics_controller_3obst SHARED
#   ${SRC_PATH}/src/fabrics_controller_3obst.cpp
# )

# add_library(fabrics_controller_5obst SHARED
#   ${SRC_PATH}/src/fabrics_controller_5obst.cpp
# )


# add_library(fabrics_controller_6obst SHARED
#   ${SRC_PATH}/src/fabrics_controller_6obst.cpp
# )

# add_library(fabrics_controller_9obst SHARED
#   ${SRC_PATH}/src/fabrics_controller_9obst.cpp
# )