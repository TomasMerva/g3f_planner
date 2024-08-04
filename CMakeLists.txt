cmake_minimum_required(VERSION 3.0.2)
project(fabrics_rollouts)
add_compile_options(-std=c++17)

find_package(yaml-cpp CONFIG REQUIRED)
get_target_property(LIBRARY_TYPE yaml-cpp::yaml-cpp TYPE)

if(LIBRARY_TYPE STREQUAL "SHARED_LIBRARY")
  if(NOT YAML_CPP_SHARED_LIBS_BUILT)
    message(FATAL_ERROR "Library type (${LIBRARY_TYPE}) contradicts config: ${YAML_CPP_SHARED_LIBS_BUILT}")
  endif()
else()
  if(YAML_CPP_SHARED_LIBS_BUILT)
    message(FATAL_ERROR "Library type (${LIBRARY_TYPE}) contradicts config: ${YAML_CPP_SHARED_LIBS_BUILT}")
  endif()
endif()


SET(SRC_PATH ${PROJECT_SOURCE_DIR}/${PROJECT_NAME})



include_directories(${SRC_PATH}/headers/)
# include_directories(${PROJECT_SOURCE_DIR}/src/)




SET(HEADERS
      ${SRC_PATH}/headers/tcp_server.h
      ${SRC_PATH}/headers/fabrics_server.h
      ${SRC_PATH}/headers/fabrics_config.h
)

SET(SOURCES
      ${SRC_PATH}/src/main.cpp
      ${SRC_PATH}/src/tcp_server.cpp
      ${SRC_PATH}/src/fabrics_server.cpp
)


add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})
