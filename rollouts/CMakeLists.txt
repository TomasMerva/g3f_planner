cmake_minimum_required(VERSION 3.0.2)
project(rollout_fabrics)
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


include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/src/)


SET(SOURCES 
      ${PROJECT_SOURCE_DIR}/include/tcp_server.h
)

SET(HEADERS
      ${PROJECT_SOURCE_DIR}/src/test.cpp
      ${PROJECT_SOURCE_DIR}/src/tcp_server.cpp
)


add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} )

# add_executable(${PROJECT_NAME} rollout_fabrics.cpp)
# target_link_libraries(${PROJECT_NAME}  ${YAML_CPP_LIBRARIES})