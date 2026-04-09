cmake_minimum_required(VERSION 3.10)
project(go2_collector_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_EXAMPLES OFF CACHE BOOL "Build Unitree SDK examples" FORCE)
set(UNITREE_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../unitree_sdk2")

add_subdirectory("${UNITREE_SDK_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/unitree_sdk2_build" EXCLUDE_FROM_ALL)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/web_ui_server.cpp")
  add_executable(go2_collector go2_collector.cpp web_ui_server.cpp)
  target_link_libraries(go2_collector PRIVATE unitree_sdk2)
endif()

add_executable(go2_bridge go2_bridge.cpp)
target_link_libraries(go2_bridge PRIVATE unitree_sdk2)

add_executable(go2_mode_probe go2_mode_probe.cpp)
target_link_libraries(go2_mode_probe PRIVATE unitree_sdk2)

add_executable(go2_video_probe go2_video_probe.cpp)
target_link_libraries(go2_video_probe PRIVATE unitree_sdk2)
