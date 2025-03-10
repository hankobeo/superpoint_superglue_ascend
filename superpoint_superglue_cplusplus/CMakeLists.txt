# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)


# project information
project(detect)


# set ascend path
set(ToolChainRoot "/home/bingo/Ascend/ascend-toolkit/6.3.RC1.alpha001/aarch64-linux") #   /home/HwHiAiUser/Ascend/ascend-toolkit/latest
set(ToolChainLib ${ToolChainRoot}/devlib)
set(TooChainInclude ${ToolChainRoot}/include)

# set compiler
add_compile_options(-std=c++11)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_SOURCE_DIR}/build)

# 设置编译选项
set(CMAKE_BUILD_TYPE "Debug")  
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
# set(CMAKE_BUILD_TYPE "Release")  
# set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

message(STATUS "Current build type: ${CMAKE_BUILD_TYPE}")

# set ascend path
include_directories(${TooChainInclude} ${CMAKE_SOURCE_DIR}/../third_party/opencv-3.4.16/install-aarch64/include)   # ${OpencvInclude}
link_directories(${ToolChainLib} ${CMAKE_SOURCE_DIR}/../third_party/opencv-3.4.16/install-aarch64/lib)

add_subdirectory("./demo")
