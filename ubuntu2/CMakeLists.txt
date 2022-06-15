cmake_minimum_required(VERSION 2.8)

project( xuexiaoqiuSLAM )



add_library(hello SHARED lib.cpp)
add_executable(slam main.cpp lib.h)
target_link_libraries(slam hello)