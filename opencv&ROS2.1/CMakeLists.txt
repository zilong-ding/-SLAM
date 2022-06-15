cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11")
PROJECT(OPENCV-L)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

ADD_EXECUTABLE(OPENCV-L main.cpp)
target_link_libraries(OPENCV-L ${OpenCV_LIBS})