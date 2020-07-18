cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "CudaOpticalFlow")

project( ${PROJECT_NAME} )


set(CMAKE_CXX_FLAGS "-std=c++11")


# find_package( OpenCV REQUIRED )
find_package( OpenCV REQUIRED opencv_optflow) # not CUDA optical flow


# message("CV version: " ${OpenCV_VERSION})
# message("optflow: " ${OpenCV_CUDAOPTFLOW_FOUND})
# message("include dirs: " ${OpenCV_INCLUDE_DIRS})
# message("install path: " ${OpenCV_INSTALL_PATH})
# message("modules: " ${OpenCV_LIB_COMPONENTS})

add_executable( ${PROJECT_NAME} main.cpp OpenCVHelper.hpp )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )