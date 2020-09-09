cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "OpticalFlow")

project( ${PROJECT_NAME} )



set(CMAKE_CXX_FLAGS "-std=c++11")

# for finding open cv on mac
if (APPLE) 
	set (OPENCV_EXTRA_MODULES_PATH "/Library/OpenCV/opencv_contrib/modules")

	set(OpenCV_DIR "/Library/OpenCV/opencv-4.4.0/build_opencv")
endif()


# find_package( OpenCV REQUIRED )
find_package( OpenCV REQUIRED opencv_optflow opencv_xfeatures2d) # not CUDA optical flow


# message("CV version: " ${OpenCV_VERSION})
# message("optflow: " ${OpenCV_CUDAOPTFLOW_FOUND})
# message("include dirs: " ${OpenCV_INCLUDE_DIRS})
# message("install path: " ${OpenCV_INSTALL_PATH})
# message("modules: " ${OpenCV_LIB_COMPONENTS})

file(GLOB_RECURSE HELPER_FILES ${PROJECT_SOURCE_DIR}/*.hpp)

# set(PROJECT_INCLUDES "${PROJECT_INCLUDES_H}" "${PROJECT_INCLUDES_HPP}")


add_executable( "ofComparison" single_comparison_and_vis.cpp ${HELPER_FILES} )
target_link_libraries( "ofComparison" ${OpenCV_LIBS} )

add_executable( "ofBatchComparison" batch_comparison.cpp ${HELPER_FILES} )
target_link_libraries( "ofBatchComparison" ${OpenCV_LIBS} )

add_executable( "visMean" vis_mean_optical_flow.cpp ${HELPER_FILES} )
target_link_libraries( "visMean" ${OpenCV_LIBS} )

add_executable( "pathFlow" mean_flow_on_path.cpp ${HELPER_FILES} )
target_link_libraries( "pathFlow" ${OpenCV_LIBS} )

add_executable( "trackMinMax" minima_and_maxima_tracking.cpp ${HELPER_FILES} )
target_link_libraries( "trackMinMax" ${OpenCV_LIBS} )
