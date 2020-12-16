cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "SIV")

project( ${PROJECT_NAME} )


set(CMAKE_CXX_FLAGS "-std=c++11")

# for finding open cv on mac
# if (APPLE) 
# 	set (OPENCV_EXTRA_MODULES_PATH "/Library/OpenCV/opencv_contrib/modules")

# 	set(OpenCV_DIR "/Library/OpenCV/opencv-4.4.0/build_opencv")
# endif()


# find_package( OpenCV REQUIRED )
find_package( OpenCV REQUIRED opencv_optflow opencv_xfeatures2d) # not CUDA optical flow


message("CV version: " ${OpenCV_VERSION})
message("CV libs : " ${OpenCV_LIBS})
# message("optflow: " ${OpenCV_CUDAOPTFLOW_FOUND})
# message("include dirs: " ${OpenCV_INCLUDE_DIRS})
# message("install path: " ${OpenCV_INSTALL_PATH})
# message("modules: " ${OpenCV_LIB_COMPONENTS})

file(GLOB_RECURSE HELPER_FILES ${PROJECT_SOURCE_DIR}/helpers/* ${PROJECT_SOURCE_DIR}/feature_matching/*)

# message(${HELPER_FILES})


add_executable( "image_pair_optical_flow_estimation_and_vis" image_pair_optical_flow_estimation_and_vis.cpp ${HELPER_FILES} )
target_link_libraries( "image_pair_optical_flow_estimation_and_vis" ${OpenCV_LIBS} )

add_executable( "batch_optical_flow_estimation" batch_optical_flow_estimation.cpp ${HELPER_FILES} )
target_link_libraries( "batch_optical_flow_estimation" ${OpenCV_LIBS} )

add_executable( "visualise_mean_optical_flow" visualise_mean_optical_flow.cpp ${HELPER_FILES} )
target_link_libraries( "visualise_mean_optical_flow" ${OpenCV_LIBS} )

add_executable( "mean_flow_on_path" mean_flow_on_path.cpp ${HELPER_FILES} )
target_link_libraries( "mean_flow_on_path" ${OpenCV_LIBS} )

add_executable( "minima_and_maxima_sift_feature_matching" minima_and_maxima_sift_feature_matching.cpp ${HELPER_FILES} )
target_link_libraries( "minima_and_maxima_sift_feature_matching" ${OpenCV_LIBS} )

add_executable( "of_lukas_kanade" of_lukas_kanade.cpp ${HELPER_FILES} )
target_link_libraries( "of_lukas_kanade" ${OpenCV_LIBS} )

add_executable( "depth_image_of" depth_image_optical_flow.cpp ${HELPER_FILES} )
target_link_libraries( "depth_image_of" ${OpenCV_LIBS} )