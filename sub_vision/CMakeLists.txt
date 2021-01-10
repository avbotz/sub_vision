cmake_minimum_required(VERSION 3.5)
project(sub_vision)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sub_vision_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

set(SPINNAKER_INCLUDE_DIR "/usr/include/spinnaker" CACHE PATH "Spinnaker Include Directory")
set(SPINNAKER_LIB_DIR "/usr/lib" CACHE PATH "Spinnaker Libs Directory")

# find_library(FLYCAPTURE2 flycapture)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
    ${SPINNAKER_INCLUDE_DIR})
link_directories(${SPINNAKER_LIB_DIR})

add_executable(vision_node
    src/vision.cpp
    src/service.cpp
    src/filters.cpp
    src/log.cpp
    src/gate.cpp
    src/bins.cpp
    src/model.cpp
    src/tensor.cpp
    src/target.cpp
)

add_executable(acquisition_node
    src/acquisition.cpp
    src/log.cpp
    src/filters.cpp
)

foreach(node IN ITEMS vision_node acquisition_node)
    target_include_directories(${node}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )

    ament_target_dependencies(${node}
        "rclcpp"
        "sub_vision_interfaces"
        "cv_bridge"
        "image_transport"
        "geometry_msgs"
    )

    target_link_libraries(${node}
        ${OpenCV_LIBARIES}
        Spinnaker
        -ltensorflow
    )

    install(TARGETS
        ${node}
        DESTINATION lib/${PROJECT_NAME}
    )
endforeach()

# Install launch files
# install(DIRECTORY
#     launch
#     DESTINATION share/${PROJECT_NAME}/
# )

ament_package()
