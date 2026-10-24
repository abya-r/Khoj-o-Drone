cmake_minimum_required(VERSION 3.8)
project(swift_pico)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(waypoint_navigation REQUIRED)

# Install Python scripts as executables
install(PROGRAMS
  src/waypoint_service.py
  src/pico_client.py
  src/pico_server.py
  src/pico_controller_PID.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files and other directories if needed
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
