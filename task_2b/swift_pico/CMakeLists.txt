cmake_minimum_required(VERSION 3.8)
project(swift_pico)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(waypoint_navigation REQUIRED)

# Installed WITHOUT .py so that "ros2 run swift_pico pico_server" works
install(PROGRAMS src/waypoint_service.py DESTINATION lib/${PROJECT_NAME} RENAME waypoint_service)
install(PROGRAMS src/pico_client.py      DESTINATION lib/${PROJECT_NAME} RENAME pico_client)
install(PROGRAMS src/pico_server.py      DESTINATION lib/${PROJECT_NAME} RENAME pico_server)
# only keep this line if src/pico_controller_PID.py really exists
install(PROGRAMS src/pico_controller_PID.py DESTINATION lib/${PROJECT_NAME} RENAME pico_controller_PID)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
