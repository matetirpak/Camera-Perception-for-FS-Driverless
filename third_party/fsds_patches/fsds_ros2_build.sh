#!/bin/bash
# Builds the FSDS ROS 2 workspace (fs_msgs and fsds_ros2_bridge). Copy this to ros2/build.sh in the FSDS repo.
# The system python is set explicitly so CMake doesn't pick up another python without the ROS packages.
# Needs AirSim/AirLib/deps/eigen3 (downloaded by AirSim/setup.sh).
set -e
cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3 "$@"
