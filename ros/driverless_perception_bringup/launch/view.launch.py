"""Look at the running simulator in Foxglove. Starts the FSDS ROS 2 bridge, the ground truth node and foxglove_bridge.

Start the simulator first (scripts/sim.sh start TrainingMap), then run

    ros2 launch driverless_perception_bringup view.launch.py

and connect Foxglove to ws://localhost:8765. Arguments: fsds_bridge, foxglove (both default true) and port.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fsds_launch = os.path.join(get_package_share_directory("fsds_ros2_bridge"), "launch", "fsds_ros2_bridge.launch.py")
    foxglove_launch = os.path.join(get_package_share_directory("foxglove_bridge"), "launch", "foxglove_bridge_launch.xml")
    return LaunchDescription([
        DeclareLaunchArgument("fsds_bridge", default_value="true"),
        DeclareLaunchArgument("foxglove", default_value="true"),
        DeclareLaunchArgument("port", default_value="8765"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(fsds_launch),
                                 condition=IfCondition(LaunchConfiguration("fsds_bridge"))),
        Node(package="driverless_perception_race", executable="ground_truth_node", name="ground_truth",
             output="screen"),
        IncludeLaunchDescription(AnyLaunchDescriptionSource(foxglove_launch),
                                 launch_arguments={"port": LaunchConfiguration("port")}.items(),
                                 condition=IfCondition(LaunchConfiguration("foxglove"))),
    ])
