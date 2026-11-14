"""Record a simulator run. The car is driven by ground truth for a number of laps and everything goes into an MCAP bag.

The simulator has to be running already, scripts/record_run.sh starts a fresh one for you. Example:

    ros2 launch driverless_perception_bringup record.launch.py laps:=1 output:=runs/my_run

Once the race supervisor has counted the laps and the car stands still, the supervisor exits. That shuts down this
launch file and closes the bag. Open the .mcap in Foxglove or replay it with scripts/replay_run.sh.

Arguments: laps (default 10), speed of the oracle driver in m/s (default 6.0), output directory for the bag which
must not exist yet (default runs/<date>_<time>), raw_images to also record the raw camera images at roughly
20 MB/s per camera (default false) and foxglove to start foxglove_bridge and watch the run live (default false).

mono:=true also runs the mono pipeline on cam_mono and records its cones, stereo:=true the stereo pipeline on
cam_left and cam_right. Both need detector and keypoints (model files), mono_conf is the detector threshold and
ultralytics_dir the repo's .ultralytics folder. image_rate caps how many frames per second are compressed into the
bag (default 10).
"""
import json
import os
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

FSDS_SETTINGS = os.path.expanduser("~/Formula-Student-Driverless-Simulator/settings.json")

# Everything except the raw camera images. The latched topics (track, GT cones, markers, race state) are recorded
# once at the start, Foxglove keeps showing the last message of each topic.
TOPICS = [
    "^/tf$", "^/tf_static$",
    "^/fsds/(imu|gps|gss|wheel_states|control_command)$",
    "^/fsds/signal/.*",
    "^/fsds/testing_only/.*",
    "^/fsds/lidar/.*",
    "^/fsds/[^/]+/camera_info$",
    "^/fsds/[^/]+/image_color/compressed$",
    "^/ground_truth/.*",
    "^/race/.*",
    "^/perception/.*",
]
RAW_IMAGES = "^/fsds/[^/]+/image_color$"


def camera_names() -> list[str]:
    """The cameras the FSDS bridge will publish, taken from the settings file it reads."""
    with open(FSDS_SETTINGS) as f:
        settings = json.load(f)
    return list(settings["Vehicles"]["FSCar"].get("Cameras", {}).keys())


def recorder(context):
    output = LaunchConfiguration("output").perform(context)
    if not output:
        output = os.path.join("runs", time.strftime("%Y-%m-%d_%H%M%S"))
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    topics = TOPICS + ([RAW_IMAGES] if LaunchConfiguration("raw_images").perform(context) == "true" else [])
    cmd = ["ros2", "bag", "record", "--storage", "mcap", "--storage-preset-profile", "zstd_fast",
           "--output", output, "--regex", "|".join(topics)]
    return [ExecuteProcess(cmd=cmd, output="screen", name="recorder")]


def generate_launch_description():
    fsds_launch = os.path.join(get_package_share_directory("fsds_ros2_bridge"), "launch", "fsds_ros2_bridge.launch.py")
    foxglove_launch = os.path.join(get_package_share_directory("foxglove_bridge"), "launch",
                                   "foxglove_bridge_launch.xml")
    cameras = camera_names()

    # typed, so that speed:=5 isn't handed to the driver as an integer, which it refuses
    laps = ParameterValue(LaunchConfiguration("laps"), value_type=int)
    speed = ParameterValue(LaunchConfiguration("speed"), value_type=float)
    supervisor = Node(package="driverless_perception_race", executable="race_supervisor_node", name="race_supervisor",
                      output="screen", parameters=[{"laps": laps}])

    return LaunchDescription([
        DeclareLaunchArgument("laps", default_value="10"),
        DeclareLaunchArgument("speed", default_value="6.0"),
        DeclareLaunchArgument("output", default_value=""),
        DeclareLaunchArgument("raw_images", default_value="false"),
        DeclareLaunchArgument("foxglove", default_value="false"),
        DeclareLaunchArgument("mono", default_value="false"),
        DeclareLaunchArgument("stereo", default_value="false"),
        DeclareLaunchArgument("image_rate", default_value="10.0"),
        DeclareLaunchArgument("detector", default_value=""),
        DeclareLaunchArgument("keypoints", default_value=""),
        DeclareLaunchArgument("mono_conf", default_value="0.44"),
        DeclareLaunchArgument("ultralytics_dir", default_value=""),

        IncludeLaunchDescription(PythonLaunchDescriptionSource(fsds_launch),
                                 launch_arguments={"mission_name": "trackdrive"}.items()),
        Node(package="driverless_perception_race", executable="ground_truth_node", name="ground_truth",
             output="screen"),
        Node(package="driverless_perception_race", executable="oracle_driver_node", name="oracle_driver",
             output="screen", parameters=[{"speed": speed}]),
        Node(package="driverless_perception_race", executable="image_compressor_node", name="image_compressor",
             output="screen",
             parameters=[{"cameras": cameras,
                          "max_rate": ParameterValue(LaunchConfiguration("image_rate"), value_type=float)}]),
        Node(package="driverless_perception_ros", executable="mono_node", name="mono", output="screen",
             condition=IfCondition(LaunchConfiguration("mono")),
             parameters=[{"camera": "cam_mono", "detector": LaunchConfiguration("detector"),
                          "keypoints": LaunchConfiguration("keypoints"),
                          "conf": ParameterValue(LaunchConfiguration("mono_conf"), value_type=float),
                          "ultralytics_dir": LaunchConfiguration("ultralytics_dir")}]),
        Node(package="driverless_perception_ros", executable="stereo_node", name="stereo", output="screen",
             condition=IfCondition(LaunchConfiguration("stereo")),
             parameters=[{"detector": LaunchConfiguration("detector"),
                          "keypoints": LaunchConfiguration("keypoints"),
                          "conf": ParameterValue(LaunchConfiguration("mono_conf"), value_type=float),
                          "ultralytics_dir": LaunchConfiguration("ultralytics_dir")}]),
        supervisor,
        OpaqueFunction(function=recorder),
        IncludeLaunchDescription(AnyLaunchDescriptionSource(foxglove_launch),
                                 condition=IfCondition(LaunchConfiguration("foxglove"))),
        RegisterEventHandler(OnProcessExit(target_action=supervisor,
                                           on_exit=[EmitEvent(event=Shutdown(reason="run finished"))])),
    ])
