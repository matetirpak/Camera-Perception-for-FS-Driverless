"""ROS 2 nodes for the perception pipelines. They read the /fsds sensor topics, call driverless_perception and
publish a driverless_perception_msgs/ConeArray on /perception/<lidar|mono|stereo>/cones.

The nodes get added as the pipelines are built (see docs/roadmap.md): mono_node, stereo_node, lidar_node, a node for
synthetic LiDAR intensity and one that switches sensors off during a run.
"""
