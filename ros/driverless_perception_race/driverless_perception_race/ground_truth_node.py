"""Publishes ground truth for the rest of the stack. We don't estimate the car's pose, it comes from here.

The FSDS bridge only publishes static transforms, so this node turns /fsds/testing_only/odom into the transform
fsds/map -> fsds/FSCar. It also republishes the latched ground truth track as a ConeArray (/ground_truth/cones) and
as cylinder markers (/ground_truth/cone_markers) for Foxglove and RViz, both in fsds/map.
"""
from __future__ import annotations

import rclpy
from driverless_perception_msgs.msg import ConeArray as ConeArrayMsg
from driverless_perception_ros.conversions import stamp_to_ns, to_msg
from fs_msgs.msg import Track
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from driverless_perception.common.types import Cone, ConeArray, ConeColor, Source

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
PROB_INDEX = {ConeColor.BLUE: 0, ConeColor.YELLOW: 1, ConeColor.ORANGE_BIG: 2, ConeColor.ORANGE_SMALL: 2}
# RGBA per ConeColor and marker size (diameter, height) [m]: small cone 228 x 325 mm, big orange 285 x 505 mm
STYLE = {
    ConeColor.BLUE: ((0.1, 0.35, 0.95, 1.0), (0.228, 0.325)),
    ConeColor.YELLOW: ((0.98, 0.8, 0.05, 1.0), (0.228, 0.325)),
    ConeColor.ORANGE_BIG: ((1.0, 0.45, 0.0, 1.0), (0.285, 0.505)),
    ConeColor.ORANGE_SMALL: ((1.0, 0.45, 0.0, 1.0), (0.228, 0.325)),
    ConeColor.UNKNOWN: ((0.6, 0.6, 0.6, 1.0), (0.228, 0.325)),
}


class GroundTruthNode(Node):
    def __init__(self):
        super().__init__("ground_truth")
        self.map_frame = self.declare_parameter("map_frame", "fsds/map").value
        self.car_frame = self.declare_parameter("car_frame", "fsds/FSCar").value
        self.tf = TransformBroadcaster(self)
        self.cones_pub = self.create_publisher(ConeArrayMsg, "/ground_truth/cones", LATCHED)
        self.markers_pub = self.create_publisher(MarkerArray, "/ground_truth/cone_markers", LATCHED)
        self.create_subscription(Odometry, "/fsds/testing_only/odom", self.on_odom, 10)
        self.create_subscription(Track, "/fsds/testing_only/track", self.on_track, LATCHED)

    def on_odom(self, msg: Odometry) -> None:
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.car_frame
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
        t.transform.rotation = q
        self.tf.sendTransform(t)

    def on_track(self, msg: Track) -> None:
        stamp_ns = stamp_to_ns(self.get_clock().now().to_msg())
        cones = []
        for c in msg.track:
            probs = [0.0, 0.0, 0.0, 0.0]  # blue, yellow, orange, unknown
            probs[PROB_INDEX.get(ConeColor(c.color), 3)] = 1.0
            cones.append(Cone(c.location.x, c.location.y, color_probs=probs))
        gt = ConeArray(stamp_ns, Source.GROUND_TRUTH, cones, frame_id=self.map_frame)
        out = to_msg(gt)
        for m, c in zip(out.cones, msg.track):  # keep the exact GT class (small vs big orange)
            m.color = c.color
        self.cones_pub.publish(out)

        markers = MarkerArray()
        for i, c in enumerate(msg.track):
            (r, g, b, a), (d, h) = STYLE[ConeColor(c.color)]
            m = Marker()
            m.header.frame_id = self.map_frame
            m.ns, m.id, m.type, m.action = "gt_cones", i, Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = c.location.x, c.location.y, h / 2
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = d
            m.scale.z = h
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            markers.markers.append(m)
        self.markers_pub.publish(markers)
        self.get_logger().info(f"ground truth track: {len(msg.track)} cones")


def main() -> None:
    rclpy.init()
    node = GroundTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
