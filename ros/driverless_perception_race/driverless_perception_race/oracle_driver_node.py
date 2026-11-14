"""Drives the car around the ground-truth centre line so the perception pipelines only have to observe.

Steering is pure pursuit with a speed dependent lookahead, the target speed comes from a curvature based speed
profile. The node brakes to a stop once /race/state says "stopping" or "finished".
"""
from __future__ import annotations

import numpy as np
import rclpy
from fs_msgs.msg import ControlCommand, Track
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from driverless_perception.common.geometry import yaw_from_quaternion
from driverless_perception.common.types import ConeColor
from driverless_perception.sim.ground_truth import centerline, nearest_index, pure_pursuit, speed_profile

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class OracleDriver(Node):
    def __init__(self):
        super().__init__("oracle_driver")
        self.v_max = self.declare_parameter("speed", 6.0).value
        self.a_lat = self.declare_parameter("max_lateral_accel", 4.0).value
        self.lookahead_min = self.declare_parameter("lookahead_min", 3.0).value
        self.lookahead_time = self.declare_parameter("lookahead_time", 0.5).value
        self.kp = self.declare_parameter("throttle_gain", 0.25).value

        self.path = None
        self.v_target = None
        self.odom = None
        self.stop = False
        self.cmd_pub = self.create_publisher(ControlCommand, "/fsds/control_command", 1)
        self.create_subscription(Track, "/fsds/testing_only/track", self.on_track, LATCHED)
        self.create_subscription(Odometry, "/fsds/testing_only/odom", self.on_odom, 10)
        self.create_subscription(String, "/race/state", self.on_state, LATCHED)
        self.create_timer(0.05, self.step)

    def on_track(self, msg: Track) -> None:
        xy = np.array([[c.location.x, c.location.y] for c in msg.track])
        colors = np.array([c.color for c in msg.track])
        if not ((colors == ConeColor.BLUE).any() and (colors == ConeColor.YELLOW).any()):
            self.get_logger().error("track has no blue/yellow cones, cannot build a centre line")
            return
        self.path = centerline(xy, colors)
        self.v_target = speed_profile(self.path, self.v_max, self.a_lat)
        self.get_logger().info(f"centre line with {len(self.path)} points, target speed "
                               f"{self.v_target.min():.1f} to {self.v_target.max():.1f} m/s")

    def on_odom(self, msg: Odometry) -> None:
        self.odom = msg

    def on_state(self, msg: String) -> None:
        self.stop = msg.data in ("stopping", "finished")

    def step(self) -> None:
        if self.path is None or self.odom is None:
            return
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        pose = (p.x, p.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        v = self.odom.twist.twist.linear.x

        cmd = ControlCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        lookahead = max(self.lookahead_min, self.lookahead_time * v)
        cmd.steering = pure_pursuit(self.path, pose, lookahead=lookahead)
        if self.stop:
            cmd.throttle, cmd.brake = 0.0, 1.0
        else:
            target = float(self.v_target[nearest_index(self.path, p.x, p.y)])
            error = target - v
            cmd.throttle = float(np.clip(self.kp * error, 0.0, 0.6))
            cmd.brake = float(np.clip(-0.3 * error, 0.0, 1.0)) if error < -0.5 else 0.0
        self.cmd_pub.publish(cmd)


def main() -> None:
    rclpy.init()
    node = OracleDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
