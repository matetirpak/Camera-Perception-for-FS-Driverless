"""Counts laps, stops the car after the requested number and ends the run.

Laps are counted from our own crossings of the start line (the segment between the big orange cones) using the
ground-truth odometry. The FSDS referee is not reliable for this: its first lap entry can be a partial lap
depending on where the car spawns, and it keeps running across resets. A crossing only counts as a lap if the car
drove at least half the track length since the previous one, so the first crossing right after the start does not.

When the run is over the node publishes /fsds/signal/finished and exits, which shuts down the launch file.
"""
from __future__ import annotations

import json

import numpy as np
import rclpy
from fs_msgs.msg import ExtraInfo, FinishedSignal, Track
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

from driverless_perception.sim.ground_truth import centerline, crossed, start_line

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class RaceSupervisor(Node):
    def __init__(self):
        super().__init__("race_supervisor")
        self.laps = int(self.declare_parameter("laps", 10).value)
        self.state = None
        self.line = None
        self.track_length = None
        self.prev_pos = None
        self.dist_since_line = 0.0
        self.last_line_time = None
        self.lap_times: list[float] = []
        self.stopped_since = None
        self.referee = None

        self.state_pub = self.create_publisher(String, "/race/state", LATCHED)
        self.lap_pub = self.create_publisher(Int32, "/race/lap", LATCHED)
        self.finished_pub = self.create_publisher(FinishedSignal, "/fsds/signal/finished", 1)
        self.create_subscription(Track, "/fsds/testing_only/track", self.on_track, LATCHED)
        self.create_subscription(Odometry, "/fsds/testing_only/odom", self.on_odom, 10)
        self.create_subscription(ExtraInfo, "/fsds/testing_only/extra_info", self.on_extra_info, 10)
        self.set_state("waiting")
        self.lap_pub.publish(Int32(data=0))
        self.done = False

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.state_pub.publish(String(data=state))
            self.get_logger().info(f"state: {state}")

    def on_track(self, msg: Track) -> None:
        xy = np.array([[c.location.x, c.location.y] for c in msg.track])
        colors = np.array([c.color for c in msg.track])
        self.line = start_line(xy, colors)
        path = centerline(xy, colors)
        self.track_length = float(np.sum(np.linalg.norm(np.diff(np.vstack([path, path[:1]]), axis=0), axis=1)))
        self.get_logger().info(f"track length {self.track_length:.0f} m, {self.laps} laps")

    def on_extra_info(self, msg: ExtraInfo) -> None:
        self.referee = {"laps": list(msg.laps), "doo_counter": int(msg.doo_counter)}

    def on_odom(self, msg: Odometry) -> None:
        if self.line is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        speed = abs(msg.twist.twist.linear.x)
        if self.state == "waiting":
            self.last_line_time = t
            self.set_state("racing")
        if self.prev_pos is not None:
            self.dist_since_line += float(np.linalg.norm(pos - self.prev_pos))
            if self.state == "racing" and crossed(self.prev_pos, pos, *self.line):
                if self.dist_since_line > 0.5 * self.track_length:
                    self.lap_times.append(t - self.last_line_time)
                    self.lap_pub.publish(Int32(data=len(self.lap_times)))
                    self.get_logger().info(f"lap {len(self.lap_times)}: {self.lap_times[-1]:.2f} s")
                self.dist_since_line = 0.0
                self.last_line_time = t
                if len(self.lap_times) >= self.laps:
                    self.set_state("stopping")
        self.prev_pos = pos

        if self.state == "stopping":
            if speed < 0.1:
                self.stopped_since = self.stopped_since or t
                if t - self.stopped_since > 1.0:
                    self.finish()
            else:
                self.stopped_since = None

    def finish(self) -> None:
        self.set_state("finished")
        self.finished_pub.publish(FinishedSignal())
        summary = {"laps": len(self.lap_times), "lap_times_s": [round(x, 3) for x in self.lap_times],
                   "referee": self.referee}
        self.get_logger().info("run summary: " + json.dumps(summary))
        self.done = True


def main() -> None:
    rclpy.init()
    node = RaceSupervisor()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        # give the latched state and the finished signal a moment to reach the recorder
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
