"""The stereo pipeline as a ROS node: cones found in the left image, matched and triangulated with the right one.

Reads /fsds/cam_left and /fsds/cam_right (image_color, or image_color/compressed with ``compressed``), their
camera_info and the ground truth odometry. Only images from the same render are paired, which their stamps show:
less than ``max_pair_dt`` apart. Both then get the car's pose at the left image's stamp, like a synchronised rig;
giving each image the pose at its own stamp was three times less accurate. Publishes ConeArray on
/perception/stereo/cones (frame fsds/FSCar, stamped with the left image), markers for Foxglove and where the time
went for every pair on /perception/stereo/timing (see ``timing``; "wait for pair" is the time from the left image's
arrival until the right one and the odometry were there too). Only the newest pair is processed.

Parameters: as for mono_node, plus ``max_range`` (20 m like mono), ``max_pair_dt`` and ``compressed``.
"""
from __future__ import annotations

import os
import time
from collections import deque

import numpy as np
import rclpy
from driverless_perception_msgs.msg import ConeArray, FrameTiming
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from driverless_perception.common.cameras import Intrinsics, camera_pose
from driverless_perception.common.geometry import interpolate_pose_3d
from driverless_perception.common.timing import StageTimer
from driverless_perception.common.types import Cone, Source
from driverless_perception.common.types import ConeArray as Cones
from driverless_perception.vision.mono import color_probs
from driverless_perception.vision.stereo.pipeline import View, position_sigma

from .conversions import (
    IMAGE_QOS,
    compressed_to_bgr,
    cone_markers,
    image_to_bgr,
    stamp_to_ns,
    to_msg,
    transform_to_matrix,
)
from .timing import arrival, pipeline_stages, receive, timing_msg

CAMERAS = ("cam_left", "cam_right")


class StereoNode(Node):
    def __init__(self):
        super().__init__("stereo_node")
        detector = self.declare_parameter("detector", "").value
        keypoints = self.declare_parameter("keypoints", "").value
        conf = float(self.declare_parameter("conf", 0.44).value)
        min_height = int(self.declare_parameter("min_height", 20).value)
        pnp = self.declare_parameter("pnp", "upright").value
        jpeg_quality = int(self.declare_parameter("jpeg_quality", 95).value)
        half = bool(self.declare_parameter("half", True).value)
        compile_mode = self.declare_parameter("compile", "reduce-overhead").value
        device = self.declare_parameter("device", "cuda").value
        ultralytics_dir = self.declare_parameter("ultralytics_dir", "").value
        self.car_frame = self.declare_parameter("car_frame", "fsds/FSCar").value
        max_range = float(self.declare_parameter("max_range", 20.0).value)
        # the images of one render are read back one after another, up to about 16 ms apart; the next render is a whole
        # frame (about 70 ms) later
        self.max_pair_dt = int(1e9 * float(self.declare_parameter("max_pair_dt", 0.025).value))
        compressed = bool(self.declare_parameter("compressed", False).value)
        if not detector or not keypoints:
            raise ValueError("the detector and keypoints parameters need model files")
        if ultralytics_dir:
            os.environ["YOLO_CONFIG_DIR"] = ultralytics_dir
        from ultralytics import YOLO

        from driverless_perception.vision.keypoints.train import load_model
        from driverless_perception.vision.mono import MonoPipeline
        from driverless_perception.vision.stereo.pipeline import StereoPipeline

        self.timer = StageTimer()
        # compressed images already are JPEG
        mono = MonoPipeline(load_model(keypoints, device), device, min_height, YOLO(detector), conf, timer=self.timer,
                            pnp=pnp, jpeg_quality=None if compressed else jpeg_quality or None, half=half,
                            compile=compile_mode)
        if half or compile_mode:
            self.get_logger().info("warming up the detector")
            mono.warm_up()
        self.pipeline = StereoPipeline(mono, max_range)
        self.intrinsics = {}
        self.mounts = {}
        self.images = {camera: deque(maxlen=5) for camera in CAMERAS}
        self.odom = deque(maxlen=400)
        self.last_left = 0
        self.times = []
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.pub = self.create_publisher(ConeArray, "/perception/stereo/cones", 10)
        self.markers = self.create_publisher(MarkerArray, "/perception/stereo/markers", 10)
        self.timing = self.create_publisher(FrameTiming, "/perception/stereo/timing", 10)
        self.create_subscription(Odometry, "/fsds/testing_only/odom", self.on_odom, 100)
        for camera in CAMERAS:
            self.create_subscription(CameraInfo, f"/fsds/{camera}/camera_info",
                                     lambda msg, camera=camera: self.on_info(camera, msg), 10)
            if compressed:
                self.create_subscription(CompressedImage, f"/fsds/{camera}/image_color/compressed",
                                         lambda data, info, camera=camera: self.on_image(camera, data, info,
                                                                                         CompressedImage),
                                         10, raw=True)
            else:
                self.create_subscription(Image, f"/fsds/{camera}/image_color",
                                         lambda data, info, camera=camera: self.on_image(camera, data, info, Image),
                                         IMAGE_QOS, raw=True)
        self.decode = compressed_to_bgr if compressed else image_to_bgr
        self.get_logger().info(f"stereo pipeline up to {max_range:.0f} m, detector {os.path.basename(detector)}")

    def on_info(self, camera: str, msg: CameraInfo) -> None:
        # the bridge's camera_info has the principal point in the middle and square pixels, like Intrinsics
        fov = np.degrees(2 * np.arctan(msg.width / 2 / msg.k[0]))
        self.intrinsics[camera] = Intrinsics(msg.width, msg.height, float(fov))

    def on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.odom.append((stamp_to_ns(msg.header.stamp), [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
        self.process()

    def on_image(self, camera: str, data: bytes, info: dict, msg_type) -> None:
        received = receive(data, info, msg_type)
        msg = received.msg
        if camera not in self.mounts:
            try:
                t = self.tf.lookup_transform(self.car_frame, msg.header.frame_id, Time()).transform
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"no transform {self.car_frame} <- {msg.header.frame_id} yet: {e}",
                                       throttle_duration_sec=5)
                return
            self.mounts[camera] = transform_to_matrix(t)
        self.images[camera].append(received)
        self.process()

    def car_pose(self, stamp: int) -> tuple[np.ndarray, np.ndarray] | None:
        """The car's position and rotation matrix at ``stamp``, interpolated from the odometry."""
        times = np.array([o[0] for o in self.odom])
        return interpolate_pose_3d(stamp, times, np.array([o[1] for o in self.odom]),
                                   np.array([o[2] for o in self.odom]))

    def view(self, camera: str, pose: tuple[np.ndarray, np.ndarray], image: np.ndarray) -> View:
        position, orientation = camera_pose(*pose, *self.mounts[camera])
        # the car's origin is on the ground
        return View(image, self.intrinsics[camera], position, orientation, float(pose[0][2]))

    def process(self) -> None:
        """Runs the pipeline on the newest left image once a right image from the same render has come and the
        odometry has caught up."""
        if len(self.intrinsics) < 2 or len(self.mounts) < 2 or not self.images["cam_left"]:
            return
        left = self.images["cam_left"][-1]
        t_left = left.stamp_ns
        if t_left <= self.last_left or not self.images["cam_right"] or not self.odom:
            return
        right = min(self.images["cam_right"], key=lambda r: abs(r.stamp_ns - t_left))
        if abs(right.stamp_ns - t_left) > self.max_pair_dt or self.odom[-1][0] < t_left:
            return
        self.last_left = t_left
        pose = self.car_pose(t_left)
        if pose is None:
            return
        # from the moment the left image was ready until its partner and the odometry had come as well
        left_ready = left.taken_ns + 1e6 * left.deserialize_ms
        wait_ms = (time.time_ns() - left_ready) / 1e6
        start = time.perf_counter()
        views = [self.view(c, pose, self.decode(r.msg)) for c, r in (("cam_left", left), ("cam_right", right))]
        decode_ms = 1000 * (time.perf_counter() - start)
        if any(v.image is None for v in views):
            return
        position, rotation = pose
        self.timer.frame()
        found = self.pipeline(*views)
        stages, boxes = pipeline_stages(self.timer.frame())
        self.timer.samples.clear()
        publish_start = time.perf_counter()
        cones = []
        for c in found:
            x, y, _ = rotation.T @ (np.array([*c.position, position[2]]) - position)
            sigma = position_sigma(float(np.hypot(x, y)))
            cones.append(Cone(float(x), float(y), color_probs(c.color, c.score), np.eye(2) * sigma ** 2))
        self.pub.publish(to_msg(Cones(t_left, Source.STEREO, cones, self.car_frame)))
        self.markers.publish(cone_markers(cones, left.msg.header.stamp, self.car_frame, "stereo", Marker.CYLINDER))
        publish_ms = 1000 * (time.perf_counter() - publish_start)
        self.timing.publish(timing_msg(t_left, [*arrival(left), ("wait for pair", wait_ms),
                                                *arrival(right, "right image/"), ("decode", decode_ms), *stages,
                                                ("publish", publish_ms)], boxes))
        self.times.append(time.perf_counter() - start)
        if len(self.times) == 100:
            self.get_logger().info(f"{1000 * np.median(self.times):.0f} ms per pair (median of 100)")
            self.times.clear()


def main() -> None:
    rclpy.init()
    node = StereoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
