"""The mono pipeline as a ROS node: detector, keypoints and PnP on one camera, cones in the car frame.

Reads /fsds/<camera>/image_color and camera_info, publishes driverless_perception_msgs/ConeArray on
/perception/mono/cones (frame fsds/FSCar, stamped with the image's time, source MONO), markers for Foxglove on
/perception/mono/markers and where the time went for every frame on /perception/mono/timing (see ``timing``).

Parameters: ``camera`` (cam_mono), ``detector`` and ``keypoints`` (model files), ``conf`` (detector threshold),
``min_height`` (smallest box in pixels that gets keypoints), ``pnp`` (upright, sqpnp or ground, see
``vision.pnp``), ``edge_px`` (boxes this close to the left or right border are left out), ``jpeg_quality``
(keypoints from a JPEG version of the area around each box like the training captures, 0 switches it off,
``vision.mono.jpeg_like``), ``half`` and ``compile`` (the detector in half precision and through torch.compile
with that mode, by default true and reduce-overhead, false and an empty string for plain fp32; compiling happens at
start-up and takes several seconds), ``device`` and ``ultralytics_dir`` (where
ultralytics keeps its settings, the repo's .ultralytics).
"""
from __future__ import annotations

import os
import time

import numpy as np
import rclpy
from driverless_perception_msgs.msg import ConeArray, FrameTiming
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
from visualization_msgs.msg import MarkerArray

from driverless_perception.common.timing import StageTimer
from driverless_perception.common.types import Cone, Source
from driverless_perception.common.types import ConeArray as Cones
from driverless_perception.vision.mono import color_probs, position_sigma

from .conversions import IMAGE_QOS, cone_markers, image_to_bgr, to_msg, transform_to_matrix
from .timing import arrival, pipeline_stages, receive, timing_msg


class MonoNode(Node):
    def __init__(self):
        super().__init__("mono_node")
        self.camera = self.declare_parameter("camera", "cam_mono").value
        detector = self.declare_parameter("detector", "").value
        keypoints = self.declare_parameter("keypoints", "").value
        conf = float(self.declare_parameter("conf", 0.44).value)
        min_height = int(self.declare_parameter("min_height", 20).value)
        pnp = self.declare_parameter("pnp", "upright").value
        jpeg_quality = int(self.declare_parameter("jpeg_quality", 95).value)
        edge_px = float(self.declare_parameter("edge_px", 2.0).value)
        half = bool(self.declare_parameter("half", True).value)
        compile_mode = self.declare_parameter("compile", "reduce-overhead").value
        device = self.declare_parameter("device", "cuda").value
        ultralytics_dir = self.declare_parameter("ultralytics_dir", "").value
        self.car_frame = self.declare_parameter("car_frame", "fsds/FSCar").value
        if not detector or not keypoints:
            raise ValueError("the detector and keypoints parameters need model files")
        if ultralytics_dir:
            os.environ["YOLO_CONFIG_DIR"] = ultralytics_dir
        from ultralytics import YOLO

        from driverless_perception.vision.keypoints.train import load_model
        from driverless_perception.vision.mono import MonoPipeline

        self.timer = StageTimer()
        self.pipeline = MonoPipeline(load_model(keypoints, device), device, min_height, YOLO(detector), conf,
                                     timer=self.timer, pnp=pnp, edge_px=edge_px,
                                     jpeg_quality=jpeg_quality or None, half=half, compile=compile_mode)
        if half or compile_mode:
            self.get_logger().info("warming up the detector")
            self.pipeline.warm_up()
        self.K = None
        self.extrinsic = None
        self.times = []
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.pub = self.create_publisher(ConeArray, "/perception/mono/cones", 10)
        self.markers = self.create_publisher(MarkerArray, "/perception/mono/markers", 10)
        self.timing = self.create_publisher(FrameTiming, "/perception/mono/timing", 10)
        self.create_subscription(CameraInfo, f"/fsds/{self.camera}/camera_info", self.on_info, 10)
        self.create_subscription(Image, f"/fsds/{self.camera}/image_color", self.on_image, IMAGE_QOS,
                                 raw=True)
        self.get_logger().info(f"mono pipeline on {self.camera}, detector {os.path.basename(detector)}, PnP {pnp}")

    def on_info(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, float).reshape(3, 3)

    def camera_to_car(self, frame: str):
        """Rotation and translation from the camera's optical frame to the car frame (a static transform)."""
        if self.extrinsic is None:
            try:
                t = self.tf.lookup_transform(self.car_frame, frame, Time()).transform
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"no transform {self.car_frame} <- {frame} yet: {e}", throttle_duration_sec=5)
                return None
            self.extrinsic = transform_to_matrix(t)
        return self.extrinsic

    def on_image(self, data: bytes, info: dict) -> None:
        received = receive(data, info, Image)
        msg = received.msg
        extrinsic = self.camera_to_car(msg.header.frame_id)
        start = time.perf_counter()
        image = image_to_bgr(msg)
        decode_ms = 1000 * (time.perf_counter() - start)
        if self.K is None or extrinsic is None or image is None:
            return
        R, t = extrinsic
        # "up" of the car in the optical frame: the upright cone doesn't depend on the car's pitch and roll
        up = R.T @ np.array([0.0, 0.0, 1.0])
        self.timer.frame()
        found = self.pipeline(image, self.K, up)
        stages, boxes = pipeline_stages(self.timer.frame())
        self.timer.samples.clear()
        publish_start = time.perf_counter()
        cones = []
        for c in found:
            x, y, _ = R @ c.position + t
            sigma = position_sigma(float(np.linalg.norm(c.position)))
            cones.append(Cone(float(x), float(y), color_probs(c.color, c.score), np.eye(2) * sigma ** 2))
        out = Cones(received.stamp_ns, Source.MONO, cones, self.car_frame)
        self.pub.publish(to_msg(out))
        self.markers.publish(cone_markers(cones, msg.header.stamp, self.car_frame, "mono"))
        publish_ms = 1000 * (time.perf_counter() - publish_start)
        self.timing.publish(timing_msg(received.stamp_ns, [*arrival(received), ("decode", decode_ms), *stages,
                                                            ("publish", publish_ms)], boxes))
        self.times.append(time.perf_counter() - start)
        if len(self.times) == 100:
            self.get_logger().info(f"{1000 * np.median(self.times):.0f} ms per image (median of 100)")
            self.times.clear()


def main() -> None:
    rclpy.init()
    node = MonoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
