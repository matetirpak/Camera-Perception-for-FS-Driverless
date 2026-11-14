"""Republishes FSDS camera images as JPEG so a recorded run stays small.

Raw images are about 20 MB/s per camera, JPEG at 10 frames per second a few percent of that, and Foxglove shows
sensor_msgs/CompressedImage directly. For every camera this reads /fsds/<name>/image_color and writes
/fsds/<name>/image_color/compressed.

The rate limit counts frames, not images: once an image is let through, the other cameras of the same render (any
image within ``frame_window``) are let through too. Throttling each camera on its own left only 39% of the
recorded frames with a stereo pair. The pipelines read the raw images anyway, this is for watching a run.
"""
from __future__ import annotations

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

CHANNELS = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}


class ImageCompressor(Node):
    def __init__(self):
        super().__init__("image_compressor")
        cameras = self.declare_parameter("cameras", ["cam1", "cam2"]).value
        self.quality = int(self.declare_parameter("jpeg_quality", 80).value)
        self.min_period = 1.0 / float(self.declare_parameter("max_rate", 10.0).value)
        # images whose stamps are this close come from the same render (see docs/notes/simulator-notes.md)
        self.frame_window = float(self.declare_parameter("frame_window", 0.025).value)
        self.frame_start = None
        self.last_frame = -1e9
        self.sent_in_frame: set[str] = set()
        self.pubs = {}
        for name in cameras:
            topic = f"/fsds/{name}/image_color"
            self.pubs[topic] = self.create_publisher(CompressedImage, topic + "/compressed", 5)
            self.create_subscription(Image, topic, lambda msg, t=topic: self.on_image(t, msg), 5)
        self.get_logger().info(f"compressing {list(self.pubs)} at up to {1 / self.min_period:.0f} frames per "
                               f"second, cameras within {1000 * self.frame_window:.0f} ms kept together")

    def on_image(self, topic: str, msg: Image) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.frame_start is not None and abs(t - self.frame_start) <= self.frame_window:
            if topic in self.sent_in_frame:
                return
        elif t - self.last_frame >= self.min_period:
            self.frame_start, self.last_frame = t, t
            self.sent_in_frame = set()
        else:
            return
        channels = CHANNELS.get(msg.encoding)
        if channels is None:
            self.get_logger().warn(f"unsupported encoding {msg.encoding} on {topic}", once=True)
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // channels, channels)[:, : msg.width]
        if msg.encoding.startswith("rgb"):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if channels == 3 else cv2.COLOR_RGBA2BGR)
        elif channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        out = CompressedImage()
        out.header = msg.header
        out.format = "jpeg"
        out.data = jpg.tobytes()
        self.pubs[topic].publish(out)
        self.sent_in_frame.add(topic)


def main() -> None:
    rclpy.init()
    node = ImageCompressor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
