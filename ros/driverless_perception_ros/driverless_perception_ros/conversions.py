"""driverless_perception.common.types <-> driverless_perception_msgs conversions, and other message helpers."""
from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time
from driverless_perception_msgs.msg import Cone as ConeMsg
from driverless_perception_msgs.msg import ConeArray as ConeArrayMsg
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from driverless_perception.common.types import CONE_RGB, Cone, ConeArray, Source

_CHANNELS = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}

# For the raw camera images. A 1280x960 image is 3.7 MB, DDS splits it into many fragments, and with best effort
# (the usual sensor data profile) losing one fragment loses the whole image: 7% of the mono images never arrived in
# a race, and stereo, which needs both images of a pair, lost about a quarter. Reliable delivery resends the
# fragments. Keeping the last two means a node that falls behind gets the newest image, not a backlog.
IMAGE_QOS = QoSProfile(depth=2, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
MARKER_COLOURS = {int(c): rgb for c, rgb in CONE_RGB.items()}


def image_to_bgr(msg) -> np.ndarray | None:
    """sensor_msgs/Image to an (H, W, 3) BGR array, None for encodings other than 8 bit colour."""
    import cv2

    channels = _CHANNELS.get(msg.encoding)
    if channels is None:
        return None
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // channels, channels)[:, : msg.width]
    if msg.encoding.startswith("rgb"):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR if channels == 3 else cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if channels == 4 else np.ascontiguousarray(img)


def compressed_to_bgr(msg) -> np.ndarray | None:
    """sensor_msgs/CompressedImage (jpeg or png) to an (H, W, 3) BGR array."""
    import cv2

    return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)


def transform_to_matrix(transform) -> tuple[np.ndarray, np.ndarray]:
    """geometry_msgs/Transform as rotation matrix and translation."""
    from scipy.spatial.transform import Rotation

    q, p = transform.rotation, transform.translation
    return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([p.x, p.y, p.z])


def cone_markers(cones: list[Cone], stamp: Time, frame_id: str, ns: str, shape: int = Marker.SPHERE) -> MarkerArray:
    """One marker per cone in its most likely colour, after a marker that clears the previous ones."""
    markers = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)
    for i, c in enumerate(cones):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = frame_id
        m.ns, m.id, m.type, m.action = ns, i, shape, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(c.x), float(c.y), 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        r, g, b = MARKER_COLOURS[int(c.color)]
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.9
        markers.markers.append(m)
    return markers


def stamp_to_ns(stamp: Time) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def ns_to_stamp(ns: int) -> Time:
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def to_msg(cones: ConeArray) -> ConeArrayMsg:
    msg = ConeArrayMsg()
    msg.header.stamp = ns_to_stamp(cones.stamp_ns)
    msg.header.frame_id = cones.frame_id
    msg.source = int(cones.source)
    for c in cones.cones:
        m = ConeMsg()
        m.position.x, m.position.y = float(c.x), float(c.y)
        m.covariance = [float(v) for v in np.asarray(c.covariance).reshape(4)]
        m.color_probs = [float(v) for v in np.asarray(c.color_probs).reshape(4)]
        m.color = int(c.color)
        msg.cones.append(m)
    return msg


def from_msg(msg: ConeArrayMsg) -> ConeArray:
    cones = [Cone(m.position.x, m.position.y, np.array(m.color_probs, dtype=float),
                  np.array(m.covariance, dtype=float).reshape(2, 2)) for m in msg.cones]
    return ConeArray(stamp_to_ns(msg.header.stamp), Source(msg.source), cones, msg.header.frame_id)
