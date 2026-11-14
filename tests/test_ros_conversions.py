import os

import numpy as np
import pytest

from driverless_perception.common.types import Cone, ConeArray, Source

# importing the message packages without this repo's workspace sourced takes the interpreter down with it, so the
# module keeps away from them until the environment is there, at collection time as well
WORKSPACE = any("driverless_perception" in p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(":"))
pytestmark = pytest.mark.skipif(not WORKSPACE, reason="needs this repo's ROS workspace sourced")


def test_stamp_conversion_roundtrip():
    from driverless_perception_ros.conversions import ns_to_stamp, stamp_to_ns

    assert stamp_to_ns(ns_to_stamp(1_789_251_053_153_512_471)) == 1_789_251_053_153_512_471


def test_cone_array_survives_the_message():
    from driverless_perception_ros.conversions import from_msg, to_msg

    cones = [Cone(3.0, -1.5, np.array([0.7, 0.1, 0.1, 0.1]), np.diag([0.04, 0.04])),
             Cone(9.0, 2.0, np.array([0.1, 0.8, 0.05, 0.05]), np.diag([0.09, 0.09]))]
    original = ConeArray(1_000_000_007, Source.STEREO, cones, "fsds/FSCar")
    back = from_msg(to_msg(original))
    assert back.stamp_ns == original.stamp_ns and back.source == Source.STEREO
    assert back.frame_id == "fsds/FSCar" and len(back.cones) == 2
    for a, b in zip(original.cones, back.cones):
        assert np.isclose(a.x, b.x) and np.isclose(a.y, b.y)
        assert np.allclose(a.color_probs, b.color_probs)
        assert np.allclose(np.asarray(a.covariance).reshape(2, 2), b.covariance)
        assert a.color == b.color


def test_markers_clear_the_previous_ones_and_colour_by_class():
    from driverless_perception_ros.conversions import cone_markers, ns_to_stamp

    cones = [Cone(1.0, 0.0, np.array([0.9, 0.1, 0.0, 0.0]), np.eye(2))]
    markers = cone_markers(cones, ns_to_stamp(5), "fsds/FSCar", "stereo")
    assert markers.markers[0].action == 3  # DELETEALL
    assert len(markers.markers) == 2
    marker = markers.markers[1]
    assert marker.ns == "stereo" and marker.header.frame_id == "fsds/FSCar"
    assert marker.color.b > marker.color.r and marker.color.a > 0  # a blue cone


def test_image_conversion_handles_encodings_and_padding():
    from driverless_perception_ros.conversions import image_to_bgr
    from sensor_msgs.msg import Image

    msg = Image()
    msg.height, msg.width, msg.encoding = 2, 3, "rgb8"
    msg.step = 12  # a row is padded beyond width * channels
    rgb = np.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90], [0, 0, 0]],
                    [[11, 21, 31], [41, 51, 61], [71, 81, 91], [0, 0, 0]]], np.uint8)
    msg.data = rgb.tobytes()
    bgr = image_to_bgr(msg)
    assert bgr.shape == (2, 3, 3)
    assert list(bgr[0, 0]) == [30, 20, 10]  # rgb to bgr, the padding dropped
    msg.encoding = "32FC1"
    assert image_to_bgr(msg) is None


def test_a_raw_image_splits_into_bridge_transport_queue_and_deserialisation():
    from driverless_perception_ros.conversions import ns_to_stamp
    from driverless_perception_ros.timing import arrival, receive
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image

    msg = Image(height=2, width=2, encoding="bgr8", step=6, data=bytes(12))
    msg.header.stamp = ns_to_stamp(1_000_000_000)
    info = {"source_timestamp": 1_004_000_000, "received_timestamp": 1_005_500_000}
    r = receive(serialize_message(msg), info, Image)
    assert r.stamp_ns == 1_000_000_000 and r.msg.width == 2
    stages = dict(arrival(r, "right image/"))
    assert stages["right image/bridge"] == 4.0 and stages["right image/transport"] == 1.5
    assert stages["right image/deserialize"] >= 0


def test_timing_message_drops_the_box_count_and_ends_with_the_total():
    from driverless_perception_ros.timing import TOTAL, pipeline_stages, timing_msg

    stages, boxes = pipeline_stages({"detector": 6.0, "detector inference": 4.0, "boxes": 7, "pnp": 0.4})
    assert stages == [("detector", 6.0), ("detector/inference", 4.0), ("pnp", 0.4)] and boxes == 7
    msg = timing_msg(1_000_000_000, stages, boxes)
    assert [s.name for s in msg.stages] == ["detector", "detector/inference", "pnp", TOTAL]
    assert msg.boxes == 7 and msg.stages[-1].ms > 0
