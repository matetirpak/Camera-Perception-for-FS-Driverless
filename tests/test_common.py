import math

import numpy as np

from driverless_perception.common.cameras import Intrinsics, world_to_optical
from driverless_perception.common.fsds_settings import FsdsSettings
from driverless_perception.common.geometry import car_to_world, interpolate_pose, world_to_car
from driverless_perception.common.types import ConeColor


def test_rpc_colour_order_differs_from_fs_msgs():
    assert ConeColor.from_fsds_rpc(0) == ConeColor.YELLOW
    assert ConeColor.from_fsds_rpc(1) == ConeColor.BLUE
    assert ConeColor.BLUE == 0 and ConeColor.YELLOW == 1


def test_settings_intrinsics_and_lidar_resolution(default_settings):
    s = FsdsSettings(default_settings)
    cam = s.cameras["cam1"]
    assert math.isclose(cam.fx, 392.5)  # same as the bridge publishes for this camera
    lidar = s.lidars["Lidar1"]
    assert math.isclose(math.degrees(lidar.vertical_resolution_rad), 10 / 3)
    assert math.isclose(math.degrees(lidar.horizontal_resolution_rad), 180 / 1024)


def test_camera_projection_centre_and_left():
    camera = Intrinsics(785, 785, 90.0)
    position, level = np.array([0.0, 0.0, 0.8]), np.array([1.0, 0.0, 0.0, 0.0])
    uv = camera.project(world_to_optical(np.array([[10.0, 0.0, 0.8], [10.0, 2.0, 0.8]]), position, level))
    assert np.allclose(uv[0], [392.5, 392.5])  # straight ahead of the camera -> image centre
    assert uv[1, 0] < 392.5  # a point to the left (y > 0) lands left of centre


def test_world_car_roundtrip_and_interpolation():
    pose = (3.0, -2.0, 0.7)
    pts = np.array([[1.0, 2.0], [-4.0, 0.5]])
    assert np.allclose(car_to_world(world_to_car(pts, pose), pose), pts)
    times = np.array([0, 10])
    poses = np.array([[0.0, 0.0, math.pi - 0.1], [1.0, 0.0, -math.pi + 0.1]])
    x, _, yaw = interpolate_pose(5, times, poses)
    assert math.isclose(x, 0.5) and math.isclose(abs(yaw), math.pi, abs_tol=1e-9)  # wraps through +-pi
