import math

import numpy as np

from driverless_perception.common.cameras import (
    BODY_TO_OPTICAL,
    Intrinsics,
    camera_pose,
    optical_to_world,
    quat_to_matrix,
    world_to_optical,
)
from driverless_perception.common.geometry import (
    car_to_world,
    interpolate_pose,
    interpolate_pose_3d,
    world_to_car,
    yaw_from_quaternion,
)

LEVEL = np.array([1.0, 0.0, 0.0, 0.0])


def yaw_quaternion(deg):
    """(x, y, z, w) of a rotation about z, the order ROS uses."""
    a = math.radians(deg) / 2
    return [0.0, 0.0, math.sin(a), math.cos(a)]


def test_yaw_from_quaternion_over_the_whole_circle():
    for deg in (0, 45, 90, -45, -90, 179, -179):
        assert math.isclose(yaw_from_quaternion(*yaw_quaternion(deg)), math.radians(deg), abs_tol=1e-9)
    # +-180 degrees is the same rotation, atan2 returns +pi for both
    assert math.isclose(abs(yaw_from_quaternion(*yaw_quaternion(180))), math.pi, abs_tol=1e-9)


def test_car_frame_transforms_and_their_inverse():
    points = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert np.allclose(world_to_car(points, (0.0, 0.0, 0.0)), points)
    assert np.allclose(world_to_car(points, (1.0, 2.0, 0.0)), [[0.0, -2.0], [-1.0, 0.0]])
    # the car looks along +y, so a point on +x is to its right, which is -y in the car frame
    assert np.allclose(world_to_car(points, (0.0, 0.0, math.pi / 2)), [[0.0, -1.0], [2.0, 0.0]])
    for pose in ((0.0, 0.0, 0.0), (3.0, -2.0, 0.7), (-1.0, 5.0, -2.5)):
        assert np.allclose(car_to_world(world_to_car(points, pose), pose), points)
        assert np.allclose(world_to_car(car_to_world(points, pose), pose), points)


def test_pose_interpolation_lerps_and_clamps():
    times = np.array([0, 10, 30])
    poses = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 1.0], [30.0, 5.0, 1.0]])
    assert np.allclose(interpolate_pose(0, times, poses), poses[0])
    assert np.allclose(interpolate_pose(10, times, poses), poses[1])
    assert np.allclose(interpolate_pose(5, times, poses), [5.0, 2.5, 0.5])
    assert np.allclose(interpolate_pose(20, times, poses), [20.0, 5.0, 1.0])
    # outside the span the ends are held instead of extrapolated
    assert np.allclose(interpolate_pose(-100, times, poses), poses[0])
    assert np.allclose(interpolate_pose(100, times, poses), poses[2])


def test_pose_interpolation_takes_the_short_way_round():
    times = np.array([0, 10])
    poses = np.array([[0.0, 0.0, math.pi - 0.1], [0.0, 0.0, -math.pi + 0.1]])
    yaw = interpolate_pose(5, times, poses)[2]
    # halfway between the two is +-pi, not zero, which is what the wrap has to give
    assert math.isclose(abs(math.atan2(math.sin(yaw), math.cos(yaw))), math.pi, abs_tol=1e-9)
    poses = np.array([[0.0, 0.0, -0.2], [0.0, 0.0, 0.2]])
    assert math.isclose(interpolate_pose(5, times, poses)[2], 0.0, abs_tol=1e-9)


def test_pose_interpolation_in_3d_lerps_position_and_slerps_rotation():
    times = np.array([0, 100])
    positions = np.array([[0.0, 0.0, 0.25], [10.0, 4.0, 0.35]])
    quaternions = np.array([yaw_quaternion(0), yaw_quaternion(90)])
    position, rotation = interpolate_pose_3d(25, times, positions, quaternions)
    assert np.allclose(position, [2.5, 1.0, 0.275])
    assert math.isclose(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])), 22.5, abs_tol=1e-6)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert interpolate_pose_3d(-1, times, positions, quaternions) is None
    assert interpolate_pose_3d(101, times, positions, quaternions) is None
    assert interpolate_pose_3d(0, times[:1], positions[:1], quaternions[:1]) is None


def test_intrinsics_follow_from_the_field_of_view():
    camera = Intrinsics(1280, 960, 90.0)
    assert math.isclose(camera.f, 640.0)  # (w / 2) / tan(45 deg)
    assert (camera.cx, camera.cy) == (640.0, 480.0)
    assert np.allclose(camera.matrix, [[640.0, 0.0, 640.0], [0.0, 640.0, 480.0], [0.0, 0.0, 1.0]])
    narrow = Intrinsics(1280, 960, 60.0)
    assert math.isclose(narrow.f, 640.0 / math.tan(math.radians(30)))


def test_projection_puts_the_axis_in_the_middle_and_keeps_the_directions():
    camera = Intrinsics(800, 600, 90.0)  # f = 400
    points = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [-1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [0.0, 0.0, 5.0]])
    uv = camera.project(points)
    assert np.allclose(uv[0], [400.0, 300.0])
    assert np.allclose(uv[1], [440.0, 300.0])  # optical x is to the right
    assert np.allclose(uv[2], [360.0, 300.0])
    assert np.allclose(uv[3], [400.0, 340.0])  # optical y is down
    assert np.allclose(uv[4], [400.0, 300.0])  # on the axis, distance does not move it


def test_quaternion_to_matrix_turns_the_axes_as_expected():
    assert np.allclose(quat_to_matrix([1.0, 0.0, 0.0, 0.0]), np.eye(3))
    a = math.sin(math.pi / 4)
    about_z = quat_to_matrix([a, 0.0, 0.0, a])  # 90 degrees about z
    assert np.allclose(about_z @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)
    about_y = quat_to_matrix([a, 0.0, a, 0.0])
    assert np.allclose(about_y @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], atol=1e-9)
    about_x = quat_to_matrix([a, a, 0.0, 0.0])
    assert np.allclose(about_x @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(about_z @ about_z.T, np.eye(3), atol=1e-9)
    assert math.isclose(np.linalg.det(about_z), 1.0, abs_tol=1e-9)


def test_optical_frame_conventions_and_roundtrip():
    position = np.array([2.0, 1.0, 1.25])
    ahead, left, above = [12.0, 1.0, 1.25], [2.0, 4.0, 1.25], [2.0, 1.0, 3.25]
    opt = world_to_optical(np.array([ahead, left, above]), position, LEVEL)
    assert np.allclose(opt[0], [0.0, 0.0, 10.0])  # straight ahead is along the optical z
    assert np.allclose(opt[1], [-3.0, 0.0, 0.0])  # to the left of the car is negative optical x
    assert np.allclose(opt[2], [0.0, -2.0, 0.0])  # above is negative optical y
    turned = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    for orientation in (LEVEL, turned):
        points = np.array([ahead, left, above])
        assert np.allclose(optical_to_world(world_to_optical(points, position, orientation), position, orientation),
                           points)


def test_camera_pose_combines_the_car_pose_and_the_mount():
    # the bridge publishes the camera's optical frame: z along the car's x, x along -y, y along -z
    optical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    mount = np.array([-0.3, 0.16, 1.0])
    position, orientation = camera_pose(np.array([1.0, 2.0, 0.25]), np.eye(3), optical, mount)
    assert np.allclose(position, [0.7, 2.16, 1.25])
    assert np.allclose(orientation, [1.0, 0.0, 0.0, 0.0])
    turned = quat_to_matrix([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    position, orientation = camera_pose(np.zeros(3), turned, optical, mount)
    assert np.allclose(position, [-0.16, -0.3, 1.0])
    assert np.allclose(orientation, [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    # a camera looking forward has its body frame aligned with the car, which is what BODY_TO_OPTICAL undoes
    assert np.allclose(optical @ BODY_TO_OPTICAL, np.eye(3))
