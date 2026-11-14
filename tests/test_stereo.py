import cv2
import numpy as np

from driverless_perception.common.cameras import Intrinsics, camera_pose, world_to_optical
from driverless_perception.common.cone_model import BODY_BOTTOM, TOP, body_radius
from driverless_perception.common.geometry import interpolate_pose_3d
from driverless_perception.vision.stereo.matching import match_features
from driverless_perception.vision.stereo.propagation import propagate_box
from driverless_perception.vision.stereo.triangulation import (
    cone_from_points,
    fundamental,
    projection,
    triangulate,
)

INTR = Intrinsics(1280, 960, 64.5)
GROUND = 0.25


def yaw(deg):
    a = np.radians(deg) / 2
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


# the right image is taken a little later, the car has moved on and turned
LEFT = (INTR, np.array([0.0, 0.16, GROUND + 1.0]), yaw(0.0))
RIGHT = (INTR, np.array([0.08, -0.155, GROUND + 1.0]), yaw(1.5))


def pixels(points, camera):
    intr, position, orientation = camera
    return intr.project(world_to_optical(points, position, orientation))


def some_points(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(3, 12, n), rng.uniform(-3, 3, n), rng.uniform(GROUND, GROUND + 0.3, n)])


def test_projection_matrix_matches_the_camera_model():
    pts = some_points()
    uv = np.hstack([pts, np.ones((len(pts), 1))]) @ projection(*LEFT).T
    assert np.allclose(uv[:, :2] / uv[:, 2:], pixels(pts, LEFT))


def test_triangulation_with_two_poses():
    pts = some_points()
    assert np.allclose(triangulate(pixels(pts, LEFT), pixels(pts, RIGHT), LEFT, RIGHT), pts, atol=1e-6)


def test_matches_lie_on_their_epipolar_lines():
    pts = some_points()
    left = np.hstack([pixels(pts, LEFT), np.ones((len(pts), 1))])
    right = np.hstack([pixels(pts, RIGHT), np.ones((len(pts), 1))])
    F = fundamental(LEFT, RIGHT)
    lines = left @ F.T
    d = np.abs(np.sum(lines * right, axis=1)) / np.hypot(lines[:, 0], lines[:, 1])
    assert d.max() < 1e-6


def test_propagated_box_holds_the_cone():
    cone = np.array([7.0, 1.0])
    a = np.linspace(0, 2 * np.pi, 36)
    ring = np.column_stack([cone[0] + 0.114 * np.cos(a), cone[1] + 0.114 * np.sin(a), np.full_like(a, GROUND)])
    uv = pixels(np.vstack([ring, ring + [0, 0, 0.325]]), RIGHT)
    # PnP put the cone 0.4 m too close
    guess = cone * (1 - 0.4 / np.linalg.norm(cone))
    box = propagate_box(guess, GROUND, RIGHT[1], RIGHT[2], INTR)
    assert box is not None
    assert (uv.min(axis=0) >= box[:2]).all() and (uv.max(axis=0) <= box[2:]).all()
    assert propagate_box(np.array([-5.0, 0.0]), GROUND, RIGHT[1], RIGHT[2], INTR) is None


def test_cone_position_from_points_on_its_front():
    rng = np.random.default_rng(1)
    cone, camera = np.array([8.0, 2.0]), LEFT[1][:2]
    ray = (cone - camera) / np.linalg.norm(cone - camera)
    side = np.array([-ray[1], ray[0]])
    z = rng.uniform(BODY_BOTTOM, TOP, 300)
    s = rng.uniform(-1, 1, 300)
    r = body_radius(z)
    xy = cone + (s * r)[:, None] * side - (np.sqrt(1 - s ** 2) * r)[:, None] * ray
    points = np.column_stack([xy, GROUND + z])
    tarmac = np.column_stack([cone + rng.normal(0, 0.3, (100, 2)), np.full(100, GROUND + 0.005)])
    wrong = np.column_stack([cone + rng.uniform(1.5, 3, (40, 2)), np.full(40, GROUND + 0.2)])
    guess = cone + [0.3, -0.2]
    estimate, on_cone = cone_from_points(np.vstack([points, tarmac, wrong]), GROUND, guess, camera)
    assert on_cone[:300].all() and not on_cone[300:].any()
    assert np.linalg.norm(estimate - cone) < 0.01
    estimate, on_cone = cone_from_points(tarmac, GROUND, guess, camera)
    assert estimate is None and not on_cone.any()


def texture(seed=0, shape=(960, 1280)):
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (shape[0] // 4, shape[1] // 4)).astype(np.float32)
    return cv2.resize(cv2.GaussianBlur(noise, (0, 0), 1.0), shape[::-1]).astype(np.uint8)


def test_orb_matches_follow_the_disparity():
    patch = texture(1)[400:480, 600:660]
    left, right = texture(0), texture(2)
    left[400:480, 600:660] = patch
    right[400:480, 580:640] = patch
    pts_left, pts_right, info = match_features(left, [600, 400, 660, 480], right, [560, 390, 680, 490])
    assert len(pts_left) >= 10 and info["ratio_test"] == len(pts_left)
    assert np.allclose(np.median(pts_left - pts_right, axis=0), [20, 0], atol=0.5)

    # rectified pair, the epipolar lines are rows: a patch 10 rows lower can't match
    right = texture(2)
    right[410:490, 580:640] = patch
    F = np.array([[0.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert len(match_features(left, [600, 400, 660, 480], right, [560, 390, 680, 500])[0]) >= 10
    assert len(match_features(left, [600, 400, 660, 480], right, [560, 390, 680, 500], F)[0]) == 0


def test_camera_pose_from_the_car_pose_and_mount():
    # the bridge's optical frame for a camera looking forward: z along the car's x, x along -y, y along -z
    optical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    position, orientation = camera_pose(np.array([1.0, 2.0, 0.25]), np.eye(3), optical, np.array([-0.3, 0.16, 1.0]))
    assert np.allclose(position, [0.7, 2.16, 1.25]) and np.allclose(orientation, [1, 0, 0, 0])
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    position, orientation = camera_pose(np.zeros(3), turned, optical, np.array([-0.3, 0.16, 1.0]))
    assert np.allclose(position, [-0.16, -0.3, 1.0]) and np.allclose(orientation, yaw(90))


def test_pose_interpolation_in_3d():
    times = np.array([0, 100])
    positions = np.array([[0.0, 0.0, 0.25], [1.0, 0.0, 0.25]])
    quaternions = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    position, rotation = interpolate_pose_3d(50, times, positions, quaternions)
    assert np.allclose(position, [0.5, 0.0, 0.25])
    assert np.isclose(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])), 45)
    assert interpolate_pose_3d(150, times, positions, quaternions) is None
