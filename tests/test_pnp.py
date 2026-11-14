import numpy as np

from driverless_perception.common.cameras import Intrinsics, up_in_optical, world_to_optical
from driverless_perception.common.cone_model import silhouette_keypoints
from driverless_perception.vision.pnp import solve_cone_position, solve_upright_positions

INTR = Intrinsics(1280, 960, 54.5)
K = np.array([[INTR.f, 0, INTR.cx], [0, INTR.f, INTR.cy], [0, 0, 1]])
CAMERA = np.array([0.0, 0.0, 1.25])
LEVEL = np.array([1.0, 0.0, 0.0, 0.0])


def keypoints_of(cone):
    world = silhouette_keypoints(cone, 0.25, CAMERA)
    return INTR.project(world_to_optical(world, CAMERA, LEVEL))


def test_exact_keypoints_give_the_cone_position():
    for cone in ([6.0, 0.5], [12.0, -2.0], [18.0, 3.0]):
        t = solve_cone_position(keypoints_of(cone), K)
        truth = world_to_optical(np.array([[*cone, 0.25]]), CAMERA, LEVEL)[0]
        # the edge points sit a few millimetres in front of the axis plane, nothing else is off
        assert t is not None and np.linalg.norm(t - truth) < 0.01


def test_noisy_keypoints_and_an_outlier():
    rng = np.random.default_rng(0)
    kp = keypoints_of([10.0, 1.0]) + rng.normal(0, 0.3, (7, 2))
    kp[5] += [15, -10]
    t = solve_cone_position(kp, K)
    truth = world_to_optical(np.array([[10.0, 1.0, 0.25]]), CAMERA, LEVEL)[0]
    assert t is not None and np.linalg.norm(t - truth) < 0.5


def test_degenerate_keypoints_give_no_position():
    from driverless_perception.common.cone_model import model_points

    # a collapsed regression: all seven points on top of each other, which makes SQPnP assert internally
    centre = np.array([[640.0, 480.0]] * 7)
    spread = model_points()[:, [0, 2]] - model_points()[:, [0, 2]].mean(axis=0)
    assert solve_cone_position(centre + 0.01 * spread, K) is None
    assert solve_cone_position(centre, K) is None


def test_a_cone_behind_the_camera_gives_no_position():
    # keypoints in no sensible arrangement, which PnP happily solves as a cone behind the camera
    scrambled = np.array([[655.13, 912.45], [184.52, 910.7], [399.14, 406.39], [1059.46, 392.83],
                          [703.48, 26.46], [964.5, 516.62], [422.06, 756.89]])
    assert solve_cone_position(scrambled, K) is None
    assert solve_cone_position(keypoints_of([8.0, 0.0]), K)[2] > 0


# upright cones

def tilted(pitch_deg, roll_deg):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_euler("xyz", [roll_deg, pitch_deg, 0.0], degrees=True).as_quat()
    return np.array([w, x, y, z])


def scene(orientation, cones=([3.0, 1.0], [8.0, -1.5], [14.0, 2.0], [22.0, -0.5])):
    kp = np.stack([INTR.project(world_to_optical(silhouette_keypoints(c, 0.25, CAMERA), CAMERA, orientation))
                   for c in cones])
    truth = world_to_optical(np.array([[*c, 0.25] for c in cones]), CAMERA, orientation)
    return kp, truth


def test_level_camera_looks_along_minus_y_for_up():
    assert np.allclose(up_in_optical(LEVEL), [0.0, -1.0, 0.0])


def test_upright_and_ground_recover_exact_cones_from_a_tilted_camera():
    # pitched and rolled like the car under braking in a corner; the solver uses the labeller's own geometry, so
    # exact keypoints have to come back exactly, with and without the ground plane
    orientation = tilted(1.0, 0.75)
    kp, truth = scene(orientation)
    up = up_in_optical(orientation)
    for height in (None, CAMERA[2] - 0.25):
        estimates = solve_upright_positions(kp, K, up, height)
        assert all(np.linalg.norm(e - t) < 1e-4 for e, t in zip(estimates, truth))


def test_upright_drops_an_outlier_keypoint():
    kp, truth = scene(LEVEL, cones=([10.0, 1.0],))
    rng = np.random.default_rng(0)
    kp = kp + rng.normal(0, 0.3, kp.shape)
    kp[0, 5] += [15, -10]
    estimate = solve_upright_positions(kp, K, up_in_optical(LEVEL))[0]
    assert estimate is not None and np.linalg.norm(estimate - truth[0]) < 0.3


def test_upright_has_no_range_bias_on_noisy_keypoints_where_sqpnp_does():
    # the reason for the upright solver: with noisy keypoints a free pose leans the cone and puts far cones too
    # close, a cone that has to stand upright cannot do that
    kp, truth = scene(LEVEL, cones=([18.0, 1.0],))
    rng = np.random.default_rng(1)
    up = up_in_optical(LEVEL)
    upright, free = [], []
    for _ in range(300):
        noisy = kp[0] + rng.normal(0, 0.7, (7, 2))
        upright.append(np.linalg.norm(solve_upright_positions(noisy[None], K, up)[0]))
        t = solve_cone_position(noisy, K)
        if t is not None:
            free.append(np.linalg.norm(t))
    true_range = np.linalg.norm(truth[0])
    assert abs(np.median(upright) - true_range) < 0.1
    assert np.median(free) - true_range < -0.2


def test_upright_gives_nothing_for_no_keypoints_and_none_behind_the_camera():
    assert solve_upright_positions(np.zeros((0, 7, 2)), K, [0.0, -1.0, 0.0]) == []
    scrambled = np.array([[655.13, 912.45], [184.52, 910.7], [399.14, 406.39], [1059.46, 392.83],
                          [703.48, 26.46], [964.5, 516.62], [422.06, 756.89]])
    result = solve_upright_positions(scrambled[None], K, [0.0, -1.0, 0.0])[0]
    assert result is None or result[2] > 0
