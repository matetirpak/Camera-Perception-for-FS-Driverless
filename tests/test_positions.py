import numpy as np

from driverless_perception.evaluation.mapping import build_map
from driverless_perception.evaluation.positions import position_error, spread_by_distance, summarise
from driverless_perception.vision.keypoints.cross_ratio import LEFT_EDGE, cross_ratio

CAMERA = np.array([0.0, 0.0])
GROUND = 0.25


def test_error_splits_into_range_lateral_and_height():
    # the cone stands 10 m straight ahead, the estimate is 0.3 m too close, 0.1 m to the left and 2 cm too high
    error = position_error([9.7, 0.1, GROUND + 0.02], [10.0, 0.0], GROUND, CAMERA)
    assert np.isclose(error.distance_m, 10.0) and np.isclose(error.estimate_m, np.hypot(9.7, 0.1))
    assert np.isclose(error.range_m, np.hypot(9.7, 0.1) - 10.0, atol=1e-6)
    assert np.isclose(error.lateral_m, 0.1) and np.isclose(error.vertical_m, 0.02)
    assert np.isclose(error.error_xy_m, np.hypot(0.3, 0.1))
    assert np.isclose(error.error_3d_m, np.linalg.norm([0.3, 0.1, 0.02]))
    # without a height the estimate sits on the ground, where the true cone base is
    assert position_error([9.7, 0.1], [10.0, 0.0], GROUND, CAMERA).vertical_m == 0.0


def test_summary_and_spread_group_by_true_distance():
    errors = [position_error([d - 0.1, 0.0], [d, 0.0], GROUND, CAMERA) for d in (3.0, 4.0, 7.0, 8.0)]
    bins = summarise(errors, (0, 5, 10))
    assert [b["cones"] for b in bins] == [2, 2]
    assert np.isclose(bins[0]["median_m"], 0.1) and np.isclose(bins[0]["range_bias_m"], -0.1)
    assert np.isclose(bins[0]["lateral_rmse_m"], 0.0)
    spread = spread_by_distance(errors, (4.0,), tolerance=0.25)
    assert spread[0]["cones"] == 1 and np.allclose(spread[0]["estimates_m"], [3.9])


def test_range_bias_is_a_median_and_survives_outliers():
    # nine cones a little too close and one wildly too far: the bias follows the nine, the spread notices the one
    errors = [position_error([10.0 - 0.1, 0.0], [10.0, 0.0], GROUND, CAMERA) for _ in range(9)]
    errors.append(position_error([13.0, 0.0], [10.0, 0.0], GROUND, CAMERA))
    stats = summarise(errors, (0, 20))[0]
    assert np.isclose(stats["range_bias_m"], -0.1)  # a mean would give +0.21
    assert np.isclose(stats["median_m"], 0.1)
    # the 90th percentile interpolates between the ninth and tenth of ten values: 0.1 + 0.1 * (3.0 - 0.1)
    assert np.isclose(stats["p90_m"], 0.39)
    assert np.isclose(stats["range_spread_m"], 0.93)


def test_map_is_the_median_of_the_observations():
    track = np.array([[10.0, 0.0], [10.0, 3.0]])
    colors = np.array([0, 1])
    times = np.array([0, 2_000_000_000])
    poses = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # the car stands at the origin: three views of the first cone, one of the second, and one invented cone
    frames = [(1_000_000_000, np.array([[10.2, 0.0], [10.0, 3.0], [4.0, -6.0]]), np.array([0, 1, 0])),
              (1_100_000_000, np.array([[9.9, 0.0]]), np.array([0])),
              (1_200_000_000, np.array([[10.1, 0.0]]), np.array([1]))]
    result = build_map(frames, times, poses, track, colors, gate=1.0, min_observations=3)
    assert result["mapped"] == 1 and result["missed"] == 1
    cone = result["cones"][0]
    assert cone["cone"] == 0 and cone["observations"] == 3
    assert np.isclose(cone["x"], 10.1) and np.isclose(cone["error_m"], 0.1)
    # two of the three say blue, and the cone that belongs to no true one is counted apart
    assert cone["colour"] == "blue" and cone["true_colour"] == "blue"
    assert result["spurious_observations"] == 1
    assert np.isclose(result["rmse_m"], 0.1) and result["colour_accuracy"] == 1.0


def test_cross_ratio_of_a_straight_edge():
    # four points on a line, spaced so the cross ratio is easy to check by hand
    points = np.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 4.0]]])
    assert np.isclose(cross_ratio(points, LEFT_EDGE)[0], (2 / 4) / (1 / 3))
