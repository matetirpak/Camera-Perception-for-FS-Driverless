import cv2
import numpy as np
import torch

from driverless_perception.common.cone_model import cross_ratio_3d
from driverless_perception.common.geometry import world_to_car
from driverless_perception.common.types import ConeColor
from driverless_perception.evaluation.keypoints import evaluate_keypoints, report_markdown
from driverless_perception.evaluation.mapping import build_map
from driverless_perception.evaluation.positions import position_error, spread_by_distance, summarise
from driverless_perception.evaluation.race import CameraView, score_frames, split_cones
from driverless_perception.vision.keypoints.crops import CropCache

CAMERA = np.array([0.0, 0.0])
GROUND = 0.25


def test_error_of_a_cone_off_to_the_side_splits_range_from_lateral():
    truth = [6.0, 8.0]  # 10 m away but not straight ahead: ray (0.6, 0.8), across (-0.8, 0.6)
    # 0.3 m too close and 0.4 m to the side along "across", plus 0.05 m too low, works out to (5.5, 8.0, 0.2)
    error = position_error([5.5, 8.0, 0.2], truth, GROUND, CAMERA)
    assert np.isclose(error.distance_m, 10.0)
    assert np.isclose(error.lateral_m, 0.4) and np.isclose(error.vertical_m, -0.05)
    expected_estimate = np.hypot(5.5, 8.0)
    assert np.isclose(error.estimate_m, expected_estimate)
    assert np.isclose(error.range_m, expected_estimate - 10.0) and error.range_m < 0.0
    assert np.isclose(error.error_xy_m, 0.5)
    assert np.isclose(error.error_3d_m, np.linalg.norm([-0.5, 0.0, -0.05]))

    # the same lateral offset but 0.3 m too far instead of too close gives a positive range error
    far = position_error([5.86, 8.48, 0.25], truth, GROUND, CAMERA)
    assert np.isclose(far.lateral_m, 0.4) and far.range_m > 0.0
    assert np.isclose(far.range_m, np.hypot(5.86, 8.48) - 10.0)


def test_summary_and_spread_handle_an_empty_bin():
    errors = [position_error([d - 0.1, 0.0], [d, 0.0], GROUND, CAMERA) for d in (3.0, 8.0)]
    bins = summarise(errors, (0, 5, 10, 15))
    assert [b["cones"] for b in bins] == [1, 1, 0]
    assert bins[2]["distance_m"] == [10, 15] and "median_m" not in bins[2]
    spread = spread_by_distance(errors, (3.0, 20.0), tolerance=0.25)
    assert spread[0]["cones"] == 1 and np.allclose(spread[0]["estimates_m"], [2.9])
    assert spread[1]["cones"] == 0 and spread[1]["estimates_m"] == []


def test_map_transforms_a_moving_car_and_applies_gate_min_observations_and_colour_voting():
    # track cones: 0 and 3 blue, 1 yellow, 2 big orange (not a landmark), 4 yellow, 5 blue
    track = np.array([[10.0, 0.0], [10.0, 3.0], [20.0, 5.0], [-5.0, -5.0], [15.0, -2.0], [0.0, 10.0]])
    colors = np.array([ConeColor.BLUE, ConeColor.YELLOW, ConeColor.ORANGE_BIG, ConeColor.BLUE, ConeColor.YELLOW,
                       ConeColor.BLUE])
    times = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000])
    # the car drives forward, turns 90 degrees, then drives forward again
    poses = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, np.pi / 2], [4.0, 2.0, np.pi / 2],
                      [4.0, 4.0, np.pi / 2]])
    frames = [
        # cone 0 seen 0.1 m short, then 0.1 m long, then exactly; cone 2 (excluded) and cone 4 (2 mislabels) seen too
        (0, np.array([[10.1, 0.0], [20.2, 5.0], [15.2, -2.0], [0.0, 10.4]]),
         np.array([ConeColor.BLUE, ConeColor.ORANGE_BIG, ConeColor.BLUE, ConeColor.BLUE])),
        (1_000_000_000, np.array([[7.9, 0.0], [17.9, 5.1], [13.2, -2.0]]),
         np.array([ConeColor.BLUE, ConeColor.ORANGE_BIG, ConeColor.BLUE])),
        (2_000_000_000, np.array([[0.0, -6.0], [5.0, -16.0], [10.4, 4.0]]),
         np.array([ConeColor.YELLOW, ConeColor.ORANGE_BIG, ConeColor.BLUE])),
        # cone 1 only gets 2 observations (below min_observations); one spurious point far from any cone
        (3_000_000_000, np.array([[1.2, -6.0], [-4.0, -11.2], [0.0, 0.0]]),
         np.array([ConeColor.YELLOW, ConeColor.YELLOW, ConeColor.ORANGE_SMALL])),
        # a point 1.2 m from cone 0, just outside the 1.0 m gate, so it counts as spurious rather than a match
        (4_000_000_000, np.array([[-1.0, -5.7], [6.4, 4.0], [-4.0, -7.2]]),
         np.array([ConeColor.YELLOW, ConeColor.BLUE, ConeColor.YELLOW])),
    ]
    result = build_map(frames, times, poses, track, colors, gate=1.0, min_observations=3)

    assert result["frames"] == 5 and result["gate_m"] == 1.0 and result["min_observations"] == 3
    assert result["expected"] == 5  # every cone but the big orange one (index 2)
    assert result["mapped"] == 3 and result["missed"] == 2  # cone 1 too few observations, cone 3 never seen
    assert result["spurious_observations"] == 2
    assert {c["cone"] for c in result["cones"]} == {0, 4, 5}

    cones = {c["cone"]: c for c in result["cones"]}
    assert cones[0]["observations"] == 3
    assert np.isclose(cones[0]["x"], 10.0) and np.isclose(cones[0]["y"], 0.0)
    assert np.isclose(cones[0]["error_m"], 0.0)
    assert cones[0]["colour"] == "blue" and cones[0]["true_colour"] == "blue"

    # cone 4 is really yellow but 2 of its 3 observations were mislabelled blue, so the vote comes out wrong
    assert np.isclose(cones[4]["error_m"], 0.2)
    assert cones[4]["colour"] == "blue" and cones[4]["true_colour"] == "yellow"

    assert np.isclose(cones[5]["error_m"], 0.4)
    assert cones[5]["colour"] == "blue" and cones[5]["true_colour"] == "blue"

    errors = (0.0, 0.2, 0.4)
    assert np.isclose(result["rmse_m"], np.sqrt(np.mean(np.square(errors))))
    assert np.isclose(result["median_m"], 0.2)
    assert np.isclose(result["p90_m"], 0.36)  # linear interpolation between the sorted 0.2 and 0.4
    assert np.isclose(result["colour_accuracy"], 2 / 3)  # cones 0 and 5 right, cone 4 wrong


def test_evaluate_keypoints_reports_hand_computed_errors():
    truth = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 4.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    # crop A: every point pushed 2 px right, a uniform translation the cross ratio can't see
    predicted_a = truth + [2.0, 0.0]
    # crop B: only the left stripe top (index 1) is wrong, by (3, 4) px, breaking the left edge's straight line
    predicted_b = truth.copy()
    predicted_b[1] = truth[1] + [3.0, 4.0]

    # box (0, 0, 80, 80) makes plain_crop's window transform the identity, so truth passes through unchanged
    box = np.array([0.0, 0.0, 80.0, 80.0])
    image = np.full((80, 80, 3), 128, np.uint8)
    jpeg = cv2.imencode(".jpg", image)[1].tobytes()
    cache = CropCache(
        jpegs=[jpeg, jpeg], boxes=np.array([box, box]), keypoints=np.array([truth, truth]),
        classes=np.array([ConeColor.BLUE, ConeColor.YELLOW]), distances=np.array([3.0, 12.0]),
        # image_boxes are 160x160, twice the 80x80 crop, so 1 crop pixel is 2 image pixels
        image_boxes=np.array([[0.0, 0.0, 160.0, 160.0], [0.0, 0.0, 160.0, 160.0]]))

    class FakeModel:
        def __init__(self, predicted):
            self.predicted = torch.tensor(np.stack(predicted), dtype=torch.float32)

        def __call__(self, x):
            return self.predicted[: x.shape[0]]

    result = evaluate_keypoints(cache, FakeModel([predicted_a, predicted_b]), device="cpu")

    assert result["crops"] == 2
    # crop A: every one of the 7 points is off by exactly 2 px. crop B: only point 1 is off, by hypot(3, 4) = 5 px
    errors = np.array([[2.0] * 7, [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    scale = 2.0
    assert np.isclose(result["mse_px2"], (errors ** 2).mean(axis=1).mean())
    assert np.isclose(result["median_px"], np.median(errors))
    assert np.isclose(result["median_image_px"], np.median(errors.mean(axis=1) * scale))

    per_kp = {k["index"]: k for k in result["per_keypoint"]}
    assert np.isclose(per_kp[1]["mse_px2"], (2.0 ** 2 + 5.0 ** 2) / 2) and np.isclose(per_kp[1]["median_px"], 3.5)
    assert np.isclose(per_kp[0]["mse_px2"], (2.0 ** 2 + 0.0 ** 2) / 2) and np.isclose(per_kp[0]["median_px"], 1.0)

    by_distance = {tuple(b["distance_m"]): b for b in result["by_distance"]}
    near = by_distance[(0, 5)]
    assert near["crops"] == 1 and np.isclose(near["mse_px2"], 4.0) and np.isclose(near["median_px"], 2.0)
    assert np.isclose(near["median_image_px"], 4.0) and np.isclose(near["box_height_px"], 160.0)
    far = by_distance[(10, 15)]
    assert far["crops"] == 1 and np.isclose(far["mse_px2"], 25.0 / 7.0) and np.isclose(far["median_px"], 0.0)
    assert by_distance[(5, 10)]["crops"] == 0

    by_class = {b["class"]: b for b in result["by_class"]}
    assert by_class["blue"]["crops"] == 1 and np.isclose(by_class["blue"]["mse_px2"], 4.0)
    assert by_class["yellow"]["crops"] == 1 and np.isclose(by_class["yellow"]["median_px"], 0.0)

    # both edges of the shared truth are exactly on a line (0, 1, 2, 4), giving the same cross ratio as the
    # test_positions.py cross-ratio check: (2 / 4) / (1 / 3) = 1.5, for both crops and both edges
    assert result["cross_ratio"]["model"] == cross_ratio_3d()
    assert np.isclose(result["cross_ratio"]["labels"]["median"], 1.5)
    model = cross_ratio_3d()
    assert np.isclose(result["cross_ratio"]["labels"]["deviation"], abs(1.5 - model))
    # crop A is a pure translation so its ratios are still 1.5; crop B's left edge ratio works out to sqrt(5) / 6
    predicted_ratios = np.array([1.5, np.sqrt(5) / 6, 1.5, 1.5])
    assert np.isclose(result["cross_ratio"]["predicted"]["median"], np.median(predicted_ratios))
    assert np.isclose(result["cross_ratio"]["predicted"]["deviation"], np.median(np.abs(predicted_ratios - model)))
    assert result["cross_ratio"]["predicted_values"] == [1.5, round(float(np.sqrt(5) / 6), 4), 1.5, 1.5]


def test_keypoints_report_markdown_renders_the_given_numbers():
    result = {
        "crops": 2, "mse_px2": 53.0 / 14.0, "median_px": 2.0, "median_image_px": 19.0 / 7.0,
        "per_keypoint": [{"index": i, "name": n, "mse_px2": 2.0, "median_px": 1.0}
                         for i, n in enumerate(["top", "left stripe top", "left stripe bottom", "left bottom",
                                                "right stripe top", "right stripe bottom", "right bottom"])],
        "by_distance": [
            {"distance_m": [0, 5], "crops": 1, "mse_px2": 4.0, "median_px": 2.0, "median_image_px": 4.0,
             "box_height_px": 160.0},
            {"distance_m": [5, 10], "crops": 0},
        ],
        "cross_ratio": {"model": 1.5960, "predicted": {"median": 1.5, "deviation": 0.0961},
                        "labels": {"median": 1.5, "deviation": 0.0960}},
    }
    text = report_markdown(result)
    assert "mean squared error is 3.79 px^2" in text
    assert "median 2.00 px per point, 2.71 px back in the camera" in text
    assert "| 0-5 m | 1 | 160 px | 2.00 px | 4.00 px | 4.00 px^2 |" in text
    assert "5-10 m" not in text  # the empty bin is left out of the table
    assert "The cross ratio of the cone model is 1.5960." in text
    assert "1.5000 (median, typically 0.0961 away from the" in text
    assert "the true keypoints 1.5000." in text


def test_split_cones_uses_several_cameras_and_the_first_ones_range():
    cam_a = CameraView(0.0, 0.0, np.radians(170))
    cam_b = CameraView(1.0, 0.0, np.radians(170))
    cam_c = CameraView(3.0, 0.0, np.radians(170))
    track = np.array([[6.0, 0.0], [2.0, 4.0]])
    colors = np.full(2, ConeColor.BLUE)

    # cone (2, 4) sits behind camera c (its local x is 2 - 3 < 0), so 3 cameras see less than 2 of them would
    expected, others = split_cones(track, colors, (0.0, 0.0, 0.0), (cam_a, cam_b, cam_c), max_range=20.0)
    assert np.allclose(expected, [[6.0, 0.0]]) and np.allclose(others, [[2.0, 4.0]])

    # max_range is measured only from the first camera: 6.0 m from a, 5.0 m from b, 3.0 m from c
    excluded, _ = split_cones(track, colors, (0.0, 0.0, 0.0), (cam_a, cam_b, cam_c), max_range=5.5)
    assert len(excluded) == 0
    included, _ = split_cones(track, colors, (0.0, 0.0, 0.0), (cam_b, cam_a, cam_c), max_range=5.5)
    assert np.allclose(included, [[6.0, 0.0]])
    also_included, _ = split_cones(track, colors, (0.0, 0.0, 0.0), (cam_c, cam_a, cam_b), max_range=5.5)
    assert np.allclose(also_included, [[6.0, 0.0]])


def test_score_frames_skips_frames_outside_the_odometry_span():
    times = np.array([0, 1_000_000_000])
    poses = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    track = np.array([[5.0, 0.0]])
    colors = np.array([ConeColor.BLUE])
    camera = CameraView(0.0, 0.0, np.radians(170))
    truth = world_to_car(track, (0.0, 0.0, 0.0))

    # one frame before the odometry starts, one inside it, one after it ends
    frames = [(-1_000_000, truth), (500_000_000, truth), (2_000_000_000, truth)]
    result = score_frames(frames, times, poses, track, colors, camera, max_range=10.0)
    assert result["frames"] == 1
    assert result["all"]["expected"] == 1 and result["all"]["detections"] == 1
    assert result["all"]["recall"] == 1.0 and result["all"]["precision"] == 1.0
