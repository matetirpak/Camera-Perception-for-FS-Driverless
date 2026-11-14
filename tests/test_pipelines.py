import cv2
import numpy as np

from driverless_perception.common.cameras import Intrinsics, world_to_optical
from driverless_perception.common.cone_model import silhouette_keypoints
from driverless_perception.common.types import ConeColor
from driverless_perception.vision.mono import MonoPipeline, color_probs, position_sigma
from driverless_perception.vision.pnp import solve_upright_positions
from driverless_perception.vision.stereo.pipeline import StereoPipeline, View
from driverless_perception.vision.stereo.pipeline import position_sigma as stereo_position_sigma

INTR = Intrinsics(640, 480, 60.0)
GROUND = 0.25
LEVEL = np.array([1.0, 0.0, 0.0, 0.0])
LEFT_POS = np.array([0.0, 0.16, GROUND + 1.0])
RIGHT_POS = np.array([0.0, -0.16, GROUND + 1.0])
K = INTR.matrix


def test_position_sigma_matches_the_documented_formula_and_is_monotone():
    d = np.linspace(0.0, 25.0, 251)
    assert np.allclose(position_sigma(d), 0.05 + 0.0018 * d ** 2)
    assert np.all(np.diff(position_sigma(d)) >= 0)


def test_color_probs_slots_sum_to_one_and_orange_shares_a_slot():
    blue = color_probs(ConeColor.BLUE, 0.9)
    assert np.isclose(blue[0], 0.9) and np.allclose(blue[[1, 2, 3]], 0.1 / 3) and np.isclose(blue.sum(), 1.0)
    big = color_probs(ConeColor.ORANGE_BIG, 0.7)
    small = color_probs(ConeColor.ORANGE_SMALL, 0.7)
    assert np.argmax(big) == 2 and np.allclose(big, small)


def test_usable_checks_class_and_minimum_height():
    pipe = MonoPipeline(keypoint_model=None, detector=None, min_height=20)
    assert pipe.usable([0, 0, 10, 20], ConeColor.BLUE)
    assert not pipe.usable([0, 0, 10, 19.999], ConeColor.BLUE)
    assert not pipe.usable([0, 0, 10, 50], ConeColor.ORANGE_BIG)
    assert pipe.usable([0, 0, 10, 50], ConeColor.ORANGE_SMALL)


class FakeTensor:
    """Just enough of a torch/ultralytics tensor for MonoPipeline.detect: .cpu().numpy()."""

    def __init__(self, array):
        self.array = array

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeBoxes:
    def __init__(self, boxes, classes, scores):
        self.xyxy, self.cls, self.conf = FakeTensor(boxes), FakeTensor(classes), FakeTensor(scores)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeDetector:
    """A detector returning fixed boxes, classes and scores, in the ultralytics result shape."""

    def __init__(self, boxes, classes, scores):
        self.boxes, self.classes, self.scores = boxes, classes, scores
        self.calls = []

    def predict(self, image, imgsz, conf, verbose, **options):
        self.calls.append((image.shape, options))
        return [FakeResult(FakeBoxes(self.boxes, self.classes, self.scores))]


def test_detect_filters_by_class_and_height_with_a_fake_detector():
    boxes = np.array([[0.0, 0.0, 10.0, 30.0], [0.0, 0.0, 10.0, 5.0], [0.0, 0.0, 10.0, 40.0]])
    classes = np.array([ConeColor.BLUE, ConeColor.YELLOW, ConeColor.ORANGE_BIG])
    scores = np.array([0.9, 0.8, 0.95])
    pipe = MonoPipeline(keypoint_model=None, detector=FakeDetector(boxes, classes, scores), min_height=20)
    kept_boxes, kept_classes, kept_scores = pipe.detect(np.zeros((10, 10, 3), np.uint8))
    # box 1 is a small cone tall enough, box 2 is too short, box 3 is tall but not a small cone
    assert kept_boxes.shape == (1, 4) and np.array_equal(kept_boxes[0], boxes[0])
    assert kept_classes.tolist() == [ConeColor.BLUE] and kept_scores.tolist() == [0.9]


def test_half_and_compile_only_on_the_gpu_and_warm_up_uses_them():
    detector = FakeDetector(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))
    on_cpu = MonoPipeline(None, "cpu", detector=detector, half=True, compile="reduce-overhead")
    assert on_cpu.detector_options() == {}
    on_gpu = MonoPipeline(None, "cuda", detector=detector, half=True, compile="reduce-overhead")
    assert on_gpu.detector_options() == {"quantize": 16, "compile": "reduce-overhead"}
    assert MonoPipeline(None, "cuda", detector=detector, compile="").detector_options() == {}
    # warming up runs the detector on a blank image of the camera's size with the same options
    on_gpu.warm_up(960, 1280, runs=2)
    assert detector.calls == [((960, 1280, 3), {"quantize": 16, "compile": "reduce-overhead"})] * 2
    on_gpu.detect(np.zeros((10, 10, 3), np.uint8))
    assert detector.calls[-1][1] == {"quantize": 16, "compile": "reduce-overhead"}


def test_an_exported_detector_gets_no_precision_or_compile_options():
    detector = FakeDetector(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))
    # ultralytics keeps the path of an exported model (TensorRT engine, ONNX) as a string in .model
    detector.model = "models/detection/yolo26s_fsoco_fsds.engine"
    engine = MonoPipeline(None, "cuda", detector=detector, half=True, compile="reduce-overhead")
    assert engine.detector_options() == {}


def test_positions_recovers_true_position_for_several_cones():
    # regression guard: keypoints from common.cameras and K from Intrinsics must share one pixel convention
    camera = np.array([0.0, 0.0, GROUND + 1.0])
    keypoints, truths = [], []
    for cone in ([6.0, 0.5], [11.0, -2.0], [16.0, 3.0]):
        world = silhouette_keypoints(cone, GROUND, camera)
        keypoints.append(INTR.project(world_to_optical(world, camera, LEVEL)))
        truths.append(world_to_optical(np.array([[*cone, GROUND]]), camera, LEVEL)[0])
    positions = MonoPipeline.positions(np.array(keypoints), INTR.matrix)
    for t, truth in zip(positions, truths):
        # the edge keypoints sit a few millimetres in front of the axis plane, nothing else is off
        assert t is not None and np.linalg.norm(t - truth) < 0.01


def texture(shape, seed):
    """Smooth seeded noise the size of a cone's box, with enough local contrast for ORB to find corners on it."""
    rng = np.random.default_rng(seed)
    h, w = shape
    coarse = rng.uniform(0, 255, (max(h // 4, 2), max(w // 4, 2))).astype(np.float32)
    return cv2.GaussianBlur(cv2.resize(coarse, (w, h)), (0, 0), 1.2).astype(np.uint8)


def true_box(cone_xy, position, pad=4):
    """Pixel box of a cone's true silhouette as seen from ``position``, padded like a detector box would be."""
    world = silhouette_keypoints(cone_xy, GROUND, position)
    uv = INTR.project(world_to_optical(world, position, LEVEL))
    lo, hi = uv.min(axis=0) - pad, uv.max(axis=0) + pad
    return uv, np.array([lo[0], lo[1], hi[0], hi[1]])


def scene_for(cone_xy, seed):
    """Left and right BGR images with the same random patch pasted onto the cone's true box in each, at the
    right disparity; the left box (as a detector would report it) and both images' true keypoints."""
    uv_left, box_left = true_box(cone_xy, LEFT_POS)
    uv_right, box_right = true_box(cone_xy, RIGHT_POS)
    x1, y1, x2, y2 = box_left.round().astype(int)
    w, h = x2 - x1, y2 - y1
    patch = texture((h, w), seed)[:, :, None]
    left_image = np.full((INTR.height, INTR.width, 3), 110, np.uint8)
    right_image = np.full((INTR.height, INTR.width, 3), 110, np.uint8)
    left_image[y1:y2, x1:x2] = patch
    rx1, ry1 = box_right[:2].round().astype(int)
    right_image[ry1:ry1 + h, rx1:rx1 + w] = patch
    return left_image, right_image, box_left, uv_left, uv_right


class FakeMono:
    """Stands in for MonoPipeline: a fixed detection and the exact projected keypoints, the real upright PnP on top
    so the stereo pipeline is exercised with clean geometry instead of a trained keypoint network."""

    def __init__(self, box, keypoints, cls=ConeColor.BLUE, score=0.9):
        self.box, self.kp, self.cls, self.score = box, keypoints, cls, score

    def detect(self, image):
        return np.array([self.box]), np.array([self.cls]), np.array([self.score])

    def keypoints(self, image, boxes):
        return np.array([self.kp])

    def locate(self, keypoints, K, up=None, camera_height=None):
        return solve_upright_positions(keypoints, K, up)


def test_estimate_recovers_a_synthetic_cone_and_the_right_box_holds_it():
    cone = np.array([5.0, 0.5])
    left_image, right_image, box, uv_left, uv_right = scene_for(cone, seed=1)
    left = View(left_image, INTR, LEFT_POS, LEVEL, GROUND)
    right = View(right_image, INTR, RIGHT_POS, LEVEL, GROUND)
    pipe = StereoPipeline(FakeMono(box, uv_left))
    est = pipe.estimate(left, right, np.array([box]))[0]
    assert est.points > 0 and est.matches >= est.points
    assert est.right_box is not None
    assert (uv_right.min(axis=0) >= est.right_box[:2]).all() and (uv_right.max(axis=0) <= est.right_box[2:]).all()
    # triangulation has to beat a mono-only guess biased by [0.3, -0.2] m, i.e. off by 0.36 m
    biased = cone + [0.3, -0.2]
    assert np.linalg.norm(est.stereo - cone) < np.linalg.norm(biased - cone)
    # within 3 sigma of the stereo module's own documented accuracy at 5 m (position_sigma(5) = 0.0525 m)
    dist = float(np.hypot(*(cone - LEFT_POS[:2])))
    assert np.linalg.norm(est.stereo - cone) < 3 * stereo_position_sigma(dist)


def test_call_drops_a_cone_beyond_max_range():
    cone = np.array([14.0, 1.0])
    left_image, right_image, box, uv_left, _ = scene_for(cone, seed=1)
    left = View(left_image, INTR, LEFT_POS, LEVEL, GROUND)
    right = View(right_image, INTR, RIGHT_POS, LEVEL, GROUND)
    pipe = StereoPipeline(FakeMono(box, uv_left), max_range=10.0)
    est = pipe.estimate(left, right, np.array([box]))[0]
    # triangulation succeeds on its own, the cone is only dropped for being 14 m out with a 10 m max_range
    assert est.stereo is not None
    assert pipe(left, right) == []


def test_call_drops_cones_whose_pnp_failed():
    box = np.array([100.0, 100.0, 130.0, 140.0])
    left_image = np.zeros((INTR.height, INTR.width, 3), np.uint8)
    right_image = np.zeros((INTR.height, INTR.width, 3), np.uint8)
    left = View(left_image, INTR, LEFT_POS, LEVEL, GROUND)
    right = View(right_image, INTR, RIGHT_POS, LEVEL, GROUND)

    class NoPnP(FakeMono):
        def locate(self, keypoints, K, up=None, camera_height=None):
            return [None]

    pipe = StereoPipeline(NoPnP(box, np.zeros((7, 2))))
    est = pipe.estimate(left, right, np.array([box]))[0]
    assert est.mono is None and est.stereo is None and est.matches == 0 and est.points == 0
    assert pipe(left, right) == []


def test_mono_leaves_out_boxes_at_the_left_and_right_border():
    pipe = MonoPipeline(None, edge_px=2.0)
    boxes = np.array([[0.0, 100, 40, 200], [1.9, 100, 40, 200], [2.0, 100, 40, 200], [600, 100, 640, 200],
                      [1250, 100, 1278, 200], [1250, 100, 1278.1, 200]])
    assert pipe.inside(boxes, 1280).tolist() == [False, False, True, True, True, False]
    assert MonoPipeline(None, edge_px=0.0).inside(boxes, 1280).all()


def test_mono_pnp_mode_is_checked_and_upright_needs_up():
    import pytest

    with pytest.raises(ValueError):
        MonoPipeline(None, pnp="ransac")
    kp = np.zeros((1, 7, 2))
    with pytest.raises(ValueError):
        MonoPipeline(None, pnp="upright").locate(kp, INTR.matrix)
    with pytest.raises(ValueError):
        MonoPipeline(None, pnp="ground").locate(kp, INTR.matrix, [0.0, -1.0, 0.0])


def test_estimate_skips_matching_beyond_the_match_range():
    cone = np.array([5.0, 0.5])
    left_image, right_image, box, uv_left, _ = scene_for(cone, seed=1)
    left = View(left_image, INTR, LEFT_POS, LEVEL, GROUND)
    right = View(right_image, INTR, RIGHT_POS, LEVEL, GROUND)
    pipe = StereoPipeline(FakeMono(box, uv_left))
    skipped = pipe.estimate(left, right, np.array([box]), match_range=4.0)[0]
    assert skipped.mono is not None and skipped.stereo is None and skipped.matches == 0
    assert pipe.estimate(left, right, np.array([box]), match_range=6.0)[0].stereo is not None
    # the pipeline itself matches up to max_range plus match_margin
    assert len(StereoPipeline(FakeMono(box, uv_left), max_range=3.0, match_margin=0.5)(left, right)) == 0
    assert len(StereoPipeline(FakeMono(box, uv_left), max_range=6.0)(left, right)) == 1


def test_jpeg_like_matches_a_whole_image_jpeg_around_the_boxes():
    from driverless_perception.vision.mono import jpeg_like

    rng = np.random.default_rng(0)
    raw = cv2.GaussianBlur(rng.integers(0, 255, (480, 640, 3)).astype(np.uint8), (0, 0), 2.0)
    whole = cv2.imdecode(cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], cv2.IMREAD_COLOR)
    boxes = np.array([[203.4, 150.2, 221.9, 177.5], [600.0, 300.0, 639.0, 479.0], [0.0, 0.0, 12.0, 30.0]])
    out = jpeg_like(raw, boxes, 95)
    for x1, y1, x2, y2 in boxes.astype(int):
        assert np.array_equal(out[y1:y2 + 1, x1:x2 + 1], whole[y1:y2 + 1, x1:x2 + 1])
    # far from every box the image is left as it was
    assert np.array_equal(out[300:400, 300:400], raw[300:400, 300:400])
