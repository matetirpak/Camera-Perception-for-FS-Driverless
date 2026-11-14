"""The mono camera pipeline of the paper: detector box, 7 keypoints on the 80x80 crop, PnP, cone position.

``MonoPipeline`` takes boxes from any detector and returns the cones' positions in the camera's optical frame, and
with a camera pose in the map frame. Only small cones get keypoints (the paper's cone model has one stripe), and only
boxes at least ``min_height`` pixels high, below that the keypoints are too close together for PnP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.timing import no_timer
from ..common.types import ConeColor
from .keypoints.crops import plain_crop
from .pnp import UPRIGHT_MODES, solve_cone_position, solve_upright_positions

SMALL_CONES = (ConeColor.BLUE, ConeColor.YELLOW, ConeColor.ORANGE_SMALL)
# slot of each colour in Cone.color_probs: blue, yellow, orange (big or small), unknown
_PROB_SLOT = {ConeColor.BLUE: 0, ConeColor.YELLOW: 1, ConeColor.ORANGE_BIG: 2, ConeColor.ORANGE_SMALL: 2,
              ConeColor.UNKNOWN: 3}


def position_sigma(distance: float) -> float:
    """Standard deviation of a mono cone position in metres. Fitted in September to the errors on the simulator
    test map (RMSE 0.09 m below 5 m, 0.14 m at 5 to 10 m, 0.34 m at 10 to 15 m and 0.6 m at 15 to 20 m), when mono
    still solved the full pose; with the upright cone the last two are 0.26 and 0.52 m, so it errs on the safe
    side there."""
    return 0.05 + 0.0018 * distance ** 2


def jpeg_like(image: np.ndarray, boxes: np.ndarray, quality: int, pad: int = 16) -> np.ndarray:
    """A copy of ``image`` in which the area around every box looks as if the whole image had been saved as JPEG.

    The keypoint network learned on JPEG captures, and on the raw images of a race it put far cones about 0.2 m too
    far away. JPEG works on blocks of 16x16 pixels from the image's corner, so a region that starts and ends on that
    grid comes out pixel for pixel like the same area of the whole image saved at once. A few small regions cost a
    fraction of a millisecond, the whole image ten."""
    import cv2

    out = image.copy()
    h, w = image.shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        x0, y0 = max(0, int(x1 - pad) // 16 * 16), max(0, int(y1 - pad) // 16 * 16)
        xe, ye = min(w, -(-int(x2 + 1 + pad) // 16) * 16), min(h, -(-int(y2 + 1 + pad) // 16) * 16)
        ok, buf = cv2.imencode(".jpg", image[y0:ye, x0:xe], [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out[y0:ye, x0:xe] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def color_probs(color: int, score: float) -> np.ndarray:
    """The detector's class gets its confidence, the other three share the rest."""
    probs = np.full(4, (1.0 - score) / 3)
    probs[_PROB_SLOT[ConeColor(color)]] = score
    return probs


@dataclass
class MonoCone:
    """A cone found by the pipeline: base in the camera's optical frame, detector class and score, box."""

    position: np.ndarray
    color: int
    score: float
    box: np.ndarray


PNP_MODES = ("sqpnp", *UPRIGHT_MODES)


class MonoPipeline:
    """``pnp`` chooses how the keypoints become a position: ``sqpnp`` solves the full pose of the cone, ``upright``
    only the position of an upright cone, ``ground`` only its position on the ground plane (see ``vision.pnp``). The
    last two need the camera's "up" and, for ``ground``, its height above the ground.

    Boxes closer than ``edge_px`` to the left or right image border are left out of the whole pipeline: a cone cut
    off by the frame gets keypoints for a cone that isn't there, and in a race those were the worst positions.
    Stereo uses ``detect`` directly and keeps them, the search box in the right image still holds the cone.

    ``jpeg_quality`` is for raw camera images: the keypoints are then taken from a JPEG version of the area around
    each box (``jpeg_like``), like the captures the network was trained on. Off for images that already are JPEG.

    ``half`` runs the detector in half precision and ``compile`` through ``torch.compile`` with that mode.
    "reduce-overhead" also records CUDA graphs, which removes most of the kernel launch overhead that dominates a
    single 1280 px image; together they make YOLO26s about twice as fast on the GPU. Compiling takes several seconds
    on the first image, ``warm_up`` does that before the first real frame. A TensorRT engine
    (``scripts/export_engine.py``) can be given as the detector instead, both options are then ignored."""

    def __init__(self, keypoint_model, device: str = "cpu", min_height: int = 20, detector=None, conf: float = 0.25,
                 imgsz: int = 1280, half: bool = False, timer=no_timer, pnp: str = "upright", edge_px: float = 2.0,
                 jpeg_quality: int | None = None, compile: str | None = None):
        if pnp not in PNP_MODES:
            raise ValueError(f"pnp has to be one of {PNP_MODES}, not {pnp!r}")
        self.model, self.device, self.min_height = keypoint_model, device, min_height
        # half precision and CUDA graphs only make sense on the GPU, on the CPU they are left out. An exported model
        # such as a TensorRT engine (ultralytics keeps its path as a string in .model) has its precision built in.
        gpu = str(device).startswith("cuda")
        self.exported = isinstance(getattr(detector, "model", None), str)
        self.detector, self.conf, self.imgsz = detector, conf, imgsz
        self.half = half and gpu and not self.exported
        self.compile = (compile or None) if gpu and not self.exported else None
        self.pnp, self.edge_px, self.jpeg_quality = pnp, edge_px, jpeg_quality
        # ``common.timing.StageTimer`` to measure the stages, see scripts/benchmark_latency.py
        self.timer = timer

    def __call__(self, image: np.ndarray, K: np.ndarray, up=None, camera_height: float | None = None) -> list[MonoCone]:
        """The whole pipeline on one BGR image, with the detector given at construction."""
        boxes, classes, scores = self.detect(image)
        inside = self.inside(boxes, image.shape[1])
        boxes, classes, scores = boxes[inside], classes[inside], scores[inside]
        keypoints = self.keypoints(image, boxes)
        with self.timer("pnp"):
            positions = self.locate(keypoints, K, up, camera_height)
        return [MonoCone(p, int(c), float(s), b) for p, c, s, b in zip(positions, classes, scores, boxes)
                if p is not None]

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boxes, classes and scores of the detector, only those the keypoint network can take."""
        with self.timer("detector"):
            prediction = self.detector.predict(image, imgsz=self.input_size(), conf=self.conf, verbose=False,
                                               **self.detector_options())[0]
            result = prediction.boxes
            boxes = result.xyxy.cpu().numpy()
        for stage, ms in getattr(prediction, "speed", {}).items():
            self.timer_add(f"detector {stage}", ms)
        classes = result.cls.cpu().numpy().astype(int)
        scores = result.conf.cpu().numpy()
        use = np.array([self.usable(b, c) for b, c in zip(boxes, classes)], bool)
        # not a time: how many boxes the per cone stages get, for the latency per cone
        self.timer_add("boxes", int(use.sum()))
        return boxes[use].reshape(-1, 4), classes[use], scores[use]

    def detector_options(self) -> dict:
        """The keyword arguments for the detector's ``predict`` that set its precision and compilation."""
        options = {}
        if self.half:
            # ultralytics calls half precision quantize=16
            options["quantize"] = 16
        if self.compile:
            options["compile"] = self.compile
        return options

    def input_size(self):
        """The input size to ask the detector for. An exported model is built for one size, which ultralytics finds
        on the first call but forgets again when the next call asks for imgsz, so from then on its own size is
        passed."""
        predictor = getattr(self.detector, "predictor", None)
        if self.exported and predictor is not None:
            return list(predictor.imgsz)
        return self.imgsz

    def warm_up(self, height: int = 960, width: int = 1280, runs: int = 3) -> None:
        """Runs the detector a few times on a blank image of the camera's size, so that compiling and recording
        the CUDA graphs happen before the first real frame and not during it."""
        if self.detector is None:
            return
        blank = np.zeros((height, width, 3), np.uint8)
        for _ in range(runs):
            self.detector.predict(blank, imgsz=self.input_size(), conf=self.conf, verbose=False,
                                  **self.detector_options())

    def timer_add(self, name: str, ms: float) -> None:
        if hasattr(self.timer, "add"):
            self.timer.add(name, ms)

    def inside(self, boxes: np.ndarray, width: int) -> np.ndarray:
        """Which of the (N, 4) boxes keep at least ``edge_px`` from the left and right border."""
        boxes = np.asarray(boxes).reshape(-1, 4)
        return (boxes[:, 0] >= self.edge_px) & (boxes[:, 2] <= width - self.edge_px)

    def usable(self, box, color: int) -> bool:
        return color in SMALL_CONES and box[3] - box[1] >= self.min_height

    def keypoints(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """(N, 7, 2) keypoints in image pixels (pixel i spans i to i + 1) for (N, 4) boxes."""
        import torch

        from .keypoints.network import to_tensor

        if len(boxes) == 0:
            return np.zeros((0, 7, 2))
        crops, inverses = [], []
        if self.jpeg_quality:
            with self.timer("jpeg"):
                image = jpeg_like(image, boxes, self.jpeg_quality)
        with self.timer("crops"):
            for box in boxes:
                crop, _, m = plain_crop(image, box)
                crops.append(crop)
                inverses.append(np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2])
        with self.timer("keypoint net"), torch.no_grad():
            pred = self.model(to_tensor(np.stack(crops)).to(self.device)).float().cpu().numpy()
        return np.stack([p @ inv[:, :2].T + inv[:, 2] for p, inv in zip(pred, inverses)])

    def locate(self, keypoints: np.ndarray, K: np.ndarray, up=None,
               camera_height: float | None = None) -> list[np.ndarray | None]:
        """Cone base in the optical frame per cone with the PnP chosen at construction, None where it fails. ``up``
        is the upward direction in the optical frame, ``camera_height`` the camera's height above the ground."""
        if self.pnp == "sqpnp":
            return self.positions(keypoints, K)
        if up is None or (self.pnp == "ground" and camera_height is None):
            raise ValueError(f"pnp {self.pnp!r} needs the camera's up direction" +
                             (" and height" if self.pnp == "ground" else ""))
        return solve_upright_positions(keypoints, K, up, camera_height if self.pnp == "ground" else None)

    @staticmethod
    def positions(keypoints: np.ndarray, K: np.ndarray) -> list[np.ndarray | None]:
        """Cone base in the optical frame per cone, None where PnP fails. ``K`` has the principal point at the image
        centre (``Intrinsics``), in the same pixel convention as the keypoints."""
        return [solve_cone_position(kp, K) for kp in keypoints]
