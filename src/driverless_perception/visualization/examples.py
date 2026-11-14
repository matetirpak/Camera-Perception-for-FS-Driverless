"""Example images: what the detector, the auto labeller, the keypoint network and the stereo matching produce,
drawn on real images and put together into one picture per subject."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..common.types import CLASS_NAMES
from .draw import CONE_BGR, GREEN, RED, WHITE, caption, draw_boxes, draw_points, png, tile


def _predict(model, image: np.ndarray, conf: float, imgsz: int = 1280):
    r = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0].boxes
    return r.xyxy.cpu().numpy(), r.cls.cpu().numpy().astype(int), r.conf.cpu().numpy()


def _yolo_boxes(label: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Boxes and classes of a YOLO label file, in pixels."""
    rows = [line.split() for line in label.read_text().split("\n") if line.strip()]
    classes = np.array([int(r[0]) for r in rows], int)
    xywh = np.array([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4) * [width, height, width, height]
    boxes = np.column_stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                             xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2])
    return boxes, classes


def fsoco_examples(dataset: Path, models: dict[str, object], conf: float, origins=("amz", "frt"),
                   width: int = 900) -> bytes:
    """A few FSOCO test images per model: labels in white, what the model found in the cone colours."""
    rows = []
    for name, model in models.items():
        images = []
        for origin in origins:
            files = sorted((dataset / "images/test").glob(f"{origin}_*.jpg"))
            path = files[len(files) // 3]
            image = cv2.imread(str(path))
            truth, _ = _yolo_boxes(dataset / "labels/test" / f"{path.stem}.txt", image.shape[1], image.shape[0])
            draw_boxes(image, truth, color=WHITE, thickness=1)
            draw_boxes(image, *_predict(model, image, conf)[:2], thickness=2)
            images.append(caption(image, f"{origin.upper()} test image, {name}"))
        rows += images
    return png(tile(rows, len(origins), width))


def sim_examples(captures: list[Path], model, conf: float, camera: str = "cam_mono", width: int = 900,
                 cols: int = 1) -> bytes:
    """One image per capture with the automatic labels in white and the detector's boxes in the cone colours."""
    images = []
    for capture in captures:
        boxes = defaultdict(list)
        with open(capture / "boxes.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row["labelled"] == "1" and row["image"].endswith(camera):
                    boxes[row["image"]].append(row)
        name = sorted(boxes)[len(boxes) // 2]
        image = cv2.imread(str(capture / "images" / f"{name}.jpg"))
        truth = np.array([[float(r[k]) for k in ("x1", "y1", "x2", "y2")] for r in boxes[name]])
        found, classes, _ = _predict(model, image, conf)
        draw_boxes(image, truth, color=WHITE, thickness=1)
        draw_boxes(image, found, classes, thickness=2)
        images.append(caption(image, f"{capture.name.split('_')[0]}: {len(truth)} labels, {len(found)} found"))
    return png(tile(images, cols, width))


def keypoint_examples(cache, model, device: str, n: int = 24, scale: int = 3, width: int = 240) -> bytes:
    """Crops with the true keypoints in green and the network's in red, spread over the distance range."""
    import torch

    from ..vision.keypoints.crops import plain_crop
    from ..vision.keypoints.network import to_tensor

    order = np.argsort(cache.distances)
    picked = order[np.linspace(0, len(order) - 1, n).astype(int)]
    crops, truths = [], []
    for i in picked:
        crop, keypoints, _ = plain_crop(cache.image(i), cache.boxes[i], cache.keypoints[i])
        crops.append(crop)
        truths.append(keypoints)
    with torch.no_grad():
        predicted = model(to_tensor(np.stack(crops)).to(device)).float().cpu().numpy()
    images = []
    for crop, truth, pred, i in zip(crops, truths, predicted, picked):
        big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        draw_points(big, truth * scale, GREEN, 3)
        draw_points(big, pred * scale, RED, 3, cross=True)
        images.append(caption(big, f"{CLASS_NAMES[int(cache.classes[i])]} at {cache.distances[i]:.1f} m", 22))
    return png(tile(images, 6, width))


def match_example(left: np.ndarray, right: np.ndarray, boxes, classes, estimates, width: int = 1500) -> bytes:
    """The two images of one frame side by side, with the matched features that ended up on a cone."""
    pair = np.hstack([left.copy(), right.copy()])
    offset = left.shape[1]
    for box, cls, e in zip(boxes, classes, estimates):
        if e.right_box is None or e.pts_left is None:
            continue
        draw_boxes(pair, [box], [cls], thickness=2)
        draw_boxes(pair, [np.array(e.right_box) + [offset, 0, offset, 0]], color=(150, 150, 150), thickness=1)
        for (ul, vl), (ur, vr), on in zip(e.pts_left, e.pts_right, e.on_cone):
            colour = CONE_BGR[int(cls)] if on else (120, 120, 120)
            cv2.line(pair, (int(ul), int(vl)), (int(ur + offset), int(vr)), colour, 1, cv2.LINE_AA)
        draw_points(pair, e.pts_left[e.on_cone], GREEN, 2)
        draw_points(pair, e.pts_right[e.on_cone] + [offset, 0], GREEN, 2)
    return png(caption(cv2.resize(pair, (width, int(pair.shape[0] * width / pair.shape[1]))),
                       "left image with the detector's boxes, right image with the propagated search boxes; "
                       "lines are ORB matches, coloured where the point landed on the cone"))
