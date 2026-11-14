"""Cone crops for the keypoint network: a cache of padded crops, and the augmentation of Dhall et al. (2019).

The network sees the detector's box resized to 80x80. Training uses the ground truth boxes, jittered the way the
paper did (rotation up to 15 degrees, scale 0.8 to 1.5, shift up to half the box) so the network copes with
imprecise boxes. Decoding full camera images for every crop would make an epoch take minutes, so every cone is cut
out once with room for the jitter (``CONTEXT`` times its box) and kept as a small JPEG.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CROP = 80
CONTEXT = 2.8
CACHE_SIZE = 160


@dataclass(frozen=True)
class Augmentation:
    rotation_deg: float = 15.0
    scale: tuple[float, float] = (0.8, 1.5)
    shift: float = 0.5


def window_transform(box, angle_deg: float = 0.0, scale: float = 1.0, shift=(0.0, 0.0),
                     size: int = CROP) -> np.ndarray:
    """2x3 affine matrix that maps a window around ``box`` (x1, y1, x2, y2) onto a size x size image.

    The window is the box scaled by ``scale`` about its centre, moved by ``shift`` (fractions of the box size) and
    turned by ``angle_deg``. Width and height are stretched separately, like resizing a box to a square.
    """
    x1, y1, x2, y2 = box
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    cx = (x1 + x2) / 2 + shift[0] * (x2 - x1)
    cy = (y1 + y2) / 2 + shift[1] * (y2 - y1)
    a = np.radians(angle_deg)
    rotate = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    stretch = np.diag([size / w, size / h])
    m = stretch @ rotate
    return np.hstack([m, (np.array([size / 2, size / 2]) - m @ [cx, cy])[:, None]])


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _warp(image: np.ndarray, matrix: np.ndarray, size: int, flags: int) -> np.ndarray:
    """``cv2.warpAffine`` with a matrix for our pixel convention. Points and boxes here have pixel i spanning i to
    i + 1, OpenCV puts pixel centres on whole numbers, which would shift scaled crops by up to half a pixel."""
    m = matrix.copy()
    m[:, 2] += m[:, :2] @ [0.5, 0.5] - 0.5
    return cv2.warpAffine(image, m, (size, size), flags=flags, borderMode=cv2.BORDER_REPLICATE)


def augmented_crop(image: np.ndarray, box, keypoints: np.ndarray, rng: np.random.Generator,
                   aug: Augmentation) -> tuple[np.ndarray, np.ndarray]:
    """A randomly jittered 80x80 crop of ``box`` and the keypoints moved along with it."""
    m = window_transform(box, rng.uniform(-aug.rotation_deg, aug.rotation_deg), rng.uniform(*aug.scale),
                         rng.uniform(-aug.shift, aug.shift, 2))
    return _warp(image, m, CROP, cv2.INTER_LINEAR), apply(m, keypoints)


def plain_crop(image: np.ndarray, box, keypoints: np.ndarray | None = None):
    """The box resized to 80x80 without jitter, what the network sees at runtime. Also returns the transform, whose
    inverse takes predicted keypoints back into the image."""
    m = window_transform(box)
    return _warp(image, m, CROP, cv2.INTER_LINEAR), (apply(m, keypoints) if keypoints is not None else None), m


@dataclass
class CropCache:
    """Padded crops of one split. ``boxes`` and ``keypoints`` (N, 7, 2) are in pixels of the cached crops,
    ``image_boxes`` keeps the same boxes in pixels of the camera image, which is what an error means for PnP."""

    jpegs: list[bytes]
    boxes: np.ndarray
    keypoints: np.ndarray
    classes: np.ndarray
    distances: np.ndarray
    image_boxes: np.ndarray = None

    def __post_init__(self):
        if self.image_boxes is None:
            self.image_boxes = np.zeros((len(self.jpegs), 4))

    def __len__(self) -> int:
        return len(self.jpegs)

    def image(self, i: int) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.jpegs[i], np.uint8), cv2.IMREAD_COLOR)

    def save(self, path: Path) -> None:
        offsets = np.cumsum([0] + [len(j) for j in self.jpegs])
        np.savez(path, data=np.frombuffer(b"".join(self.jpegs), np.uint8), offsets=offsets, boxes=self.boxes,
                 keypoints=self.keypoints, classes=self.classes, distances=self.distances,
                 image_boxes=self.image_boxes)

    @classmethod
    def load(cls, path: Path) -> CropCache:
        z = np.load(path)
        data, offsets = z["data"], z["offsets"]
        jpegs = [data[offsets[i]:offsets[i + 1]].tobytes() for i in range(len(offsets) - 1)]
        return cls(jpegs, z["boxes"], z["keypoints"], z["classes"], z["distances"], z.get("image_boxes"))


def build_cache(dataset: Path, split: str, min_height: int = 20, quality: int = 95) -> CropCache:
    """Cuts every cone of ``split`` in ``keypoints.csv`` with a box at least ``min_height`` pixels high out of its
    image, with ``CONTEXT`` times the box around it, scaled to ``CACHE_SIZE`` pixels."""
    with open(dataset / "keypoints.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split and int(r["y2"]) - int(r["y1"]) >= min_height]
    by_image = {}
    for r in rows:
        by_image.setdefault(r["image"], []).append(r)
    jpegs, boxes, keypoints, classes, distances, image_boxes = [], [], [], [], [], []
    for name, cones in by_image.items():
        image = cv2.imread(str(dataset / "images" / split / f"{name}.jpg"))
        for r in cones:
            box = np.array([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
            kp = np.array([[float(r[f"u{i}"]), float(r[f"v{i}"])] for i in range(7)])
            side = CONTEXT * max(box[2] - box[0], box[3] - box[1])
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            window = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
            m = window_transform(window, size=CACHE_SIZE)
            crop = _warp(image, m, CACHE_SIZE, cv2.INTER_AREA)
            jpegs.append(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
            corners = apply(m, box.reshape(2, 2))
            boxes.append(corners.reshape(4))
            keypoints.append(apply(m, kp))
            classes.append(int(r["class"]))
            distances.append(float(r["distance"]))
            image_boxes.append(box)
    return CropCache(jpegs, np.array(boxes).reshape(-1, 4), np.array(keypoints).reshape(-1, 7, 2),
                     np.array(classes, int), np.array(distances), np.array(image_boxes).reshape(-1, 4))
