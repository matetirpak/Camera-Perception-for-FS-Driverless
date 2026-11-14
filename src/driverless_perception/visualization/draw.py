"""Drawing boxes, keypoints and captions onto BGR images with OpenCV, and putting images together into a grid."""
from __future__ import annotations

import cv2
import numpy as np

from ..common.types import CONE_RGB

CONE_BGR = {int(c): tuple(round(255 * v) for v in reversed(rgb)) for c, rgb in CONE_RGB.items()}
GREEN, RED, WHITE = (80, 200, 60), (60, 60, 230), (255, 255, 255)


def draw_boxes(image: np.ndarray, boxes, classes=None, scores=None, color=None, thickness: int = 2) -> np.ndarray:
    """Draws in place. Boxes are (x1, y1, x2, y2) with pixel i spanning i to i + 1."""
    for i, box in enumerate(np.asarray(boxes, float).reshape(-1, 4)):
        c = color or CONE_BGR.get(int(classes[i]) if classes is not None else 4)
        x1, y1, x2, y2 = (round(v) for v in box)
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), c, thickness, cv2.LINE_AA)
        if scores is not None:
            cv2.putText(image, f"{scores[i]:.2f}", (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1,
                        cv2.LINE_AA)
    return image


def draw_points(image: np.ndarray, points, color, radius: int = 2, cross: bool = False) -> np.ndarray:
    """Draws (N, 2) points in place, as dots or small crosses."""
    for u, v in np.asarray(points, float).reshape(-1, 2):
        p = (round(u - 0.5), round(v - 0.5))
        if cross:
            cv2.drawMarker(image, p, color, cv2.MARKER_CROSS, 2 * radius + 1, 1, cv2.LINE_AA)
        else:
            cv2.circle(image, p, radius, color, -1, cv2.LINE_AA)
    return image


def caption(image: np.ndarray, text: str, height: int = 28) -> np.ndarray:
    """The image with a dark bar and ``text`` on top."""
    bar = np.full((height, image.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def tile(images: list[np.ndarray], cols: int, width: int, gap: int = 6) -> np.ndarray:
    """Images scaled to ``width`` and laid out in rows of ``cols``, on a dark background."""
    scaled = [cv2.resize(im, (width, round(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
              for im in images]
    rows = [scaled[i:i + cols] for i in range(0, len(scaled), cols)]
    row_heights = [max(im.shape[0] for im in row) for row in rows]
    out = np.full((sum(row_heights) + gap * (len(rows) + 1), cols * width + gap * (cols + 1), 3), 20, np.uint8)
    y = gap
    for row, h in zip(rows, row_heights):
        for j, im in enumerate(row):
            x = gap + j * (width + gap)
            out[y:y + im.shape[0], x:x + width] = im
        y += h + gap
    return out


def png(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode the image")
    return data.tobytes()
