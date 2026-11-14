"""Features inside the left and right box of a cone, matched between the two images.

The paper used SIFT, which beat BRISK for them. This uses ORB with small patches: on simulator cones it found far
more matches (median 78 per cone at 5 to 10 m against 11 for SIFT), and a cone box is too small for ORB's default
31 pixel patches. Matches go through the ratio test and have to lie close to their epipolar line.
"""
from __future__ import annotations

import cv2
import numpy as np

_orb = None


def _detector():
    global _orb
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=500, patchSize=11, edgeThreshold=11, fastThreshold=5)
    return _orb


def _features(gray: np.ndarray, box, margin: int = 16) -> tuple[np.ndarray, np.ndarray | None]:
    """ORB on a cut-out around the box (the whole image pyramid would cost more than the matching), features
    inside the box only."""
    h, w = gray.shape
    x1, y1 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
    x2, y2 = min(int(np.ceil(box[2])), w), min(int(np.ceil(box[3])), h)
    ox, oy = max(x1 - margin, 0), max(y1 - margin, 0)
    crop = gray[oy:min(y2 + margin, h), ox:min(x2 + margin, w)]
    mask = np.zeros(crop.shape, np.uint8)
    mask[y1 - oy:y2 - oy, x1 - ox:x2 - ox] = 255
    keypoints, descriptors = _detector().detectAndCompute(crop, mask)
    # OpenCV puts pixel centres on whole numbers, here pixel i spans i to i + 1
    return np.array([k.pt for k in keypoints], float).reshape(-1, 2) + [ox + 0.5, oy + 0.5], descriptors


def match_features(gray_left: np.ndarray, box_left, gray_right: np.ndarray, box_right,
                   F: np.ndarray | None = None, ratio: float = 0.8,
                   max_epipolar_px: float = 2.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Matched points (N, 2) in the left and the right image, and how the matching went (features per image, how
    many matches the ratio test left and how far the kept ones lie from their epipolar line). ``F`` is the
    fundamental matrix from left to right (``stereo.triangulation.fundamental``), matches further than
    ``max_epipolar_px`` from their line are dropped."""
    pl, dl = _features(gray_left, box_left)
    pr, dr = _features(gray_right, box_right)
    info = {"features_left": len(pl), "features_right": len(pr), "ratio_test": 0, "epipolar_px": np.zeros(0)}
    if dl is None or dr is None or len(pr) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), info
    left, right = [], []
    for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dl, dr, k=2):
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            left.append(pl[m[0].queryIdx])
            right.append(pr[m[0].trainIdx])
    left, right = np.array(left).reshape(-1, 2), np.array(right).reshape(-1, 2)
    info["ratio_test"] = len(left)
    if F is not None and len(left):
        lines = np.hstack([left, np.ones((len(left), 1))]) @ F.T
        d = np.abs(np.sum(lines * np.hstack([right, np.ones((len(right), 1))]), axis=1))
        d /= np.hypot(lines[:, 0], lines[:, 1])
        keep = d <= max_epipolar_px
        left, right, info["epipolar_px"] = left[keep], right[keep], d[keep]
    return left, right, info
