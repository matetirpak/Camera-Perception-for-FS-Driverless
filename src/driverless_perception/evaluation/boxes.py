"""Matching 2D boxes to ground truth, for detectors evaluated on images.

Greedy in order of score like COCO and Pascal VOC, so one pass is valid for every confidence threshold and
precision and recall become cumulative sums. Ground truth boxes can be marked as ignored: a detection that hits
only those counts neither way, which is how the unknown colour class is left out of the comparison with the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TP, FP, IGNORED = 1, 0, -1


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (N,4) and ``b`` (M,4), both x1 y1 x2 y2. Returns (N,M)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


@dataclass
class Matches:
    """Result of matching the detections of one image."""

    det_result: np.ndarray  # per detection: TP, FP or IGNORED
    det_gt: np.ndarray  # per detection: index of the ground truth box it matched, or -1
    gt_score: np.ndarray  # per ground truth box: score of the detection that found it, or -inf


def match(
    det_xyxy: np.ndarray,
    det_cls: np.ndarray,
    det_score: np.ndarray,
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    gt_ignore: np.ndarray | None = None,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> Matches:
    det_cls, det_score, gt_cls = np.asarray(det_cls), np.asarray(det_score, float), np.asarray(gt_cls)
    n_det, n_gt = len(det_score), len(gt_cls)
    gt_ignore = np.zeros(n_gt, bool) if gt_ignore is None else np.asarray(gt_ignore, bool)
    result = np.full(n_det, FP)
    det_gt = np.full(n_det, -1)
    gt_score = np.full(n_gt, -np.inf)
    if n_det == 0 or n_gt == 0:
        return Matches(result, det_gt, gt_score)

    iou = box_iou(det_xyxy, gt_xyxy)
    overlaps = iou >= iou_threshold
    if class_aware:
        overlaps &= det_cls[:, None] == gt_cls[None, :]
    taken = gt_ignore.copy()
    for d in np.argsort(-det_score, kind="stable"):
        free = overlaps[d] & ~taken
        if free.any():
            g = int(np.argmax(np.where(free, iou[d], -1.0)))
            taken[g] = True
            result[d], det_gt[d], gt_score[g] = TP, g, det_score[d]
        elif (iou[d] >= iou_threshold)[gt_ignore].any():
            result[d] = IGNORED
    return Matches(result, det_gt, gt_score)


def precision_recall(det_result: np.ndarray, det_score: np.ndarray, n_gt: int, thresholds: np.ndarray):
    """Precision and recall at each confidence threshold, from the results of all images put together."""
    det_result, det_score = np.asarray(det_result), np.asarray(det_score, float)
    keep = det_result != IGNORED
    det_result, det_score = det_result[keep], det_score[keep]
    order = np.argsort(-det_score, kind="stable")
    tp = np.cumsum(det_result[order] == TP)
    # number of detections with a score at or above each threshold
    count = np.searchsorted(-det_score[order], -np.asarray(thresholds, float), side="right")
    # with nothing left after the ignored detections there is no true positive at any threshold
    tp_at = np.where(count > 0, tp[np.maximum(count - 1, 0)], 0) if len(tp) else np.zeros(len(count), int)
    precision = np.divide(tp_at, count, out=np.ones(len(count)), where=count > 0)
    recall = tp_at / n_gt if n_gt else np.full(len(count), np.nan)
    return precision, recall


def f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    precision, recall = np.asarray(precision, float), np.asarray(recall, float)
    s = precision + recall
    return np.divide(2 * precision * recall, s, out=np.zeros_like(s), where=s > 0)
