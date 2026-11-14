import numpy as np

from driverless_perception.evaluation.boxes import FP, IGNORED, TP, box_iou, f1, match, precision_recall


def test_box_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(box_iou(a, b), [[1.0, 50 / 150, 0.0]])


def test_match_is_greedy_by_score_and_counts_duplicates_as_false():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
    det = np.array([[1, 0, 11, 10], [0, 0, 10, 10], [20, 0, 30, 10], [50, 50, 60, 60]])
    m = match(det, [0, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], gt, [0, 1])
    # the best scoring detection takes the first box, the better placed second one is a duplicate
    assert list(m.det_result) == [TP, FP, TP, FP]
    assert list(m.det_gt) == [0, -1, 1, -1]
    assert np.allclose(m.gt_score, [0.9, 0.7])


def test_match_respects_classes_unless_told_otherwise():
    gt, det = np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]])
    assert list(match(det, [1], [0.5], gt, [0]).det_result) == [FP]
    assert list(match(det, [1], [0.5], gt, [0], class_aware=False).det_result) == [TP]


def test_detections_on_ignored_boxes_are_neither_right_nor_wrong():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
    det = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
    m = match(det, [0, 0], [0.9, 0.8], gt, [0, 4], gt_ignore=[False, True])
    assert list(m.det_result) == [TP, IGNORED]
    assert m.gt_score[1] == -np.inf


def test_precision_recall_at_thresholds():
    result = np.array([TP, FP, TP, IGNORED, FP])
    score = np.array([0.9, 0.8, 0.7, 0.65, 0.6])
    p, r = precision_recall(result, score, n_gt=4, thresholds=[0.95, 0.85, 0.7, 0.5])
    assert np.allclose(p, [1.0, 1.0, 2 / 3, 2 / 4])
    assert np.allclose(r, [0.0, 0.25, 0.5, 0.5])
    assert np.allclose(f1([1.0, 0.5], [0.5, 0.0]), [2 / 3, 0.0])
