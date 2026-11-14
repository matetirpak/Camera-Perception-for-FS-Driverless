import numpy as np

from driverless_perception.common.types import ConeColor
from driverless_perception.evaluation.boxes import FP, IGNORED, TP, box_iou, f1, match, precision_recall
from driverless_perception.vision.detection.evaluate import (
    ImageLabels,
    Matched,
    _prf,
    _recall,
    best_threshold,
    breakdowns,
    match_split,
    summary_at,
)

# box_iou


def test_box_iou_zero_area_boxes_are_zero_not_nan():
    a = np.array([[0.0, 0.0, 0.0, 0.0]])
    b = np.array([[5.0, 5.0, 5.0, 5.0]])
    # both areas and the union are zero, the safe division must fall back to 0, not nan
    assert box_iou(a, b)[0, 0] == 0.0


def test_box_iou_boxes_touching_at_an_edge_do_not_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[10.0, 0.0, 20.0, 10.0]])
    assert box_iou(a, b)[0, 0] == 0.0


# match


def test_match_below_iou_threshold_is_a_false_positive_above_it_a_match():
    # a is fully inside b along x, giving iou = 4 / (100 + 40 - 40) = 0.4 exactly
    det = np.array([[0.0, 0.0, 10.0, 10.0]])
    gt = np.array([[0.0, 0.0, 4.0, 10.0]])
    below = match(det, [0], [0.9], gt, [0], iou_threshold=0.5)
    assert list(below.det_result) == [FP] and list(below.det_gt) == [-1]
    at = match(det, [0], [0.9], gt, [0], iou_threshold=0.4)
    assert list(at.det_result) == [TP] and list(at.det_gt) == [0]


def test_match_picks_the_highest_iou_free_box_not_the_first_free_one():
    # gt index 0 has the lower iou (0.9), gt index 1 the higher one (1.0)
    gt = np.array([[0.0, 0.0, 9.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    det = np.array([[0.0, 0.0, 10.0, 10.0]])
    m = match(det, [0], [0.9], gt, [0, 0])
    assert m.det_gt[0] == 1
    assert m.gt_score[0] == -np.inf and m.gt_score[1] == 0.9


def test_match_ties_in_score_are_broken_by_input_order_not_iou():
    # both detections tie at score 0.9, the first in the array (lower iou) still claims the only gt
    det = np.array([[0.0, 0.0, 9.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])
    m = match(det, [0, 0], [0.9, 0.9], gt, [0])
    assert list(m.det_result) == [TP, FP]
    assert list(m.det_gt) == [0, -1]


def test_match_ignores_regardless_of_predicted_class():
    # the detection's class does not match the ignored ground truth's class, but it is still ignored, not FP
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])
    det = np.array([[0.0, 0.0, 10.0, 10.0]])
    m = match(det, [0], [0.9], gt, [9], gt_ignore=[True])
    assert list(m.det_result) == [IGNORED]
    assert list(m.det_gt) == [-1]
    assert m.gt_score[0] == -np.inf


def test_match_with_no_ground_truth_all_detections_are_false_positives():
    det = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    gt = np.zeros((0, 4))
    m = match(det, [0, 1], [0.9, 0.5], gt, np.zeros(0, int))
    assert list(m.det_result) == [FP, FP]
    assert list(m.det_gt) == [-1, -1]
    assert len(m.gt_score) == 0


def test_match_with_no_detections_all_ground_truth_is_unfound():
    det = np.zeros((0, 4))
    gt = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]])
    m = match(det, np.zeros(0, int), np.zeros(0), gt, [0, 1])
    assert len(m.det_result) == 0 and len(m.det_gt) == 0
    assert list(m.gt_score) == [-np.inf, -np.inf]


# precision_recall


def test_precision_recall_with_no_ground_truth_gives_nan_recall():
    result = np.array([TP, FP])
    score = np.array([0.9, 0.6])
    p, r = precision_recall(result, score, n_gt=0, thresholds=[0.5, 0.7])
    assert np.allclose(p, [0.5, 1.0])
    assert np.all(np.isnan(r))


def test_precision_recall_when_every_detection_is_ignored():
    # nothing is left to be right or wrong, so precision keeps its default of 1.0 and recall stays at 0
    result = np.array([IGNORED, IGNORED])
    score = np.array([0.9, 0.8])
    p, r = precision_recall(result, score, n_gt=3, thresholds=[0.5])
    assert np.allclose(p, [1.0]) and np.allclose(r, [0.0])
    p, r = precision_recall(np.zeros(0), np.zeros(0), n_gt=3, thresholds=[0.2, 0.8])
    assert np.allclose(p, [1.0, 1.0]) and np.allclose(r, [0.0, 0.0])


def test_precision_recall_ties_at_the_threshold_are_all_counted():
    result = np.array([TP, TP, FP])
    score = np.array([0.9, 0.7, 0.7])
    p, r = precision_recall(result, score, n_gt=2, thresholds=[0.7])
    assert np.allclose(p, [2 / 3])
    assert np.allclose(r, [1.0])


# f1


def test_f1_both_zero_is_zero_not_a_division_error():
    assert np.allclose(f1(np.array([0.0]), np.array([0.0])), [0.0])


def test_f1_perfect_precision_and_recall_is_one():
    assert f1(np.array([1.0]), np.array([1.0]))[0] == 1.0


# best_threshold


def _matched(det_result, det_score, gt_ignore):
    det_result = np.asarray(det_result)
    det_score = np.asarray(det_score, float)
    gt_ignore = np.asarray(gt_ignore, bool)
    n_det, n_gt = len(det_score), len(gt_ignore)
    return Matched(det_result=det_result, det_score=det_score, det_cls=np.zeros(n_det, int),
                   det_height=np.zeros(n_det), det_group=np.full(n_det, "g"), gt_score=np.full(n_gt, -np.inf),
                   gt_cls=np.zeros(n_gt, int), gt_ignore=gt_ignore, gt_height=np.zeros(n_gt),
                   gt_group=np.full(n_gt, "g"), gt_distance=np.full(n_gt, np.nan),
                   gt_tags=[() for _ in range(n_gt)], pairs=np.zeros((0, 3)))


def test_best_threshold_finds_the_true_interior_maximum():
    # sorted by score: TP, TP, FP with n_gt=3. F1 is 2/3 above 0.9, 4/5 in (0.3, 0.7], 2/3 at or below 0.3, then 0
    # the grid step is 0.01, so the tied maximum of 0.8 starts at 0.31 and best_threshold must return its first tie
    m = _matched([TP, TP, FP], [0.9, 0.7, 0.3], [False, False, False])
    assert abs(best_threshold(m) - 0.31) < 1e-9


def test_best_threshold_with_no_detections_at_all():
    # a model that never fires has F1 zero everywhere, so the first grid point is returned
    m = _matched(np.array([]), np.array([]), [False, False])
    assert abs(best_threshold(m) - 0.05) < 1e-9


def test_best_threshold_with_no_ground_truth_at_all():
    # n_gt=0 makes recall nan everywhere, f1's safe division then falls back to 0 everywhere (nan > 0 is False),
    # so this also degenerates to the first grid point, 0.05
    m = _matched([TP, FP], [0.9, 0.5], [])
    assert abs(best_threshold(m) - 0.05) < 1e-9


# _prf


def test_prf_excludes_ignored_detections_from_both_tp_and_the_denominator():
    det_result = np.array([TP, IGNORED, FP])
    det_score = np.array([0.9, 0.8, 0.7])
    gt_score = np.array([0.9])
    gt_ignore = np.array([False])
    out = _prf(det_result, det_score, gt_score, gt_ignore, t=0.5)
    # the ignored detection does not count towards the 2 kept detections nor towards the 1 true positive
    assert out["detections"] == 2 and out["precision"] == 0.5
    assert out["cones"] == 1 and out["recall"] == 1.0
    assert np.isclose(out["f1"], 2 / 3)


def test_prf_no_detections_above_threshold_precision_is_none():
    det_result = np.array([FP])
    det_score = np.array([0.1])
    out = _prf(det_result, det_score, np.array([-np.inf]), np.array([False]), t=0.5)
    assert out["precision"] is None and out["detections"] == 0
    assert out["recall"] == 0.0 and out["f1"] is None


def test_prf_no_ground_truth_recall_is_none():
    det_result = np.array([TP])
    det_score = np.array([0.9])
    out = _prf(det_result, det_score, np.array([-np.inf, -np.inf]), np.array([True, True]), t=0.5)
    assert out["recall"] is None and out["cones"] == 0
    assert out["precision"] == 1.0 and out["f1"] is None


def test_prf_with_nothing_at_all_is_all_none():
    out = _prf(np.array([]), np.array([]), np.array([]), np.array([], bool), t=0.5)
    assert out["precision"] is None and out["recall"] is None and out["f1"] is None
    assert out["detections"] == 0 and out["cones"] == 0


# _recall


def test_recall_excludes_ignored_ground_truth_from_mask():
    gt_score = np.array([0.9, -np.inf, 0.4, 0.9])
    gt_ignore = np.array([False, True, False, False])
    mask = np.array([True, True, True, False])
    out = _recall(gt_score, gt_ignore, mask, t=0.5)
    # index 1 is masked in but ignored, so only indices 0 and 2 count, and only index 0 is found
    assert out["cones"] == 2 and out["recall"] == 0.5


def test_recall_with_an_empty_mask_is_none():
    gt_score = np.array([0.9])
    gt_ignore = np.array([False])
    out = _recall(gt_score, gt_ignore, np.array([False]), t=0.5)
    assert out["recall"] is None and out["cones"] == 0


# summary_at and breakdowns on a hand-built Matched


def _bins_example():
    # 3 detections, 4 ground truth boxes (one ignored), two groups, distances and tags mixed in on purpose
    return Matched(
        det_result=np.array([TP, FP, TP]),
        det_score=np.array([0.9, 0.8, 0.3]),
        det_cls=np.array([0, 1, 0]),
        det_height=np.array([5.0, 10.0, 20.0]),
        det_group=np.array(["g1", "g1", "g2"]),
        gt_score=np.array([0.9, -np.inf, -np.inf, 0.3]),
        gt_cls=np.array([0, 1, 0, 0]),
        gt_ignore=np.array([False, False, True, False]),
        gt_height=np.array([5.0, 12.0, 3.0, 20.0]),
        gt_group=np.array(["g1", "g1", "g2", "g2"]),
        gt_distance=np.array([5.0, 15.0, np.nan, 25.0]),
        gt_tags=[(), ("truncated",), (), ("truncated", "knocked_over")],
        pairs=np.array([[0.0, 0.0, 0.9], [0.0, 0.0, 0.3]]),
    )


def test_summary_at_per_class_and_confusion():
    m = _bins_example()
    s = summary_at(m, 0.5, {0: "blue", 1: "yellow"})
    # 2 kept detections (0.3 excluded), 1 tp -> precision 0.5, recall 1/3 (only the box found above threshold)
    assert s["all"]["precision"] == 0.5 and np.isclose(s["all"]["recall"], 1 / 3)
    # blue: only the 0.9 detection is kept, it is a tp, the ignored blue box does not count towards its 2 cones
    assert s["blue"]["precision"] == 1.0 and s["blue"]["recall"] == 0.5
    # yellow: the kept detection is a false positive, the one yellow box is not found
    assert s["yellow"]["precision"] == 0.0 and s["yellow"]["recall"] == 0.0
    # only the score 0.9 pair survives the >= t filter on pairs, giving one correct blue-blue match
    assert s["confusion"]["counts"] == [[1, 0], [0, 0]]
    assert s["colour_accuracy"] == 1.0


def test_breakdowns_by_height_uses_detections_for_precision_ground_truth_for_recall():
    m = _bins_example()
    b = breakdowns(m, 0.5)
    heights = b["by_height"]
    assert [h["height_px"] for h in heights] == [[0, 8], [8, 16], [16, 32], [32, 64], [64, None]]
    assert heights[0]["precision"] == 1.0 and heights[0]["recall"] == 1.0
    assert heights[1]["precision"] == 0.0 and heights[1]["recall"] == 0.0
    # the only detection in this bin (score 0.3) is below t, so precision is undefined, but its matching
    # ground truth box is still counted for recall (0 out of 1, not found)
    assert heights[2]["precision"] is None and heights[2]["recall"] == 0.0 and heights[2]["cones"] == 1
    assert heights[3]["detections"] == 0 and heights[3]["cones"] == 0
    assert heights[4]["detections"] == 0 and heights[4]["cones"] == 0


def test_breakdowns_by_group():
    m = _bins_example()
    b = breakdowns(m, 0.5)
    assert b["by_group"]["g1"]["precision"] == 0.5 and b["by_group"]["g1"]["recall"] == 0.5
    # g2's only kept-by-score detection is none (its one detection scores 0.3), and its one non-ignored box misses
    assert b["by_group"]["g2"]["precision"] is None and b["by_group"]["g2"]["recall"] == 0.0


def test_breakdowns_by_tag_counts_multi_tag_boxes_in_every_tag_and_drops_ignored():
    m = _bins_example()
    b = breakdowns(m, 0.5)
    # the last box carries both tags, so it is counted in both buckets, and neither found
    assert b["by_tag"]["truncated"] == {"recall": 0.0, "cones": 2}
    assert b["by_tag"]["knocked_over"] == {"recall": 0.0, "cones": 1}
    # only the first box has no tags and is not ignored; the ignored untagged box is excluded from "no_tag"
    assert b["by_tag"]["no_tag"] == {"recall": 1.0, "cones": 1}


def test_breakdowns_by_distance_ignores_nan_distances():
    m = _bins_example()
    b = breakdowns(m, 0.5)
    dist = b["by_distance"]
    assert [d["cones"] for d in dist] == [1, 1, 1, 0, 0, 0]
    assert dist[0]["recall"] == 1.0 and dist[1]["recall"] == 0.0 and dist[2]["recall"] == 0.0


# match_split paper mode


def test_paper_mode_drops_unknown_class_detections_entirely():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    labels = [ImageLabels("img1", "g", 100, 100, box, np.array([int(ConeColor.BLUE)]), [()], np.array([np.nan]))]
    detections = {"img1": (
        np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 0.0, 110.0, 10.0]]),
        np.array([int(ConeColor.BLUE), int(ConeColor.UNKNOWN)]),
        np.array([0.9, 0.8]),
    )}
    names = {int(ConeColor.BLUE): "blue"}
    normal = summary_at(match_split(labels, detections, imgsz=100), 0.5, names)
    # outside paper mode the unknown-class detection matches nothing and is a false positive
    assert normal["all"]["precision"] == 0.5 and normal["all"]["recall"] == 1.0
    paper = summary_at(match_split(labels, detections, imgsz=100, paper=True), 0.5, names)
    # in paper mode it is filtered out before matching, so it does not even count as a false positive
    assert paper["all"]["precision"] == 1.0 and paper["all"]["recall"] == 1.0


def test_paper_mode_merges_orange_small_detection_into_orange_big():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    labels = [ImageLabels("img1", "g", 100, 100, box, np.array([int(ConeColor.ORANGE_BIG)]), [()],
                          np.array([np.nan]))]
    detections = {"img1": (box, np.array([int(ConeColor.ORANGE_SMALL)]), np.array([0.9]))}
    names = {int(ConeColor.ORANGE_BIG): "orange"}
    normal = summary_at(match_split(labels, detections, imgsz=100), 0.5, names)
    # small and big orange are different classes outside paper mode: a miss and a false positive
    assert normal["all"]["precision"] == 0.0 and normal["all"]["recall"] == 0.0
    paper = summary_at(match_split(labels, detections, imgsz=100, paper=True), 0.5, names)
    # paper mode merges orange on the detection side too, not only on the ground truth side
    assert paper["all"]["precision"] == 1.0 and paper["all"]["recall"] == 1.0
