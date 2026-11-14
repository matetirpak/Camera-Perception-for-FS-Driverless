import numpy as np

from driverless_perception.evaluation.matching import match
from driverless_perception.evaluation.metrics import BinnedCounts


def test_match_empty_inputs():
    assert match([], []) == []
    assert match(np.array([]).reshape(0, 2), np.array([[1.0, 2.0]])) == []
    assert match(np.array([[1.0, 2.0]]), np.array([]).reshape(0, 2)) == []


def test_match_single_pair_within_gate():
    det = np.array([[1.0, 0.0]])
    gt = np.array([[0.0, 0.0]])
    result = match(det, gt, gate_m=2.0)
    assert len(result) == 1
    assert result[0][:2] == (0, 0)
    assert np.isclose(result[0][2], 1.0)


def test_match_single_pair_outside_gate():
    det = np.array([[1.0, 0.0]])
    gt = np.array([[0.0, 0.0]])
    result = match(det, gt, gate_m=0.5)
    assert result == []


def test_match_multiple_pairs_greedy_vs_optimal():
    det = np.array([[0.0, 0.0], [3.0, 0.0]])
    gt = np.array([[1.0, 0.0], [2.0, 0.0]])
    result = match(det, gt, gate_m=2.0)
    assert len(result) == 2
    det_indices = {r[0] for r in result}
    gt_indices = {r[1] for r in result}
    assert det_indices == {0, 1}
    assert gt_indices == {0, 1}
    total_dist = sum(r[2] for r in result)
    assert np.isclose(total_dist, 2.0)


def test_match_gate_excludes_far_pairs():
    det = np.array([[0.0, 0.0], [1.5, 0.0]])
    gt = np.array([[1.0, 0.0], [3.0, 0.0]])
    result = match(det, gt, gate_m=1.0)
    assert len(result) == 1
    assert result[0][:2] == (1, 0)


def test_match_uses_global_assignment():
    det = np.array([[0.0, 0.0], [1.2, 0.0], [3.0, 0.0]])
    gt = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    result = match(det, gt, gate_m=2.0)
    det_indices = {r[0] for r in result}
    gt_indices = {r[1] for r in result}
    assert len(result) == 3
    assert det_indices == {0, 1, 2}
    assert gt_indices == {0, 1, 2}
    costs = [r[2] for r in result]
    assert np.allclose(sorted(costs), [0.8, 1.0, 1.0])


def test_binned_counts_initialization():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0, 15.0))
    assert len(bc.tp) == 3
    assert len(bc.fp) == 3
    assert len(bc.fn) == 3
    assert len(bc.errors) == 3
    assert np.array_equal(bc.tp, [0, 0, 0])
    assert np.array_equal(bc.fp, [0, 0, 0])
    assert np.array_equal(bc.fn, [0, 0, 0])


def test_binned_counts_matched_detection_in_gate():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    det_xy = np.array([[2.0, 0.0]])
    gt_xy = np.array([[3.0, 0.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=2.0)
    assert bc.tp[0] == 1
    assert bc.fp[0] == 0
    assert bc.fn[0] == 0
    error = np.hypot(3.0 - 2.0, 0.0 - 0.0)
    assert np.isclose(bc.errors[0][0], error)


def test_binned_counts_matched_detection_bins_by_gt_range():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0, 15.0))
    det_xy = np.array([[8.0, 0.0]])
    gt_xy = np.array([[9.0, 0.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=3.0)
    assert bc.tp[0] == 0
    assert bc.tp[1] == 1
    assert bc.tp[2] == 0
    expected_error = 1.0
    assert np.isclose(bc.errors[1][0], expected_error)


def test_binned_counts_unmatched_detection_false_positive():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    det_xy = np.array([[2.0, 0.0]])
    gt_xy = np.array([[10.0, 0.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=1.0)
    assert bc.fp[0] == 1
    assert bc.tp[0] == 0
    assert len(bc.errors[0]) == 0


def test_binned_counts_unmatched_gt_false_negative():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0, 15.0))
    det_xy = np.array([[2.0, 0.0]])
    gt_xy = np.array([[9.0, 0.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=1.0)
    assert bc.fn[1] == 1
    assert bc.tp[1] == 0


def test_binned_counts_outside_outermost_bin():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    det_xy = np.array([[15.0, 0.0]])
    gt_xy = np.array([[16.0, 0.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=2.0)
    assert np.sum(bc.tp) == 0
    assert np.sum(bc.fp) == 0
    assert np.sum(bc.fn) == 0


def test_binned_counts_ignore_xy_near_unmatched_detection():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0, 15.0))
    det_xy = np.array([[2.0, 0.0]])
    gt_xy = np.array([[9.0, 0.0]])
    ignore_xy = np.array([[2.2, 0.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=0.5, ignore_xy=ignore_xy)
    assert bc.fp[0] == 0
    assert bc.tp[0] == 0
    assert bc.fn[1] == 1


def test_binned_counts_ignore_xy_far_from_detection():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    det_xy = np.array([[2.0, 0.0]])
    gt_xy = np.array([[10.0, 0.0]])
    ignore_xy = np.array([[5.0, 0.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=0.5, ignore_xy=ignore_xy)
    assert bc.fp[0] == 1


def test_binned_counts_multiple_frames_accumulate():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    bc.add_frame(np.array([[1.0, 0.0]]), np.array([[2.0, 0.0]]), gate_m=2.0)
    bc.add_frame(np.array([[3.0, 0.0]]), np.array([[4.0, 0.0]]), gate_m=2.0)
    assert bc.tp[0] == 2
    assert len(bc.errors[0]) == 2


def test_binned_counts_summary_empty_bin():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    summary = bc.summary()
    assert len(summary) == 2
    assert summary[0]["recall"] is None
    assert summary[0]["precision"] is None
    assert summary[0]["pos_rmse_m"] is None
    assert summary[0]["n_gt"] == 0


def test_binned_counts_summary_with_tp_only():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    bc.add_frame(np.array([[1.0, 0.0]]), np.array([[2.0, 0.0]]), gate_m=2.0)
    summary = bc.summary()
    assert summary[0]["n_gt"] == 1
    assert np.isclose(summary[0]["recall"], 1.0)
    assert np.isclose(summary[0]["precision"], 1.0)


def test_binned_counts_summary_recall_calculation():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    bc.add_frame(np.array([[1.0, 0.0], [1.5, 0.0]]), np.array([[2.0, 0.0], [3.0, 0.0]]), gate_m=2.0)
    summary = bc.summary()
    assert summary[0]["n_gt"] == 2
    assert np.isclose(summary[0]["recall"], 1.0)


def test_binned_counts_summary_precision_calculation():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    bc.add_frame(np.array([[1.0, 0.0], [1.5, 0.0], [2.5, 0.0]]), np.array([[2.0, 0.0]]), gate_m=2.0)
    summary = bc.summary()
    tp = summary[0]["n_gt"]
    fp = len([1.5, 2.5])
    expected_precision = tp / (tp + fp)
    assert np.isclose(summary[0]["precision"], expected_precision)


def test_binned_counts_summary_pos_rmse_single_error():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    bc.add_frame(np.array([[1.5, 0.0]]), np.array([[2.0, 0.0]]), gate_m=2.0)
    summary = bc.summary()
    expected_rmse = 0.5
    assert np.isclose(summary[0]["pos_rmse_m"], expected_rmse)


def test_binned_counts_summary_pos_rmse_multiple_errors():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    bc.add_frame(np.array([[1.0, 0.0], [2.6, 0.0]]), np.array([[2.0, 0.0], [2.0, 0.0]]), gate_m=2.0)
    summary = bc.summary()
    errors = [1.0, 0.6]
    expected_rmse = np.sqrt(np.mean(np.square(errors)))
    assert np.isclose(summary[0]["pos_rmse_m"], expected_rmse)


def test_binned_counts_summary_multiple_bins():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0, 15.0))
    bc.add_frame(np.array([[1.0, 0.0], [7.0, 0.0]]), np.array([[2.0, 0.0], [8.0, 0.0]]), gate_m=2.0)
    summary = bc.summary()
    assert len(summary) == 3
    assert summary[0]["n_gt"] == 1
    assert summary[1]["n_gt"] == 1
    assert summary[2]["n_gt"] == 0
    assert summary[2]["recall"] is None


def test_binned_counts_2d_cone_positions():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0, 15.0))
    det_xy = np.array([[3.0, 4.0]])
    gt_xy = np.array([[6.0, 8.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=10.0)
    assert bc.tp[2] == 1
    gt_distance = np.hypot(6.0, 8.0)
    assert np.isclose(gt_distance, 10.0)
    expected_error = np.hypot(3.0 - 6.0, 4.0 - 8.0)
    assert np.isclose(bc.errors[2][0], expected_error)


def test_binned_counts_list_input_converted_to_array():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    det_list = [[1.0, 0.0]]
    gt_list = [[2.0, 0.0]]
    bc.add_frame(det_list, gt_list, gate_m=2.0)
    assert bc.tp[0] == 1


def test_binned_counts_complex_matching_scenario():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0, 20.0))
    det_xy = np.array([[1.0, 0.0], [3.0, 0.0], [7.0, 0.0]])
    gt_xy = np.array([[2.0, 0.0], [8.0, 0.0], [15.0, 0.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=2.0)
    assert bc.tp[0] == 1
    assert bc.fp[0] == 1
    assert bc.tp[1] == 1
    assert bc.fn[2] == 1
    gt_distances = np.hypot(gt_xy[:, 0], gt_xy[:, 1])
    assert 10.0 < gt_distances[2] < 20.0


def test_match_with_2d_lists():
    result = match([[1.0, 0.0]], [[0.0, 0.0]], gate_m=2.0)
    assert len(result) == 1
    assert np.isclose(result[0][2], 1.0)


def test_match_no_valid_pairs_within_gate():
    det = np.array([[0.0, 0.0], [5.0, 0.0]])
    gt = np.array([[1.0, 0.0], [6.0, 0.0]])
    result = match(det, gt, gate_m=0.5)
    assert result == []


def test_binned_counts_summary_none_for_empty_bins():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0, 15.0, 20.0))
    bc.add_frame(np.array([[1.0, 0.0]]), np.array([[2.0, 0.0]]), gate_m=2.0)
    summary = bc.summary()
    assert summary[0]["recall"] == 1.0
    for i in range(1, 4):
        assert summary[i]["recall"] is None
        assert summary[i]["precision"] is None
        assert summary[i]["pos_rmse_m"] is None
        assert summary[i]["n_gt"] == 0


def test_match_asymmetric_numbers():
    det = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    gt = np.array([[0.5, 0.0], [1.5, 0.0]])
    result = match(det, gt, gate_m=1.0)
    assert len(result) == 2
    det_indices = {r[0] for r in result}
    gt_indices = {r[1] for r in result}
    assert len(det_indices) == 2
    assert len(gt_indices) == 2


def test_binned_counts_ignore_xy_empty():
    bc = BinnedCounts(edges=(0.0, 5.0, 10.0))
    det_xy = np.array([[2.0, 0.0]])
    gt_xy = np.array([[10.0, 0.0]])
    bc.add_frame(det_xy, gt_xy, gate_m=0.5, ignore_xy=np.array([]).reshape(0, 2))
    assert bc.fp[0] == 1
