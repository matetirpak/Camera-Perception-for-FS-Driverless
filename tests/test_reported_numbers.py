"""Pins the numbers README.md and docs/*.md quote to the report JSONs they were read from.

If a re-run of a script changes one of these numbers, this test fails instead of the docs silently going stale.
Each assertion names the doc paragraph or table it protects. Tolerances match the two (or one) decimal places
the docs round to, so a value still fails once it would print differently.
"""
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(relpath):
    path = ROOT / relpath
    if not path.exists():
        pytest.skip(f"{relpath} not found")
    return json.loads(path.read_text())


def test_fsoco_detector_matches_dhall_comparison_table():
    # README.md "Summary" table and docs/evaluation.md "Comparison with the paper": YOLO26s trained on FSOCO
    # alone, precision 0.92, recall 0.85. Also reports/detection/fsoco.md "Compared with Dhall et al. (2019)".
    d = _load("reports/detection/fsoco/yolo26s_fsoco.json")
    assert abs(d["paper"]["all"]["precision"] - 0.92) < 0.005
    assert abs(d["paper"]["all"]["recall"] - 0.85) < 0.005
    # README.md below the summary table: the fine-tuned model the pipelines use reaches 0.84 / 0.68 there
    finetuned = _load("reports/detection/fsoco/yolo26s_fsoco_fsds.json")
    assert abs(finetuned["paper"]["all"]["precision"] - 0.84) < 0.005
    assert abs(finetuned["paper"]["all"]["recall"] - 0.68) < 0.005


def test_fsoco_finetuned_detector_forgets_real_cones():
    # README.md "Limitations" and docs/walkthrough.md: FSOCO recall 0.80 down to 0.63 after fine-tuning on
    # simulator images.
    base = _load("reports/detection/fsoco/yolo26s_fsoco.json")
    finetuned = _load("reports/detection/fsoco/yolo26s_fsoco_fsds.json")
    assert abs(base["test"]["all"]["recall"] - 0.80) < 0.005
    assert abs(finetuned["test"]["all"]["recall"] - 0.63) < 0.005


def test_fsds_detector_precision_recall():
    # docs/walkthrough.md step 3: the fine-tuned yolo26s_fsoco_fsds reaches 0.94 / 0.93 on the simulator test map.
    d = _load("reports/detection/fsds/yolo26s_fsoco_fsds.json")
    assert abs(d["test"]["all"]["precision"] - 0.94) < 0.005
    assert abs(d["test"]["all"]["recall"] - 0.93) < 0.005


def _bin(bins, key, low):
    return next(b for b in bins if b[key][0] == low)


def test_detector_sizes():
    # docs/detection.md "Model size" and README.md "Limitations": the complete n run against the stopped one,
    # FSOCO recall and small-box recall of n, s and m, and the fine-tuned models on the simulator test map.
    stopped = _load("reports/detection/fsoco/yolo26n_fsoco.json")
    complete = _load("reports/detection/fsoco/yolo26n_fsoco-2.json")
    assert abs(stopped["test"]["all"]["precision"] - 0.92) < 0.005
    assert abs(complete["test"]["all"]["precision"] - 0.89) < 0.005
    assert abs(complete["test"]["all"]["recall"] - 0.77) < 0.005
    for size, recall, small in (("n", 0.76, 0.57), ("s", 0.80, 0.66), ("m", 0.81, 0.70)):
        d = _load(f"reports/detection/fsoco/yolo26{size}_fsoco.json")
        assert abs(d["test"]["all"]["recall"] - recall) < 0.005, size
        assert abs(_bin(d["by_height"], "height_px", 8)["recall"] - small) < 0.005, size
    for size, recall, far in (("n", 0.89, 0.81), ("s", 0.93, 0.88), ("m", 0.94, 0.91)):
        d = _load(f"reports/detection/fsds/yolo26{size}_fsoco_fsds.json")
        assert abs(d["test"]["all"]["recall"] - recall) < 0.005, size
        assert abs(_bin(d["by_distance"], "distance_m", 40)["recall"] - far) < 0.005, size
        # "up to 20 m every model finds all of them"
        assert _bin(d["by_distance"], "distance_m", 0)["recall"] >= 0.995, size
        assert _bin(d["by_distance"], "distance_m", 10)["recall"] >= 0.995, size
    # "in half precision m is about 1 ms slower than n"
    n = _load("reports/detection/fsds/yolo26n_fsoco_fsds.json")["latency"]["total"]
    m = _load("reports/detection/fsds/yolo26m_fsoco_fsds.json")["latency"]["total"]
    assert 0.5 <= m - n <= 1.5


def test_keypoint_network_error():
    # README.md, docs/evaluation.md, docs/mono.md and reports/mono/fsds.md: 1.72 px^2 on 3881 test crops.
    d = _load("reports/mono/fsds.json")
    assert d["keypoints_detail"]["crops"] == 3881
    assert abs(d["keypoints_detail"]["mse_px2"] - 1.72) < 0.01


def test_mono_position_error_by_distance():
    # README.md, docs/evaluation.md and docs/mono.md: whole pipeline off by 0.08 m at 5-10 m, 0.30 m at 15-20 m.
    # reports/mono/fsds.md "Detector boxes (whole pipeline)" has all four bins, checked here.
    d = _load("reports/mono/fsds.json")
    assert d["pnp"] == "upright" and d["edge_px"] == 2.0
    bins = {tuple(b["distance_m"]): b for b in d["bins"]["detector"]}
    assert abs(bins[(0, 5)]["median_m"] - 0.06) < 0.005
    assert abs(bins[(5, 10)]["median_m"] - 0.08) < 0.005
    assert abs(bins[(10, 15)]["median_m"] - 0.16) < 0.005
    assert abs(bins[(15, 20)]["median_m"] - 0.30) < 0.005


def test_pnp_comparison_table():
    # docs/mono.md "Position estimation" and docs/design-decisions.md: SQPnP against the
    # upright cone on the same keypoints, and the ground plane with the true attitude and with a level car.
    comp = _load("reports/mono/fsds.json")["pnp_comparison"]
    far = {name: next(b for b in rows if tuple(b["distance_m"]) == (15, 20)) for name, rows in comp.items()}
    assert abs(far["sqpnp true"]["median_m"] - 0.40) < 0.005 and abs(far["sqpnp true"]["range_bias_m"] + 0.30) < 0.005
    assert abs(far["upright true"]["median_m"] - 0.30) < 0.005
    assert abs(far["upright true"]["range_bias_m"] + 0.01) < 0.005
    assert abs(far["ground true"]["median_m"] - 0.05) < 0.005
    assert abs(far["ground level"]["median_m"] - 0.62) < 0.005
    mid = {name: next(b for b in rows if tuple(b["distance_m"]) == (10, 15)) for name, rows in comp.items()}
    assert abs(mid["sqpnp true"]["median_m"] - 0.20) < 0.005 and abs(mid["sqpnp true"]["range_bias_m"] + 0.16) < 0.005
    assert abs(mid["upright true"]["median_m"] - 0.16) < 0.005


def test_stereo_range_spread_beats_mono_at_5_to_10m():
    # README.md and docs/evaluation.md: stereo range spread 0.04 m against mono's 0.13 m at 5-10 m.
    d = _load("reports/stereo/fsds.json")
    b = next(b for b in d["bins"]["detector"] if tuple(b["distance_m"]) == (5, 10))
    assert abs(b["mono"]["range_spread_m"] - 0.13) < 0.005
    assert abs(b["stereo"]["range_spread_m"] - 0.04) < 0.005


def test_stereo_position_error_by_distance():
    # docs/stereo.md "Evaluation on images" table (detector boxes / whole pipeline), all four rows.
    # README.md "Results" table takes its rows from the same bins.
    d = _load("reports/stereo/fsds.json")
    bins = {tuple(b["distance_m"]): b for b in d["bins"]["detector"]}

    found = {k: b["with_stereo"] / b["cones"] for k, b in bins.items()}
    assert abs(found[(0, 5)] - 0.83) < 0.01
    assert abs(found[(5, 10)] - 0.97) < 0.01
    assert abs(found[(10, 15)] - 0.97) < 0.01
    assert abs(found[(15, 20)] - 0.90) < 0.01

    assert abs(bins[(0, 5)]["mono"]["median_m"] - 0.06) < 0.005
    assert abs(bins[(0, 5)]["stereo"]["median_m"] - 0.03) < 0.005
    assert abs(bins[(5, 10)]["mono"]["median_m"] - 0.08) < 0.005
    assert abs(bins[(5, 10)]["stereo"]["median_m"] - 0.04) < 0.005
    assert abs(bins[(10, 15)]["mono"]["median_m"] - 0.16) < 0.005
    assert abs(bins[(10, 15)]["stereo"]["median_m"] - 0.07) < 0.005
    assert abs(bins[(15, 20)]["mono"]["median_m"] - 0.31) < 0.005
    assert abs(bins[(15, 20)]["stereo"]["median_m"] - 0.11) < 0.005

    assert abs(bins[(0, 5)]["mono"]["range_spread_m"] - 0.09) < 0.005
    assert abs(bins[(0, 5)]["stereo"]["range_spread_m"] - 0.03) < 0.005
    assert abs(bins[(10, 15)]["mono"]["range_spread_m"] - 0.27) < 0.005
    assert abs(bins[(10, 15)]["stereo"]["range_spread_m"] - 0.11) < 0.005
    assert abs(bins[(15, 20)]["mono"]["range_spread_m"] - 0.40) < 0.005
    assert abs(bins[(15, 20)]["stereo"]["range_spread_m"] - 0.20) < 0.005


def test_stereo_stage_by_stage_table():
    # docs/stereo.md stage by stage table, ground truth boxes: box IoU, features, matches, epipolar residual.
    d = _load("reports/stereo/fsds.json")
    stages = {tuple(s["distance_m"]): s for s in d["stages"]}

    s = stages[(0, 5)]
    assert abs(s["iou_median"] - 0.40) < 0.005 and s["features_median"] == 354 and s["matches_median"] == 67
    assert s["on_cone_median"] == 49 and abs(s["epipolar_px_median"] - 0.54) < 0.005
    assert abs(s["point_error_median_m"] - 0.07) < 0.005

    s = stages[(15, 20)]
    assert abs(s["iou_median"] - 0.30) < 0.005 and s["features_median"] == 37 and s["matches_median"] == 15
    assert s["on_cone_median"] == 11 and abs(s["epipolar_px_median"] - 0.21) < 0.005
    assert abs(s["point_error_median_m"] - 0.22) < 0.005
    assert abs(s["median_minus_mean_m"] - 0.07) < 0.005


def test_map_of_the_10lap_race():
    # README.md "Results", docs/stereo.md "Cone map" and docs/evaluation.md:
    # stereo 0.03 m (176 of 180), mono 0.14 m (163 of 180), every colour right.
    d = _load("reports/map/live_competitionmap3_10laps.json")
    assert d["mono"]["mapped"] == 163 and d["mono"]["expected"] == 180
    assert abs(d["mono"]["rmse_m"] - 0.14) < 0.005
    assert d["stereo"]["mapped"] == 176 and d["stereo"]["expected"] == 180
    assert abs(d["stereo"]["rmse_m"] - 0.03) < 0.005
    assert d["mono"]["colour_accuracy"] == 1.0 and d["stereo"]["colour_accuracy"] == 1.0


def test_depth_test_mono_bias():
    # docs/mono.md "Standing depth test": the mono camera is 0.05 m long at 5.5 m and 0.11 m short at 18 m.
    d = _load("reports/depth/fsds.json")
    cones = {c["cone"]: c for c in d["cones"]}

    near = cones[0]["series"]["mono camera PnP"]
    near_bias = sorted(near)[len(near) // 2] - cones[0]["distance_m"]
    assert abs(near_bias - 0.05) < 0.005

    far = cones[7]["series"]["mono camera PnP"]
    far_bias = sorted(far)[len(far) // 2] - cones[7]["distance_m"]
    assert abs(far_bias - (-0.11)) < 0.005


def test_depth_test_stereo_bias_up_to_13m():
    # docs/mono.md: stereo stays within 0.10 m of the true distance up to 13 m (worst cone).
    d = _load("reports/depth/fsds.json")
    worst = 0.0
    for c in d["cones"]:
        if c["distance_m"] > 13.0:
            continue
        series = c["series"].get("stereo")
        if not series:
            continue
        bias = abs(statistics.median(series) - c["distance_m"])
        worst = max(worst, bias)
    assert abs(worst - 0.10) < 0.005


def test_10lap_race_mono():
    # README.md race table (mono row) and docs/mono.md: 13.5 Hz, 35 / 41 ms, recall 0.93, precision 0.98, the same in
    # every lap, 0.77 below 5 m (cones cut off by the image border), 0.90 / 0.93 at 15-20 m, RMSE 0.10 to 0.41 m.
    d = _load("runs/live_competitionmap3_10laps/mono_metrics.json")
    assert abs(d["rate_hz"] - 13.5) < 0.05
    assert round(d["latency_ms"]["median"]) == 35 and round(d["latency_ms"]["p95"]) == 41
    assert abs(d["all"]["recall"] - 0.93) < 0.005 and abs(d["all"]["precision"] - 0.98) < 0.005
    assert all(0.915 <= lap["recall"] < 0.935 for lap in d["laps"]) and len(d["laps"]) == 10
    bins = {tuple(b["bin_m"]): b for b in d["bins"]}
    assert abs(bins[(0, 5)]["recall"] - 0.77) < 0.005
    assert abs(bins[(0, 5)]["pos_rmse_m"] - 0.10) < 0.005
    assert abs(bins[(5, 10)]["pos_rmse_m"] - 0.14) < 0.005
    far = bins[(15, 20.0)]
    assert abs(far["pos_rmse_m"] - 0.41) < 0.005
    assert abs(far["recall"] - 0.90) < 0.005 and abs(far["precision"] - 0.93) < 0.005
    # README.md and docs/mono.md: with a 0.5 m gate 0.86 / 0.91 overall and 0.68 / 0.70 at 15-20 m, RMSE 0.26 m there
    strict = d["strict"]
    assert strict["gate_m"] == 0.5
    assert abs(strict["all"]["recall"] - 0.86) < 0.005 and abs(strict["all"]["precision"] - 0.91) < 0.005
    far = {tuple(b["bin_m"]): b for b in strict["bins"]}[(15, 20.0)]
    assert abs(far["recall"] - 0.68) < 0.005 and abs(far["precision"] - 0.70) < 0.005
    assert abs(far["pos_rmse_m"] - 0.26) < 0.005


def test_10lap_race_stereo():
    # README.md race table (stereo row) and docs/stereo.md: 13.5 Hz, 53 / 62 ms, recall 0.85, precision 1.00 up to
    # 20 m; recall 0.98 to 0.99 up to 15 m and 0.38 at 15-20 m, RMSE 0.04, 0.05, 0.12 and 0.20 m; with a 0.5 m gate
    # 0.84 / 0.99.
    d = _load("runs/live_competitionmap3_10laps/stereo_metrics.json")
    assert d["max_range_m"] == 20.0
    assert abs(d["rate_hz"] - 13.5) < 0.05
    assert round(d["latency_ms"]["median"]) == 53 and round(d["latency_ms"]["p95"]) == 62
    assert abs(d["all"]["recall"] - 0.85) < 0.005 and abs(d["all"]["precision"] - 1.00) < 0.005
    assert abs(d["strict"]["all"]["recall"] - 0.84) < 0.005 and abs(d["strict"]["all"]["precision"] - 0.99) < 0.005
    bins = {tuple(b["bin_m"]): b for b in d["bins"]}
    assert all(0.975 <= bins[k]["recall"] < 0.995 for k in ((0, 5), (5, 10), (10, 15)))
    assert abs(bins[(15, 20.0)]["recall"] - 0.38) < 0.005
    for k, rmse in (((0, 5), 0.04), ((5, 10), 0.05), ((10, 15), 0.12), ((15, 20.0), 0.20)):
        assert abs(bins[k]["pos_rmse_m"] - rmse) < 0.006


def test_10lap_race_timing():
    # README.md "Latency" and docs/stereo.md: the bridge takes 13 ms (median) and 18 ms (p95) from the stamp to
    # publishing, stereo waits 9 ms (p95) for the partner image, the detector takes 7.3 (stereo) and 8.4 ms (mono)
    # next to the other node, ORB matching 10.5 ms.
    d = _load("runs/live_competitionmap3_10laps/stereo_metrics.json")["timing"]
    assert round(d["bridge"]["p50"]) == 13 and round(d["bridge"]["p95"]) == 18
    assert round(d["wait for pair"]["p95"]) == 9
    assert abs(d["detector"]["p50"] - 7.3) < 0.05 and abs(d["orb match"]["p50"] - 10.5) < 0.05
    mono = _load("runs/live_competitionmap3_10laps/mono_metrics.json")["timing"]
    assert abs(mono["detector"]["p50"] - 8.4) < 0.05


def test_latency_benchmark():
    # README.md "Latency" table: mono 9.6 / 13.5 ms, stereo 23.2 / 31.8 ms, both 33.3 / 42.3 ms (p50 / p99), the
    # detector 4.3 ms and ORB matching 9.6 ms as the biggest stages, with the detector in fp16 and compiled.
    d = _load("reports/latency/fsds.json")
    assert d["half"] and d["compile"] == "reduce-overhead"
    assert abs(d["mono"]["total"]["p50"] - 9.6) < 0.05 and abs(d["mono"]["total"]["p99"] - 13.5) < 0.05
    assert abs(d["stereo"]["total"]["p50"] - 23.2) < 0.05 and abs(d["stereo"]["total"]["p99"] - 31.8) < 0.05
    assert abs(d["both"]["p50"] - 33.3) < 0.05 and abs(d["both"]["p99"] - 42.3) < 0.05
    stages = {k: {s["stage"]: s for s in d[k]["stages"]} for k in ("mono", "stereo")}
    assert abs(stages["mono"]["detector"]["p50"] - 4.3) < 0.05
    assert abs(stages["stereo"]["orb match"]["p50"] - 9.6) < 0.05


def test_detector_compile_speedup():
    # README.md "Latency" and docs/detection.md "Inference speed": the detector alone at 1280 px takes 7.6 ms in
    # fp32 and 3.8 ms in fp16 through torch.compile with CUDA graphs, 6.5 ms compiled in fp32, 6.9 ms in fp16 alone,
    # and as a TensorRT engine 3.1 ms in fp16 and 2.6 ms in INT8 with 8% fewer boxes.
    rows = _load("reports/latency/fsds.json")["sweep"]
    sweep = {(r["imgsz"], r["half"], r.get("compile")): r for r in rows if not r.get("engine")}
    engine = {r["engine"]: r for r in rows if r.get("engine")}
    assert abs(sweep[(1280, False, None)]["total"]["p50"] - 7.6) < 0.05
    assert abs(sweep[(1280, True, None)]["total"]["p50"] - 6.9) < 0.05
    assert abs(sweep[(1280, False, "reduce-overhead")]["total"]["p50"] - 6.5) < 0.05
    assert abs(sweep[(1280, True, "reduce-overhead")]["total"]["p50"] - 3.8) < 0.05
    assert abs(engine["fp16"]["total"]["p50"] - 3.1) < 0.05 and abs(engine["int8"]["total"]["p50"] - 2.6) < 0.05
    fewer = 1 - engine["int8"]["boxes"] / engine["fp16"]["boxes"]
    assert 0.07 <= fewer < 0.085


def test_tensorrt_in_the_pipelines():
    # README.md "Latency" and docs/detection.md: with the fp16 engine both pipelines take 31.7 / 40.0 ms (p50 / p99),
    # and mono finds 3,792 cones with 0.29 m at 15-20 m, against 3,878 and 0.30 m for the compiled model.
    latency = _load("reports/tensorrt/latency/fsds.json")
    assert latency["engine"] and latency["detector"].endswith(".engine")
    assert abs(latency["both"]["p50"] - 31.7) < 0.05 and abs(latency["both"]["p99"] - 40.0) < 0.05
    trt = _load("reports/tensorrt/mono/fsds.json")
    compiled = _load("reports/mono/fsds.json")
    assert trt["counts"]["detector_cones"] == 3792 and compiled["counts"]["detector_cones"] == 3878
    far = next(b for b in trt["bins"]["detector"] if tuple(b["distance_m"]) == (15, 20))
    assert abs(far["median_m"] - 0.29) < 0.005
