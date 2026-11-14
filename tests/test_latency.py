import json

import numpy as np
import pytest

from driverless_perception.common.timing import StageTimer, no_timer, percentiles
from driverless_perception.evaluation.latency import budget, per_cone, report_markdown, stage_table
from driverless_perception.vision.detection.evaluate import find_threshold
from driverless_perception.vision.detection.train import free_path


def test_timer_sums_a_stage_per_frame_and_skips_warm_up():
    timer = StageTimer()
    timer.enabled = False
    with timer("a"):
        pass
    timer.frame()
    timer.enabled = True
    for _ in range(3):
        with timer("a"):
            pass
    timer.add("a", 5.0)
    timer.add("boxes", 3)
    frame = timer.frame()
    assert len(timer.samples["a"]) == 1 and frame["a"] >= 5.0 and frame["boxes"] == 3
    assert timer.frame() == {}
    with no_timer("anything"):
        pass


def test_percentiles():
    p = percentiles(np.arange(1, 101))
    assert p["n"] == 100 and p["p50"] == 50.5 and p["max"] == 100 and abs(p["p99"] - 99.01) < 1e-9
    assert percentiles([]) == {"n": 0}


def test_per_cone_fits_fixed_and_marginal_cost():
    fit = per_cone([0, 2, 4, 6], [10, 13, 16, 19])
    assert fit["fixed_ms"] == pytest.approx(10) and fit["per_cone_ms"] == pytest.approx(1.5)
    assert per_cone([3, 3], [8, 10]) == {"fixed_ms": 9.0, "per_cone_ms": 0.0}


def test_budget_of_a_10_hz_loop():
    b = budget(10.0, 75.0, 91.0)
    assert b["period_ms"] == 100 and b["left_after_processing_ms"] == 25 and b["processing_share"] == 0.75
    assert b["left_after_race_ms"] == pytest.approx(9.0)
    assert "race_ms" not in budget(20.0, 75.0)
    assert budget(20.0, 75.0)["left_after_processing_ms"] == -25


def test_stage_table_keeps_pipeline_order_and_drops_missing():
    rows = stage_table({"pnp": [1.0], "detector": [2.0, 4.0]}, ("detector", "crops", "pnp"))
    assert [r["stage"] for r in rows] == ["detector", "pnp"] and rows[0]["mean"] == 3.0


def test_report_renders_every_section():
    stage = {"stage": "detector", **percentiles([1.0, 2.0])}
    pipe = {"stages": [stage], "detector_parts": [], "total": {**stage, "stage": "total"},
            "per_cone": {"fixed_ms": 1.0, "per_cone_ms": 0.5}, "cones_mean": 4.0, "cones_max": 9}
    sweep = {"imgsz": 640, "half": True, "boxes": 3.0,
             **{k: percentiles([1.0]) for k in ("preprocess", "inference", "postprocess", "total")}}
    result = {"frames": 2, "capture": "c", "device": "gpu", "detector": "d", "imgsz": 1280, "keypoints": "k",
              "warmup": 1, "mono": pipe, "stereo": pipe, "both": {**stage, "stage": "both"},
              "budget": [budget(10, 50, 90)], "race": {"stereo": {"rate_hz": 9.3, "median": 52, "p95": 90}},
              "sweep": [sweep]}
    text = report_markdown(result)
    for part in ("## Mono", "## Stereo", "| 10.0 Hz | 100 ms | 50 ms | 50% | 50 ms | 90 ms | 90% | 10 ms |",
                 "stereo 9.3 Hz", "| 640 px | fp16 |"):
        assert part in text


def test_free_path_never_returns_an_existing_file(tmp_path):
    model = tmp_path / "fsds.pt"
    assert free_path(model) == model
    model.touch()
    assert free_path(model) == tmp_path / "fsds-2.pt"
    (tmp_path / "fsds-2.pt").touch()
    assert free_path(model) == tmp_path / "fsds-3.pt"
    run = tmp_path / "run"
    run.mkdir()
    assert free_path(run) == tmp_path / "run-2"


def test_threshold_comes_from_the_first_report_folder_that_has_it(tmp_path):
    for folder, value in (("candidate", 0.4), ("published", 0.44)):
        d = tmp_path / folder / "detection" / "fsds"
        d.mkdir(parents=True)
        (d / "det.json").write_text(json.dumps({"threshold": value}))
    assert find_threshold("models/det.pt", tmp_path / "candidate", tmp_path / "published") == 0.4
    assert find_threshold("det", tmp_path / "nothing", tmp_path / "published") == 0.44
    with pytest.raises(FileNotFoundError):
        find_threshold("other", tmp_path / "published")


def test_precision_label_names_compilation_and_tensorrt():
    from driverless_perception.evaluation.latency import precision_label

    assert precision_label({"half": False}) == "fp32"
    assert precision_label({"half": True, "compile": "reduce-overhead"}) == "fp16, compiled"
    assert precision_label({"half": True, "compile": None, "engine": "int8"}) == "TensorRT, int8"
