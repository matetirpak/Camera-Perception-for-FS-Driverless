import numpy as np

from driverless_perception.common.types import ConeColor
from driverless_perception.vision.detection.evaluate import (
    PAPER_CLASSES,
    ImageLabels,
    breakdowns,
    match_split,
    report_markdown,
    summary_at,
)
from driverless_perception.vision.detection.fsoco import CLASS_NAMES

BLUE, YELLOW, ORANGE_BIG, ORANGE_SMALL, UNKNOWN = (int(c) for c in ConeColor)


def example():
    # one 1280x640 image, so heights at a 1280 input are the heights in pixels
    boxes = np.array([[0, 0, 10, 20], [100, 0, 110, 40], [200, 0, 210, 80], [300, 0, 310, 20], [400, 0, 410, 20]])
    labels = [ImageLabels("ugr_amz_00001", "amz", 1280, 640, boxes.astype(float),
                          np.array([BLUE, YELLOW, ORANGE_SMALL, UNKNOWN, BLUE]),
                          [(), ("truncated",), (), (), ()], np.array([5.0, 12.0, 25.0, 45.0, 70.0]))]
    detections = {"ugr_amz_00001": (
        np.array([[0, 0, 10, 20], [100, 0, 110, 40], [200, 0, 210, 80], [300, 0, 310, 20], [600, 0, 610, 20]], float),
        np.array([BLUE, BLUE, ORANGE_BIG, YELLOW, YELLOW]),
        np.array([0.9, 0.8, 0.7, 0.6, 0.3]),
    )}
    return labels, detections


def test_summary_counts_colours_and_misses():
    labels, detections = example()
    m = match_split(labels, detections, imgsz=1280)
    s = summary_at(m, 0.5, CLASS_NAMES)
    # only the first blue box is right: yellow was called blue, small orange big orange, unknown yellow
    assert s["all"]["precision"] == 1 / 4 and s["all"]["recall"] == 1 / 5
    assert s["blue"]["recall"] == 1 / 2
    assert s["colour_accuracy"] == 1 / 4
    assert s["confusion"]["counts"][YELLOW][BLUE] == 1


def test_paper_mode_merges_orange_and_ignores_unknown():
    labels, detections = example()
    m = match_split(labels, detections, imgsz=1280, paper=True)
    s = summary_at(m, 0.5, PAPER_CLASSES)
    # blue and orange are right now, the yellow detection on the unknown cone doesn't count at all
    assert s["all"]["precision"] == 2 / 3 and s["all"]["recall"] == 2 / 4
    assert s["orange"]["recall"] == 1.0


def test_breakdowns_and_report():
    labels, detections = example()
    m = match_split(labels, detections, imgsz=1280)
    b = breakdowns(m, 0.5)
    assert [x["cones"] for x in b["by_height"]] == [0, 0, 3, 1, 1]
    assert b["by_tag"]["truncated"] == {"recall": 0.0, "cones": 1}
    assert b["by_group"]["amz"]["recall"] == 1 / 5
    # only the blue cone at 5 m was found
    assert [x["cones"] for x in b["by_distance"]] == [1, 1, 1, 0, 1, 1]
    assert b["by_distance"][0]["recall"] == 1.0 and b["by_distance"][1]["recall"] == 0.0
    results = {"model": "yolo26n_fsoco", "dataset": "fsds", "threshold": 0.5, "test": summary_at(m, 0.5, CLASS_NAMES),
               "paper": summary_at(match_split(labels, detections, 1280, paper=True), 0.5, PAPER_CLASSES), **b}
    text = report_markdown([results])
    assert "| yolo26n_fsoco | 0.50 | 0.25 | 0.20 |" in text
    assert "Dhall et al. 2019, YOLOv2" in text
    assert "## Recall by distance" in text and "## By map" in text
    assert "| model | 0-10 m | 10-20 m | 20-30 m | 30-40 m | 40-60 m | 60+ m |" in text


def test_fsoco_images_have_no_distances():
    labels, detections = example()
    labels[0].distance = np.full(5, np.nan)
    assert "by_distance" not in breakdowns(match_split(labels, detections, imgsz=1280), 0.5)
