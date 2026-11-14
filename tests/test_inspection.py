import numpy as np

from driverless_perception.sim.autolabel import LabelPolicy
from driverless_perception.visualization.inspection import (
    annotate,
    band,
    keypoint_error,
    keypoint_panel,
    label_text,
    overlap,
    pick,
    put_text,
    read_boxes,
    reject_reason,
    spread,
    write_index,
    yolo_rows,
)


def row(**kw):
    base = {"class": "1", "x1": "10", "y1": "10", "x2": "30", "y2": "40", "distance": "9.55", "visible": "0.96",
            "contrast": "85.2", "truncated": "0", "labelled": "1"}
    return base | {k: str(v) for k, v in kw.items()}


def test_reject_reason_names_the_threshold_that_failed():
    policy = LabelPolicy()
    assert reject_reason(row(y2=12), policy) == "h 2 px"
    assert reject_reason(row(visible=0.21), policy) == "vis 0.21"
    assert reject_reason(row(contrast=8), policy) == "con 8"
    assert reject_reason(row(), policy) == "dropped"


def test_label_text_uses_what_the_dataset_has():
    assert label_text(row()) == ["yellow 9.6 m", "vis 0.96 con 85"]
    fsoco = {"class": "0", "x1": "0", "y1": "0", "x2": "5", "y2": "5", "tags": "truncated knocked_over"}
    assert label_text(fsoco) == ["blue", "truncated knocked_over"]
    assert label_text(row(truncated=1))[1].endswith("truncated")


def test_labels_are_coloured_and_rejected_boxes_grey():
    image = np.zeros((80, 80, 3), np.uint8)
    annotate(image, [row(), row(x1=50, x2=70, labelled=0)], LabelPolicy(), min_text_height=100)
    label_px, rejected_px = image[10, 20], image[10, 60]
    assert label_px[1] > label_px[0] and label_px[2] > label_px[0]
    assert rejected_px[0] == rejected_px[1] == rejected_px[2] > 0
    assert not image[25, 20].any()


def test_text_that_would_cover_other_text_is_left_out():
    image = np.zeros((100, 200, 3), np.uint8)
    taken = []
    assert put_text(image, ["yellow 9.6 m"], 10, 50, (255, 255, 255), taken=taken)
    assert not put_text(image, ["blue 9.7 m"], 20, 52, (255, 255, 255), taken=taken)
    assert put_text(image, ["blue 20 m"], 10, 95, (255, 255, 255), taken=taken)
    assert overlap((0, 0, 10, 10), (5, 5, 15, 15)) and not overlap((0, 0, 10, 10), (10, 0, 20, 10))


def test_band_keeps_the_strip_with_the_boxes():
    image = np.zeros((200, 50, 3), np.uint8)
    assert band(image, [row(y1=100, y2=120)], margin=10).shape[0] == 40
    assert band(image, []).shape[0] == 200


def test_yolo_rows_are_pixel_boxes(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("2 0.5 0.5 0.2 0.4\n")
    (r,) = yolo_rows(label, 100, 50)
    assert r["class"] == "2" and [float(r[k]) for k in ("x1", "y1", "x2", "y2")] == [40, 15, 60, 35]
    assert yolo_rows(tmp_path / "missing.txt", 100, 50) == []


def test_read_boxes_groups_rows_and_copes_with_crlf(tmp_path):
    path = tmp_path / "boxes.csv"
    path.write_bytes(b"image,class,labelled\r\na,1,1\r\na,0,0\r\nb,1,1\r\n")
    rows = read_boxes(path)
    assert [len(rows["a"]), len(rows["b"])] == [2, 1]
    assert rows["a"][1]["labelled"] == "0"


def test_keypoint_panel_scales_and_marks_the_points():
    crop = np.zeros((80, 80, 3), np.uint8)
    truth = np.array([[40.5, 10.5]] + [[20.5 + 5 * i, 60.5] for i in range(6)])
    panel = keypoint_panel(crop, truth, truth + 1, scale=4, numbered=False)
    assert panel.shape == (320, 320, 3)
    assert panel[42, 162].any()
    assert keypoint_error(truth, truth + [3, 4]) == 5.0


def test_pick_and_spread_are_deterministic():
    items = list(range(100))
    assert pick(items, 5, seed=1) == pick(items, 5, seed=1) == sorted(pick(items, 5, seed=1))
    assert pick(items[:3], 5) == [0, 1, 2]
    values = np.array([5.0, 1.0, 3.0, 9.0, 7.0])
    assert list(spread(values, 3)) == [1, 0, 3]


def test_index_lists_the_examples(tmp_path):
    path = write_index(tmp_path, "t", "uv run x", [{"file": "a.png", "labels": 3}], ["note"])
    text = path.read_text()
    assert "    uv run x" in text and "note" in text and "| a.png | 3 |" in text
