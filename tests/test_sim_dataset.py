import csv

import cv2
import numpy as np
import pytest
import yaml

from driverless_perception.common.types import ConeColor
from driverless_perception.sim.autolabel import LabelPolicy
from driverless_perception.sim.dataset import export_captures, read_split


def fake_capture(root, name, map_name):
    capture = root / name
    (capture / "images").mkdir(parents=True)
    run = {"name": name, "map": map_name, "cameras": {"cam": {"width": 64, "height": 48, "fov_deg": 60.0}}}
    (capture / "run.yaml").write_text(yaml.safe_dump(run))
    images = [f"{name}_00000_cam", f"{name}_00001_cam"]
    for image in images:
        cv2.imwrite(str(capture / "images" / f"{image}.jpg"), np.zeros((48, 64, 3), np.uint8))
    with open(capture / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "camera"])
        w.writerows([[image, "cam"] for image in images])
    with open(capture / "boxes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "x1", "y1", "x2", "y2", "distance", "visible", "contrast", "truncated", "cone",
                    "labelled"])
        # a clear cone and one lost in fog in the first image, nothing in the second
        w.writerow([images[0], int(ConeColor.BLUE), 16, 12, 32, 36, 8.0, 0.95, 60.0, 0, 3, 1])
        w.writerow([images[0], int(ConeColor.YELLOW), 40, 20, 44, 26, 55.0, 0.9, 5.0, 0, 7, 1])
    return capture


def test_export_applies_the_policy_and_splits_by_map(tmp_path):
    captures = [fake_capture(tmp_path / "captures", "a", "TrainingMap"),
                fake_capture(tmp_path / "captures", "b", "CompetitionMap3")]
    split = {"TrainingMap": "train", "CompetitionMap3": "test"}
    out = tmp_path / "dataset"
    summary = export_captures(captures, split, out, LabelPolicy(min_contrast=15))

    label = (out / "labels" / "train" / "a_00000_cam.txt").read_text().split()
    assert label == [str(int(ConeColor.BLUE)), "0.375000", "0.500000", "0.250000", "0.500000"]
    assert (out / "labels" / "train" / "a_00001_cam.txt").read_text() == ""
    assert (out / "images" / "test" / "b_00000_cam.jpg").exists()
    # hard linked, not copied
    assert (out / "images" / "train" / "a_00000_cam.jpg").stat().st_ino == (tmp_path / "captures" / "a" / "images"
                                                                            / "a_00000_cam.jpg").stat().st_ino
    assert summary["train"]["images"] == 2 and summary["train"]["labels"][ConeColor.BLUE] == 1
    assert summary["train"]["labels"][ConeColor.YELLOW] == 0
    with open(out / "boxes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["split"], r["labelled"]) for r in rows] == [("train", "1"), ("train", "0"), ("test", "1"), ("test", "0")]
    assert yaml.safe_load((out / "fsds.yaml").read_text())["test"] == "images/test"

    with pytest.raises(ValueError):
        export_captures(captures, {"TrainingMap": "train"}, tmp_path / "other", LabelPolicy())


def test_keypoints_for_fully_visible_small_cones(tmp_path):
    capture = fake_capture(tmp_path / "captures", "a", "TrainingMap")
    run = yaml.safe_load((capture / "run.yaml").read_text())
    run["mounts"] = {"cam": {"X": 0.0, "Y": 0.0, "Z": 1.0, "Roll": 0, "Pitch": 0, "Yaw": 0}}
    (capture / "run.yaml").write_text(yaml.safe_dump(run))
    # a level camera 1 m above the ground at the origin, the blue cone (index 3) 8 m ahead
    with open(capture / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "camera", "x", "y", "z", "qw", "qx", "qy", "qz"])
        w.writerows([[f"a_0000{i}_cam", "cam", 0, 0, 1.25, 1, 0, 0, 0] for i in range(2)])
    with open(capture / "cones.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cone", "x", "y", "class"])
        w.writerows([[i, 8.0, 0.0 if i == 3 else 5.0, 0] for i in range(8)])
    out = tmp_path / "dataset"
    summary = export_captures([capture], {"TrainingMap": "train"}, out, LabelPolicy(min_contrast=15))
    with open(out / "keypoints.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    # only the clear blue cone, the yellow one failed the contrast
    assert len(rows) == summary["train"]["keypoint_cones"] == 1
    u = [float(rows[0][f"u{i}"]) for i in range(7)]
    v = [float(rows[0][f"v{i}"]) for i in range(7)]
    assert u[1] < u[0] < u[4] and v[0] < v[1] < v[2] < v[3]
    assert abs(u[0] - 32.0) < 0.5


def test_read_split(tmp_path):
    path = tmp_path / "split.yaml"
    path.write_text("train: [TrainingMap, CompetitionMap1]\nval: [CompetitionMapTestday2]\ntest: [CompetitionMap3]\n")
    assert read_split(path)["CompetitionMap3"] == "test"
    path.write_text("train: [TrainingMap]\nval: [TrainingMap]\ntest: []\n")
    with pytest.raises(ValueError):
        read_split(path)
