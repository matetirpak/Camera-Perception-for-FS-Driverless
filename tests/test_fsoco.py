import json

import cv2
import numpy as np
import pytest

from driverless_perception.common.types import ConeColor
from driverless_perception.vision.detection.fsoco import (
    BORDER,
    convert_image,
    group_split,
    image_origin,
    read_annotation,
    read_split,
    yolo_lines,
)


def write_sample(root, team, name, width, height, objects):
    (root / team / "ann").mkdir(parents=True, exist_ok=True)
    (root / team / "img").mkdir(parents=True, exist_ok=True)
    ann = root / team / "ann" / f"{name}.json"
    ann.write_text(json.dumps({"size": {"width": width, "height": height}, "objects": objects}))
    return ann


def box(cls, x1, y1, x2, y2, tags=()):
    return {"classTitle": cls, "points": {"exterior": [[x1, y1], [x2, y2]]}, "tags": [{"name": t} for t in tags]}


def test_read_annotation_removes_the_border(tmp_path):
    ann = write_sample(tmp_path, "amz", "a.png", 1000 + 2 * BORDER, 500 + 2 * BORDER, [
        box("blue_cone", BORDER + 10, BORDER + 20, BORDER + 30, BORDER + 60, ["truncated"]),
        box("large_orange_cone", BORDER + 990, BORDER + 400, BORDER + 1010, BORDER + 520),
        box("yellow_cone", 5, 5, 100, 100),
    ])
    s = read_annotation(ann)
    assert (s.team, s.name, s.width, s.height) == ("amz", "amz_a", 1000, 500)
    assert s.image == tmp_path / "amz" / "img" / "a.png"
    # the yellow box lies completely in the border and is dropped, the orange one is clipped
    assert len(s.boxes) == 2
    b = s.boxes[0]
    assert (b.color, b.x1, b.y1, b.x2, b.y2, b.tags) == (ConeColor.BLUE, 10, 20, 30, 60, ("truncated",))
    assert (s.boxes[1].x2, s.boxes[1].y2) == (1000, 500)


def test_yolo_lines_use_cone_color_ids(tmp_path):
    ann = write_sample(tmp_path, "t", "a.png", 200 + 2 * BORDER, 100 + 2 * BORDER, [
        box("yellow_cone", BORDER + 50, BORDER + 25, BORDER + 150, BORDER + 75),
    ])
    assert yolo_lines(read_annotation(ann)) == [f"{int(ConeColor.YELLOW)} 0.500000 0.500000 0.500000 0.500000"]


def test_convert_image_crops_and_shrinks(tmp_path):
    image = np.zeros((600 + 2 * BORDER, 3000 + 2 * BORDER, 3), np.uint8)
    image[BORDER:-BORDER, BORDER:-BORDER] = 200
    src, dst = tmp_path / "in.png", tmp_path / "out" / "in.jpg"
    cv2.imwrite(str(src), image)
    convert_image(src, dst, (3000, 600), long_side=1500)
    out = cv2.imread(str(dst))
    assert out.shape == (300, 1500, 3)
    assert out.min() > 190  # no black border left
    with pytest.raises(ValueError):
        convert_image(src, dst, (3000, 601))


def test_image_origin_comes_from_the_file_name():
    assert image_origin("amz_00805.png") == "amz_wide"
    assert image_origin("amz_00012.png") == "amz"
    assert image_origin("BME_0042.jpg") == "bme"
    assert image_origin("ka_00181.png") == "ka"
    with pytest.raises(ValueError):
        image_origin("image.png")


def test_group_split_places_fixed_groups_first():
    rng = np.random.default_rng(1)
    counts = {f"g{i}": np.concatenate([[rng.integers(100, 400)], rng.integers(0, 3000, 5)]) for i in range(30)}
    split = group_split(counts, val=("g1",), test=("g3",), trials=500)
    assert "g1" in split["val"] and "g3" in split["test"]
    groups = [g for name in ("train", "val", "test") for g in split[name]]
    assert sorted(groups) == sorted(counts)
    assert split == group_split(counts, val=("g1",), test=("g3",), trials=500)
    total = sum(c[0] for c in counts.values())
    for name in ("val", "test"):
        share = sum(counts[g][0] for g in split[name]) / total
        assert 0.05 < share < 0.2
    with pytest.raises(ValueError):
        group_split(counts, val=("g1",), test=("g1",))


def test_read_split_rejects_duplicates(tmp_path):
    path = tmp_path / "split.yaml"
    path.write_text("train: [a, b]\nval: [c]\ntest: [a]\n")
    with pytest.raises(ValueError):
        read_split(path)
