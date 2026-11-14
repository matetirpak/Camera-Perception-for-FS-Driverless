import numpy as np

from driverless_perception.visualization.draw import caption, draw_boxes, draw_points, png, tile
from driverless_perception.visualization.figures import _centres, _label
from driverless_perception.visualization.foxglove import Annotations, _json_schema, sphere_cloud


def test_boxes_and_points_land_where_they_should():
    image = np.zeros((40, 60, 3), np.uint8)
    draw_boxes(image, [[10, 12, 20, 22]], [0], thickness=1)
    assert image[12, 10].any() and image[21, 19].any() and not image[16, 15].any()
    draw_points(image, [[30.5, 30.5]], (255, 255, 255), 1)
    assert image[30, 30].all()


def test_tiling_and_encoding():
    images = [np.full((20, 40, 3), v, np.uint8) for v in (50, 100, 150)]
    grid = tile([caption(im, "x") for im in images], 2, 80)
    assert grid.shape[1] == 2 * 80 + 3 * 6
    assert png(grid).startswith(b"\x89PNG")


def test_bin_labels_and_centres():
    bins = [{"d": [0, 5]}, {"d": [5, 10]}, {"d": [60, None]}]
    assert np.allclose(_centres(bins, "d"), [2.5, 7.5, 70])
    assert [_label(b["d"], " m") for b in bins] == ["0-5 m", "5-10 m", "60+ m"]


def test_annotations_and_scene_carry_the_stamp():
    ann = Annotations(1_500_000_000)
    ann.box([1, 2, 3, 4], (1.0, 0.0, 0.0, 1.0))
    ann.points([[5, 6], [7, 8]], (0.0, 1.0, 0.0, 1.0))
    ann.text(1, 2, "blue 0.90", (1.0, 1.0, 1.0, 1.0))
    assert [len(p.points) for p in ann.msg.points] == [4, 2]
    assert ann.msg.texts[0].timestamp.ToNanoseconds() == 1_500_000_000
    update = sphere_cloud(7, "fsds/map", "matches", np.zeros((3, 3)), [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
    assert len(update.entities[0].spheres) == 3
    assert update.entities[0].spheres[1].color.g == 1.0


def test_json_schema_of_a_metrics_message():
    schema = _json_schema({"recall": 1.0, "missed": None, "name": "mono", "rows": [{"lap": 1}]})
    assert schema["properties"]["recall"]["type"] == ["number", "null"]
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["rows"]["items"]["properties"]["lap"]["type"] == ["number", "null"]
