import numpy as np
import pytest
import torch

from driverless_perception.common.cone_model import cross_ratio_3d
from driverless_perception.vision.keypoints.crops import (
    CROP,
    Augmentation,
    CropCache,
    augmented_crop,
    plain_crop,
    window_transform,
)
from driverless_perception.vision.keypoints.network import KeypointNet, cross_ratio, keypoint_loss, to_tensor


def test_plain_crop_maps_the_box_onto_the_crop():
    box = (100.0, 50.0, 140.0, 110.0)
    m = window_transform(box)
    corners = np.array([[100.0, 50.0], [140.0, 110.0], [120.0, 80.0]])
    assert np.allclose(corners @ m[:, :2].T + m[:, 2], [[0, 0], [CROP, CROP], [CROP / 2, CROP / 2]])


def test_keypoints_move_with_the_image():
    # a bright dot at a keypoint has to end up where the transformed keypoint says
    image = np.zeros((200, 200, 3), np.uint8)
    image[118:123, 128:133] = 255
    # centre of the dot, pixel i spans i to i + 1
    keypoints = np.tile([[130.5, 120.5]], (7, 1))
    rng = np.random.default_rng(3)
    for _ in range(20):
        crop, kp = augmented_crop(image, (100.0, 80.0, 150.0, 160.0), keypoints, rng, Augmentation())
        weight = crop[..., 0].astype(float)
        if weight.sum() and 6 <= kp[0, 0] < CROP - 6 and 6 <= kp[0, 1] < CROP - 6:
            vv, uu = np.mgrid[:CROP, :CROP] + 0.5
            centre = [(uu * weight).sum() / weight.sum(), (vv * weight).sum() / weight.sum()]
            assert np.allclose(centre, kp[0], atol=0.5)
    crop, kp, _ = plain_crop(image, (100.0, 80.0, 150.0, 160.0), keypoints)
    assert crop.shape == (CROP, CROP, 3) and np.allclose(kp[0], [(130.5 - 100) * 80 / 50, (120.5 - 80) * 80 / 80])


def test_cache_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (160, 160, 3), dtype=np.uint8) for _ in range(3)]
    import cv2

    cache = CropCache([cv2.imencode(".jpg", c)[1].tobytes() for c in crops], np.ones((3, 4)), np.ones((3, 7, 2)),
                      np.array([0, 1, 0]), np.array([5.0, 9.0, 12.0]))
    cache.save(tmp_path / "c.npz")
    loaded = CropCache.load(tmp_path / "c.npz")
    assert len(loaded) == 3 and loaded.image(1).shape == (160, 160, 3)
    assert loaded.distances.tolist() == [5.0, 9.0, 12.0]


def test_network_shape_and_loss():
    model = KeypointNet().eval()
    out = model(to_tensor(np.zeros((2, CROP, CROP, 3), np.uint8)))
    assert out.shape == (2, 7, 2)
    target = torch.tensor([[[40, 10], [34, 30], [32, 50], [30, 70], [46, 30], [48, 50], [50, 70]]], dtype=torch.float32)
    # a perfect prediction leaves only the cross ratio term, which is small
    perfect = keypoint_loss(target, target, cross_ratio_3d())
    assert perfect < 1e-3
    assert keypoint_loss(target + 2, target, cross_ratio_3d()) == pytest.approx(7 * 8, rel=1e-3)
    assert cross_ratio(target, (0, 1, 2, 3)).shape == (1,)
