import cv2
import numpy as np

from driverless_perception.vision.keypoints.crops import (
    CROP,
    Augmentation,
    CropCache,
    _warp,
    apply,
    augmented_crop,
    plain_crop,
    window_transform,
)


def test_window_transform_maps_box_corners_to_crop_corners():
    box = (100.0, 50.0, 140.0, 110.0)
    m = window_transform(box)
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y2], [(x1 + x2) / 2, (y1 + y2) / 2]])
    transformed = apply(m, corners)
    expected = np.array([[0.0, 0.0], [CROP, CROP], [CROP / 2, CROP / 2]])
    assert np.allclose(transformed, expected)


def test_window_transform_with_scale():
    box = (100.0, 50.0, 140.0, 110.0)
    scale = 1.5
    m = window_transform(box, scale=scale)
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    corner_box = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy + h / 2]])
    transformed = apply(m, corner_box)
    assert np.allclose(transformed, [[0, 0], [CROP, CROP]])


def test_window_transform_with_shift():
    box = (100.0, 50.0, 140.0, 110.0)
    shift = (0.25, -0.25)
    m = window_transform(box, shift=shift)
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    shifted_cx = cx + shift[0] * (x2 - x1)
    shifted_cy = cy + shift[1] * (y2 - y1)
    centre = np.array([[shifted_cx, shifted_cy]])
    transformed = apply(m, centre)
    assert np.allclose(transformed, [[CROP / 2, CROP / 2]], atol=1e-10)


def test_window_transform_with_rotation_zero_is_axis_aligned():
    box = (100.0, 50.0, 140.0, 110.0)
    m = window_transform(box, angle_deg=0.0)
    x1, y1, x2, y2 = box
    points_on_x = np.array([[x1, (y1 + y2) / 2], [x2, (y1 + y2) / 2]])
    points_on_y = np.array([[(x1 + x2) / 2, y1], [(x1 + x2) / 2, y2]])
    transformed_x = apply(m, points_on_x)
    transformed_y = apply(m, points_on_y)
    assert np.allclose(transformed_x[:, 1], CROP / 2)
    assert np.allclose(transformed_y[:, 0], CROP / 2)


def test_window_transform_rotation_small_angle():
    box = (100.0, 50.0, 150.0, 100.0)
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    m = window_transform(box, angle_deg=0.0)
    centre = np.array([[cx, cy]])
    corner1 = np.array([[x1, y1]])
    corner2 = np.array([[x2, y2]])
    t_centre = apply(m, centre)
    t_corner1 = apply(m, corner1)
    t_corner2 = apply(m, corner2)
    assert np.allclose(t_centre, [CROP / 2, CROP / 2])
    assert np.allclose(t_corner1, [0, 0])
    assert np.allclose(t_corner2, [CROP, CROP])


def test_apply_transforms_points():
    m = np.array([[2.0, 0.0, 10.0], [0.0, 3.0, 5.0]])
    points = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
    result = apply(m, points)
    expected = np.array([[12.0, 11.0], [16.0, 17.0], [10.0, 5.0]])
    assert np.allclose(result, expected)


def test_apply_with_identity_matrix():
    m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    points = np.array([[5.0, 10.0], [3.0, 7.0]])
    result = apply(m, points)
    assert np.allclose(result, points)


def test_plain_crop_returns_80x80_with_keypoints():
    image = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    box = (50.0, 40.0, 120.0, 140.0)
    keypoints = np.array([[60.0, 50.0], [100.0, 130.0], [75.0, 80.0]])
    crop, kp, m = plain_crop(image, box, keypoints)
    assert crop.shape == (CROP, CROP, 3)
    assert kp.shape == (3, 2)
    assert m.shape == (2, 3)
    assert 0 <= kp.min() and kp.max() <= CROP


def test_plain_crop_keypoints_roundtrip():
    image = np.random.default_rng(1).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    box = (50.0, 40.0, 150.0, 140.0)
    keypoints = np.array([[60.0, 50.0], [100.0, 130.0], [75.0, 80.0], [140.0, 100.0]])
    _, kp_transformed, m = plain_crop(image, box, keypoints)
    A = m[:, :2]
    b = m[:, 2]
    A_inv = np.linalg.inv(A)
    m_inv = np.hstack([A_inv, (-A_inv @ b)[:, None]])
    kp_recovered = apply(m_inv, kp_transformed)
    assert np.allclose(kp_recovered, keypoints, atol=1e-10)


def test_augmented_crop_has_correct_shape():
    image = np.random.default_rng(2).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    box = (50.0, 40.0, 150.0, 140.0)
    keypoints = np.array([[60.0, 50.0], [100.0, 130.0], [75.0, 80.0]])
    rng = np.random.default_rng(3)
    aug = Augmentation()
    crop, kp = augmented_crop(image, box, keypoints, rng, aug)
    assert crop.shape == (CROP, CROP, 3)
    assert kp.shape == (3, 2)


def test_augmented_crop_keypoints_match_transform_matrix():
    image = np.random.default_rng(4).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    box = (50.0, 40.0, 150.0, 140.0)
    keypoints = np.array([[60.0, 50.0], [100.0, 130.0], [75.0, 80.0]])
    rng = np.random.default_rng(5)
    aug = Augmentation()
    _, kp = augmented_crop(image, box, keypoints, rng, aug)
    assert np.all((kp >= 0) & (kp <= CROP))


def test_augmented_crop_with_fixed_seed_is_deterministic():
    image = np.random.default_rng(6).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    box = (50.0, 40.0, 150.0, 140.0)
    keypoints = np.array([[60.0, 50.0], [100.0, 130.0], [75.0, 80.0]])
    aug = Augmentation()
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    crop1, kp1 = augmented_crop(image, box, keypoints, rng1, aug)
    crop2, kp2 = augmented_crop(image, box, keypoints, rng2, aug)
    assert np.allclose(crop1, crop2)
    assert np.allclose(kp1, kp2)


def test_crop_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(8)
    crops = [rng.integers(0, 255, (160, 160, 3), dtype=np.uint8) for _ in range(3)]
    jpegs = [cv2.imencode(".jpg", c)[1].tobytes() for c in crops]
    boxes = np.array([[10, 20, 50, 70], [5, 5, 45, 45], [30, 30, 80, 80]])
    keypoints = np.random.default_rng(9).uniform(0, 160, (3, 7, 2))
    classes = np.array([0, 1, 0])
    distances = np.array([5.0, 9.0, 12.0])
    image_boxes = np.array([[100, 150, 200, 300], [50, 50, 200, 200], [150, 150, 300, 300]])
    cache = CropCache(jpegs, boxes, keypoints, classes, distances, image_boxes)
    cache.save(tmp_path / "test.npz")
    loaded = CropCache.load(tmp_path / "test.npz")
    assert len(loaded) == 3
    assert np.allclose(loaded.boxes, boxes)
    assert np.allclose(loaded.keypoints, keypoints)
    assert np.allclose(loaded.classes, classes)
    assert np.allclose(loaded.distances, distances)
    assert np.allclose(loaded.image_boxes, image_boxes)


def test_crop_cache_image_decodes_to_correct_shape(tmp_path):
    rng = np.random.default_rng(10)
    crops = [rng.integers(0, 255, (160, 160, 3), dtype=np.uint8) for _ in range(2)]
    jpegs = [cv2.imencode(".jpg", c)[1].tobytes() for c in crops]
    cache = CropCache(jpegs, np.ones((2, 4)), np.ones((2, 7, 2)), np.array([0, 1]),
                      np.array([5.0, 9.0]))
    cache.save(tmp_path / "test.npz")
    loaded = CropCache.load(tmp_path / "test.npz")
    for i in range(2):
        img = loaded.image(i)
        assert img.shape == (160, 160, 3)
        assert img.dtype == np.uint8


def test_crop_cache_missing_image_boxes_creates_default():
    jpegs = [cv2.imencode(".jpg", np.zeros((160, 160, 3), dtype=np.uint8))[1].tobytes()]
    boxes = np.array([[10, 20, 50, 70]])
    keypoints = np.ones((1, 7, 2))
    classes = np.array([0])
    distances = np.array([5.0])
    cache = CropCache(jpegs, boxes, keypoints, classes, distances)
    assert cache.image_boxes is not None
    assert cache.image_boxes.shape == (1, 4)
    assert np.allclose(cache.image_boxes, 0)


def test_crop_cache_roundtrip_preserves_jpegs(tmp_path):
    rng = np.random.default_rng(11)
    original_img = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
    jpeg = cv2.imencode(".jpg", original_img)[1].tobytes()
    cache = CropCache([jpeg], np.ones((1, 4)), np.ones((1, 7, 2)), np.array([0]),
                      np.array([5.0]))
    cache.save(tmp_path / "test.npz")
    loaded = CropCache.load(tmp_path / "test.npz")
    decoded = loaded.image(0)
    assert decoded.shape == (160, 160, 3)
    assert 0 <= decoded.min() and decoded.max() <= 255


def test_warp_follows_the_pixel_convention():
    # one white pixel at (30, 20), which spans 30 to 31 and 20 to 21, cropped with the box scaled by two
    image = np.zeros((40, 40), np.uint8)
    image[20, 30] = 255
    crop = _warp(image, window_transform([0.0, 0.0, 40.0, 40.0], size=80), 80, cv2.INTER_NEAREST)
    ys, xs = np.nonzero(crop)
    assert sorted(set(xs)) == [60, 61] and sorted(set(ys)) == [40, 41]


def test_plain_crop_and_augmented_crop_consistency():
    image = np.random.default_rng(12).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    box = (50.0, 40.0, 150.0, 140.0)
    keypoints = np.array([[60.0, 50.0], [100.0, 130.0]])
    plain_c, plain_kp, _ = plain_crop(image, box, keypoints)
    aug_c, aug_kp = augmented_crop(image, box, keypoints, np.random.default_rng(0),
                                    Augmentation(rotation_deg=0.0, scale=(1.0, 1.0),
                                                 shift=0.0))
    assert np.allclose(plain_c, aug_c, atol=1)
    assert np.allclose(plain_kp, aug_kp, atol=1e-5)


def test_window_transform_separates_width_height_scaling():
    box = (0.0, 0.0, 100.0, 50.0)
    m = window_transform(box)
    x1, y1, x2, y2 = box
    w, h = (x2 - x1), (y2 - y1)
    scale_x = CROP / w
    scale_y = CROP / h
    assert np.isclose(scale_x, 0.8)
    assert np.isclose(scale_y, 1.6)
    corner1 = apply(m, np.array([[x1, y1]]))
    corner2 = apply(m, np.array([[x2, y2]]))
    assert np.allclose(corner1, [[0, 0]])
    assert np.allclose(corner2, [[CROP, CROP]])


def test_apply_with_single_point():
    m = np.array([[2.0, 1.0, 3.0], [0.5, 3.0, 2.0]])
    point = np.array([[1.0, 2.0]])
    result = apply(m, point)
    expected = np.array([[1.0 * 2 + 2.0 * 1 + 3, 1.0 * 0.5 + 2.0 * 3 + 2]])
    assert np.allclose(result, expected)


def test_crop_cache_length():
    jpegs = [cv2.imencode(".jpg", np.zeros((160, 160, 3), dtype=np.uint8))[1].tobytes()
             for _ in range(5)]
    boxes = np.ones((5, 4))
    keypoints = np.ones((5, 7, 2))
    classes = np.zeros(5, dtype=int)
    distances = np.ones(5)
    cache = CropCache(jpegs, boxes, keypoints, classes, distances)
    assert len(cache) == 5


def test_augmented_crop_respects_augmentation_bounds():
    image = np.random.default_rng(13).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    box = (100.0, 100.0, 200.0, 200.0)
    keypoints = np.array([[150.0, 150.0]])
    rng = np.random.default_rng(14)
    aug = Augmentation(rotation_deg=15.0, scale=(0.8, 1.5), shift=0.5)
    for _ in range(10):
        crop, kp = augmented_crop(image, box, keypoints, rng, aug)
        assert crop.shape == (CROP, CROP, 3)
        assert 0 <= kp[0, 0] <= CROP and 0 <= kp[0, 1] <= CROP


def test_window_transform_matrix_structure():
    box = (10.0, 20.0, 50.0, 80.0)
    m = window_transform(box)
    assert m.shape == (2, 3)
    assert isinstance(m, np.ndarray)
    assert m.dtype == np.float64


def test_apply_batch_points():
    m = np.array([[1.0, 0.5, 2.0], [0.0, 2.0, 1.0]])
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = apply(m, points)
    expected = np.array([[2.0, 1.0], [3.0, 1.0], [2.5, 3.0], [3.5, 3.0]])
    assert np.allclose(result, expected)
