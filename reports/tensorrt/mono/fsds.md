# Mono cone positions on simulator images

Written by `scripts/evaluate_mono.py`. Test map CompetitionMap3 at dusk. Keypoint network `fsds`,
detector `yolo26s_fsoco_fsds.engine`, PnP `upright`, detector boxes within 2 px of the left or right border left out.
The error is the distance between the estimated and the true cone base,
horizontally and in 3D, and split into its parts: range (estimated minus true distance from the camera,
negative means too close), lateral (across the line of sight) and vertical. Dhall et al. (2019) report
about 0.5 m at 10 m and 1 m at 16 m.

## Keypoint network

On 3881 test crops the mean squared error is 1.72 px^2 in the 80x80 crop
(median 0.96 px per point, 0.45 px back in the camera
image). Dhall et al. reached 3.78 px^2 on their test crops.

| keypoint | top | left stripe top | left stripe bottom | left bottom | right stripe top | right stripe bottom | right bottom |
|---|---|---|---|---|---|---|---|
| median error | 1.04 px | 0.85 px | 0.87 px | 1.11 px | 0.88 px | 0.90 px | 1.14 px |

| distance | crops | box height | median error | in the image | MSE |
|---|---|---|---|---|---|
| 0-5 m | 434 | 98 px | 0.95 px | 1.20 px | 1.50 px^2 |
| 5-10 m | 1521 | 50 px | 0.86 px | 0.54 px | 1.22 px^2 |
| 10-15 m | 1263 | 29 px | 0.96 px | 0.35 px | 1.66 px^2 |
| 15-20 m | 622 | 22 px | 1.30 px | 0.36 px | 3.07 px^2 |
| 20-30 m | 41 | 20 px | 1.33 px | 0.32 px | 4.18 px^2 |

The cross ratio of the cone model is 1.4217. The network's keypoints give
1.4248 (median, typically 0.0034 away from the
model), the true keypoints 1.4263.


## Exact keypoints (PnP and cone model only)

| distance | cones | median | RMSE | 90% | 3D median | range bias | range spread | lateral RMSE | vertical bias |
|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 462 | 0.00 m | 0.00 m | 0.00 m | 0.00 m | -0.00 m | 0.00 m | 0.00 m | 0.00 m |
| 5-10 m | 1500 | 0.00 m | 0.00 m | 0.00 m | 0.00 m | -0.00 m | 0.00 m | 0.00 m | 0.00 m |
| 10-15 m | 1262 | 0.00 m | 0.00 m | 0.00 m | 0.00 m | -0.00 m | 0.00 m | 0.00 m | 0.00 m |
| 15-20 m | 616 | 0.00 m | 0.00 m | 0.00 m | 0.00 m | -0.00 m | 0.00 m | 0.00 m | 0.00 m |
| 20-30 m | 41 | 0.00 m | 0.00 m | 0.00 m | 0.00 m | -0.00 m | 0.00 m | 0.00 m | 0.00 m |

## Ground truth boxes, keypoint network

| distance | cones | median | RMSE | 90% | 3D median | range bias | range spread | lateral RMSE | vertical bias |
|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 462 | 0.06 m | 0.09 m | 0.15 m | 0.06 m | 0.05 m | 0.08 m | 0.00 m | -0.01 m |
| 5-10 m | 1500 | 0.08 m | 0.13 m | 0.21 m | 0.08 m | 0.04 m | 0.12 m | 0.00 m | -0.01 m |
| 10-15 m | 1262 | 0.16 m | 0.26 m | 0.43 m | 0.16 m | -0.00 m | 0.26 m | 0.00 m | -0.00 m |
| 15-20 m | 616 | 0.29 m | 0.49 m | 0.80 m | 0.29 m | -0.00 m | 0.49 m | 0.00 m | -0.00 m |
| 20-30 m | 41 | 0.30 m | 0.65 m | 1.16 m | 0.30 m | 0.18 m | 0.64 m | 0.00 m | -0.01 m |

## Detector boxes, keypoint network (whole pipeline)

| distance | cones | median | RMSE | 90% | 3D median | range bias | range spread | lateral RMSE | vertical bias |
|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 427 | 0.06 m | 0.09 m | 0.15 m | 0.06 m | 0.05 m | 0.08 m | 0.00 m | -0.01 m |
| 5-10 m | 1495 | 0.08 m | 0.14 m | 0.23 m | 0.08 m | 0.04 m | 0.13 m | 0.00 m | -0.01 m |
| 10-15 m | 1285 | 0.16 m | 0.27 m | 0.44 m | 0.16 m | -0.00 m | 0.27 m | 0.00 m | -0.00 m |
| 15-20 m | 561 | 0.29 m | 0.50 m | 0.76 m | 0.29 m | -0.01 m | 0.50 m | 0.00 m | -0.00 m |
| 20-30 m | 24 | 0.39 m | 0.68 m | 1.13 m | 0.39 m | 0.17 m | 0.68 m | 0.01 m | -0.01 m |

## Solver comparison

The detector's keypoints of the whole pipeline above, solved four ways: PnP of the full pose (SQPnP, the
paper's way), an upright cone with only its base position unknown (what the pipeline uses), and that
cone held on the ground plane, once with the camera's true attitude and once with the car taken as
level. Median error and range bias per distance.

| distance | SQPnP | upright | ground, true attitude | ground, car level |
|---|---|---|---|---|
| 0-5 m | 0.05 / 0.02 m | 0.06 / 0.05 m | 0.01 / -0.01 m | 0.03 / -0.00 m |
| 5-10 m | 0.09 / -0.03 m | 0.08 / 0.04 m | 0.01 / -0.01 m | 0.10 / -0.10 m |
| 10-15 m | 0.21 / -0.16 m | 0.16 / -0.00 m | 0.03 / -0.01 m | 0.36 / -0.35 m |
| 15-20 m | 0.37 / -0.28 m | 0.29 / -0.01 m | 0.05 / -0.02 m | 0.63 / -0.56 m |
| 20-30 m | 0.46 / -0.24 m | 0.39 / 0.17 m | 0.07 / -0.02 m | 1.02 / -0.35 m |

Held on the ground the position comes from where the cone stands in the image rather than from its size,
which is only better while the attitude is known to about a tenth of a degree. The simulator has a flat
ground and gives the exact attitude, a car doesn't, so the pipeline uses the upright cone.

## Counts

- exact: 3881 cones, PnP failed for 0
- boxes: 3881 cones, PnP failed for 0
- detector: 3792 cones, PnP failed for 0
- detector: 3823 usable boxes, 31 of them matched no labelled cone
