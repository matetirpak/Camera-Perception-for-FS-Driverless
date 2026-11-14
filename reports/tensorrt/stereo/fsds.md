# Stereo cone positions on simulator images

Written by `scripts/evaluate_stereo.py`. Test map CompetitionMap3 at dusk, left and right image of each
frame with the pose each was rendered from. Detector `yolo26s_fsoco_fsds.engine` (threshold
0.41), keypoint network `fsds`. The error is the horizontal distance
between estimated and true cone base, split into the range part (estimated minus true distance from the
camera, median = bias, standard deviation = spread) and the lateral part across the line of sight. Mono
and stereo are compared on the cones where both gave a position. The pipeline only reports cones up to
10 m, the rows beyond show why.

## Ground truth boxes

| distance | cones | PnP | stereo | matches | on cone | median mono / stereo | 90% mono / stereo | range bias mono / stereo | range spread mono / stereo | lateral RMSE mono / stereo |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 175 | 175 | 160 | 67 | 49 | 0.05 / 0.03 m | 0.14 / 0.06 m | 0.04 / 0.02 m | 0.07 / 0.02 m | 0.00 / 0.03 m |
| 5-10 m | 543 | 543 | 530 | 46 | 36 | 0.08 / 0.03 m | 0.21 / 0.07 m | 0.03 / 0.01 m | 0.12 / 0.04 m | 0.00 / 0.02 m |
| 10-15 m | 440 | 440 | 434 | 24 | 20 | 0.17 / 0.07 m | 0.45 / 0.17 m | -0.02 / -0.01 m | 0.27 / 0.11 m | 0.00 / 0.02 m |
| 15-20 m | 166 | 166 | 154 | 15 | 11 | 0.30 / 0.12 m | 0.71 / 0.30 m | -0.10 / 0.00 m | 0.44 / 0.26 m | 0.00 / 0.03 m |

## Detector boxes (whole pipeline)

| distance | cones | PnP | stereo | matches | on cone | median mono / stereo | 90% mono / stereo | range bias mono / stereo | range spread mono / stereo | lateral RMSE mono / stereo |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 207 | 207 | 170 | 64 | 43 | 0.06 / 0.03 m | 0.15 / 0.07 m | 0.05 / 0.02 m | 0.09 / 0.02 m | 0.00 / 0.04 m |
| 5-10 m | 554 | 554 | 535 | 46 | 36 | 0.08 / 0.04 m | 0.22 / 0.07 m | 0.03 / 0.02 m | 0.13 / 0.04 m | 0.00 / 0.02 m |
| 10-15 m | 461 | 461 | 445 | 24 | 20 | 0.16 / 0.07 m | 0.44 / 0.17 m | -0.02 / -0.01 m | 0.27 / 0.11 m | 0.00 / 0.03 m |
| 15-20 m | 146 | 146 | 131 | 16 | 12 | 0.30 / 0.12 m | 0.66 / 0.34 m | -0.09 / 0.00 m | 0.42 / 0.31 m | 0.00 / 0.03 m |

## Stage by stage (ground truth boxes)

How each step did its job: the propagated box against the true box of the same cone in the right
image, what the matching found, and how far the points that ended up on the cone lie from its axis.

| distance | cones | box IoU | box holds the cone | features | after ratio test | matches | on cone | epipolar | point error | median vs mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 175 | 0.40 | 100% | 354 | 99 | 67 | 49 | 0.54 px | 0.07 m | 0.01 m |
| 5-10 m | 543 | 0.36 | 100% | 169 | 57 | 46 | 36 | 0.40 px | 0.10 m | 0.02 m |
| 10-15 m | 440 | 0.32 | 100% | 64 | 27 | 24 | 20 | 0.36 px | 0.17 m | 0.04 m |
| 15-20 m | 166 | 0.30 | 100% | 37 | 17 | 15 | 11 | 0.21 px | 0.22 m | 0.07 m |

## Counts and time

- detector: 1418 usable boxes in 210 left images, 50 of them matched no labelled cone
- keypoints, PnP and stereo together: 17.2 ms per frame, 2.5 ms per box, without the detector
