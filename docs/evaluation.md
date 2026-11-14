# Evaluation

Each stage of the camera pipelines is evaluated separately, so an error can be traced to the stage that caused it.
Every stage has a script that writes a report, a figure with the same numbers, and a tab in the Foxglove file
([results.md](results.md)).

| stage | what is measured | command | report | tab |
|---|---|---|---|---|
| dataset and labels | images and boxes per origin, map, class and distance | `prepare_fsoco.py`, `prepare_fsds.py` | `data/processed/*/boxes.csv` | Datasets |
| detection | precision, recall, F1, mAP, per class, distance, box height and tag, latency | `evaluate_detector.py` | [reports/detection/](../reports/detection/) | Detection |
| keypoints | error per keypoint, with distance, in the crop and back in the image, cross ratio | `evaluate_mono.py` | [reports/mono/fsds.md](../reports/mono/fsds.md) | Keypoints |
| mono positions | position error (3D, range, lateral, vertical) with exact keypoints, true boxes and detector boxes, and the same keypoints solved as full pose, upright cone and on the ground plane | `evaluate_mono.py` | [reports/mono/fsds.md](../reports/mono/fsds.md) | Mono |
| standing depth test | bias and repeatability on cones at known distances | `depth_test.py` | [reports/depth/fsds.md](../reports/depth/fsds.md) | Mono |
| stereo stages | propagated box against the true one, features and matches, epipolar residual, error of a single triangulated point, median against mean | `evaluate_stereo.py` | [reports/stereo/fsds.md](../reports/stereo/fsds.md) | Stereo |
| stereo positions | the same error split next to PnP on the same cones | `evaluate_stereo.py` | [reports/stereo/fsds.md](../reports/stereo/fsds.md) | Stereo |
| race | recall, precision and error per range, per lap and per image, rate, latency and the time of every stage from the simulator's stamp to the published cones | `score_run.py` | [reports/mono/](../reports/mono/), [reports/stereo/](../reports/stereo/) | Race, Race scores |
| map | landmark RMSE, missed cones and colour accuracy over a whole race | `build_map.py` | [reports/map/](../reports/map/) | Map |

## Error components

`evaluation/positions.py` measures every position the same way, from where the camera stood: range (estimated minus
true distance, negative means the cone was put too close), lateral (across the line of sight) and vertical. The 3D
error is all three together. The camera pipelines are almost exact sideways (lateral RMSE 0.00 to 0.04 m) and
nearly all of their error is in the distance, which is why the plots show the range part.

## Per cone table

The "Per cone" tab of the results file shows, for the image under the playhead, one row per cone the detector
found: its colour and score, the height of its box, what PnP and stereo made of it, how many features matched and
how many of them landed on the cone, and how far each estimate is from the nearest true cone. The images next to
it draw the same cones, so a row with a large error can be looked at directly.

## Comparison with the paper

| what | Gosala et al. / Dhall et al. | here |
|---|---|---|
| detector on real images | precision 0.84, recall 0.71 | 0.92 and 0.85 on the FSOCO test cameras (their three classes) |
| keypoint network | 3.78 px^2 on their test crops | 1.72 px^2 on simulator crops |
| mono position error | about 0.5 m at 10 m, 1 m at 16 m | 0.08 m at 5 to 10 m, 0.30 m at 15 to 20 m |
| stereo against PnP | triangulated depths scatter much less | range spread 0.04 m against 0.13 m at 5 to 10 m |
| map from stereo only | RMSE 0.25 m, with their SLAM | 0.03 m with the ground truth pose |
| map from the LiDAR | RMSE 0.20 m with all colours right (their Fig. 12; the 96% colour accuracy of Fig. 8a is a separate close range test) | the LiDAR pipeline comes later |

These are simulator numbers with the ground truth pose, so they should be better than the paper's. The relations
are what should agree (stereo steadier than PnP while driving, error growing with distance). For the map in
particular, AMZ measured the position of their own car against a laser reference at 0.2 m RMSE (Kabzan et al. 2020),
an error of the same order as their 0.25 m map that the ground truth pose here doesn't have. There is also no
published result on FSOCO to compare the detector with ([detection.md](detection.md)), so that row is only a rough
comparison.
