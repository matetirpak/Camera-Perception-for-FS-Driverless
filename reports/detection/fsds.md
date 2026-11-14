# Cone detection on simulator images

Written by `scripts/evaluate_detector.py`. Simulator images labelled from the ground truth (see
`docs/detection.md`), tested on CompetitionMap3 at dusk, a map no model saw during training (see
`configs/detection/fsds_split.yaml`). The threshold comes from val, CompetitionMapTestday2 in fog.
Precision and recall use one confidence threshold per model, the one with the best F1 on val. A
detection counts if its box overlaps a labelled cone of the same class with an IoU of at least 0.5.

## Overall

| model | threshold | precision | recall | F1 | colour accuracy | mAP50 | mAP50-95 | ms per image |
|---|---|---|---|---|---|---|---|---|
| yolo26m_fsoco | 0.29 | 0.89 | 0.88 | 0.89 | 0.973 | 0.927 | 0.700 | 7.6 |
| yolo26m_fsoco_fsds | 0.50 | 0.95 | 0.94 | 0.95 | 1.000 | 0.958 | 0.789 | 7.5 |
| yolo26n_fsoco | 0.35 | 0.86 | 0.82 | 0.84 | 0.962 | 0.871 | 0.590 | 6.3 |
| yolo26n_fsoco-2 | 0.36 | 0.89 | 0.82 | 0.85 | 0.981 | 0.881 | 0.611 | 6.3 |
| yolo26n_fsoco-2_fsds | 0.47 | 0.94 | 0.88 | 0.91 | 0.999 | 0.924 | 0.699 | 6.4 |
| yolo26n_fsoco_fsds | 0.44 | 0.94 | 0.89 | 0.91 | 0.999 | 0.925 | 0.694 | 6.4 |
| yolo26s_fsoco | 0.28 | 0.86 | 0.84 | 0.85 | 0.944 | 0.907 | 0.655 | 6.7 |
| yolo26s_fsoco_fsds | 0.41 | 0.94 | 0.93 | 0.93 | 0.999 | 0.953 | 0.752 | 6.3 |

Colour accuracy is the share of found cones that got the right class. Latency is the median over 200 runs
on one 1280x960 image (the size of the gotthard.json cameras) with PyTorch in half precision, including
pre- and postprocessing.

## Compared with Dhall et al. (2019)

Their YOLOv2 had three classes, so here both orange classes are merged and cones labelled as unknown are
ignored. Their test set was their own, so this is only a rough comparison.

| model | precision | recall |
|---|---|---|
| yolo26m_fsoco | 0.89 | 0.88 |
| yolo26m_fsoco_fsds | 0.95 | 0.94 |
| yolo26n_fsoco | 0.86 | 0.82 |
| yolo26n_fsoco-2 | 0.89 | 0.82 |
| yolo26n_fsoco-2_fsds | 0.94 | 0.88 |
| yolo26n_fsoco_fsds | 0.94 | 0.89 |
| yolo26s_fsoco | 0.86 | 0.84 |
| yolo26s_fsoco_fsds | 0.94 | 0.93 |
| Dhall et al. 2019, YOLOv2 | 0.84 | 0.71 |

## Per class

AP50 from ultralytics, recall at the chosen threshold.

| model | blue | yellow | orange_big |
|---|---|---|---|
| yolo26m_fsoco | 0.90 / 0.84 | 0.92 / 0.92 | 0.96 / 0.94 |
| yolo26m_fsoco_fsds | 0.95 / 0.92 | 0.97 / 0.95 | 0.95 / 0.96 |
| yolo26n_fsoco | 0.86 / 0.77 | 0.87 / 0.86 | 0.89 / 0.86 |
| yolo26n_fsoco-2 | 0.87 / 0.79 | 0.88 / 0.84 | 0.90 / 0.86 |
| yolo26n_fsoco-2_fsds | 0.91 / 0.87 | 0.93 / 0.89 | 0.92 / 0.85 |
| yolo26n_fsoco_fsds | 0.91 / 0.88 | 0.94 / 0.90 | 0.93 / 0.83 |
| yolo26s_fsoco | 0.88 / 0.77 | 0.90 / 0.90 | 0.94 / 0.91 |
| yolo26s_fsoco_fsds | 0.94 / 0.92 | 0.96 / 0.94 | 0.96 / 0.95 |

## Recall by distance

Distance from the camera to the cone.

| model | 0-10 m | 10-20 m | 20-30 m | 30-40 m | 40-60 m | 60+ m |
|---|---|---|---|---|---|---|
| yolo26m_fsoco | 1.00 | 0.99 | 0.97 | 0.92 | 0.78 | 0.28 |
| yolo26m_fsoco_fsds | 1.00 | 1.00 | 1.00 | 0.98 | 0.91 | 0.47 |
| yolo26n_fsoco | 1.00 | 0.98 | 0.92 | 0.84 | 0.63 | 0.15 |
| yolo26n_fsoco-2 | 1.00 | 0.99 | 0.94 | 0.86 | 0.59 | 0.13 |
| yolo26n_fsoco-2_fsds | 1.00 | 1.00 | 0.98 | 0.93 | 0.76 | 0.22 |
| yolo26n_fsoco_fsds | 1.00 | 1.00 | 0.98 | 0.94 | 0.81 | 0.25 |
| yolo26s_fsoco | 1.00 | 0.96 | 0.90 | 0.86 | 0.72 | 0.22 |
| yolo26s_fsoco_fsds | 1.00 | 1.00 | 0.99 | 0.97 | 0.88 | 0.44 |
| cones | 2150 | 2425 | 2099 | 2107 | 2284 | 923 |

## Recall by box height

Height of the labelled box in pixels at the network input.

| model | 0-8 px | 8-16 px | 16-32 px | 32-64 px | 64+ px |
|---|---|---|---|---|---|
| yolo26m_fsoco | 0.62 | 0.95 | 0.99 | 1.00 | 1.00 |
| yolo26m_fsoco_fsds | 0.77 | 0.99 | 1.00 | 1.00 | 1.00 |
| yolo26n_fsoco | 0.47 | 0.88 | 0.97 | 1.00 | 1.00 |
| yolo26n_fsoco-2 | 0.43 | 0.90 | 0.98 | 1.00 | 1.00 |
| yolo26n_fsoco-2_fsds | 0.59 | 0.95 | 0.99 | 1.00 | 1.00 |
| yolo26n_fsoco_fsds | 0.64 | 0.96 | 0.99 | 1.00 | 1.00 |
| yolo26s_fsoco | 0.56 | 0.89 | 0.94 | 1.00 | 1.00 |
| yolo26s_fsoco_fsds | 0.74 | 0.98 | 1.00 | 1.00 | 1.00 |
| cones | 3084 | 3675 | 2721 | 1574 | 934 |

## By map

Precision / recall.

| model | CompetitionMap3 |
|---|---|
| yolo26m_fsoco | 0.89 / 0.88 |
| yolo26m_fsoco_fsds | 0.95 / 0.94 |
| yolo26n_fsoco | 0.86 / 0.82 |
| yolo26n_fsoco-2 | 0.89 / 0.82 |
| yolo26n_fsoco-2_fsds | 0.94 / 0.88 |
| yolo26n_fsoco_fsds | 0.94 / 0.89 |
| yolo26s_fsoco | 0.86 / 0.84 |
| yolo26s_fsoco_fsds | 0.94 / 0.93 |

## Recall by tag

| model | truncated | no tag |
|---|---|---|
| yolo26m_fsoco | 0.60 | 0.90 |
| yolo26m_fsoco_fsds | 0.67 | 0.95 |
| yolo26n_fsoco | 0.49 | 0.83 |
| yolo26n_fsoco-2 | 0.47 | 0.84 |
| yolo26n_fsoco-2_fsds | 0.52 | 0.90 |
| yolo26n_fsoco_fsds | 0.53 | 0.91 |
| yolo26s_fsoco | 0.56 | 0.85 |
| yolo26s_fsoco_fsds | 0.63 | 0.94 |
| cones | 580 | 11408 |
