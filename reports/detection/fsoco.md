# Cone detection on FSOCO

Written by `scripts/evaluate_detector.py`. The test split holds the images from AMZ's own folder and from
FRT, cameras the models never saw during training (see `configs/detection/fsoco_split.yaml`).
Precision and recall use one confidence threshold per model, the one with the best F1 on val. A
detection counts if its box overlaps a labelled cone of the same class with an IoU of at least 0.5.

## Overall

| model | threshold | precision | recall | F1 | colour accuracy | mAP50 | mAP50-95 | ms per image |
|---|---|---|---|---|---|---|---|---|
| yolo26m_fsoco | 0.36 | 0.90 | 0.81 | 0.85 | 0.971 | 0.671 | 0.441 | 7.5 |
| yolo26m_fsoco_fsds | 0.57 | 0.82 | 0.65 | 0.73 | 0.926 | 0.488 | 0.295 | 7.5 |
| yolo26n_fsoco | 0.47 | 0.92 | 0.76 | 0.83 | 0.973 | 0.648 | 0.412 | 6.2 |
| yolo26n_fsoco-2 | 0.40 | 0.89 | 0.77 | 0.83 | 0.959 | 0.623 | 0.402 | 6.3 |
| yolo26n_fsoco-2_fsds | 0.55 | 0.83 | 0.59 | 0.69 | 0.910 | 0.427 | 0.238 | 6.4 |
| yolo26n_fsoco_fsds | 0.52 | 0.81 | 0.59 | 0.68 | 0.904 | 0.431 | 0.246 | 6.5 |
| yolo26s_fsoco | 0.37 | 0.89 | 0.80 | 0.84 | 0.968 | 0.654 | 0.438 | 6.3 |
| yolo26s_fsoco_fsds | 0.53 | 0.81 | 0.63 | 0.71 | 0.917 | 0.480 | 0.288 | 6.3 |

Colour accuracy is the share of found cones that got the right class. Latency is the median over 200 runs
on one 1280x960 image (the size of the gotthard.json cameras) with PyTorch in half precision, including
pre- and postprocessing.

## Compared with Dhall et al. (2019)

Their YOLOv2 had three classes, so here both orange classes are merged and cones labelled as unknown are
ignored. Their test set was their own, so this is only a rough comparison.

| model | precision | recall |
|---|---|---|
| yolo26m_fsoco | 0.92 | 0.86 |
| yolo26m_fsoco_fsds | 0.87 | 0.72 |
| yolo26n_fsoco | 0.94 | 0.81 |
| yolo26n_fsoco-2 | 0.92 | 0.83 |
| yolo26n_fsoco-2_fsds | 0.87 | 0.64 |
| yolo26n_fsoco_fsds | 0.86 | 0.65 |
| yolo26s_fsoco | 0.92 | 0.85 |
| yolo26s_fsoco_fsds | 0.84 | 0.68 |
| Dhall et al. 2019, YOLOv2 | 0.84 | 0.71 |

## Per class

AP50 from ultralytics, recall at the chosen threshold.

| model | blue | yellow | orange_big | orange_small | unknown |
|---|---|---|---|---|---|
| yolo26m_fsoco | 0.95 / 0.87 | 0.96 / 0.92 | 0.54 / 0.42 | 0.86 / 0.80 | 0.04 / 0.00 |
| yolo26m_fsoco_fsds | 0.85 / 0.77 | 0.88 / 0.85 | 0.35 / 0.46 | 0.36 / 0.00 | 0.00 / 0.00 |
| yolo26n_fsoco | 0.93 / 0.82 | 0.94 / 0.86 | 0.48 / 0.38 | 0.80 / 0.71 | 0.09 / 0.00 |
| yolo26n_fsoco-2 | 0.93 / 0.83 | 0.93 / 0.87 | 0.41 / 0.30 | 0.81 / 0.76 | 0.04 / 0.01 |
| yolo26n_fsoco-2_fsds | 0.82 / 0.69 | 0.81 / 0.79 | 0.27 / 0.32 | 0.23 / 0.00 | 0.00 / 0.00 |
| yolo26n_fsoco_fsds | 0.78 / 0.70 | 0.82 / 0.78 | 0.25 / 0.30 | 0.30 / 0.00 | 0.00 / 0.00 |
| yolo26s_fsoco | 0.95 / 0.86 | 0.95 / 0.89 | 0.50 / 0.40 | 0.83 / 0.77 | 0.05 / 0.01 |
| yolo26s_fsoco_fsds | 0.84 / 0.76 | 0.83 / 0.82 | 0.38 / 0.36 | 0.36 / 0.00 | 0.00 / 0.00 |

## Recall by box height

Height of the labelled box in pixels at the network input.

| model | 0-8 px | 8-16 px | 16-32 px | 32-64 px | 64+ px |
|---|---|---|---|---|---|
| yolo26m_fsoco | 0.13 | 0.70 | 0.91 | 0.95 | 0.86 |
| yolo26m_fsoco_fsds | 0.11 | 0.57 | 0.72 | 0.78 | 0.67 |
| yolo26n_fsoco | 0.06 | 0.57 | 0.89 | 0.94 | 0.85 |
| yolo26n_fsoco-2 | 0.09 | 0.60 | 0.89 | 0.94 | 0.86 |
| yolo26n_fsoco-2_fsds | 0.04 | 0.44 | 0.68 | 0.76 | 0.65 |
| yolo26n_fsoco_fsds | 0.08 | 0.47 | 0.67 | 0.73 | 0.64 |
| yolo26s_fsoco | 0.13 | 0.66 | 0.90 | 0.95 | 0.86 |
| yolo26s_fsoco_fsds | 0.08 | 0.52 | 0.70 | 0.77 | 0.68 |
| cones | 786 | 5469 | 6464 | 3193 | 2058 |

## By camera origin

Precision / recall.

| model | amz | frt |
|---|---|---|
| yolo26m_fsoco | 0.92 / 0.88 | 0.81 / 0.62 |
| yolo26m_fsoco_fsds | 0.82 / 0.73 | 0.83 / 0.42 |
| yolo26n_fsoco | 0.95 / 0.83 | 0.80 / 0.55 |
| yolo26n_fsoco-2 | 0.93 / 0.85 | 0.75 / 0.57 |
| yolo26n_fsoco-2_fsds | 0.83 / 0.66 | 0.83 / 0.39 |
| yolo26n_fsoco_fsds | 0.81 / 0.66 | 0.80 / 0.40 |
| yolo26s_fsoco | 0.93 / 0.87 | 0.78 / 0.60 |
| yolo26s_fsoco_fsds | 0.81 / 0.71 | 0.84 / 0.40 |

## Recall by tag

| model | truncated | knocked over | sticker band removed | no tag |
|---|---|---|---|---|
| yolo26m_fsoco | 0.71 | 0.33 | 0.94 | 0.87 |
| yolo26m_fsoco_fsds | 0.55 | 0.05 | 0.80 | 0.72 |
| yolo26n_fsoco | 0.58 | 0.28 | 0.96 | 0.82 |
| yolo26n_fsoco-2 | 0.59 | 0.30 | 0.98 | 0.83 |
| yolo26n_fsoco-2_fsds | 0.38 | 0.06 | 0.79 | 0.66 |
| yolo26n_fsoco_fsds | 0.40 | 0.06 | 0.77 | 0.66 |
| yolo26s_fsoco | 0.69 | 0.35 | 0.95 | 0.85 |
| yolo26s_fsoco_fsds | 0.51 | 0.05 | 0.79 | 0.70 |
| cones | 1950 | 1617 | 1470 | 13138 |
