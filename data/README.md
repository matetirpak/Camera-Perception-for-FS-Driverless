# data

Not tracked by git. `make data` (`scripts/link_data.sh`) links the datasets into `raw/`:

- FSOCO bounding boxes (real camera images of cones), downloaded from https://fsoco.github.io/fsoco-dataset/download
  and found in `fsoco/data/fsoco_bounding_boxes_train` two levels above the repository or wherever `FSOCO_DIR`
  points
- ConeScenes (real LiDAR scans with intensity), found the same way or through `CONESCENES_DIR`; nothing uses it yet
- the cone maps that come with FSDS

`interim/` holds intermediate results such as the auto labelled simulator captures (`fsds_captures/`, written by
`scripts/autolabel_fsds.py`) and the standing depth test captures (`fsds_static/`). `processed/` holds the datasets
that are ready for training: `fsoco_yolo/`, `fsds_yolo/` and the keypoint crops in `fsds_keypoints/`.
