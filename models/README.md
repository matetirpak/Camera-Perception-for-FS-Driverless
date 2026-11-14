# models

Not tracked by git. Pretrained starting points and trained weights, one folder each:

- `yolo26/` the COCO weights the detectors start from (`yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt` from
  https://github.com/ultralytics/assets/releases)
- `detection/` the detectors, copied here by `scripts/train_detector.py` and named after their config (a name that
  is taken becomes `<name>-2`). The pipelines use `yolo26s_fsoco_fsds.pt`. `scripts/export_engine.py` writes a
  TensorRT engine (`.engine`, with an `.onnx` on the way) next to the weights.
- `keypoints/` the keypoint network, `fsds.pt` from `make keypoints`

A `lidar_colour/` folder for the LiDAR colour classifier will follow with the LiDAR pipeline.
